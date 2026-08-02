from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import YouTubeASLPoseDataset
from .features import STREAM_JOINTS
from .masking import sample_span_mask
from .model import SkeletonBert, SkeletonBertConfig
from .utils import atomic_torch_save, choose_device, load_json, seed_everything


def move_batch(
    batch: dict, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    streams = {name: batch[name].to(device, non_blocking=True) for name in STREAM_JOINTS}
    observed = {
        name: batch[f"{name}_observed"].to(device, non_blocking=True)
        for name in STREAM_JOINTS
    }
    valid = batch["valid"].to(device, non_blocking=True)
    return streams, observed, valid


def masked_reconstruction_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    supervised: torch.Tensor,
    velocity_weight: float,
    observed: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced masked-frame coordinate and within-span velocity reconstruction."""
    if not supervised.any():
        raise ValueError("No frames selected for masked reconstruction")
    losses: dict[str, torch.Tensor] = {}
    for name in STREAM_JOINTS:
        frame_error = F.smooth_l1_loss(predictions[name], targets[name], reduction="none")
        joint_seen = (
            observed[name]
            if observed is not None
            else torch.ones_like(targets[name][..., 0], dtype=torch.bool)
        )
        coordinate_mask = supervised.unsqueeze(-1) & joint_seen
        if coordinate_mask.any():
            coordinate_loss = frame_error[
                coordinate_mask.unsqueeze(-1).expand_as(frame_error)
            ].mean()
        else:
            coordinate_loss = frame_error.new_zeros(())

        predicted_velocity = torch.diff(predictions[name][..., :2], dim=1)
        target_velocity = torch.diff(targets[name][..., :2], dim=1)
        velocity_frames = (supervised[:, 1:] & supervised[:, :-1]).unsqueeze(-1)
        velocity_seen = joint_seen[:, 1:] & joint_seen[:, :-1]
        velocity_mask = velocity_frames & velocity_seen
        if velocity_mask.any():
            velocity_error = F.smooth_l1_loss(
                predicted_velocity, target_velocity, reduction="none"
            )
            velocity_loss = velocity_error[
                velocity_mask.unsqueeze(-1).expand_as(velocity_error)
            ].mean()
        else:
            velocity_loss = coordinate_loss.new_zeros(())
        losses[name] = coordinate_loss + velocity_weight * velocity_loss

    total = torch.stack(list(losses.values())).mean()
    return total, {name: float(value.detach()) for name, value in losses.items()}


def cosine_schedule(
    optimizer: AdamW, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def validate(
    model: SkeletonBert,
    loader: DataLoader,
    device: torch.device,
    mask_probability: float,
    span_length: int,
    velocity_weight: float,
) -> float:
    model.eval()
    losses: list[float] = []
    generator = torch.Generator().manual_seed(0)
    for batch in loader:
        streams, observed, valid = move_batch(batch, device)
        mask = sample_span_mask(valid, mask_probability, span_length, generator)
        predictions = model(streams, valid, mask)
        loss, _ = masked_reconstruction_loss(
            predictions, streams, mask & valid, velocity_weight, observed
        )
        losses.append(float(loss))
    return sum(losses) / max(len(losses), 1)


def run_training(config_path: Path, resume: Path | None = None) -> None:
    config = load_json(config_path)
    seed_everything(int(config["seed"]))
    device = choose_device()
    print(f"Using device: {device}")

    data_config = config["data"]
    train_data = YouTubeASLPoseDataset(
        archive=data_config["train_archive"],
        annotations=data_config["train_annotations"],
        max_frames=data_config["max_frames"],
        training=True,
        limit_clips=data_config.get("limit_train_clips"),
    )
    val_data = YouTubeASLPoseDataset(
        archive=data_config.get("val_archive", data_config["train_archive"]),
        annotations=data_config["val_annotations"],
        max_frames=data_config["max_frames"],
        training=False,
        limit_clips=data_config.get("limit_val_clips"),
    )
    train_config = config["training"]
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=pin_memory,
    )

    model_config = SkeletonBertConfig(**config["model"])
    model = SkeletonBert(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    accumulation = int(train_config["gradient_accumulation"])
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_steps = int(
        train_config.get("scheduler_total_steps", steps_per_epoch * train_config["epochs"])
    )
    scheduler = cosine_schedule(optimizer, train_config["warmup_steps"], total_steps)
    amp_enabled = bool(train_config["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val = float(checkpoint.get("best_val", best_val))
        print(f"Resumed from epoch {start_epoch}")

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_config = config["masking"]

    for epoch in range(start_epoch, train_config["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        logged_batches = 0
        for batch_index, batch in enumerate(train_loader):
            streams, observed, valid = move_batch(batch, device)
            mask = sample_span_mask(
                valid, mask_config["mask_probability"], mask_config["mean_span_length"]
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                predictions = model(streams, valid, mask)
                loss, stream_losses = masked_reconstruction_loss(
                    predictions,
                    streams,
                    mask & valid,
                    mask_config["velocity_weight"],
                    observed,
                )
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, len(train_loader) - group_start)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
            running += float(loss.detach())
            logged_batches += 1

            last_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % accumulation == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % train_config["log_every"] == 0:
                    details = " ".join(
                        f"{key}={value:.4f}" for key, value in stream_losses.items()
                    )
                    print(
                        f"epoch={epoch + 1} step={global_step} "
                        f"loss={running / max(logged_batches, 1):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} {details}"
                    )
                    running = 0.0
                    logged_batches = 0

        val_loss = validate(
            model,
            val_loader,
            device,
            mask_config["mask_probability"],
            mask_config["mean_span_length"],
            mask_config["velocity_weight"],
        )
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f}")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": min(best_val, val_loss),
            "model_config": model_config.to_dict(),
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            atomic_torch_save(payload, output_dir / "best.pt")
        if (epoch + 1) % train_config["save_every"] == 0:
            atomic_torch_save(payload, output_dir / f"epoch_{epoch + 1:03d}.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain BERT on masked skeleton trajectories")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain.json"))
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.config, args.resume)


if __name__ == "__main__":
    main()
