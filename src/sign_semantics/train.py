from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, **{name: 0.0 for name in STREAM_JOINTS}}
    clips = 0
    generator = torch.Generator().manual_seed(0)
    progress = tqdm(total=len(loader.dataset), desc="validation", unit="clips", leave=False)
    for batch in loader:
        streams, observed, valid = move_batch(batch, device)
        mask = sample_span_mask(valid, mask_probability, span_length, generator)
        predictions = model(streams, valid, mask)
        loss, stream_losses = masked_reconstruction_loss(
            predictions, streams, mask & valid, velocity_weight, observed
        )
        batch_clips = int(valid.shape[0])
        clips += batch_clips
        totals["loss"] += float(loss) * batch_clips
        for name, value in stream_losses.items():
            totals[name] += value * batch_clips
        progress.update(batch_clips)
    progress.close()
    return {name: value / max(clips, 1) for name, value in totals.items()}


def run_training(config_path: Path, resume: Path | None = None) -> None:
    config = load_json(config_path)
    seed_everything(int(config["seed"]))
    device = choose_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}", flush=True)

    data_config = config["data"]
    print("Indexing training clips from the shard ZIP...", flush=True)
    train_data = YouTubeASLPoseDataset(
        archive=data_config["train_archive"],
        annotations=data_config["train_annotations"],
        max_frames=data_config["max_frames"],
        training=True,
        limit_clips=data_config.get("limit_train_clips"),
    )
    print(f"Indexed {len(train_data):,} training clips.", flush=True)
    print("Indexing validation clips from the shard ZIP...", flush=True)
    val_data = YouTubeASLPoseDataset(
        archive=data_config.get("val_archive", data_config["train_archive"]),
        annotations=data_config["val_annotations"],
        max_frames=data_config["max_frames"],
        training=False,
        limit_clips=data_config.get("limit_val_clips"),
    )
    print(f"Indexed {len(val_data):,} validation clips.", flush=True)
    train_config = config["training"]
    pin_memory = device.type == "cuda"
    worker_options = {}
    if data_config["num_workers"] > 0:
        worker_options = {
            "persistent_workers": True,
            "prefetch_factor": int(data_config.get("prefetch_factor", 2)),
        }
    train_loader = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
        **worker_options,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=pin_memory,
        **worker_options,
    )
    print(
        f"DataLoaders ready: batch_size={train_config['batch_size']}, "
        f"workers={data_config['num_workers']}, train_batches={len(train_loader):,}.",
        flush=True,
    )

    model_config = SkeletonBertConfig(**config["model"])
    model = SkeletonBert(model_config).to(device)
    print("Model is on the device; preparing optimizer.", flush=True)
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
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
        print(f"Resumed from epoch {start_epoch}", flush=True)

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_config = config["masking"]

    for epoch in range(start_epoch, train_config["epochs"]):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        logged_batches = 0
        epoch_totals = {"loss": 0.0, **{name: 0.0 for name in STREAM_JOINTS}}
        epoch_clips = 0
        progress = tqdm(
            total=len(train_data),
            desc=f"epoch {epoch + 1}/{train_config['epochs']}",
            unit="clips",
            dynamic_ncols=True,
        )
        print("Waiting for the first training batch...", flush=True)
        for batch_index, batch in enumerate(train_loader):
            if batch_index == 0:
                print("First batch loaded; GPU training has started.", flush=True)
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
            batch_clips = int(valid.shape[0])
            epoch_clips += batch_clips
            epoch_totals["loss"] += float(loss.detach()) * batch_clips
            for name, value in stream_losses.items():
                epoch_totals[name] += value * batch_clips
            progress.update(batch_clips)

            last_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % accumulation == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                progress_every = int(train_config.get("progress_every", 5))
                if global_step % progress_every == 0 or last_batch:
                    elapsed = time.perf_counter() - epoch_started
                    clips_per_second = epoch_clips / max(elapsed, 1e-9)
                    eta_seconds = (len(train_data) - epoch_clips) / max(
                        clips_per_second, 1e-9
                    )
                    progress_record = {
                        "epoch": epoch + 1,
                        "optimizer_step": global_step,
                        "clips": epoch_clips,
                        "total_clips": len(train_data),
                        "percent": round(100 * epoch_clips / len(train_data), 2),
                        "clips_per_second": round(clips_per_second, 3),
                        "eta_seconds": round(eta_seconds),
                        "train_loss_so_far": round(
                            epoch_totals["loss"] / max(epoch_clips, 1), 6
                        ),
                    }
                    print(
                        "training_progress="
                        + json.dumps(progress_record, sort_keys=True),
                        flush=True,
                    )
                if global_step % train_config["log_every"] == 0:
                    progress.set_postfix(
                        step=global_step,
                        loss=f"{running / max(logged_batches, 1):.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    running = 0.0
                    logged_batches = 0
        progress.close()

        train_metrics = {
            name: value / max(epoch_clips, 1) for name, value in epoch_totals.items()
        }
        val_metrics = validate(
            model,
            val_loader,
            device,
            mask_config["mask_probability"],
            mask_config["mean_span_length"],
            mask_config["velocity_weight"],
        )
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_metrics = {
            "epoch": epoch + 1,
            "optimizer_steps": global_step,
            "train_clips": epoch_clips,
            "seconds": round(epoch_seconds, 1),
            "clips_per_second": round(epoch_clips / max(epoch_seconds, 1e-9), 3),
            **{f"train_{name}": round(value, 6) for name, value in train_metrics.items()},
            **{f"val_{name}": round(value, 6) for name, value in val_metrics.items()},
        }
        print("epoch_metrics=" + json.dumps(epoch_metrics, sort_keys=True), flush=True)
        val_loss = val_metrics["loss"]
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
