from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import How2SignPoseDataset
from .features import STREAM_JOINTS
from .masking import sample_span_mask
from .model import ModelConfig, MultiStreamSignTransformer
from .targets import ClusterTargeter
from .utils import atomic_torch_save, choose_device, load_json, seed_everything


def move_batch(batch: dict, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    streams = {name: batch[name].to(device, non_blocking=True) for name in STREAM_JOINTS}
    valid = batch["valid"].to(device, non_blocking=True)
    return streams, valid


def masked_prediction_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    supervised: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not supervised.any():
        raise ValueError("No frames selected for masked prediction")
    losses = {name: F.cross_entropy(logits[name][supervised], targets[name][supervised]) for name in logits}
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
    model: MultiStreamSignTransformer,
    loader: DataLoader,
    targeter: ClusterTargeter,
    device: torch.device,
    mask_probability: float,
    span_length: int,
) -> float:
    model.eval()
    losses: list[float] = []
    generator = torch.Generator().manual_seed(0)
    for batch in loader:
        streams, valid = move_batch(batch, device)
        mask = sample_span_mask(valid, mask_probability, span_length, generator)
        targets = targeter.assign(streams)
        logits = model(streams, valid, mask)
        loss, _ = masked_prediction_loss(logits, targets, mask & valid)
        losses.append(float(loss))
    return sum(losses) / max(len(losses), 1)


def infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def run_training(config_path: Path, resume: Path | None = None) -> None:
    config = load_json(config_path)
    seed_everything(int(config["seed"]))
    device = choose_device()
    print(f"Using device: {device}")

    data_config = config["data"]
    train_data = How2SignPoseDataset(
        data_config["train_manifest"], data_config["max_frames"], training=True
    )
    val_data = How2SignPoseDataset(
        data_config["val_manifest"], data_config["max_frames"], training=False
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=pin_memory,
    )

    targeter = ClusterTargeter(data_config["centers_path"], device)
    model_config = ModelConfig(**config["model"])
    model = MultiStreamSignTransformer(model_config, targeter.cluster_sizes).to(device)
    train_config = config["training"]
    optimizer = AdamW(
        model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"]
    )
    steps_per_epoch = max(1, len(train_loader) // train_config["gradient_accumulation"])
    total_steps = steps_per_epoch * train_config["epochs"]
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
    accumulation = int(train_config["gradient_accumulation"])
    mask_config = config["masking"]

    for epoch in range(start_epoch, train_config["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            streams, valid = move_batch(batch, device)
            mask = sample_span_mask(
                valid, mask_config["mask_probability"], mask_config["mean_span_length"]
            )
            targets = targeter.assign(streams)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast:
                logits = model(streams, valid, mask)
                loss, stream_losses = masked_prediction_loss(logits, targets, mask & valid)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            running += float(loss.detach())

            if (batch_index + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % train_config["log_every"] == 0:
                    mean = running / train_config["log_every"]
                    details = " ".join(f"{key}={value:.3f}" for key, value in stream_losses.items())
                    print(
                        f"epoch={epoch + 1} step={global_step} loss={mean:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} {details}"
                    )
                    running = 0.0

        val_loss = validate(
            model,
            val_loader,
            targeter,
            device,
            mask_config["mask_probability"],
            mask_config["mean_span_length"],
        )
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f}")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": min(best_val, val_loss),
            "model_config": config["model"],
            "cluster_sizes": targeter.cluster_sizes,
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            atomic_torch_save(payload, output_dir / "best.pt")
        if (epoch + 1) % train_config["save_every"] == 0:
            atomic_torch_save(payload, output_dir / f"epoch_{epoch + 1:03d}.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the sign-language Transformer")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain.json"))
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.config, args.resume)


if __name__ == "__main__":
    main()

