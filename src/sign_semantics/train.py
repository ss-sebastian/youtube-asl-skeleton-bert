from __future__ import annotations

import argparse
import csv
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
from .model import SkeletonBert, SkeletonBertConfig, masked_mean
from .utils import atomic_torch_save, choose_device, load_json, seed_everything

PCK_THRESHOLDS = (0.1, 0.2)
OBJECTIVES = ("masked_reconstruction", "next_frame", "contrastive")


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


def next_frame_objective(
    model: SkeletonBert,
    streams: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    valid: torch.Tensor,
    velocity_weight: float,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Predict frame t+1 from frames up to t using causal self-attention."""
    inputs = {name: values[:, :-1] for name, values in streams.items()}
    targets = {name: values[:, 1:] for name, values in streams.items()}
    target_observed = {name: values[:, 1:] for name, values in observed.items()}
    input_valid = valid[:, :-1]
    supervised = input_valid & valid[:, 1:]
    if not supervised.any():
        raise ValueError("Next-frame prediction requires at least one two-frame clip")
    predictions = model(inputs, input_valid, causal=True)
    loss, stream_losses = masked_reconstruction_loss(
        predictions,
        targets,
        supervised,
        velocity_weight,
        target_observed,
    )
    statistics = masked_keypoint_statistics(
        predictions, targets, supervised, target_observed
    )
    return loss, stream_losses, statistics


def _random_like(
    reference: torch.Tensor,
    generator: torch.Generator | None,
    normal: bool,
) -> torch.Tensor:
    """Draw deterministically on CPU for validation, directly on-device otherwise."""
    if generator is None:
        return torch.randn_like(reference) if normal else torch.rand_like(reference)
    sampler = torch.randn if normal else torch.rand
    return sampler(
        reference.shape, dtype=reference.dtype, generator=generator, device="cpu"
    ).to(reference.device)


def contrastive_view(
    streams: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    valid: torch.Tensor,
    coordinate_jitter_std: float,
    landmark_dropout_probability: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Create a mild trajectory view without changing order or handedness."""
    if coordinate_jitter_std < 0:
        raise ValueError("coordinate_jitter_std must be non-negative")
    if not 0 <= landmark_dropout_probability < 1:
        raise ValueError("landmark_dropout_probability must be in [0, 1)")
    view: dict[str, torch.Tensor] = {}
    for name, values in streams.items():
        seen = observed[name] & valid.unsqueeze(-1)
        augmented = values.clone()
        if coordinate_jitter_std:
            noise = _random_like(values, generator, normal=True)
            augmented = augmented + noise * coordinate_jitter_std * seen.unsqueeze(-1)
        if landmark_dropout_probability:
            draws = _random_like(seen.float(), generator, normal=False)
            keep = (draws >= landmark_dropout_probability) & seen
            augmented = torch.where(keep.unsqueeze(-1), augmented, torch.zeros_like(augmented))
        view[name] = augmented
    return view


def contrastive_objective(
    model: SkeletonBert,
    streams: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    valid: torch.Tensor,
    config: dict,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Symmetric in-batch InfoNCE over two corrupted views of each sentence."""
    temperature = float(config.get("temperature", 0.1))
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    view_kwargs = {
        "coordinate_jitter_std": float(config.get("coordinate_jitter_std", 0.01)),
        "landmark_dropout_probability": float(
            config.get("landmark_dropout_probability", 0.05)
        ),
        "generator": generator,
    }
    first = contrastive_view(streams, observed, valid, **view_kwargs)
    second = contrastive_view(streams, observed, valid, **view_kwargs)
    mask_probability = float(config.get("view_mask_probability", 0.2))
    span_length = int(config.get("view_mean_span_length", 10))
    first_mask = sample_span_mask(valid, mask_probability, span_length, generator)
    second_mask = sample_span_mask(valid, mask_probability, span_length, generator)
    first_hidden = model.encode(first, valid, first_mask)
    second_hidden = model.encode(second, valid, second_mask)
    assert isinstance(first_hidden, torch.Tensor)
    assert isinstance(second_hidden, torch.Tensor)
    first_embedding = F.normalize(masked_mean(first_hidden, valid), dim=-1)
    second_embedding = F.normalize(masked_mean(second_hidden, valid), dim=-1)
    logits = first_embedding @ second_embedding.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    similarities = first_embedding @ second_embedding.T
    positive_similarity = similarities.diagonal().mean()
    if similarities.shape[0] > 1:
        off_diagonal = ~torch.eye(
            similarities.shape[0], dtype=torch.bool, device=similarities.device
        )
        negative_similarity = similarities[off_diagonal].mean()
    else:
        negative_similarity = similarities.new_zeros(())
    retrieval_accuracy = 0.5 * (
        (logits.argmax(dim=1) == labels).float().mean()
        + (logits.argmax(dim=0) == labels).float().mean()
    )
    embedding_std = torch.cat([first_embedding, second_embedding], dim=0).std(
        dim=0, unbiased=False
    ).mean()
    metrics = {
        "positive_similarity": float(positive_similarity.detach()),
        "negative_similarity": float(negative_similarity.detach()),
        "retrieval_accuracy": float(retrieval_accuracy.detach()),
        "embedding_std": float(embedding_std.detach()),
    }
    return loss, metrics, {}


def compute_objective(
    model: SkeletonBert,
    streams: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    valid: torch.Tensor,
    objective_config: dict,
    masking_config: dict,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Run one of the matched-backbone pretraining objectives."""
    name = str(objective_config.get("name", "masked_reconstruction"))
    if name == "masked_reconstruction":
        mask = sample_span_mask(
            valid,
            float(masking_config["mask_probability"]),
            int(masking_config["mean_span_length"]),
            generator,
        )
        predictions = model(streams, valid, mask)
        loss, metrics = masked_reconstruction_loss(
            predictions,
            streams,
            mask & valid,
            float(masking_config["velocity_weight"]),
            observed,
        )
        statistics = masked_keypoint_statistics(
            predictions, streams, mask & valid, observed
        )
        return loss, metrics, statistics
    if name == "next_frame":
        return next_frame_objective(
            model,
            streams,
            observed,
            valid,
            float(objective_config.get("velocity_weight", masking_config["velocity_weight"])),
        )
    if name == "contrastive":
        return contrastive_objective(
            model, streams, observed, valid, objective_config, generator
        )
    raise ValueError(f"Unknown objective {name!r}; expected one of {OBJECTIVES}")


def masked_keypoint_statistics(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    supervised: torch.Tensor,
    observed: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Accumulate exact masked-joint errors in shoulder-width-normalized units."""
    statistics: dict[str, float] = {}
    for stream in ("all", *STREAM_JOINTS):
        statistics[f"{stream}_distance_sum"] = 0.0
        statistics[f"{stream}_distance_squared_sum"] = 0.0
        statistics[f"{stream}_count"] = 0.0
        for threshold in PCK_THRESHOLDS:
            statistics[f"{stream}_within_{threshold}"] = 0.0

    for name in STREAM_JOINTS:
        point_mask = supervised.unsqueeze(-1) & observed[name]
        distances = torch.linalg.vector_norm(
            predictions[name].float() - targets[name].float(), dim=-1
        )
        selected = distances[point_mask]
        if selected.numel() == 0:
            continue
        distance_sum = float(selected.detach().sum())
        distance_squared_sum = float(selected.detach().square().sum())
        count = float(selected.numel())
        for stream in ("all", name):
            statistics[f"{stream}_distance_sum"] += distance_sum
            statistics[f"{stream}_distance_squared_sum"] += distance_squared_sum
            statistics[f"{stream}_count"] += count
            for threshold in PCK_THRESHOLDS:
                statistics[f"{stream}_within_{threshold}"] += float(
                    (selected.detach() <= threshold).sum()
                )
    return statistics


def add_statistics(total: dict[str, float], update: dict[str, float]) -> None:
    for name, value in update.items():
        total[name] = total.get(name, 0.0) + value


def summarize_keypoint_statistics(statistics: dict[str, float]) -> dict[str, float]:
    """Return MPJPE, RMSE, and PCK for all points and each landmark stream."""
    metrics: dict[str, float] = {}
    for stream in ("all", *STREAM_JOINTS):
        count = max(statistics.get(f"{stream}_count", 0.0), 1.0)
        prefix = "" if stream == "all" else f"{stream}_"
        metrics[f"{prefix}mpjpe"] = (
            statistics.get(f"{stream}_distance_sum", 0.0) / count
        )
        metrics[f"{prefix}rmse"] = math.sqrt(
            statistics.get(f"{stream}_distance_squared_sum", 0.0) / count
        )
        for threshold in PCK_THRESHOLDS:
            label = str(threshold).replace(".", "_")
            metrics[f"{prefix}pck_{label}"] = (
                statistics.get(f"{stream}_within_{threshold}", 0.0) / count
            )
    return metrics


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
    objective_config: dict,
    masking_config: dict,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0}
    keypoint_totals: dict[str, float] = {}
    clips = 0
    generator = torch.Generator().manual_seed(0)
    progress = tqdm(total=len(loader.dataset), desc="validation", unit="clips", leave=False)
    for batch in loader:
        streams, observed, valid = move_batch(batch, device)
        loss, objective_metrics, keypoint_statistics = compute_objective(
            model,
            streams,
            observed,
            valid,
            objective_config,
            masking_config,
            generator,
        )
        add_statistics(keypoint_totals, keypoint_statistics)
        batch_clips = int(valid.shape[0])
        clips += batch_clips
        totals["loss"] += float(loss) * batch_clips
        for name, value in objective_metrics.items():
            totals[name] = totals.get(name, 0.0) + value * batch_clips
        progress.update(batch_clips)
    progress.close()
    metrics = {name: value / max(clips, 1) for name, value in totals.items()}
    if keypoint_totals:
        metrics.update(summarize_keypoint_statistics(keypoint_totals))
    return metrics


def run_training(config_path: Path, resume: Path | None = None) -> None:
    config = load_json(config_path)
    objective_config = config.get("objective", {"name": "masked_reconstruction"})
    objective_name = str(objective_config.get("name", "masked_reconstruction"))
    if objective_name not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective_name!r}; expected one of {OBJECTIVES}")
    seed_everything(int(config["seed"]))
    device = choose_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}; objective={objective_name}", flush=True)

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

    model_values = dict(config["model"])
    model_values["causal_attention"] = objective_name == "next_frame"
    model_config = SkeletonBertConfig(**model_values)
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
        checkpoint_objective = checkpoint.get("objective", "masked_reconstruction")
        if checkpoint_objective != objective_name:
            raise ValueError(
                f"Checkpoint objective {checkpoint_objective!r} does not match "
                f"configured objective {objective_name!r}"
            )
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        logged_batches = 0
        epoch_totals: dict[str, float] = {"loss": 0.0}
        epoch_keypoint_totals: dict[str, float] = {}
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
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                loss, objective_metrics, batch_keypoint_statistics = compute_objective(
                    model,
                    streams,
                    observed,
                    valid,
                    objective_config,
                    mask_config,
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
            for name, value in objective_metrics.items():
                epoch_totals[name] = epoch_totals.get(name, 0.0) + value * batch_clips
            add_statistics(epoch_keypoint_totals, batch_keypoint_statistics)
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
        if epoch_keypoint_totals:
            train_metrics.update(summarize_keypoint_statistics(epoch_keypoint_totals))
        val_metrics = validate(
            model,
            val_loader,
            device,
            objective_config,
            mask_config,
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
        if device.type == "cuda":
            epoch_metrics["gpu_peak_memory_gb"] = round(
                torch.cuda.max_memory_allocated(device) / 1024**3, 3
            )
        else:
            epoch_metrics["gpu_peak_memory_gb"] = 0.0
        summary = (
            f"Epoch {epoch + 1} summary: objective={objective_name}, "
            f"train_loss={epoch_metrics['train_loss']:.6f}, "
            f"val_loss={epoch_metrics['val_loss']:.6f}"
        )
        if "val_mpjpe" in epoch_metrics:
            summary += (
                f", val_mpjpe={epoch_metrics['val_mpjpe']:.4f}, "
                f"val_pck@0.1={epoch_metrics['val_pck_0_1']:.3f}, "
                f"val_pck@0.2={epoch_metrics['val_pck_0_2']:.3f}"
            )
        if "val_retrieval_accuracy" in epoch_metrics:
            summary += (
                f", val_retrieval_accuracy="
                f"{epoch_metrics['val_retrieval_accuracy']:.3f}, "
                f"val_positive_similarity="
                f"{epoch_metrics['val_positive_similarity']:.3f}"
            )
        summary += (
            f", seconds={epoch_metrics['seconds']:.1f}, "
            f"clips_per_second={epoch_metrics['clips_per_second']:.3f}"
        )
        print(summary, flush=True)
        print("epoch_metrics=" + json.dumps(epoch_metrics, sort_keys=True), flush=True)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        metrics_csv = output_dir / "metrics.csv"
        with metrics_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(epoch_metrics))
            if metrics_csv.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(epoch_metrics)
        val_loss = val_metrics["loss"]
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": min(best_val, val_loss),
            "model_config": model_config.to_dict(),
            "objective": objective_name,
            "objective_config": objective_config,
        }
        atomic_torch_save(payload, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            atomic_torch_save(payload, output_dir / "best.pt")
        save_every = int(train_config["save_every"])
        if save_every > 0 and (epoch + 1) % save_every == 0:
            atomic_torch_save(payload, output_dir / f"epoch_{epoch + 1:03d}.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain a Transformer on skeleton trajectories")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain.json"))
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.config, args.resume)


if __name__ == "__main__":
    main()
