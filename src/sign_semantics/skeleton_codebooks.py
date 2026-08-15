from __future__ import annotations

"""Fit and apply fixed, frame-local skeleton cluster targets."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import YouTubeASLPoseDataset


PARTS = ("body", "right_hand", "left_hand", "face")


def split_parts(
    streams: dict[str, torch.Tensor], observed: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {
            "body": streams["body"],
            "right_hand": streams["hands"][:, :, :21],
            "left_hand": streams["hands"][:, :, 21:],
            "face": streams["face"],
        },
        {
            "body": observed["body"],
            "right_hand": observed["hands"][:, :, :21],
            "left_hand": observed["hands"][:, :, 21:],
            "face": observed["face"],
        },
    )


def frame_descriptors(values: torch.Tensor) -> torch.Tensor:
    """Flatten frame-local normalized coordinates; never use neighbouring frames."""
    return values.flatten(start_dim=-2)


class SkeletonCodebooks:
    def __init__(self, path: str | Path, device: torch.device) -> None:
        arrays = np.load(path)
        self.means = {
            name: torch.from_numpy(arrays[f"{name}_mean"]).to(device) for name in PARTS
        }
        self.scales = {
            name: torch.from_numpy(arrays[f"{name}_scale"]).to(device) for name in PARTS
        }
        self.centroids = {
            name: torch.from_numpy(arrays[f"{name}_centroids"]).to(device) for name in PARTS
        }

    @torch.no_grad()
    def targets(self, name: str, descriptors: torch.Tensor) -> torch.Tensor:
        standardized = (descriptors.float() - self.means[name]) / self.scales[name]
        distances = torch.cdist(standardized, self.centroids[name].float())
        return distances.argmin(dim=-1)


def fit_codebooks(
    archive: Path,
    annotations: Path,
    output: Path,
    max_frames: int,
    max_sampled_frames: int,
    clusters: int,
    batch_size: int,
    num_workers: int,
    limit_clips: int | None = None,
) -> None:
    dataset = YouTubeASLPoseDataset(
        archive,
        annotations,
        max_frames=max_frames,
        training=False,
        limit_clips=limit_clips,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    samples: dict[str, list[np.ndarray]] = {name: [] for name in PARTS}
    counts = {name: 0 for name in PARTS}
    generator = torch.Generator().manual_seed(42)
    for batch in tqdm(loader, desc="Sampling frame-local skeleton targets"):
        streams = {name: batch[name] for name in ("body", "hands", "face")}
        observed = {name: batch[f"{name}_observed"] for name in streams}
        parts, seen = split_parts(streams, observed)
        valid = batch["valid"]
        for name in PARTS:
            # Exclude severely missing frames so zero-padding does not become a dominant unit.
            usable = valid & (seen[name].float().mean(dim=-1) >= 0.5)
            descriptors = frame_descriptors(parts[name])[usable]
            remaining = max_sampled_frames - counts[name]
            if remaining <= 0 or descriptors.numel() == 0:
                continue
            if len(descriptors) > remaining:
                choice = torch.randperm(len(descriptors), generator=generator)[:remaining]
                descriptors = descriptors[choice]
            samples[name].append(descriptors.numpy().astype(np.float32, copy=False))
            counts[name] += len(descriptors)
        if all(value >= max_sampled_frames for value in counts.values()):
            break

    payload: dict[str, np.ndarray] = {}
    for index, name in enumerate(PARTS):
        if not samples[name]:
            raise RuntimeError(f"No usable frames collected for {name}")
        values = np.concatenate(samples[name], axis=0)
        if len(values) < clusters:
            raise RuntimeError(f"Only {len(values)} usable {name} frames for {clusters} clusters")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-5] = 1.0
        standardized = (values - mean) / scale
        estimator = MiniBatchKMeans(
            n_clusters=clusters,
            batch_size=4096,
            n_init=3,
            random_state=42 + index,
            reassignment_ratio=0.01,
        ).fit(standardized)
        payload[f"{name}_mean"] = mean
        payload[f"{name}_scale"] = scale
        payload[f"{name}_centroids"] = estimator.cluster_centers_.astype(np.float32)
        print(f"{name}: fitted {clusters} clusters from {len(values):,} frames", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"Saved frozen skeleton codebooks to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--max-sampled-frames", type=int, default=100_000)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-clips", type=int)
    args = parser.parse_args()
    fit_codebooks(**vars(args))


if __name__ == "__main__":
    main()
