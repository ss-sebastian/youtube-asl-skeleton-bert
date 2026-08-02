from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .features import STREAM_JOINTS, cluster_features_numpy
from .utils import read_jsonl


def reservoir_frames(
    records: list[dict], stream: str, maximum: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    total_frames = sum(int(record.get("num_frames", 0)) for record in records)
    sampling_probability = min(1.0, maximum * 1.15 / max(total_frames, 1))
    for record in tqdm(records, desc=f"Sampling {stream}"):
        with np.load(record["path"]) as payload:
            features = cluster_features_numpy(payload[stream].astype(np.float32))
        selected = rng.random(features.shape[0]) < sampling_probability
        if selected.any():
            chunks.append(features[selected])
    if not chunks:
        raise ValueError(f"No frames available for {stream}")
    samples = np.concatenate(chunks, axis=0)
    if samples.shape[0] > maximum:
        samples = samples[rng.choice(samples.shape[0], size=maximum, replace=False)]
    rng.shuffle(samples)
    return samples


def fit_clusters(
    manifest: Path,
    output: Path,
    n_clusters: int,
    max_frames: int,
    batch_size: int,
    seed: int,
) -> None:
    records = read_jsonl(manifest)
    centers: dict[str, np.ndarray] = {}
    for offset, stream in enumerate(STREAM_JOINTS):
        samples = reservoir_frames(records, stream, max_frames, seed + offset)
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=seed,
            n_init="auto",
            reassignment_ratio=0.01,
        )
        model.fit(samples)
        centers[stream] = model.cluster_centers_.astype(np.float32)
        print(f"{stream}: fitted {n_clusters} clusters on {len(samples)} frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **centers)
    print(f"Saved cluster targets to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit SHuBERT-style pose cluster targets")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=500)
    parser.add_argument("--max-sampled-frames", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fit_clusters(
        args.manifest,
        args.output,
        args.n_clusters,
        args.max_sampled_frames,
        args.batch_size,
        args.seed,
    )


if __name__ == "__main__":
    main()
