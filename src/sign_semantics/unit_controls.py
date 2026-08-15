from __future__ import annotations

"""Matched trained, random, and raw-kinematic controls for sign unit discovery."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import YouTubeASLPoseDataset
from .features import STREAM_JOINTS
from .model import SkeletonBert
from .unit_probe import (
    _collapse_runs,
    _load_contrastive_model,
    _load_standard_skeleton,
    _pool_patches,
    annotation_source_groups,
    evaluate_lexical_units,
    leave_one_signer_out_scores,
    probe_codebook_stability,
    signer_conditional_permutation_p,
)
from .utils import choose_device, seed_everything


def _raw_patch_features(
    streams: dict[str, torch.Tensor],
    valid: torch.Tensor,
    width: int,
    stride: int,
) -> list[np.ndarray]:
    frames = torch.cat(
        [streams[name].flatten(start_dim=2) for name in STREAM_JOINTS], dim=-1
    ).float()
    velocity = torch.diff(frames, dim=1, prepend=frames[:, :1]).abs()
    pooled: list[np.ndarray] = []
    for row in range(len(frames)):
        length = int(valid[row].sum())
        starts = list(range(0, max(length - width + 1, 1), stride))
        final = max(0, length - width)
        if not starts or starts[-1] != final:
            starts.append(final)
        values = []
        for start in starts:
            stop = min(start + width, length)
            values.append(
                torch.cat(
                    [frames[row, start:stop].mean(dim=0), velocity[row, start:stop].mean(dim=0)]
                )
            )
        patches = F.normalize(torch.stack(values), dim=-1)
        pooled.append(patches.cpu().numpy())
    return pooled


@torch.no_grad()
def extract_matched_control_features(
    checkpoint: Path,
    archive: Path,
    annotations: Path,
    output_dir: Path,
    sample_clips: int,
    layer: int,
    patch_frames: int,
    patch_stride: int,
    batch_size: int,
    seed: int,
) -> None:
    device = choose_device()
    trained = _load_contrastive_model(checkpoint, device)
    seed_everything(seed)
    random_model = SkeletonBert(trained.config).to(device).eval()
    dataset = YouTubeASLPoseDataset(
        archive, annotations, max_frames=trained.config.max_frames, training=False
    )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(sample_clips, len(indices))]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    group_map = annotation_source_groups(annotations)
    values: dict[str, list[np.ndarray]] = {
        "trained_contrastive": [],
        "random_transformer": [],
        "raw_kinematics": [],
    }
    clip_ids: list[str] = []
    source_ids: list[str] = []
    offsets = [0]
    for batch in tqdm(loader, desc="Extracting three matched unit controls"):
        streams = {name: batch[name].to(device, non_blocking=True) for name in STREAM_JOINTS}
        valid = batch["valid"].to(device, non_blocking=True)
        trained_layers = trained.encode(
            streams, valid, causal=False, return_all_layers=True
        )
        random_layers = random_model.encode(
            streams, valid, causal=False, return_all_layers=True
        )
        assert isinstance(trained_layers, list) and isinstance(random_layers, list)
        batch_features = {
            "trained_contrastive": _pool_patches(
                trained_layers[layer - 1], valid, patch_frames, patch_stride
            ),
            "random_transformer": _pool_patches(
                random_layers[layer - 1], valid, patch_frames, patch_stride
            ),
            "raw_kinematics": _raw_patch_features(
                streams, valid, patch_frames, patch_stride
            ),
        }
        for row, clip_id in enumerate(batch["id"]):
            clip_id = str(clip_id)
            length = len(batch_features["trained_contrastive"][row])
            if not all(len(batch_features[name][row]) == length for name in values):
                raise RuntimeError("Control representations produced different patch counts")
            clip_ids.append(clip_id)
            source_ids.append(group_map.get(clip_id, "__unknown_source__"))
            offsets.append(offsets[-1] + length)
            for name in values:
                values[name].append(batch_features[name][row].astype(np.float16))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, parts in values.items():
        np.savez_compressed(
            output_dir / f"{name}_continuous_features.npz",
            features=np.concatenate(parts).astype(np.float16),
            clip_ids=np.asarray(clip_ids),
            source_ids=np.asarray(source_ids),
            offsets=np.asarray(offsets, dtype=np.int64),
            layer=layer,
            patch_frames=patch_frames,
            patch_stride=patch_stride,
        )


@torch.no_grad()
def evaluate_raw_lexical_units(
    codebook,
    manifest: Path,
    root: Path,
    output_dir: Path,
    patch_frames: int,
    patch_stride: int,
    permutations: int,
    seed: int,
) -> dict:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    histograms = []
    for row in tqdm(rows, desc="Testing raw units on held-out ASL Citizen"):
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        streams, valid = _load_standard_skeleton(path, 256)
        patches = _raw_patch_features(
            {name: value.unsqueeze(0) for name, value in streams.items()},
            valid.unsqueeze(0),
            patch_frames,
            patch_stride,
        )[0]
        units = _collapse_runs(codebook.predict(patches))
        histogram = np.bincount(units, minlength=codebook.n_clusters).astype(np.float32)
        histogram /= np.linalg.norm(histogram).clip(min=1e-8)
        histograms.append(histogram)
    matrix = np.stack(histograms)
    concepts = np.asarray([row["id"].strip().lower() for row in rows])
    signers = np.asarray([row["participant_id"].strip() for row in rows])
    sample_ids = np.asarray([row.get("sample_id", Path(row["path"]).stem) for row in rows])
    np.savez_compressed(
        output_dir / "lexical_unit_histograms.npz",
        features=matrix,
        concepts=concepts,
        signers=signers,
        sample_ids=sample_ids,
    )
    top1, top5, mrr, separation = leave_one_signer_out_scores(
        matrix, concepts, signers
    )
    p_value = signer_conditional_permutation_p(
        matrix, concepts, signers, top1, permutations, seed
    )
    metrics = {
        "tokens": len(rows),
        "concepts": len(set(concepts.tolist())),
        "signers": len(set(signers.tolist())),
        "chance_top1": 1 / len(set(concepts.tolist())),
        "top1": top1,
        "top5": top5,
        "mrr": mrr,
        "same_vs_different_distance_separation": separation,
        "top1_permutation_p": p_value,
        "permutations": permutations,
        "permutation_scheme": "concept_labels_shuffled_within_signer",
    }
    (output_dir / "lexical_unit_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched falsification controls for sign units")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--lexical-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-clips", type=int, default=5000)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--patch-frames", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    extract_matched_control_features(
        args.checkpoint,
        args.archive,
        args.annotations,
        args.output_dir,
        args.sample_clips,
        args.layer,
        args.patch_frames,
        args.patch_stride,
        args.batch_size,
        args.seed,
    )
    comparison = []
    for name in ("trained_contrastive", "random_transformer", "raw_kinematics"):
        condition_dir = args.output_dir / name
        codebook, continuous = probe_codebook_stability(
            args.output_dir / f"{name}_continuous_features.npz",
            condition_dir,
            args.clusters,
            args.seed,
        )
        if name == "raw_kinematics":
            lexical = evaluate_raw_lexical_units(
                codebook,
                args.lexical_manifest,
                args.root,
                condition_dir,
                args.patch_frames,
                args.patch_stride,
                args.permutations,
                args.seed,
            )
        else:
            checkpoint = args.checkpoint
            if name == "random_transformer":
                # Save the exact random model used for continuous extraction.
                device = choose_device()
                trained = _load_contrastive_model(args.checkpoint, device)
                seed_everything(args.seed)
                random_model = SkeletonBert(trained.config)
                random_checkpoint = args.output_dir / "random_model.pt"
                torch.save(
                    {
                        "model": random_model.state_dict(),
                        "model_config": trained.config.to_dict(),
                        "objective": "contrastive",
                    },
                    random_checkpoint,
                )
                checkpoint = random_checkpoint
            lexical = evaluate_lexical_units(
                checkpoint,
                codebook,
                args.lexical_manifest,
                args.root,
                condition_dir,
                args.layer,
                args.patch_frames,
                args.patch_stride,
                args.permutations,
                args.seed,
            )
        comparison.append(
            {
                "representation": name,
                "split_half_ami": continuous["split_half_ami"],
                "centroid_cosine": continuous["matched_centroid_cosine_mean"],
                "effective_units": continuous["effective_units"],
                "lexical_top1": lexical["top1"],
                "lexical_top5": lexical["top5"],
                "lexical_mrr": lexical["mrr"],
                "distance_separation": lexical["same_vs_different_distance_separation"],
                "signer_conditional_p": lexical["top1_permutation_p"],
            }
        )
    random_top1 = next(row["lexical_top1"] for row in comparison if row["representation"] == "random_transformer")
    raw_top1 = next(row["lexical_top1"] for row in comparison if row["representation"] == "raw_kinematics")
    for row in comparison:
        row["top1_delta_vs_random"] = row["lexical_top1"] - random_top1
        row["top1_delta_vs_raw"] = row["lexical_top1"] - raw_top1
    with (args.output_dir / "control_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print("unit_control_comparison=" + json.dumps(comparison, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
