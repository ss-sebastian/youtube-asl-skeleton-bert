from __future__ import annotations

"""Pilot discovery and falsification tests for recurring continuous-sign units."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import orjson
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import YouTubeASLPoseDataset
from .features import COORDINATE_DIM, STREAM_JOINTS
from .model import SkeletonBert, SkeletonBertConfig
from .utils import choose_device, seed_everything


def _load_contrastive_model(checkpoint_path: Path, device: torch.device) -> SkeletonBert:
    """Load the matched vanilla backbone without depending on later RSA modules."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("objective") != "contrastive":
        raise ValueError("The unit pilot requires the trained vanilla contrastive checkpoint")
    values = dict(checkpoint["model_config"])
    values.pop("architecture", None)
    model = SkeletonBert(SkeletonBertConfig(**values)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _load_standard_skeleton(
    path: Path, max_frames: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with np.load(path) as payload:
        arrays = {name: np.asarray(payload[name], dtype=np.float32) for name in STREAM_JOINTS}
    length = len(arrays["body"])
    for name, joints in STREAM_JOINTS.items():
        if arrays[name].shape != (length, joints, COORDINATE_DIM):
            raise ValueError(f"{path}:{name} has an invalid shape")
    if length > max_frames:
        indices = np.linspace(0, length - 1, max_frames).round().astype(np.int64)
        arrays = {name: values[indices] for name, values in arrays.items()}
        length = max_frames
    return {name: torch.from_numpy(values) for name, values in arrays.items()}, torch.ones(length, dtype=torch.bool)


def annotation_source_groups(path: str | Path) -> dict[str, str]:
    """Map clip IDs to source-video IDs; these are not asserted to be signers."""
    payload = orjson.loads(Path(path).read_bytes())
    result: dict[str, str] = {}
    for source_id, video in payload.items():
        if not isinstance(video, dict):
            continue
        order = video.get("clip_order")
        clips = order if isinstance(order, list) else [key for key in video if key != "clip_order"]
        for clip_id in clips:
            clip_id = str(clip_id)
            if clip_id in result and result[clip_id] != str(source_id):
                raise ValueError(f"Clip {clip_id!r} occurs under multiple source videos")
            result[clip_id] = str(source_id)
    return result


def _stable_half(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return digest[0] % 2


def _pool_patches(hidden: torch.Tensor, valid: torch.Tensor, width: int, stride: int) -> list[np.ndarray]:
    pooled: list[np.ndarray] = []
    for row in range(hidden.shape[0]):
        length = int(valid[row].sum())
        if length == 0:
            pooled.append(np.empty((0, hidden.shape[-1]), dtype=np.float32))
            continue
        starts = list(range(0, max(length - width + 1, 1), stride))
        final = max(0, length - width)
        if not starts or starts[-1] != final:
            starts.append(final)
        parts = [hidden[row, start : min(start + width, length)].mean(dim=0) for start in starts]
        values = torch.stack(parts)
        values = F.normalize(values.float(), dim=-1)
        pooled.append(values.cpu().numpy())
    return pooled


@torch.no_grad()
def extract_continuous_patch_features(
    checkpoint: Path,
    archive: Path,
    annotations: Path,
    output: Path,
    sample_clips: int,
    layer: int,
    patch_frames: int,
    patch_stride: int,
    batch_size: int,
    seed: int,
) -> None:
    seed_everything(seed)
    device = choose_device()
    model = _load_contrastive_model(checkpoint, device)
    if not 1 <= layer <= model.config.num_hidden_layers:
        raise ValueError(f"layer must be in [1, {model.config.num_hidden_layers}]")
    dataset = YouTubeASLPoseDataset(
        archive, annotations, max_frames=model.config.max_frames, training=False
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
    groups = annotation_source_groups(annotations)
    features: list[np.ndarray] = []
    clip_ids: list[str] = []
    source_ids: list[str] = []
    sequence_offsets = [0]
    for batch in tqdm(loader, desc="Extracting continuous local patches"):
        streams = {name: batch[name].to(device, non_blocking=True) for name in STREAM_JOINTS}
        valid = batch["valid"].to(device, non_blocking=True)
        layers = model.encode(
            streams,
            valid,
            causal=model.config.causal_attention,
            return_all_layers=True,
        )
        assert isinstance(layers, list)
        pooled = _pool_patches(layers[layer - 1], valid, patch_frames, patch_stride)
        for clip_id, values in zip(batch["id"], pooled, strict=True):
            clip_id = str(clip_id)
            if len(values) == 0:
                continue
            features.append(values.astype(np.float16))
            clip_ids.append(clip_id)
            source_ids.append(groups.get(clip_id, "__unknown_source__"))
            sequence_offsets.append(sequence_offsets[-1] + len(values))
    if not features:
        raise RuntimeError("No patch features were extracted")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(features).astype(np.float16),
        clip_ids=np.asarray(clip_ids),
        source_ids=np.asarray(source_ids),
        offsets=np.asarray(sequence_offsets, dtype=np.int64),
        checkpoint=str(checkpoint),
        layer=layer,
        patch_frames=patch_frames,
        patch_stride=patch_stride,
        note="clips longer than model max_frames use the pretraining evaluation resampling rule",
    )


def _fit_codebook(values: np.ndarray, clusters: int, seed: int) -> MiniBatchKMeans:
    if len(values) < clusters * 10:
        raise ValueError(f"Need at least {clusters * 10} patches for {clusters} clusters")
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=min(4096, len(values)),
        n_init=5,
        max_iter=200,
        reassignment_ratio=0.01,
    )
    model.fit(values)
    return model


def _collapse_runs(units: np.ndarray) -> list[int]:
    return [int(value) for index, value in enumerate(units) if index == 0 or value != units[index - 1]]


def probe_codebook_stability(
    features_path: Path,
    output_dir: Path,
    clusters: int,
    seed: int,
) -> tuple[MiniBatchKMeans, dict]:
    with np.load(features_path) as payload:
        values = np.asarray(payload["features"], dtype=np.float32)
        clip_ids = payload["clip_ids"].astype(str)
        source_ids = payload["source_ids"].astype(str)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
    halves = np.asarray([_stable_half(source, seed) for source in source_ids])
    if len(set(halves.tolist())) != 2:
        raise ValueError("Source split produced only one half")
    patch_halves = np.concatenate(
        [np.full(offsets[i + 1] - offsets[i], halves[i]) for i in range(len(clip_ids))]
    )
    first = _fit_codebook(values[patch_halves == 0], clusters, seed)
    second = _fit_codebook(values[patch_halves == 1], clusters, seed + 1)
    anchor_rng = np.random.default_rng(seed)
    anchor_index = anchor_rng.choice(len(values), min(100_000, len(values)), replace=False)
    first_labels = first.predict(values[anchor_index])
    second_labels = second.predict(values[anchor_index])
    ami = float(adjusted_mutual_info_score(first_labels, second_labels))
    first_centres = first.cluster_centers_
    second_centres = second.cluster_centers_
    first_centres /= np.linalg.norm(first_centres, axis=1, keepdims=True).clip(min=1e-8)
    second_centres /= np.linalg.norm(second_centres, axis=1, keepdims=True).clip(min=1e-8)
    similarity = first_centres @ second_centres.T
    rows, columns = linear_sum_assignment(-similarity)
    matched = similarity[rows, columns]

    final = _fit_codebook(values, clusters, seed)
    labels = final.predict(values)
    cluster_clips: dict[int, set[str]] = defaultdict(set)
    cluster_sources: dict[int, set[str]] = defaultdict(set)
    sequences = []
    for i, clip_id in enumerate(clip_ids):
        units = _collapse_runs(labels[offsets[i] : offsets[i + 1]])
        sequences.append({"id": clip_id, "group": source_ids[i], "units": units})
        for unit in set(units):
            cluster_clips[unit].add(clip_id)
            cluster_sources[unit].add(source_ids[i])
    source_counts = np.asarray([len(cluster_sources[i]) for i in range(clusters)])
    clip_counts = np.asarray([len(cluster_clips[i]) for i in range(clusters)])
    usage = np.bincount(labels, minlength=clusters)
    probabilities = usage / usage.sum()
    entropy = float(-(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])).sum())
    metrics = {
        "clusters": clusters,
        "clips": int(len(clip_ids)),
        "sources": int(len(set(source_ids))),
        "patches": int(len(values)),
        "split_half_ami": ami,
        "matched_centroid_cosine_mean": float(matched.mean()),
        "matched_centroid_cosine_median": float(np.median(matched)),
        "unit_entropy_bits": entropy,
        "effective_units": float(2**entropy),
        "dead_units": int((usage == 0).sum()),
        "median_clips_per_unit": float(np.median(clip_counts)),
        "median_sources_per_unit": float(np.median(source_counts)),
        "units_in_at_least_5_sources": int((source_counts >= 5).sum()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "codebook.npz", centers=final.cluster_centers_)
    (output_dir / "continuous_unit_sequences.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sequences),
        encoding="utf-8",
    )
    group_sizes = Counter(row["group"] for row in sequences)
    usable = [row for row in sequences if group_sizes[row["group"]] >= 2]
    train_sequences = [
        row for row in usable if int.from_bytes(hashlib.sha256(row["group"].encode()).digest()[:2], "big") % 10 != 0
    ]
    val_sequences = [row for row in usable if row not in train_sequences]
    for name, rows_to_write in (("train", train_sequences), ("val", val_sequences)):
        (output_dir / f"{name}_unit_sequences.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_to_write),
            encoding="utf-8",
        )
    metrics["context_train_sentences"] = len(train_sequences)
    metrics["context_val_sentences"] = len(val_sequences)
    metrics["single_sentence_groups_excluded"] = sum(size == 1 for size in group_sizes.values())
    (output_dir / "continuous_unit_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "unit_recurrence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unit", "patches", "clips", "sources"])
        writer.writeheader()
        for unit in range(clusters):
            writer.writerow(
                {"unit": unit, "patches": int(usage[unit]), "clips": int(clip_counts[unit]), "sources": int(source_counts[unit])}
            )
    return final, metrics


@torch.no_grad()
def evaluate_lexical_units(
    checkpoint: Path,
    codebook: MiniBatchKMeans,
    manifest: Path,
    root: Path,
    output_dir: Path,
    layer: int,
    patch_frames: int,
    patch_stride: int,
    permutations: int,
    seed: int,
) -> dict:
    device = choose_device()
    model = _load_contrastive_model(checkpoint, device)
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    histograms = []
    for row in tqdm(rows, desc="Testing units on held-out ASL Citizen"):
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        streams, valid = _load_standard_skeleton(path, model.config.max_frames)
        batched = {name: value.unsqueeze(0).to(device) for name, value in streams.items()}
        valid_batch = valid.unsqueeze(0).to(device)
        layers = model.encode(
            batched, valid_batch, causal=model.config.causal_attention, return_all_layers=True
        )
        assert isinstance(layers, list)
        patches = _pool_patches(layers[layer - 1], valid_batch, patch_frames, patch_stride)[0]
        units = codebook.predict(patches)
        collapsed = _collapse_runs(units)
        histogram = np.bincount(collapsed, minlength=codebook.n_clusters).astype(np.float32)
        histogram /= np.linalg.norm(histogram).clip(min=1e-8)
        histograms.append(histogram)
    matrix = np.stack(histograms)
    concepts = np.asarray([row["id"].strip().lower() for row in rows])
    signers = np.asarray([row["participant_id"].strip() for row in rows])

    def score(labels: np.ndarray) -> tuple[float, float, float, float]:
        ranks = []
        same_distances = []
        different_distances = []
        for signer in sorted(set(signers.tolist())):
            query_indices = np.flatnonzero(signers == signer)
            train_indices = np.flatnonzero(signers != signer)
            candidates, inverse = np.unique(labels[train_indices], return_inverse=True)
            sums = np.zeros((len(candidates), matrix.shape[1]), dtype=np.float32)
            np.add.at(sums, inverse, matrix[train_indices])
            counts = np.bincount(inverse, minlength=len(candidates)).astype(np.float32)
            prototypes = sums / counts[:, None].clip(min=1)
            prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-8)
            similarities = matrix[query_indices] @ prototypes.T
            for row_index, query_index in enumerate(query_indices):
                matches = np.flatnonzero(candidates == labels[query_index])
                if len(matches) == 0:
                    ranks.append(len(candidates) + 1)
                    different_distances.extend((1 - similarities[row_index]).tolist())
                    continue
                target = int(matches[0])
                order = np.argsort(-similarities[row_index])
                ranks.append(int(np.flatnonzero(order == target)[0]) + 1)
                same_distances.append(float(1 - similarities[row_index, target]))
                different_distances.extend(
                    (1 - np.delete(similarities[row_index], target)).tolist()
                )
        ranks = np.asarray(ranks)
        separation = (
            float(np.mean(different_distances) - np.mean(same_distances))
            if same_distances
            else float("nan")
        )
        return (
            float((ranks == 1).mean()),
            float((ranks <= 5).mean()),
            float((1 / ranks).mean()),
            separation,
        )

    top1, top5, mrr, separation = score(concepts)
    rng = np.random.default_rng(seed)
    null_top1 = []
    for _ in tqdm(range(permutations), desc="Lexical label permutations", leave=False):
        permuted = concepts.copy()
        rng.shuffle(permuted)
        null_top1.append(score(permuted)[0])
    p_value = (1 + sum(value >= top1 for value in null_top1)) / (permutations + 1)
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
    }
    (output_dir / "lexical_unit_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe recurring visual units in continuous signing")
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = args.output_dir / "continuous_patch_features.npz"
    extract_continuous_patch_features(
        args.checkpoint,
        args.archive,
        args.annotations,
        features,
        args.sample_clips,
        args.layer,
        args.patch_frames,
        args.patch_stride,
        args.batch_size,
        args.seed,
    )
    codebook, continuous = probe_codebook_stability(
        features, args.output_dir, args.clusters, args.seed
    )
    lexical = evaluate_lexical_units(
        args.checkpoint,
        codebook,
        args.lexical_manifest,
        args.root,
        args.output_dir,
        args.layer,
        args.patch_frames,
        args.patch_stride,
        args.permutations,
        args.seed,
    )
    # These are transparent engineering gates, not established scientific cutoffs.
    gate = {
        "thresholds_are_exploratory": True,
        "split_half_ami_at_least_0_30": continuous["split_half_ami"] >= 0.30,
        "centroid_cosine_at_least_0_70": continuous["matched_centroid_cosine_mean"] >= 0.70,
        "at_least_half_units_cross_5_sources": continuous["units_in_at_least_5_sources"] >= args.clusters / 2,
        "lexical_top1_above_chance_p_0_05": lexical["top1_permutation_p"] < 0.05 and lexical["top1"] > lexical["chance_top1"],
    }
    gate["pilot_pass"] = all(value for key, value in gate.items() if key not in {"thresholds_are_exploratory", "pilot_pass"})
    (args.output_dir / "pilot_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {"continuous": continuous, "lexical": lexical, "gate": gate}
    print("unit_probe_summary=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
