from __future__ import annotations

"""Quantitative audit of the matched real/shuffled skeleton context corpus."""

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .context_shuffle import (
    ContextBatchCollator,
    SourceGroupedBatchSampler,
    source_groups_for_dataset,
)
from .data import YouTubeASLPoseDataset
from .skeleton_codebooks import PARTS, SkeletonCodebooks
from .utils import seed_everything


def _entropy(counter: Counter) -> float:
    counts = np.asarray(list(counter.values()), dtype=float)
    if not len(counts) or counts.sum() == 0:
        return 0.0
    probability = counts / counts.sum()
    return float(-(probability * np.log2(probability)).sum())


def _aligned_counts(first: Counter, second: Counter) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(first) | set(second), key=str)
    return (
        np.asarray([first[key] for key in keys], dtype=float),
        np.asarray([second[key] for key in keys], dtype=float),
    )


def _correlation(first: Counter, second: Counter) -> float:
    left, right = _aligned_counts(first, second)
    if len(left) < 2:
        return 1.0 if np.array_equal(left, right) else float("nan")
    if left.std() == 0 or right.std() == 0:
        return 1.0 if np.array_equal(left, right) else float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _js_divergence(first: Counter, second: Counter) -> float:
    left, right = _aligned_counts(first, second)
    if left.sum() == 0 or right.sum() == 0:
        return float("nan")
    return float(jensenshannon(left / left.sum(), right / right.sum(), base=2) ** 2)


def _bigrams(sequences: list[list[int]]) -> Counter:
    return Counter(
        (first, second)
        for sequence in sequences
        for first, second in zip(sequence[:-1], sequence[1:])
    )


def _cooccurrence(sequences: list[list[int]]) -> Counter:
    result: Counter = Counter()
    for sequence in sequences:
        units = sorted(set(sequence))
        result.update(combinations(units, 2))
    return result


def _adjacent_mutual_information(sequences: list[list[int]]) -> float:
    pairs = _bigrams(sequences)
    total = sum(pairs.values())
    if total == 0:
        return 0.0
    left = Counter()
    right = Counter()
    for (first, second), count in pairs.items():
        left[first] += count
        right[second] += count
    value = 0.0
    for (first, second), count in pairs.items():
        p_joint = count / total
        value += p_joint * math.log2(
            p_joint / ((left[first] / total) * (right[second] / total))
        )
    return float(value)


def diagnose(
    archive: Path,
    annotations: Path,
    codebooks: Path,
    output_dir: Path,
    max_frames: int,
    block_frames: int,
    boundary_mask_frames: int,
    batch_size: int,
    seed: int,
    limit_clips: int | None,
) -> None:
    seed_everything(seed)
    np.random.seed(seed)
    dataset = YouTubeASLPoseDataset(
        archive,
        annotations,
        max_frames=max_frames,
        training=True,
        limit_clips=limit_clips,
    )
    sampler = SourceGroupedBatchSampler(
        source_groups_for_dataset(dataset), batch_size, seed, True
    )
    collator = ContextBatchCollator(
        "shuffled",
        block_frames,
        boundary_mask_frames,
        seed,
        SkeletonCodebooks(codebooks, torch.device("cpu")),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )

    real_sequences: dict[str, dict[str, list[int]]] = {
        part: defaultdict(list) for part in PARTS
    }
    shuffled_sequences: dict[str, dict[str, list[int]]] = {
        part: defaultdict(list) for part in PARTS
    }
    lengths = []
    spans_per_sentence = Counter()
    duration_counts = Counter()
    source_counts = Counter()
    totals = Counter()
    mapping_path = output_dir / "block_mapping.jsonl.gz"
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(mapping_path, "wt", encoding="utf-8") as mapping_handle:
        for batch in tqdm(loader, desc="Auditing context manipulation", unit="batches"):
            batch_lengths = [int(value) for value in batch["context_original_lengths"]]
            lengths.extend(batch_lengths)
            source_counts.update(str(value) for value in batch["source_group"])
            audit = batch["context_batch_audit"]
            block_weight = float(audit["blocks"])
            totals["blocks"] += block_weight
            for key in (
                "moved_block_fraction",
                "cross_sentence_fraction",
                "same_source_fraction",
            ):
                totals[key] += float(audit[key]) * block_weight
            for key in (
                "join_count",
                "real_boundary_jump_sum",
                "assigned_boundary_jump_sum",
                "real_visible_boundary_jump_sum",
                "assigned_visible_boundary_jump_sum",
                "masked_boundary_frames",
                "original_valid_frames",
            ):
                totals[key] += float(audit[key])
            for record in batch["context_mapping"]:
                mapping_handle.write(json.dumps(record, sort_keys=True) + "\n")
                clip_id = str(record["destination_clip"])
                spans_per_sentence[clip_id] += 1
                duration_counts[int(record["destination_stop"]) - int(record["destination_start"])] += 1
                real = record["real_units"]
                assigned = record["assigned_units"]
                for part in PARTS:
                    real_sequences[part][clip_id].append(int(real[part]))
                    shuffled_sequences[part][clip_id].append(int(assigned[part]))

    summary: dict[str, object] = {
        "strategy": "fixed-duration local trajectory blocks reassigned across sentences",
        "reason_not_unified_pseudo_unit_spans": (
            "existing Spatial-SHuBERT has four independent frame-level targets and no "
            "validated unified span tokenizer"
        ),
        "sentences": len(lengths),
        "frames": int(sum(lengths)),
        "block_frames": block_frames,
        "boundary_mask_frames_per_side": boundary_mask_frames,
        "boundary_mask_fraction": totals["masked_boundary_frames"]
        / max(totals["original_valid_frames"], 1),
        "blocks": int(sum(duration_counts.values())),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "length_quantiles": {
            str(q): float(np.quantile(lengths, q)) for q in (0, 0.25, 0.5, 0.75, 1)
        },
        "blocks_per_sentence_mean": float(np.mean(list(spans_per_sentence.values()))),
        "blocks_per_sentence_std": float(np.std(list(spans_per_sentence.values()))),
        "duration_distribution": {str(key): value for key, value in sorted(duration_counts.items())},
        "source_groups": len(source_counts),
        "source_sentence_distribution": dict(sorted(source_counts.items())),
        "moved_block_fraction": totals["moved_block_fraction"]
        / max(totals["blocks"], 1),
        "cross_sentence_fraction": totals["cross_sentence_fraction"]
        / max(totals["blocks"], 1),
        "same_source_fraction": totals["same_source_fraction"]
        / max(totals["blocks"], 1),
        "real_boundary_jump_mean": totals["real_boundary_jump_sum"]
        / max(totals["join_count"], 1),
        "shuffled_boundary_jump_mean": totals["assigned_boundary_jump_sum"]
        / max(totals["join_count"], 1),
        "real_visible_boundary_jump_mean": totals[
            "real_visible_boundary_jump_sum"
        ]
        / max(totals["join_count"], 1),
        "shuffled_visible_boundary_jump_mean": totals[
            "assigned_visible_boundary_jump_sum"
        ]
        / max(totals["join_count"], 1),
        "seed": seed,
        "mapping": str(mapping_path),
        "training_resources_used": "skeleton frames, structural clip IDs, source-video IDs, frozen skeleton codebooks only",
    }
    part_rows = []
    for part in PARTS:
        ids = sorted(real_sequences[part])
        real = [real_sequences[part][identifier] for identifier in ids]
        shuffled = [shuffled_sequences[part][identifier] for identifier in ids]
        real_unigrams = Counter(unit for sequence in real for unit in sequence)
        shuffled_unigrams = Counter(unit for sequence in shuffled for unit in sequence)
        real_bigrams, shuffled_bigrams = _bigrams(real), _bigrams(shuffled)
        real_cooccurrence, shuffled_cooccurrence = _cooccurrence(real), _cooccurrence(shuffled)
        row = {
            "part": part,
            "real_unigram_entropy_bits": _entropy(real_unigrams),
            "shuffled_unigram_entropy_bits": _entropy(shuffled_unigrams),
            "unigram_correlation": _correlation(real_unigrams, shuffled_unigrams),
            "unigram_js_divergence": _js_divergence(real_unigrams, shuffled_unigrams),
            "real_adjacent_mutual_information_bits": _adjacent_mutual_information(real),
            "shuffled_adjacent_mutual_information_bits": _adjacent_mutual_information(shuffled),
            "bigram_correlation": _correlation(real_bigrams, shuffled_bigrams),
            "bigram_js_divergence": _js_divergence(real_bigrams, shuffled_bigrams),
            "sentence_cooccurrence_correlation": _correlation(
                real_cooccurrence, shuffled_cooccurrence
            ),
            "sentence_cooccurrence_js_divergence": _js_divergence(
                real_cooccurrence, shuffled_cooccurrence
            ),
            "unigram_counts_exactly_preserved": real_unigrams == shuffled_unigrams,
        }
        part_rows.append(row)
    summary["part_statistics"] = part_rows
    summary["all_part_unigrams_exactly_preserved"] = all(
        bool(row["unigram_counts_exactly_preserved"]) for row in part_rows
    )
    finite_bigram = [
        float(row["bigram_correlation"])
        for row in part_rows
        if np.isfinite(float(row["bigram_correlation"]))
    ]
    finite_cooccurrence = [
        float(row["sentence_cooccurrence_correlation"])
        for row in part_rows
        if np.isfinite(float(row["sentence_cooccurrence_correlation"]))
    ]
    visible_ratio = summary["shuffled_visible_boundary_jump_mean"] / max(
        summary["real_visible_boundary_jump_mean"], 1e-8
    )
    gate = {
        "thresholds_are_engineering_checks_not_scientific_cutoffs": True,
        "unigram_marginals_exact": summary["all_part_unigrams_exactly_preserved"],
        "cross_sentence_fraction_at_least_0_90": summary[
            "cross_sentence_fraction"
        ]
        >= 0.90,
        "mean_bigram_correlation_below_0_95": bool(finite_bigram)
        and float(np.mean(finite_bigram)) < 0.95,
        "mean_sentence_cooccurrence_correlation_below_0_95": bool(
            finite_cooccurrence
        )
        and float(np.mean(finite_cooccurrence)) < 0.95,
        "visible_boundary_jump_ratio_between_0_5_and_2": 0.5
        <= visible_ratio
        <= 2.0,
        "visible_boundary_jump_ratio": visible_ratio,
    }
    gate["diagnostic_pass"] = all(
        bool(value)
        for key, value in gate.items()
        if key
        not in {
            "thresholds_are_engineering_checks_not_scientific_cutoffs",
            "visible_boundary_jump_ratio",
            "diagnostic_pass",
        }
    )
    summary["gate"] = gate
    (output_dir / "context_manipulation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "context_manipulation_part_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(part_rows[0]))
        writer.writeheader()
        writer.writerows(part_rows)
    if not summary["all_part_unigrams_exactly_preserved"]:
        raise AssertionError("Block reassignment failed to preserve unit marginals")
    if not gate["diagnostic_pass"]:
        raise RuntimeError(
            "Context manipulation failed a pre-training engineering gate; "
            "full training must not start. See context_manipulation_summary.json"
        )
    print("context_manipulation=" + json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--codebooks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--block-frames", type=int, default=16)
    parser.add_argument("--boundary-mask-frames", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-clips", type=int)
    args = parser.parse_args()
    diagnose(**vars(args))


if __name__ == "__main__":
    main()
