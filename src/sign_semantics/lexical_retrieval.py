"""Strict leave-one-signer-out lexical retrieval evaluation.

This module deliberately treats a signer as the unit that must be absent from
prototype construction.  It is therefore not a leave-one-token-out analysis:
for every held-out signer, *all* of that signer's tokens are excluded before a
concept prototype is calculated.  Candidate labels are every concept for
which the remaining signers can build a prototype, so the held-out signer's
presented concept inventory is never leaked into the retrieval decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import STREAM_JOINTS


@dataclass(frozen=True)
class TokenMetadata:
    """One token's concept, unique sample ID, signer and source NPZ path."""

    concept: str
    sample_id: str
    signer: str
    path: Path


def _concept(value: object) -> str:
    result = str(value).strip().lower()
    if not result:
        raise ValueError("Empty concept ID")
    return result


def _sample_id(row: dict[str, str]) -> str:
    result = row.get("sample_id", "").strip()
    if not result:
        result = Path(row["path"]).stem
    if not result:
        raise ValueError("Empty sample_id")
    return result


def load_token_manifest(path: Path) -> list[TokenMetadata]:
    """Load the authoritative token-to-signer mapping without fallbacks.

    ``participant_id`` is mandatory: inferring it from filenames could silently
    reintroduce a signer leak.  ``sample_id`` may be omitted only when it is
    unambiguously the source path stem, matching lexical extraction's default.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "path", "participant_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must be non-empty and contain {sorted(required)}")
    metadata: list[TokenMetadata] = []
    for row in rows:
        signer = row["participant_id"].strip()
        if not signer:
            raise ValueError(f"{path}: empty participant_id for sample {_sample_id(row)!r}")
        metadata.append(
            TokenMetadata(
                concept=_concept(row["id"]),
                sample_id=_sample_id(row),
                signer=signer,
                path=Path(row["path"]),
            )
        )
    sample_ids = [item.sample_id for item in metadata]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path}: duplicate sample_id values")
    return metadata


def _layer_keys(payload: np.lib.npyio.NpzFile, source: Path) -> list[str]:
    keys = sorted(
        (key for key in payload.files if re.fullmatch(r"layer_\d+", key)),
        key=lambda value: int(value.split("_")[1]),
    )
    if not keys:
        raise ValueError(f"{source}: expected one or more layer_XX arrays")
    return keys


def load_token_layers(path: Path, metadata: list[TokenMetadata]) -> dict[str, np.ndarray]:
    """Load token-level representations and exactly align them to a manifest.

    A partial intersection is intentionally an error.  It is otherwise too
    easy to compare different controls on a different set of signer tokens.
    """
    with np.load(path, allow_pickle=False) as payload:
        if "ids" not in payload or "sample_ids" not in payload:
            raise ValueError(f"{path}: token NPZ must contain ids and sample_ids")
        ids = [_concept(value) for value in payload["ids"]]
        sample_ids = [str(value).strip() for value in payload["sample_ids"]]
        if not sample_ids or any(not value for value in sample_ids):
            raise ValueError(f"{path}: sample_ids must be non-empty")
        if len(ids) != len(sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{path}: ids/sample_ids must have equal, unique token rows")
        index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
        expected = {item.sample_id for item in metadata}
        observed = set(index)
        if expected != observed:
            raise ValueError(
                f"{path}: token sample IDs do not exactly match manifest "
                f"(missing={len(expected - observed)}, extra={len(observed - expected)})"
            )
        ordering = np.asarray([index[item.sample_id] for item in metadata], dtype=np.int64)
        for item, source_index in zip(metadata, ordering, strict=True):
            if ids[int(source_index)] != item.concept:
                raise ValueError(
                    f"{path}: concept mismatch for {item.sample_id}: "
                    f"NPZ={ids[int(source_index)]!r}, manifest={item.concept!r}"
                )
        layers: dict[str, np.ndarray] = {}
        for key in _layer_keys(payload, path):
            values = np.asarray(payload[key], dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != len(sample_ids):
                raise ValueError(f"{path}:{key} must have shape (n_tokens, n_features)")
            values = values[ordering]
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{key} contains non-finite values")
            layers[key] = values
    return layers


def raw_kinematic_layer(metadata: list[TokenMetadata], root: Path) -> np.ndarray:
    """Create a fixed pooled raw-kinematics baseline from portable input NPZs.

    The representation concatenates each coordinate's temporal mean, standard
    deviation, mean absolute velocity, and velocity standard deviation.  This
    is intentionally model-free and uses only the same normalized coordinates
    supplied to the encoders.
    """
    vectors: list[np.ndarray] = []
    for item in metadata:
        path = item.path if item.path.is_absolute() else root / item.path
        with np.load(path, allow_pickle=False) as payload:
            streams: list[np.ndarray] = []
            length: int | None = None
            for name in STREAM_JOINTS:
                if name not in payload:
                    raise ValueError(f"{path}: raw baseline requires {name}")
                values = np.asarray(payload[name], dtype=np.float64)
                if values.ndim != 3 or values.shape[-1] != 2:
                    raise ValueError(f"{path}:{name} must have shape (frames,joints,2)")
                if length is None:
                    length = len(values)
                elif len(values) != length:
                    raise ValueError(f"{path}: stream frame counts differ")
                streams.append(values.reshape(len(values), -1))
        assert length is not None
        if length < 1:
            raise ValueError(f"{path}: raw baseline cannot use an empty sequence")
        frames = np.concatenate(streams, axis=1)
        velocity = np.diff(frames, axis=0) if len(frames) > 1 else np.zeros_like(frames)
        vector = np.concatenate(
            [frames.mean(0), frames.std(0), np.abs(velocity).mean(0), velocity.std(0)]
        )
        if not np.isfinite(vector).all():
            raise ValueError(f"{path}: raw baseline contains non-finite values")
        vectors.append(vector)
    return np.stack(vectors)


def _cosine_scores(queries: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    prototype_norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if np.any(query_norm == 0) or np.any(prototype_norm == 0):
        raise ValueError("Cosine retrieval cannot use a zero-norm token or prototype")
    return (queries / query_norm) @ (prototypes / prototype_norm).T


def _metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def evaluate_cross_signer_retrieval(
    values: np.ndarray,
    metadata: list[TokenMetadata],
    permutations: int = 10_000,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one representation layer with complete signer holdouts.

    Prototype labels are permuted only after strict signer-exclusive prototypes
    and candidate pools have been fixed.  This conditional label test changes
    the concept-to-prototype association while retaining the actual score and
    signer structure.
    """
    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    if values.ndim != 2 or len(values) != len(metadata):
        raise ValueError("values must be a two-dimensional array aligned to metadata")
    if len(metadata) < 2 or len({item.signer for item in metadata}) < 2:
        raise ValueError("Need tokens from at least two signers")
    if not np.isfinite(values).all():
        raise ValueError("values contains non-finite entries")

    concepts = np.asarray([item.concept for item in metadata])
    signers = np.asarray([item.signer for item in metadata])
    rank_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    actual_ranks: list[np.ndarray] = []
    same_distances: list[np.ndarray] = []
    different_distances: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []

    for signer in sorted(set(signers)):
        test_mask = signers == signer
        train_mask = ~test_mask
        training_concepts = set(concepts[train_mask])
        candidates = sorted(training_concepts)
        if not candidates:
            audits.append(
                {
                    "test_signer": signer,
                    "test_tokens": int(test_mask.sum()),
                    "training_tokens": int(train_mask.sum()),
                    "eligible_candidate_concepts": 0,
                    "evaluated_tokens": 0,
                    "prototype_test_signer_tokens": 0,
                    "strict_no_signer_leakage": True,
                }
            )
            continue
        query_indices = np.flatnonzero(test_mask & np.isin(concepts, candidates))
        prototype_indices: list[np.ndarray] = []
        for candidate in candidates:
            indices = np.flatnonzero(train_mask & (concepts == candidate))
            if not len(indices):  # Guard against future changes to candidate construction.
                raise AssertionError("Candidate prototype unexpectedly has no training tokens")
            if np.any(signers[indices] == signer):
                raise AssertionError("Signer leakage while constructing a concept prototype")
            prototype_indices.append(indices)
        prototypes = np.stack([values[indices].mean(axis=0) for indices in prototype_indices])
        scores = _cosine_scores(values[query_indices], prototypes)
        order = np.argsort(-scores, axis=1, kind="stable")
        inverse_order = np.empty_like(order)
        inverse_order[np.arange(len(order))[:, None], order] = np.arange(len(candidates))
        candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
        targets = np.asarray([candidate_index[concepts[index]] for index in query_indices])
        ranks = inverse_order[np.arange(len(query_indices)), targets] + 1
        actual_ranks.append(ranks)
        rank_chunks.append((inverse_order + 1, scores, targets))
        same_similarity = scores[np.arange(len(query_indices)), targets]
        same_distances.append(1.0 - same_similarity)
        if len(candidates) > 1:
            different = scores.copy()
            different[np.arange(len(query_indices)), targets] = np.nan
            different_distances.append(1.0 - different[~np.isnan(different)])
        signer_metrics = _metrics_from_ranks(ranks)
        audits.append(
            {
                "test_signer": signer,
                "test_tokens": int(test_mask.sum()),
                "training_tokens": int(train_mask.sum()),
                "eligible_candidate_concepts": len(candidates),
                "evaluated_tokens": len(query_indices),
                "prototype_test_signer_tokens": 0,
                "strict_no_signer_leakage": True,
                **signer_metrics,
            }
        )

    if not actual_ranks:
        raise ValueError("No held-out signer has a concept also represented by other signers")
    ranks = np.concatenate(actual_ranks)
    result: dict[str, Any] = _metrics_from_ranks(ranks)
    result["n_queries"] = int(len(ranks))
    result["n_signers"] = int(sum(row["evaluated_tokens"] > 0 for row in audits))
    result["same_cosine_distance"] = float(np.concatenate(same_distances).mean())
    if different_distances:
        result["different_cosine_distance"] = float(np.concatenate(different_distances).mean())
        result["different_minus_same_distance"] = (
            result["different_cosine_distance"] - result["same_cosine_distance"]
        )
    else:
        result["different_cosine_distance"] = float("nan")
        result["different_minus_same_distance"] = float("nan")
    result["permutations"] = permutations

    if permutations:
        rng = np.random.default_rng(seed)
        null = {
            "top1": np.empty(permutations),
            "top5": np.empty(permutations),
            "mrr": np.empty(permutations),
        }
        null_distance_delta = np.full(permutations, np.nan)
        for permutation_index in range(permutations):
            null_ranks: list[np.ndarray] = []
            null_same: list[np.ndarray] = []
            null_different: list[np.ndarray] = []
            for rank_positions, scores, targets in rank_chunks:
                label_to_prototype = rng.permutation(rank_positions.shape[1])
                mapped = label_to_prototype[targets]
                null_ranks.append(rank_positions[np.arange(len(targets)), mapped])
                null_same.append(1.0 - scores[np.arange(len(targets)), mapped])
                if scores.shape[1] > 1:
                    different = scores.copy()
                    different[np.arange(len(targets)), mapped] = np.nan
                    null_different.append(1.0 - different[~np.isnan(different)])
            permuted_ranks = np.concatenate(null_ranks)
            null_metrics = _metrics_from_ranks(permuted_ranks)
            for name, value in null_metrics.items():
                null[name][permutation_index] = value
            if null_different:
                null_distance_delta[permutation_index] = (
                    np.concatenate(null_different).mean() - np.concatenate(null_same).mean()
                )
        for name, distribution in null.items():
            result[f"{name}_permutation_p"] = float(
                (1 + np.count_nonzero(distribution >= result[name])) / (permutations + 1)
            )
            result[f"{name}_permutation_null_mean"] = float(distribution.mean())
        if np.isfinite(result["different_minus_same_distance"]):
            result["distance_permutation_p"] = float(
                (
                    1
                    + np.count_nonzero(
                        null_distance_delta >= result["different_minus_same_distance"]
                    )
                )
                / (permutations + 1)
            )
            result["distance_permutation_null_mean"] = float(np.nanmean(null_distance_delta))
        else:
            result["distance_permutation_p"] = float("nan")
            result["distance_permutation_null_mean"] = float("nan")
    else:
        for name in ("top1", "top5", "mrr"):
            result[f"{name}_permutation_p"] = float("nan")
            result[f"{name}_permutation_null_mean"] = float("nan")
        result["distance_permutation_p"] = float("nan")
        result["distance_permutation_null_mean"] = float("nan")
    return result, audits


def _parse_representation(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("representation must be NAME=TOKEN_NPZ")
    name, filename = value.split("=", 1)
    if not name.strip() or not filename.strip():
        raise argparse.ArgumentTypeError("representation must be NAME=TOKEN_NPZ")
    return name.strip(), Path(filename)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_lexical_retrieval(
    manifest: Path,
    representations: list[tuple[str, Path]],
    output_dir: Path,
    permutations: int = 10_000,
    seed: int = 0,
    raw_kinematics: bool = False,
    root: Path = Path("."),
) -> None:
    """Run every supplied model/layer under identical strict signer holdouts."""
    metadata = load_token_manifest(manifest)
    if not representations and not raw_kinematics:
        raise ValueError("Provide --representation and/or --raw-kinematics")
    names = [name for name, _ in representations]
    if raw_kinematics:
        names.append("raw_kinematics")
    if len(names) != len(set(names)):
        raise ValueError("Representation names must be unique")

    all_layers: list[tuple[str, str, np.ndarray]] = []
    for name, path in representations:
        for layer, values in load_token_layers(path, metadata).items():
            all_layers.append((name, layer, values))
    if raw_kinematics:
        all_layers.append(("raw_kinematics", "raw_kinematics", raw_kinematic_layer(metadata, root)))

    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for model, layer, values in all_layers:
        # Different deterministic seeds avoid identical shuffled label draws
        # across model/layer outputs while remaining fully reproducible.
        layer_seed = seed + sum(ord(character) for character in f"{model}:{layer}")
        result, audits = evaluate_cross_signer_retrieval(values, metadata, permutations, layer_seed)
        summary_rows.append({"model": model, "layer": layer, **result})
        audit_rows.extend({"model": model, "layer": layer, **row} for row in audits)

    # The layerwise tests form one confirmatory family.  Preserve each exact
    # conditional-permutation p value, and add a dependence-robust Bonferroni
    # FWER correction across every supplied model/layer.  This prevents a best
    # layer selected from the same table from being reported with an
    # uncorrected p value.
    family_size = len(summary_rows)
    for row in summary_rows:
        for metric in ("top1", "top5", "mrr", "distance"):
            key = f"{metric}_permutation_p"
            value = float(row[key])
            row[f"{key}_fwer"] = (
                min(1.0, value * family_size) if np.isfinite(value) else float("nan")
            )
        row["fwer_method"] = "bonferroni_all_models_layers"
        row["fwer_family_size"] = family_size
    _write_csv(output_dir / "cross_signer_lexical_retrieval.csv", summary_rows)
    _write_csv(output_dir / "cross_signer_lexical_retrieval_audit.csv", audit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "design": (
            "leave-one-signer-out prototypes; candidate labels = every concept with a "
            "prototype among the remaining signers"
        ),
        "manifest": str(manifest),
        "representations": [{"name": name, "path": str(path)} for name, path in representations],
        "raw_kinematics": raw_kinematics,
        "permutations": permutations,
        "seed": seed,
        "n_tokens": len(metadata),
        "n_signers": len({item.signer for item in metadata}),
    }
    (output_dir / "cross_signer_lexical_retrieval_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict cross-signer lexical retrieval with signer-exclusive concept prototypes"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Token manifest with id,path,participant_id"
    )
    parser.add_argument(
        "--representation",
        type=_parse_representation,
        action="append",
        default=[],
        metavar="NAME=TOKEN_NPZ",
        help="Repeat for masked, next_frame, contrastive, random, or other token-level NPZs",
    )
    parser.add_argument(
        "--masked", type=Path, help="Convenience alias for --representation masked=..."
    )
    parser.add_argument(
        "--next-frame", type=Path, dest="next_frame", help="Alias for next_frame=..."
    )
    parser.add_argument("--contrastive", type=Path, help="Alias for contrastive=...")
    parser.add_argument("--random", type=Path, help="Alias for random=...")
    parser.add_argument(
        "--raw-kinematics",
        action="store_true",
        help="Add pooled standardized-coordinate and velocity baseline from manifest NPZ paths",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="Root for relative manifest paths"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    representations = list(args.representation)
    for name in ("masked", "next_frame", "contrastive", "random"):
        path = getattr(args, name)
        if path is not None:
            representations.append((name, path))
    run_lexical_retrieval(
        args.manifest,
        representations,
        args.output_dir,
        args.permutations,
        args.seed,
        args.raw_kinematics,
        args.root,
    )


if __name__ == "__main__":
    main()
