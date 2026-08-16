"""Primary paired lexical comparison: real context minus shuffled context."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sign_semantics.lexical_retrieval import (
    TokenMetadata,
    _cosine_scores,
    load_token_layers,
    load_token_manifest,
    raw_kinematic_layer,
)


def token_outcomes(values: np.ndarray, metadata: list[TokenMetadata]) -> dict[str, np.ndarray]:
    concepts = np.asarray([item.concept for item in metadata])
    signers = np.asarray([item.signer for item in metadata])
    outcomes = {
        name: np.full(len(metadata), np.nan)
        for name in ("top1", "top5", "mrr", "distance_delta")
    }
    for signer in sorted(set(signers)):
        test, train = signers == signer, signers != signer
        candidates = sorted(set(concepts[train]))
        query = np.flatnonzero(test & np.isin(concepts, candidates))
        prototypes = np.stack(
            [values[train & (concepts == concept)].mean(0) for concept in candidates]
        )
        scores = _cosine_scores(values[query], prototypes)
        order = np.argsort(-scores, axis=1, kind="stable")
        inverse = np.empty_like(order)
        inverse[np.arange(len(order))[:, None], order] = np.arange(len(candidates))
        lookup = {concept: index for index, concept in enumerate(candidates)}
        targets = np.asarray([lookup[concepts[index]] for index in query])
        ranks = inverse[np.arange(len(query)), targets] + 1
        same_distance = 1 - scores[np.arange(len(query)), targets]
        other = scores.copy()
        other[np.arange(len(query)), targets] = np.nan
        outcomes["top1"][query] = ranks == 1
        outcomes["top5"][query] = ranks <= 5
        outcomes["mrr"][query] = 1 / ranks
        outcomes["distance_delta"][query] = np.nanmean(1 - other, axis=1) - same_distance
    if any(np.isnan(value).any() for value in outcomes.values()):
        raise ValueError("Every held-out token must have a prototype among other signers")
    return outcomes


def run(args: argparse.Namespace) -> None:
    metadata = load_token_manifest(args.manifest)
    layers = {
        "real": load_token_layers(args.real, metadata),
        "shuffled": load_token_layers(args.shuffled, metadata),
        "random": load_token_layers(args.random, metadata),
        "raw_kinematics": {"raw_kinematics": raw_kinematic_layer(metadata, args.root)},
    }
    if list(layers["real"]) != list(layers["shuffled"]):
        raise ValueError("Real and shuffled representations must have identical layers")
    outcomes = {
        model: {
            layer: token_outcomes(values, metadata)
            for layer, values in model_layers.items()
        }
        for model, model_layers in layers.items()
    }
    layer_rows = []
    for model, model_layers in outcomes.items():
        for layer, metrics in model_layers.items():
            layer_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    **{name: float(value.mean()) for name, value in metrics.items()},
                    "n_tokens": len(metadata),
                    "n_signers": len({item.signer for item in metadata}),
                }
            )

    signer_names = sorted({item.signer for item in metadata})
    signer_indices = {
        signer: np.flatnonzero([item.signer == signer for item in metadata])
        for signer in signer_names
    }
    rng = np.random.default_rng(args.seed)
    test_rows = []
    real_layers = list(layers["real"])
    for metric in ("top1", "top5", "mrr", "distance_delta"):
        differences = np.stack(
            [
                outcomes["real"][layer][metric]
                - outcomes["shuffled"][layer][metric]
                for layer in real_layers
            ]
        )
        observed = differences.mean(axis=1)
        signer_contribution = np.stack(
            [
                differences[:, indices].sum(axis=1) / differences.shape[1]
                for indices in signer_indices.values()
            ],
            axis=1,
        )
        exceed = np.zeros(len(real_layers), dtype=np.int64)
        exceed_fwer = np.zeros(len(real_layers), dtype=np.int64)
        remaining = args.permutations
        while remaining:
            batch = min(1000, remaining)
            signs = rng.choice((-1.0, 1.0), size=(len(signer_names), batch))
            null = signer_contribution @ signs
            exceed += np.count_nonzero(null >= observed[:, None], axis=1)
            exceed_fwer += np.count_nonzero(
                null.max(axis=0)[None, :] >= observed[:, None], axis=1
            )
            remaining -= batch
        for index, layer in enumerate(real_layers):
            test_rows.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "real_value": float(outcomes["real"][layer][metric].mean()),
                    "shuffled_value": float(outcomes["shuffled"][layer][metric].mean()),
                    "real_minus_shuffled": float(observed[index]),
                    "signer_cluster_permutation_p_one_sided": float(
                        (exceed[index] + 1) / (args.permutations + 1)
                    ),
                    "signer_cluster_permutation_p_fwer": float(
                        (exceed_fwer[index] + 1) / (args.permutations + 1)
                    ),
                    "fwer_family": "all_6_layers_within_metric_real_minus_shuffled",
                    "permutations": args.permutations,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("lexical_layerwise.csv", layer_rows),
        ("lexical_real_minus_shuffled_tests.csv", test_rows),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved lexical context comparison to {args.output_dir}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--real", type=Path, required=True)
    result.add_argument("--shuffled", type=Path, required=True)
    result.add_argument("--random", type=Path, required=True)
    result.add_argument("--root", type=Path, default=Path("."))
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--permutations", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=71)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
