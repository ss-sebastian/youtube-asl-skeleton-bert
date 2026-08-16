"""Primary/robustness human-RSA test: real context minus shuffled context."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata


def layer_keys(payload: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(
        (key for key in payload.files if re.fullmatch(r"layer_\d+", key)),
        key=lambda key: int(key.split("_")[1]),
    )


def representation_vectors(
    path: Path, common: list[str], metric: str
) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value).strip().lower() for value in payload["ids"]]
        lookup = {identifier: index for index, identifier in enumerate(ids)}
        order = [lookup[identifier] for identifier in common]
        layers = layer_keys(payload)
        vectors = np.stack(
            [pdist(np.asarray(payload[layer], dtype=float)[order], metric=metric) for layer in layers]
        )
    return layers, vectors


def centered_ranks(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([rankdata(vector) for vector in np.atleast_2d(vectors)])
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError("A human/model RDM is constant")
    return values, norms


def run(args: argparse.Namespace) -> None:
    paths = [args.human_rdm, args.real, args.shuffled, args.random, args.raw]
    id_sets = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            id_sets.append({str(value).strip().lower() for value in payload["ids"]})
    common = sorted(set.intersection(*id_sets))
    if len(common) < 3:
        raise ValueError("Human and all four model/control files share fewer than three codes")

    with np.load(args.human_rdm, allow_pickle=False) as payload:
        ids = [str(value).strip().lower() for value in payload["ids"]]
        lookup = {identifier: index for index, identifier in enumerate(ids)}
        order = [lookup[identifier] for identifier in common]
        human_matrix = np.asarray(payload["rdm"], dtype=float)[np.ix_(order, order)]
    triangle = np.triu_indices(len(common), 1)
    human_vector = human_matrix[triangle]

    models = {}
    for name, path in (
        ("real", args.real),
        ("shuffled", args.shuffled),
        ("random", args.random),
        ("raw_kinematics", args.raw),
    ):
        layers, vectors = representation_vectors(path, common, args.metric)
        ranks, norms = centered_ranks(vectors)
        models[name] = {"layers": layers, "ranks": ranks, "norms": norms}
    if models["real"]["layers"] != models["shuffled"]["layers"]:
        raise ValueError("Real and shuffled models must expose identical layers")

    human_rank, human_norm = centered_ranks(human_vector)
    human_rank, human_norm = human_rank[0], human_norm[0]
    rho = {
        name: (model["ranks"] @ human_rank) / (model["norms"] * human_norm)
        for name, model in models.items()
    }
    observed = rho["real"] - rho["shuffled"]
    human_rank_matrix = np.zeros_like(human_matrix)
    human_rank_matrix[triangle] = human_rank
    human_rank_matrix[(triangle[1], triangle[0])] = human_rank
    rng = np.random.default_rng(args.seed)
    exceed = np.zeros(len(observed), dtype=np.int64)
    exceed_fwer = np.zeros(len(observed), dtype=np.int64)
    raw_exceed = {
        name: np.zeros(len(model["layers"]), dtype=np.int64)
        for name, model in models.items()
    }
    raw_exceed_fwer = {
        name: np.zeros(len(model["layers"]), dtype=np.int64)
        for name, model in models.items()
    }
    for _ in range(args.permutations):
        permutation = rng.permutation(len(common))
        permuted = human_rank_matrix[np.ix_(permutation, permutation)][triangle]
        null_real = (models["real"]["ranks"] @ permuted) / (
            models["real"]["norms"] * human_norm
        )
        null_shuffled = (models["shuffled"]["ranks"] @ permuted) / (
            models["shuffled"]["norms"] * human_norm
        )
        null_by_model = {
            name: (model["ranks"] @ permuted) / (model["norms"] * human_norm)
            for name, model in models.items()
        }
        for name, null_model in null_by_model.items():
            raw_exceed[name] += np.abs(null_model) >= np.abs(rho[name])
            raw_exceed_fwer[name] += np.max(np.abs(null_model)) >= np.abs(rho[name])
        null = null_real - null_shuffled
        exceed += null >= observed
        exceed_fwer += np.max(null) >= observed
    layer_rows = []
    for name, model in models.items():
        for index, layer in enumerate(model["layers"]):
            layer_rows.append(
                {
                    "model": name,
                    "layer": layer,
                    "n_codes": len(common),
                    "spearman_rho": float(rho[name][index]),
                    "permutation_p_two_sided": float(
                        (raw_exceed[name][index] + 1) / (args.permutations + 1)
                    ),
                    "permutation_p_fwer_within_representation": float(
                        (raw_exceed_fwer[name][index] + 1)
                        / (args.permutations + 1)
                    ),
                    "target": args.target_name,
                    "metric": args.metric,
                }
            )
    test_rows = []
    for index, layer in enumerate(models["real"]["layers"]):
        test_rows.append(
            {
                "layer": layer,
                "n_codes": len(common),
                "real_spearman_rho": float(rho["real"][index]),
                "shuffled_spearman_rho": float(rho["shuffled"][index]),
                "real_minus_shuffled_rho": float(observed[index]),
                "permutation_p_one_sided": float(
                    (exceed[index] + 1) / (args.permutations + 1)
                ),
                "permutation_p_fwer": float(
                    (exceed_fwer[index] + 1) / (args.permutations + 1)
                ),
                "fwer_family": "all_6_layers_real_minus_shuffled",
                "target": args.target_name,
                "permutations": args.permutations,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        (f"semantic_{args.target_name}_layerwise.csv", layer_rows),
        (f"semantic_{args.target_name}_real_minus_shuffled_tests.csv", test_rows),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {args.target_name} real-minus-shuffled semantic comparison")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--human-rdm", type=Path, required=True)
    result.add_argument("--target-name", required=True)
    result.add_argument("--real", type=Path, required=True)
    result.add_argument("--shuffled", type=Path, required=True)
    result.add_argument("--random", type=Path, required=True)
    result.add_argument("--raw", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metric", choices=("cosine", "correlation"), default="cosine")
    result.add_argument("--permutations", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=73)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
