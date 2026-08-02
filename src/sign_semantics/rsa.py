from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def layer_keys(payload: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(key for key in payload.files if key.startswith("layer_"))


def rdm(features: np.ndarray, metric: str) -> np.ndarray:
    matrix = squareform(pdist(features, metric=metric))
    if not np.isfinite(matrix).all():
        raise ValueError("RDM contains non-finite values; check constant or invalid representations")
    return matrix


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def run_rsa(
    sign_path: Path,
    text_path: Path,
    output: Path,
    metric: str,
    permutations: int,
    seed: int,
) -> None:
    with np.load(sign_path) as sign, np.load(text_path) as text:
        sign_ids = [str(value) for value in sign["ids"]]
        text_ids = [str(value) for value in text["ids"]]
        common = sorted(set(sign_ids) & set(text_ids))
        if len(common) < 3:
            raise ValueError("Sign and text files share fewer than 3 concept IDs")
        sign_index = [sign_ids.index(identifier) for identifier in common]
        text_index = [text_ids.index(identifier) for identifier in common]
        sign_rdms = {
            key: rdm(sign[key][sign_index], metric) for key in layer_keys(sign)
        }
        text_rdms = {
            key: rdm(text[key][text_index], metric) for key in layer_keys(text)
        }

    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int | float]] = []
    for sign_layer, sign_matrix in sign_rdms.items():
        sign_vector = upper(sign_matrix)
        for text_layer, text_matrix in text_rdms.items():
            rho = float(spearmanr(sign_vector, upper(text_matrix)).statistic)
            exceedances = 0
            for _ in range(permutations):
                order = rng.permutation(len(common))
                permuted = text_matrix[order][:, order]
                permuted_rho = float(spearmanr(sign_vector, upper(permuted)).statistic)
                exceedances += abs(permuted_rho) >= abs(rho)
            p_value = (exceedances + 1) / (permutations + 1)
            rows.append(
                {
                    "sign_layer": sign_layer,
                    "text_layer": text_layer,
                    "n_words": len(common),
                    "spearman_rho": rho,
                    "permutation_p": p_value,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} layer comparisons across {len(common)} words to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-wise word-level sign-to-mBERT RSA")
    parser.add_argument("--sign", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("correlation", "cosine"), default="correlation")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_rsa(args.sign, args.text, args.output, args.metric, args.permutations, args.seed)


if __name__ == "__main__":
    main()
