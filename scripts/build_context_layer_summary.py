"""Build the prespecified concise layerwise real/shuffled result table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(args: argparse.Namespace) -> None:
    lexical = rows(args.lexical)
    semantic = rows(args.semantic)
    lexical_lookup = {(row["model"], row["layer"]): row for row in lexical}
    semantic_lookup = {row["layer"]: row for row in semantic}
    output = []
    for layer in sorted(semantic_lookup):
        real = lexical_lookup[("real", layer)]
        shuffled = lexical_lookup[("shuffled", layer)]
        human = semantic_lookup[layer]
        output.append(
            {
                "layer": layer,
                "real_lexical_top1": real["top1"],
                "shuffled_lexical_top1": shuffled["top1"],
                "real_lexical_mrr": real["mrr"],
                "shuffled_lexical_mrr": shuffled["mrr"],
                "real_lexical_distance_separation": real["distance_delta"],
                "shuffled_lexical_distance_separation": shuffled["distance_delta"],
                "real_deaf_ppmi_rsa": human["real_spearman_rho"],
                "shuffled_deaf_ppmi_rsa": human["shuffled_spearman_rho"],
                "real_minus_shuffled_semantic_rsa": human[
                    "real_minus_shuffled_rho"
                ],
                "semantic_difference_p_fwer": human["permutation_p_fwer"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Saved {len(output)}-layer context summary to {args.output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--lexical", type=Path, required=True)
    result.add_argument("--semantic", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
