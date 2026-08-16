"""Merge matched real/shuffled learning curves without selecting a condition."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(args: argparse.Namespace) -> None:
    real = {row["epoch"]: row for row in read(args.real)}
    shuffled = {row["epoch"]: row for row in read(args.shuffled)}
    if set(real) != set(shuffled):
        raise ValueError("Real and shuffled metrics must contain identical global epochs")
    fields = (
        "optimizer_steps",
        "train_clips",
        "train_loss",
        "val_loss",
        "train_cluster_accuracy",
        "val_cluster_accuracy",
        "train_context_blocks",
        "train_context_moved_block_fraction",
        "train_context_cross_sentence_fraction",
        "train_context_same_source_fraction",
        "seconds",
    )
    result = []
    for epoch in sorted(real, key=int):
        row = {"epoch": epoch}
        for field in fields:
            row[f"real_{field}"] = real[epoch].get(field, "")
            row[f"shuffled_{field}"] = shuffled[epoch].get(field, "")
        result.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Saved matched learning-curve table to {args.output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--real", type=Path, required=True)
    result.add_argument("--shuffled", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
