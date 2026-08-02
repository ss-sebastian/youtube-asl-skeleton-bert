from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .utils import choose_device


def read_concepts(path: Path) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identifiers.append(row["id"])
            texts.append(row["text"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Concept IDs must be unique")
    return identifiers, texts


@torch.no_grad()
def embed_concepts(
    concepts_path: Path,
    output: Path,
    model_name: str,
    batch_size: int,
) -> None:
    identifiers, texts = read_concepts(concepts_path)
    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    layer_batches: list[list[np.ndarray]] | None = None

    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = encoded.pop("special_tokens_mask").to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        states = outputs.hidden_states[1:]
        if layer_batches is None:
            layer_batches = [[] for _ in states]
        lexical_mask = encoded["attention_mask"].bool() & ~special.bool()
        weights = lexical_mask.unsqueeze(-1)
        for layer_index, state in enumerate(states):
            pooled = (state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            layer_batches[layer_index].append(pooled.cpu().numpy())

    if layer_batches is None:
        raise ValueError("Concept CSV is empty")
    arrays = {
        f"layer_{index + 1:02d}": np.concatenate(batches, axis=0)
        for index, batches in enumerate(layer_batches)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(identifiers), **arrays)
    print(f"Saved {len(identifiers)} text concepts from {model_name} to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create layer-wise mBERT concept embeddings")
    parser.add_argument("--concepts", type=Path, required=True, help="CSV with id,text columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embed_concepts(args.concepts, args.output, args.model, args.batch_size)


if __name__ == "__main__":
    main()

