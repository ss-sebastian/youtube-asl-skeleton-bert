from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .features import COORDINATE_DIM, STREAM_JOINTS
from .model import masked_mean
from .model_loading import load_trained_model
from .utils import choose_device


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must be a non-empty CSV with id,path columns")
    return rows


def _resample_indices(length: int, max_frames: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Skeleton sequence is empty")
    if length <= max_frames:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, max_frames).round().astype(np.int64)


def load_standard_skeleton(
    path: Path, max_frames: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Load the repository's portable lexical NPZ format.

    Every NPZ contains ``body`` (T,25,2), ``hands`` (T,42,2), and ``face``
    (T,37,2). Coordinates must already use the same shoulder-centred and
    shoulder-scaled convention as pretraining. Optional ``*_observed`` arrays
    document missing landmarks; the encoder itself receives zeros for missing
    coordinates, exactly as during pretraining.
    """
    with np.load(path) as payload:
        arrays: dict[str, np.ndarray] = {}
        length: int | None = None
        for name, joints in STREAM_JOINTS.items():
            values = np.asarray(payload[name], dtype=np.float32)
            if values.ndim != 3 or values.shape[1:] != (joints, COORDINATE_DIM):
                raise ValueError(f"{path}:{name} must have shape (T,{joints},{COORDINATE_DIM})")
            if length is None:
                length = len(values)
            elif len(values) != length:
                raise ValueError(f"{path} streams have different frame counts")
            arrays[name] = values
    assert length is not None
    indices = _resample_indices(length, max_frames)
    sampled = {name: torch.from_numpy(values[indices]) for name, values in arrays.items()}
    valid = torch.ones(len(indices), dtype=torch.bool)
    return sampled, valid


@torch.no_grad()
def extract_lexical_representations(
    checkpoint_path: Path,
    manifest_path: Path,
    root: Path,
    output: Path,
    max_frames: int,
    token_output: Path | None = None,
) -> None:
    device = choose_device()
    model, _ = load_trained_model(str(checkpoint_path), device)

    rows = _load_manifest(manifest_path)
    by_concept: dict[str, list[list[np.ndarray]]] = defaultdict(
        lambda: [[] for _ in range(model.config.num_hidden_layers)]
    )
    sample_ids: dict[str, list[str]] = defaultdict(list)
    token_concepts: list[str] = []
    token_sample_ids: list[str] = []
    token_participant_ids: list[str] = []
    token_layers: list[list[np.ndarray]] = [[] for _ in range(model.config.num_hidden_layers)]
    for row in tqdm(rows, desc="Extracting lexical sign representations"):
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        streams, valid = load_standard_skeleton(path, max_frames)
        batched = {name: value.unsqueeze(0).to(device) for name, value in streams.items()}
        valid_batch = valid.unsqueeze(0).to(device)
        # Preserve the attention direction learned by each objective.  In
        # particular, next-frame checkpoints are causal and must not be
        # silently evaluated with bidirectional attention.
        layers = model.encode(
            batched,
            valid_batch,
            causal=model.config.causal_attention,
            return_all_layers=True,
        )
        assert isinstance(layers, list)
        concept = row["id"].strip().lower()
        for index, state in enumerate(layers):
            representation = masked_mean(state, valid_batch).squeeze(0).cpu().numpy()
            by_concept[concept][index].append(representation)
            token_layers[index].append(representation)
        sample_id = row.get("sample_id", path.stem)
        sample_ids[concept].append(sample_id)
        token_concepts.append(concept)
        token_sample_ids.append(sample_id)
        token_participant_ids.append(row.get("participant_id", "").strip())

    concepts = sorted(by_concept)
    if len(concepts) < 3:
        raise ValueError("Need at least three lexical concepts for RSA")
    arrays = {
        f"layer_{layer + 1:02d}": np.stack(
            [np.mean(by_concept[concept][layer], axis=0) for concept in concepts]
        )
        for layer in range(model.config.num_hidden_layers)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.asarray(concepts),
        counts=np.asarray([len(sample_ids[c]) for c in concepts]),
        sample_ids=np.asarray(["|".join(sample_ids[c]) for c in concepts]),
        checkpoint=str(checkpoint_path),
        **arrays,
    )
    if token_output is None:
        token_output = output.with_name(f"{output.stem}_tokens.npz")
    token_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        token_output,
        ids=np.asarray(token_concepts),
        sample_ids=np.asarray(token_sample_ids),
        participant_ids=np.asarray(token_participant_ids),
        checkpoint=str(checkpoint_path),
        **{f"layer_{index + 1:02d}": np.stack(values) for index, values in enumerate(token_layers)},
    )
    print(f"Saved {len(concepts)} concept representations from {len(rows)} tokens to {output}")
    print(f"Saved all token-level, all-layer representations to {token_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract all six best.pt layers for lexical signs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-output", type=Path)
    parser.add_argument("--max-frames", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extract_lexical_representations(
        args.checkpoint,
        args.manifest,
        args.root,
        args.output,
        args.max_frames,
        args.token_output,
    )


if __name__ == "__main__":
    main()
