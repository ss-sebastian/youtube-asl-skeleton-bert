from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import YouTubeASLPoseDataset
from .features import STREAM_JOINTS
from .model import SkeletonBert, SkeletonBertConfig
from .utils import choose_device, read_jsonl


def load_boundaries(path: Path) -> dict[str, list[dict]]:
    """Load test-only word boundaries grouped by sentence clip ID.

    Expected JSONL fields are ``clip_id``, ``gloss``, ``start_frame`` and
    ``end_frame``. End frames are exclusive. These annotations are never read by
    the pretraining data loader.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(path):
        start = int(record["start_frame"])
        end = int(record["end_frame"])
        if end <= start:
            raise ValueError(f"Invalid boundary in {path}: {record}")
        grouped[str(record["clip_id"])].append(
            {"gloss": str(record["gloss"]), "start": start, "end": end}
        )
    return grouped


@torch.no_grad()
def extract_word_representations(
    checkpoint_path: Path,
    archive: Path,
    annotations: Path,
    boundaries_path: Path,
    output: Path,
    max_frames: int,
    batch_size: int,
    min_tokens: int,
) -> None:
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SkeletonBert(SkeletonBertConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    boundaries = load_boundaries(boundaries_path)
    dataset = YouTubeASLPoseDataset(
        archive, annotations, max_frames=max_frames, training=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    token_states: dict[str, list[list[np.ndarray]]] = defaultdict(
        lambda: [[] for _ in range(model.config.num_hidden_layers)]
    )

    for batch in tqdm(loader, desc="Extracting contextual word tokens"):
        streams = {name: batch[name].to(device) for name in STREAM_JOINTS}
        valid = batch["valid"].to(device)
        states = model.encode(streams, valid, return_all_layers=True)
        assert isinstance(states, list)
        frame_indices = batch["frame_indices"]

        for sample_index, clip_id in enumerate(batch["id"]):
            if clip_id not in boundaries:
                continue
            sampled_frames = frame_indices[sample_index]
            valid_positions = sampled_frames >= 0
            for token in boundaries[clip_id]:
                positions = (
                    valid_positions
                    & (sampled_frames >= token["start"])
                    & (sampled_frames < token["end"])
                )
                if not positions.any():
                    midpoint = (token["start"] + token["end"] - 1) / 2
                    candidates = torch.where(valid_positions)[0]
                    nearest = candidates[(sampled_frames[candidates].float() - midpoint).abs().argmin()]
                    positions[nearest] = True
                for layer_index, state in enumerate(states):
                    vector = state[sample_index, positions.to(device)].mean(dim=0)
                    token_states[token["gloss"]][layer_index].append(vector.cpu().numpy())

    selected = sorted(
        gloss for gloss, layers in token_states.items() if len(layers[0]) >= min_tokens
    )
    if len(selected) < 3:
        raise ValueError(
            f"Only {len(selected)} word types have at least {min_tokens} tokens; need at least 3"
        )
    arrays: dict[str, np.ndarray] = {}
    for layer_index in range(model.config.num_hidden_layers):
        arrays[f"layer_{layer_index + 1:02d}"] = np.stack(
            [np.mean(token_states[gloss][layer_index], axis=0) for gloss in selected]
        )
    counts = np.asarray([len(token_states[gloss][0]) for gloss in selected], dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(selected), counts=counts, **arrays)
    print(f"Saved {len(selected)} word types to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pool BERT frames inside test-only word boundaries"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-tokens", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extract_word_representations(
        args.checkpoint,
        args.archive,
        args.annotations,
        args.boundaries,
        args.output,
        args.max_frames,
        args.batch_size,
        args.min_tokens,
    )


if __name__ == "__main__":
    main()
