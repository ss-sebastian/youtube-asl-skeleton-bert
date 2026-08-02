from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import YouTubeASLPoseDataset
from .features import STREAM_JOINTS
from .model import SkeletonBert, SkeletonBertConfig, masked_mean
from .utils import choose_device


@torch.no_grad()
def extract_representations(
    checkpoint_path: Path,
    archive: Path,
    annotations: Path,
    output: Path,
    max_frames: int,
    batch_size: int,
) -> None:
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SkeletonBert(SkeletonBertConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = YouTubeASLPoseDataset(
        archive, annotations, max_frames=max_frames, training=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    identifiers: list[str] = []
    layer_batches: list[list[np.ndarray]] = [
        [] for _ in range(model.config.num_hidden_layers)
    ]
    for batch in tqdm(loader, desc="Extracting sentence representations"):
        streams = {name: batch[name].to(device) for name in STREAM_JOINTS}
        valid = batch["valid"].to(device)
        states = model.encode(streams, valid, return_all_layers=True)
        assert isinstance(states, list)
        for layer_index, state in enumerate(states):
            layer_batches[layer_index].append(masked_mean(state, valid).cpu().numpy())
        identifiers.extend(batch["id"])

    arrays = {
        f"layer_{index + 1:02d}": np.concatenate(batches, axis=0)
        for index, batches in enumerate(layer_batches)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(identifiers), **arrays)
    print(f"Saved {len(identifiers)} sentence representations to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export layer-wise sentence embeddings for RSA")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extract_representations(
        args.checkpoint,
        args.archive,
        args.annotations,
        args.output,
        args.max_frames,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
