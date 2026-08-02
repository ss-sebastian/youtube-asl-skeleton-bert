from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import STREAM_JOINTS
from .utils import read_jsonl


class How2SignPoseDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Sentence clips stored by :mod:`sign_semantics.prepare`."""

    def __init__(self, manifest: str | Path, max_frames: int, training: bool) -> None:
        self.records = read_jsonl(manifest)
        self.max_frames = max_frames
        self.training = training
        if not self.records:
            raise ValueError(f"Manifest is empty: {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _window(
        self, arrays: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        length = arrays["body"].shape[0]
        if length <= self.max_frames:
            return arrays, np.arange(length, dtype=np.int64)
        if self.training:
            start = int(np.random.randint(0, length - self.max_frames + 1))
            indices = slice(start, start + self.max_frames)
            frame_indices = np.arange(start, start + self.max_frames, dtype=np.int64)
        else:
            indices = np.linspace(0, length - 1, self.max_frames).round().astype(np.int64)
            frame_indices = indices
        return {name: value[indices] for name, value in arrays.items()}, frame_indices

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with np.load(record["path"]) as payload:
            arrays = {name: payload[name].astype(np.float32) for name in STREAM_JOINTS}
        original_length = arrays["body"].shape[0]
        arrays, frame_indices = self._window(arrays)
        length = arrays["body"].shape[0]
        item: dict[str, torch.Tensor | str] = {"id": record["id"]}
        for name, joints in STREAM_JOINTS.items():
            padded = np.zeros((self.max_frames, joints, 3), dtype=np.float32)
            padded[:length] = arrays[name]
            item[name] = torch.from_numpy(padded)
        item["valid"] = torch.arange(self.max_frames) < length
        padded_indices = torch.full((self.max_frames,), -1, dtype=torch.long)
        padded_indices[:length] = torch.from_numpy(frame_indices)
        item["frame_indices"] = padded_indices
        item["original_num_frames"] = torch.tensor(original_length, dtype=torch.long)
        return item
