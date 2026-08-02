from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import COORDINATE_DIM, FACE_LANDMARKS, STREAM_JOINTS, normalize_youtube_asl


def annotation_clip_ids(path: str | Path) -> set[str]:
    """Read only clip IDs from the official annotation file.

    Translation strings exist in the same file but are deliberately ignored by
    self-supervised pretraining.
    """
    with Path(path).open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    clip_ids: set[str] = set()
    for video in annotations.values():
        if not isinstance(video, dict):
            continue
        order = video.get("clip_order")
        if isinstance(order, list):
            clip_ids.update(str(value) for value in order)
        else:
            clip_ids.update(str(key) for key in video if key != "clip_order")
    return clip_ids


def _landmarks(frame: dict[str, Any], key: str, joints: int) -> tuple[np.ndarray, np.ndarray]:
    raw = frame.get(key) or []
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < joints or values.shape[1] < COORDINATE_DIM:
        return (
            np.zeros((joints, COORDINATE_DIM), dtype=np.float32),
            np.zeros(joints, dtype=bool),
        )
    values = values[:joints, :COORDINATE_DIM]
    seen = np.isfinite(values).all(axis=-1)
    values[~seen] = 0
    return values, seen


def parse_youtube_asl(payload: dict[str, Any], data_key: str = "cropped_keypoints") -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """Convert one released YouTube-ASL clip JSON to 208-D frame streams."""
    frames = payload.get(data_key)
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Missing non-empty {data_key!r} frame list")

    values: dict[str, list[np.ndarray]] = {name: [] for name in STREAM_JOINTS}
    masks: dict[str, list[np.ndarray]] = {name: [] for name in STREAM_JOINTS}
    for frame in frames:
        pose, pose_seen = _landmarks(frame, "pose_landmarks", 25)
        right, right_seen = _landmarks(frame, "right_hand_landmarks", 21)
        left, left_seen = _landmarks(frame, "left_hand_landmarks", 21)
        full_face, full_face_seen = _landmarks(frame, "face_landmarks", 478)

        values["body"].append(pose)
        masks["body"].append(pose_seen)
        values["hands"].append(np.concatenate([right, left], axis=0))
        masks["hands"].append(np.concatenate([right_seen, left_seen], axis=0))
        values["face"].append(full_face[FACE_LANDMARKS])
        masks["face"].append(full_face_seen[FACE_LANDMARKS])

    streams = {name: np.stack(parts) for name, parts in values.items()}
    observed = {name: np.stack(parts) for name, parts in masks.items()}
    streams = normalize_youtube_asl(streams, observed)
    return streams, observed


class YouTubeASLPoseDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Read sentence-level keypoint JSON directly from a shard ZIP or directory."""

    def __init__(
        self,
        archive: str | Path,
        annotations: str | Path,
        max_frames: int,
        training: bool,
        limit_clips: int | None = None,
        data_key: str = "cropped_keypoints",
    ) -> None:
        self.archive_source = str(archive)
        self.is_remote = self.archive_source.startswith(("http://", "https://"))
        self.archive = None if self.is_remote else Path(archive)
        self.max_frames = max_frames
        self.training = training
        self.data_key = data_key
        self._zip: zipfile.ZipFile | None = None
        allowed = annotation_clip_ids(annotations)

        if self.is_remote:
            from remotezip import RemoteZip

            with RemoteZip(self.archive_source) as handle:
                candidates = [
                    name for name in handle.namelist()
                    if name.lower().endswith(".json") and not name.endswith("/")
                ]
            self.members = [
                name for name in candidates if PurePosixPath(name).stem in allowed
            ]
            self.directory_files = None
        elif self.archive is not None and self.archive.is_file() and zipfile.is_zipfile(self.archive):
            with zipfile.ZipFile(self.archive) as handle:
                candidates = [
                    name for name in handle.namelist()
                    if name.lower().endswith(".json") and not name.endswith("/")
                ]
            self.members = [
                name for name in candidates if PurePosixPath(name).stem in allowed
            ]
            self.directory_files: list[Path] | None = None
        elif self.archive is not None and self.archive.is_dir():
            candidates = sorted(self.archive.rglob("*.json"))
            self.directory_files = [path for path in candidates if path.stem in allowed]
            self.members = [str(path) for path in self.directory_files]
        else:
            raise FileNotFoundError(f"Keypoint ZIP/directory not found: {self.archive_source}")

        self.members.sort()
        if limit_clips is not None:
            self.members = self.members[: int(limit_clips)]
            if self.directory_files is not None:
                self.directory_files = self.directory_files[: int(limit_clips)]
        if not self.members:
            raise ValueError(
                f"No clips in {self.archive_source} matched annotations from {annotations}"
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_zip"] = None
        return state

    def __del__(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __len__(self) -> int:
        return len(self.members)

    def _read_payload(self, index: int) -> dict[str, Any]:
        if self.directory_files is not None:
            with self.directory_files[index].open(encoding="utf-8") as handle:
                return json.load(handle)
        if self._zip is None:
            if self.is_remote:
                from remotezip import RemoteZip

                self._zip = RemoteZip(self.archive_source)
            else:
                self._zip = zipfile.ZipFile(self.archive)
        with self._zip.open(self.members[index]) as handle:
            return json.load(handle)

    def _window(
        self,
        streams: dict[str, np.ndarray],
        observed: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        length = streams["body"].shape[0]
        if length <= self.max_frames:
            indices: slice | np.ndarray = slice(None)
            frame_indices = np.arange(length, dtype=np.int64)
        elif self.training:
            start = int(np.random.randint(0, length - self.max_frames + 1))
            indices = slice(start, start + self.max_frames)
            frame_indices = np.arange(start, start + self.max_frames, dtype=np.int64)
        else:
            indices = np.linspace(0, length - 1, self.max_frames).round().astype(np.int64)
            frame_indices = indices
        return (
            {name: array[indices] for name, array in streams.items()},
            {name: array[indices] for name, array in observed.items()},
            frame_indices,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        streams, observed = parse_youtube_asl(self._read_payload(index), self.data_key)
        original_length = streams["body"].shape[0]
        streams, observed, frame_indices = self._window(streams, observed)
        length = streams["body"].shape[0]
        item: dict[str, torch.Tensor | str] = {
            "id": PurePosixPath(self.members[index]).stem,
        }
        for name, joints in STREAM_JOINTS.items():
            padded = np.zeros(
                (self.max_frames, joints, COORDINATE_DIM), dtype=np.float32
            )
            padded_seen = np.zeros((self.max_frames, joints), dtype=bool)
            padded[:length] = streams[name]
            padded_seen[:length] = observed[name]
            item[name] = torch.from_numpy(padded)
            item[f"{name}_observed"] = torch.from_numpy(padded_seen)
        item["valid"] = torch.arange(self.max_frames) < length
        padded_indices = torch.full((self.max_frames,), -1, dtype=torch.long)
        padded_indices[:length] = torch.from_numpy(frame_indices)
        item["frame_indices"] = padded_indices
        item["original_num_frames"] = torch.tensor(original_length, dtype=torch.long)
        return item
