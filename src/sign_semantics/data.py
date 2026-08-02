from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import orjson
import torch
from torch.utils.data import Dataset

from .features import COORDINATE_DIM, FACE_LANDMARKS, STREAM_JOINTS, normalize_youtube_asl


def annotation_clip_ids(path: str | Path) -> set[str]:
    """Read only clip IDs from the official annotation file.

    Translation strings exist in the same file but are deliberately ignored by
    self-supervised pretraining.
    """
    annotations = orjson.loads(Path(path).read_bytes())
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


def _selected_landmarks(
    frame: dict[str, Any], key: str, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Read only requested landmarks instead of converting the full 478-point face."""
    raw = frame.get(key) or []
    joints = len(indices)
    if len(raw) <= int(indices.max()):
        return (
            np.zeros((joints, COORDINATE_DIM), dtype=np.float32),
            np.zeros(joints, dtype=bool),
        )
    values = np.asarray([raw[int(index)] for index in indices], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < COORDINATE_DIM:
        return (
            np.zeros((joints, COORDINATE_DIM), dtype=np.float32),
            np.zeros(joints, dtype=bool),
        )
    values = values[:, :COORDINATE_DIM]
    seen = np.isfinite(values).all(axis=-1)
    values[~seen] = 0
    return values, seen


def parse_youtube_asl(
    payload: dict[str, Any],
    data_key: str = "cropped_keypoints",
    frame_indices: np.ndarray | None = None,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """Convert one released YouTube-ASL clip JSON to 208-D frame streams."""
    frames = payload.get(data_key)
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Missing non-empty {data_key!r} frame list")
    if frame_indices is not None:
        frames = [frames[int(index)] for index in frame_indices]

    values: dict[str, list[np.ndarray]] = {name: [] for name in STREAM_JOINTS}
    masks: dict[str, list[np.ndarray]] = {name: [] for name in STREAM_JOINTS}
    for frame in frames:
        pose, pose_seen = _landmarks(frame, "pose_landmarks", 25)
        right, right_seen = _landmarks(frame, "right_hand_landmarks", 21)
        left, left_seen = _landmarks(frame, "left_hand_landmarks", 21)
        face, face_seen = _selected_landmarks(frame, "face_landmarks", FACE_LANDMARKS)

        values["body"].append(pose)
        masks["body"].append(pose_seen)
        values["hands"].append(np.concatenate([right, left], axis=0))
        masks["hands"].append(np.concatenate([right_seen, left_seen], axis=0))
        values["face"].append(face)
        masks["face"].append(face_seen)

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
            return orjson.loads(self.directory_files[index].read_bytes())
        if self._zip is None:
            if self.is_remote:
                from remotezip import RemoteZip

                self._zip = RemoteZip(self.archive_source)
            else:
                self._zip = zipfile.ZipFile(self.archive)
        with self._zip.open(self.members[index]) as handle:
            return orjson.loads(handle.read())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        payload = self._read_payload(index)
        frames = payload.get(self.data_key)
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Missing non-empty {self.data_key!r} frame list")
        original_length = len(frames)
        if original_length <= self.max_frames:
            frame_indices = np.arange(original_length, dtype=np.int64)
        elif self.training:
            start = int(np.random.randint(0, original_length - self.max_frames + 1))
            frame_indices = np.arange(start, start + self.max_frames, dtype=np.int64)
        else:
            frame_indices = (
                np.linspace(0, original_length - 1, self.max_frames)
                .round()
                .astype(np.int64)
            )
        streams, observed = parse_youtube_asl(payload, self.data_key, frame_indices)
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
