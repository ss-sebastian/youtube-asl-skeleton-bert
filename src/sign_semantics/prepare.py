from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .features import STREAM_JOINTS, normalize_openpose

OPENPOSE_KEYS = (
    "pose_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
    "face_keypoints_2d",
)


def _reshape_keypoints(person: dict, key: str, joints: int) -> np.ndarray:
    values = person.get(key, [])
    if len(values) != joints * 3:
        return np.zeros((joints, 3), dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(joints, 3)


def _select_signer(people: list[dict]) -> dict:
    if not people:
        return {}

    def confidence(person: dict) -> float:
        values = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32)
        return float(values[2::3].sum()) if values.size else 0.0

    return max(people, key=confidence)


def parse_openpose_clip(frame_files: list[Path]) -> dict[str, np.ndarray]:
    body_frames: list[np.ndarray] = []
    hands_frames: list[np.ndarray] = []
    face_frames: list[np.ndarray] = []

    for path in frame_files:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        person = _select_signer(payload.get("people", []))
        body_frames.append(_reshape_keypoints(person, OPENPOSE_KEYS[0], STREAM_JOINTS["body"]))
        left = _reshape_keypoints(person, OPENPOSE_KEYS[1], 21)
        right = _reshape_keypoints(person, OPENPOSE_KEYS[2], 21)
        hands_frames.append(np.concatenate([left, right], axis=0))
        face_frames.append(_reshape_keypoints(person, OPENPOSE_KEYS[3], STREAM_JOINTS["face"]))

    if not body_frames:
        raise ValueError("Clip contains no JSON frames")
    body, hands, face = normalize_openpose(
        np.stack(body_frames), np.stack(hands_frames), np.stack(face_frames)
    )
    return {"body": body, "hands": hands, "face": face}


def discover_clips(input_root: Path) -> list[tuple[str, list[Path]]]:
    """Find directories containing OpenPose frame JSON files."""
    groups: dict[Path, list[Path]] = {}
    for path in input_root.rglob("*.json"):
        groups.setdefault(path.parent, []).append(path)
    clips: list[tuple[str, list[Path]]] = []
    for directory, paths in groups.items():
        relative = directory.relative_to(input_root)
        clip_id = "__".join(relative.parts) or directory.name
        clips.append((clip_id, sorted(paths)))
    return sorted(clips, key=lambda item: item[0])


def prepare_dataset(input_root: Path, output_root: Path, manifest_path: Path, min_frames: int) -> None:
    clips = discover_clips(input_root)
    if not clips:
        raise FileNotFoundError(f"No OpenPose JSON clips found below {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for clip_id, frame_files in tqdm(clips, desc="Preparing How2Sign"):
            if len(frame_files) < min_frames:
                continue
            try:
                streams = parse_openpose_clip(frame_files)
            except (json.JSONDecodeError, OSError, ValueError) as error:
                tqdm.write(f"Skipping {clip_id}: {error}")
                continue
            destination = output_root / f"{clip_id}.npz"
            np.savez_compressed(destination, **streams)
            record = {
                "id": clip_id,
                "path": str(destination.resolve()),
                "num_frames": int(streams["body"].shape[0]),
            }
            manifest.write(json.dumps(record) + "\n")
            kept += 1
    print(f"Wrote {kept} clips to {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert How2Sign OpenPose JSON to NPZ clips")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(args.input_root, args.output_root, args.manifest, args.min_frames)


if __name__ == "__main__":
    main()

