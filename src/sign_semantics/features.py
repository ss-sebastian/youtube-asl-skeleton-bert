from __future__ import annotations

import numpy as np
STREAM_JOINTS = {"body": 25, "hands": 42, "face": 70}


def normalize_openpose(
    body: np.ndarray,
    hands: np.ndarray,
    face: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize 2D coordinates using neck origin and shoulder width per frame.

    Arrays use OpenPose's ``(x, y, confidence)`` convention. Confidence is retained,
    while coordinates with zero confidence are set to zero after normalization.
    """
    streams = [body.astype(np.float32), hands.astype(np.float32), face.astype(np.float32)]
    if any(array.ndim != 3 or array.shape[-1] != 3 for array in streams):
        raise ValueError("Each stream must have shape (frames, joints, 3)")
    if not (body.shape[0] == hands.shape[0] == face.shape[0]):
        raise ValueError("All streams must contain the same number of frames")

    neck = body[:, 1, :2]
    shoulder_distance = np.linalg.norm(body[:, 2, :2] - body[:, 5, :2], axis=-1)
    valid_shoulders = (body[:, 2, 2] > 0) & (body[:, 5, 2] > 0) & (shoulder_distance > epsilon)
    fallback = np.median(shoulder_distance[valid_shoulders]) if valid_shoulders.any() else 1.0
    scale = np.where(valid_shoulders, shoulder_distance, fallback).clip(min=epsilon)

    normalized: list[np.ndarray] = []
    for array in streams:
        result = array.copy()
        result[..., :2] = (result[..., :2] - neck[:, None, :]) / scale[:, None, None]
        result[..., :2] *= (result[..., 2:3] > 0)
        result[~np.isfinite(result)] = 0
        normalized.append(result.astype(np.float32))
    return normalized[0], normalized[1], normalized[2]
