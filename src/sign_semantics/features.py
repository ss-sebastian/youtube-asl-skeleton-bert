from __future__ import annotations

import numpy as np

# The released YouTube-ASL representation uses 25 pose points, 21 points per
# hand, and the 37-point face subset used by the accompanying PoseEstimation
# code. With x/y coordinates this is exactly 104 * 2 = 208 values per frame.
STREAM_JOINTS = {"body": 25, "hands": 42, "face": 37}
COORDINATE_DIM = 2

FACE_LANDMARKS = np.asarray(
    [
        0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
        93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
        282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473,
    ],
    dtype=np.int64,
)


def _interpolate_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill missing per-frame normalization values without using text labels."""
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    frames = np.arange(len(values))
    result = values.copy()
    if values.ndim == 1:
        result[~valid] = np.interp(frames[~valid], frames[valid], values[valid])
    else:
        for coordinate in range(values.shape[1]):
            result[~valid, coordinate] = np.interp(
                frames[~valid], frames[valid], values[valid, coordinate]
            )
    return result


def normalize_youtube_asl(
    streams: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Shoulder-centre and shoulder-scale all selected 2D landmarks.

    The same global transform is applied to pose, hands, and face so that hand
    location relative to the body remains available to the self-supervised
    model. Missing landmarks remain exactly zero and are excluded from loss by
    their separate observation masks.
    """
    body = streams["body"]
    body_seen = observed["body"]
    if body.ndim != 3 or body.shape[1:] != (STREAM_JOINTS["body"], COORDINATE_DIM):
        raise ValueError("body must have shape (frames, 25, 2)")

    left_shoulder, right_shoulder = 11, 12
    valid_shoulders = body_seen[:, left_shoulder] & body_seen[:, right_shoulder]
    shoulder_center = (body[:, left_shoulder] + body[:, right_shoulder]) / 2
    shoulder_width = np.linalg.norm(
        body[:, left_shoulder] - body[:, right_shoulder], axis=-1
    )
    valid_shoulders &= shoulder_width > epsilon

    if valid_shoulders.any():
        shoulder_center = _interpolate_invalid(shoulder_center, valid_shoulders)
        shoulder_width = _interpolate_invalid(shoulder_width, valid_shoulders)
    else:
        # A clip with no usable shoulders is still loadable, but its native
        # MediaPipe [0, 1]-like coordinate scale is retained.
        shoulder_center = np.zeros_like(shoulder_center)
        shoulder_width = np.ones_like(shoulder_width)
    shoulder_width = np.clip(shoulder_width, epsilon, None)

    normalized: dict[str, np.ndarray] = {}
    for name, values in streams.items():
        result = (values - shoulder_center[:, None, :]) / shoulder_width[:, None, None]
        result = result.astype(np.float32)
        result[~observed[name]] = 0
        result[~np.isfinite(result)] = 0
        normalized[name] = result
    return normalized
