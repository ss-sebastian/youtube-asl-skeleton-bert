from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sign_semantics.data import How2SignPoseDataset
from sign_semantics.features import STREAM_JOINTS, cluster_features_numpy
from sign_semantics.masking import sample_span_mask
from sign_semantics.model import ModelConfig, MultiStreamSignTransformer, masked_mean
from sign_semantics.prepare import parse_openpose_clip
from sign_semantics.targets import ClusterTargeter
from sign_semantics.train import masked_prediction_loss


def keypoints(joints: int, frame: int) -> list[float]:
    output: list[float] = []
    for joint in range(joints):
        output.extend([float(frame + joint), float(frame - joint), 1.0])
    return output


class PipelineTest(unittest.TestCase):
    def test_openpose_to_backward_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_paths: list[Path] = []
            for frame in range(16):
                payload = {
                    "people": [
                        {
                            "pose_keypoints_2d": keypoints(25, frame),
                            "hand_left_keypoints_2d": keypoints(21, frame),
                            "hand_right_keypoints_2d": keypoints(21, frame + 1),
                            "face_keypoints_2d": keypoints(70, frame),
                        }
                    ]
                }
                path = root / f"{frame:04d}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                frame_paths.append(path)

            streams = parse_openpose_clip(frame_paths)
            npz_path = root / "clip.npz"
            np.savez_compressed(npz_path, **streams)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"id": "synthetic", "path": str(npz_path), "num_frames": 16}) + "\n",
                encoding="utf-8",
            )
            dataset = How2SignPoseDataset(manifest, max_frames=20, training=True)
            item = dataset[0]
            batch_streams = {name: item[name].unsqueeze(0) for name in STREAM_JOINTS}
            valid = item["valid"].unsqueeze(0)

            centers = {}
            for name in STREAM_JOINTS:
                features = cluster_features_numpy(streams[name])
                centers[name] = features[np.linspace(0, 15, 4).astype(int)]
            centers_path = root / "centers.npz"
            np.savez_compressed(centers_path, **centers)
            targeter = ClusterTargeter(centers_path, torch.device("cpu"))

            model = MultiStreamSignTransformer(
                ModelConfig(d_model=32, n_heads=4, n_layers=2, max_frames=20, dropout=0.0),
                targeter.cluster_sizes,
            )
            mask = sample_span_mask(valid, probability=0.5, mean_span_length=3)
            targets = targeter.assign(batch_streams)
            logits = model(batch_streams, valid, mask)
            loss, _ = masked_prediction_loss(logits, targets, mask & valid)
            loss.backward()

            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
            states = model.encode(batch_streams, valid, return_all_layers=True)
            self.assertEqual(len(states), 2)
            self.assertEqual(masked_mean(states[-1], valid).shape, (1, 32))


if __name__ == "__main__":
    unittest.main()

