from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sign_semantics.data import How2SignPoseDataset
from sign_semantics.features import STREAM_JOINTS
from sign_semantics.masking import sample_span_mask
from sign_semantics.model import SkeletonBert, SkeletonBertConfig, masked_mean
from sign_semantics.prepare import parse_openpose_clip
from sign_semantics.rsa import run_rsa
from sign_semantics.train import masked_reconstruction_loss
from sign_semantics.word_extract import load_boundaries


def keypoints(joints: int, frame: int) -> list[float]:
    output: list[float] = []
    for joint in range(joints):
        output.extend([float(frame + joint), float(frame - joint), 1.0])
    return output


class PipelineTest(unittest.TestCase):
    def test_openpose_to_skeleton_bert_backward_pass(self) -> None:
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
                json.dumps({"id": "synthetic", "path": str(npz_path), "num_frames": 16})
                + "\n",
                encoding="utf-8",
            )
            item = How2SignPoseDataset(manifest, max_frames=20, training=True)[0]
            batch_streams = {name: item[name].unsqueeze(0) for name in STREAM_JOINTS}
            valid = item["valid"].unsqueeze(0)

            model = SkeletonBert(
                SkeletonBertConfig(
                    hidden_size=32,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    intermediate_size=64,
                    max_frames=20,
                    hidden_dropout_prob=0.0,
                    attention_probs_dropout_prob=0.0,
                )
            )
            mask = sample_span_mask(valid, probability=0.5, mean_span_length=4)
            predictions = model(batch_streams, valid, mask)
            loss, _ = masked_reconstruction_loss(
                predictions, batch_streams, mask & valid, velocity_weight=0.25
            )
            loss.backward()

            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(predictions["hands"].shape, (1, 20, 42, 3))
            states = model.encode(batch_streams, valid, return_all_layers=True)
            self.assertEqual(len(states), 2)
            self.assertEqual(masked_mean(states[-1], valid).shape, (1, 32))
            self.assertEqual(item["frame_indices"][:16].tolist(), list(range(16)))

    def test_word_boundaries_and_rsa(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundaries = root / "boundaries.jsonl"
            boundaries.write_text(
                json.dumps(
                    {"clip_id": "c1", "gloss": "DOG", "start_frame": 2, "end_frame": 6}
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_boundaries(boundaries)
            self.assertEqual(loaded["c1"][0]["gloss"], "DOG")

            identifiers = np.asarray(["CAT", "DOG", "HOUSE", "RUN"])
            rng = np.random.default_rng(3)
            sign_path = root / "sign.npz"
            text_path = root / "text.npz"
            np.savez_compressed(
                sign_path, ids=identifiers, layer_01=rng.normal(size=(4, 8))
            )
            np.savez_compressed(
                text_path, ids=identifiers, layer_01=rng.normal(size=(4, 8))
            )
            output = root / "rsa.csv"
            run_rsa(sign_path, text_path, output, "cosine", permutations=5, seed=1)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_words"], "4")


if __name__ == "__main__":
    unittest.main()
