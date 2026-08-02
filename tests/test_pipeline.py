from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from sign_semantics.data import YouTubeASLPoseDataset, parse_youtube_asl
from sign_semantics.features import STREAM_JOINTS
from sign_semantics.masking import sample_span_mask
from sign_semantics.model import SkeletonBert, SkeletonBertConfig, masked_mean
from sign_semantics.rsa import run_rsa
from sign_semantics.train import masked_reconstruction_loss
from sign_semantics.word_extract import load_boundaries


def landmarks(joints: int, frame: int) -> list[list[float]]:
    return [
        [0.3 + 0.002 * joint + 0.001 * frame, 0.2 + 0.001 * joint]
        for joint in range(joints)
    ]


def youtube_asl_payload(frames: int = 16) -> dict:
    sequence = []
    for frame in range(frames):
        pose = landmarks(33, frame)
        # Make shoulder width non-zero and stable for normalization.
        pose[11] = [0.35, 0.4]
        pose[12] = [0.65, 0.4]
        sequence.append(
            {
                "pose_landmarks": pose,
                "right_hand_landmarks": landmarks(21, frame),
                "left_hand_landmarks": landmarks(21, frame + 1),
                "face_landmarks": landmarks(478, frame),
            }
        )
    return {"cropped_keypoints": sequence}


class PipelineTest(unittest.TestCase):
    def test_youtube_asl_zip_to_skeleton_bert_backward_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "raw_keypoints_1.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr("raw_keypoints/clip_001.json", json.dumps(youtube_asl_payload()))
            annotations = root / "train.json"
            annotations.write_text(
                json.dumps(
                    {
                        "video_1": {
                            "clip_order": ["clip_001"],
                            "clip_001": {"translation": "not loaded as a target"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            item = YouTubeASLPoseDataset(
                archive, annotations, max_frames=20, training=True
            )[0]
            batch_streams = {name: item[name].unsqueeze(0) for name in STREAM_JOINTS}
            observed = {
                name: item[f"{name}_observed"].unsqueeze(0) for name in STREAM_JOINTS
            }
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
            self.assertEqual(model.input_dim, 208)
            mask = sample_span_mask(valid, probability=0.5, mean_span_length=4)
            predictions = model(batch_streams, valid, mask)
            loss, _ = masked_reconstruction_loss(
                predictions,
                batch_streams,
                mask & valid,
                velocity_weight=0.25,
                observed=observed,
            )
            loss.backward()

            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(predictions["hands"].shape, (1, 20, 42, 2))
            states = model.encode(batch_streams, valid, return_all_layers=True)
            self.assertEqual(len(states), 2)
            self.assertEqual(masked_mean(states[-1], valid).shape, (1, 32))
            self.assertEqual(item["frame_indices"][:16].tolist(), list(range(16)))

    def test_missing_landmarks_are_zero_and_unsupervised(self) -> None:
        payload = youtube_asl_payload(frames=4)
        payload["cropped_keypoints"][1]["right_hand_landmarks"] = []
        streams, observed = parse_youtube_asl(payload)
        self.assertFalse(observed["hands"][1, :21].any())
        self.assertTrue(np.all(streams["hands"][1, :21] == 0))

    def test_long_validation_clip_selects_frames_before_feature_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "raw_keypoints_1.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                handle.writestr(
                    "raw_keypoints/clip_long.json",
                    json.dumps(youtube_asl_payload(frames=30)),
                )
            annotations = root / "dev.json"
            annotations.write_text(
                json.dumps({"video": {"clip_order": ["clip_long"]}}),
                encoding="utf-8",
            )

            item = YouTubeASLPoseDataset(
                archive, annotations, max_frames=10, training=False
            )[0]
            expected = np.linspace(0, 29, 10).round().astype(np.int64).tolist()
            self.assertEqual(item["frame_indices"].tolist(), expected)
            self.assertEqual(int(item["valid"].sum()), 10)

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
            np.savez_compressed(sign_path, ids=identifiers, layer_01=rng.normal(size=(4, 8)))
            np.savez_compressed(text_path, ids=identifiers, layer_01=rng.normal(size=(4, 8)))
            output = root / "rsa.csv"
            run_rsa(sign_path, text_path, output, "cosine", permutations=5, seed=1)
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_words"], "4")


if __name__ == "__main__":
    unittest.main()
