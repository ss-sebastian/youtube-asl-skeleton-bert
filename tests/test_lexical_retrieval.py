from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sign_semantics.features import STREAM_JOINTS
from sign_semantics.lexical_retrieval import (
    TokenMetadata,
    evaluate_cross_signer_retrieval,
    load_token_layers,
    load_token_manifest,
    run_lexical_retrieval,
)


class CrossSignerLexicalRetrievalTest(unittest.TestCase):
    def test_candidates_and_prototypes_exclude_entire_test_signer(self) -> None:
        metadata = [
            TokenMetadata("apple", "p1_apple", "P1", Path("p1_apple.npz")),
            TokenMetadata("apple", "p2_apple", "P2", Path("p2_apple.npz")),
            # Bell is a valid distractor for P1 even though P1 never produced it.
            TokenMetadata("bell", "p2_bell", "P2", Path("p2_bell.npz")),
            TokenMetadata("apple", "p3_apple", "P3", Path("p3_apple.npz")),
            TokenMetadata("bell", "p3_bell", "P3", Path("p3_bell.npz")),
        ]
        values = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        )

        result, audit = evaluate_cross_signer_retrieval(values, metadata, permutations=19, seed=3)

        self.assertEqual(result["n_queries"], 5)
        self.assertEqual(result["top1"], 1.0)
        p1 = next(row for row in audit if row["test_signer"] == "P1")
        self.assertEqual(p1["eligible_candidate_concepts"], 2)
        self.assertEqual(p1["evaluated_tokens"], 1)
        self.assertEqual(p1["prototype_test_signer_tokens"], 0)
        self.assertTrue(p1["strict_no_signer_leakage"])

    def test_token_npz_must_exactly_match_manifest_and_runner_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"id": "apple", "path": "a.npz", "sample_id": "a", "participant_id": "P1"},
                {"id": "bell", "path": "b.npz", "sample_id": "b", "participant_id": "P1"},
                {"id": "apple", "path": "c.npz", "sample_id": "c", "participant_id": "P2"},
                {"id": "bell", "path": "d.npz", "sample_id": "d", "participant_id": "P2"},
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            tokens = root / "tokens.npz"
            np.savez_compressed(
                tokens,
                ids=np.asarray(["bell", "apple", "bell", "apple"]),
                sample_ids=np.asarray(["d", "a", "b", "c"]),
                layer_01=np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float),
            )
            metadata = load_token_manifest(manifest)
            aligned = load_token_layers(tokens, metadata)
            self.assertTrue(np.array_equal(aligned["layer_01"][0], np.asarray([1.0, 0.0])))

            output = root / "output"
            run_lexical_retrieval(manifest, [("masked", tokens)], output, permutations=3, seed=1)
            with (output / "cross_signer_lexical_retrieval.csv").open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(
                [(row["model"], row["layer"]) for row in summary], [("masked", "layer_01")]
            )
            self.assertEqual(summary[0]["n_queries"], "4")
            self.assertEqual(summary[0]["top1"], "1.0")

            np.savez_compressed(
                root / "missing.npz",
                ids=np.asarray(["apple"]),
                sample_ids=np.asarray(["a"]),
                layer_01=np.asarray([[1.0, 0.0]]),
            )
            with self.assertRaisesRegex(ValueError, "do not exactly match manifest"):
                load_token_layers(root / "missing.npz", metadata)

    def test_raw_kinematics_control_uses_manifest_input_npzs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for sample, concept, signer, offset in (
                ("a", "apple", "P1", 1.0),
                ("b", "apple", "P2", 1.0),
                ("c", "bell", "P1", 2.0),
                ("d", "bell", "P2", 2.0),
            ):
                arrays = {
                    name: np.full((3, joints, 2), offset, dtype=np.float32)
                    for name, joints in STREAM_JOINTS.items()
                }
                np.savez_compressed(root / f"{sample}.npz", **arrays)
                rows.append(
                    {
                        "id": concept,
                        "path": f"{sample}.npz",
                        "sample_id": sample,
                        "participant_id": signer,
                    }
                )
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            output = root / "raw"
            run_lexical_retrieval(
                manifest, [], output, permutations=0, raw_kinematics=True, root=root
            )
            with (output / "cross_signer_lexical_retrieval.csv").open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(summary[0]["model"], "raw_kinematics")
            self.assertEqual(summary[0]["layer"], "raw_kinematics")


if __name__ == "__main__":
    unittest.main()
