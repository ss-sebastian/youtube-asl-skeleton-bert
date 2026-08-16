from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sign_semantics.context_shuffle import (
    ContextBatchCollator,
    SourceGroupedBatchSampler,
    assert_no_linguistic_supervision,
)
from sign_semantics.data import annotation_clip_sources
from sign_semantics.masking import sample_span_mask
from sign_semantics.skeleton_codebooks import SkeletonCodebooks
from sign_semantics.spatial_shubert_model import SpatialSHuBERT, SpatialSHuBERTConfig
from sign_semantics.train import masked_cluster_objective


def item(clip: int, frames: int = 32) -> dict:
    result: dict[str, object] = {
        "id": f"clip_{clip}",
        "source_group": "shared_source",
        "valid": torch.ones(frames, dtype=torch.bool),
        "frame_indices": torch.arange(frames),
        "original_num_frames": torch.tensor(frames),
    }
    for stream, joints in (("body", 25), ("hands", 42), ("face", 37)):
        values = torch.zeros(frames, joints, 2)
        # Every 16-frame block has a unique, easily audited identity.
        values[:16] = clip * 10 + 1
        values[16:] = clip * 10 + 2
        result[stream] = values
        result[f"{stream}_observed"] = torch.ones(frames, joints, dtype=torch.bool)
    return result


def test_block_shuffle_preserves_frames_lengths_and_local_order() -> None:
    items = [item(index) for index in range(4)]
    real = ContextBatchCollator("real", 16, 1, 7)(items)
    shuffled = ContextBatchCollator("shuffled", 16, 1, 7)(items)
    assert torch.equal(real["context_original_lengths"], shuffled["context_original_lengths"])
    assert torch.equal(real["context_boundary_mask"], shuffled["context_boundary_mask"])
    assert int(shuffled["context_boundary_mask"].sum()) == 4 * 2
    assert shuffled["context_batch_audit"]["cross_sentence_fraction"] == 1.0
    for stream in ("body", "hands", "face"):
        before = sorted(real[stream][:, 0, 0, 0].tolist() + real[stream][:, 16, 0, 0].tolist())
        after = sorted(
            shuffled[stream][:, 0, 0, 0].tolist()
            + shuffled[stream][:, 16, 0, 0].tolist()
        )
        assert before == after
        # No block is reversed or internally fragmented.
        for row in range(4):
            assert torch.unique(shuffled[stream][row, :16]).numel() == 1
            assert torch.unique(shuffled[stream][row, 16:]).numel() == 1


def test_source_grouped_sampler_is_deterministic_and_complete() -> None:
    groups = ["a", "a", "b", "b", "c"]
    first = list(SourceGroupedBatchSampler(groups, 2, 3, True))
    second = list(SourceGroupedBatchSampler(groups, 2, 3, True))
    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(len(groups)))


def test_structural_annotation_reader_ignores_translation_values(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "source_a": {
                    "clip_order": ["clip_1", "clip_2"],
                    "clip_1": {"translation": "FORBIDDEN SECRET TEXT"},
                    "clip_2": {"translation": "ANOTHER LABEL"},
                }
            }
        ),
        encoding="utf-8",
    )
    assert annotation_clip_sources(path) == {
        "clip_1": "source_a",
        "clip_2": "source_a",
    }


def test_linguistic_training_keys_are_rejected() -> None:
    assert_no_linguistic_supervision(
        {"data": {"train_annotations": "YT.translations.train.json"}}
    )
    with pytest.raises(ValueError):
        assert_no_linguistic_supervision({"training": {"gloss_labels": "x.csv"}})
    with pytest.raises(ValueError):
        assert_no_linguistic_supervision({"data": {"neutral_path": "asl_citizen.csv"}})


def test_span_mask_supports_internal_boundary_gaps() -> None:
    valid = torch.tensor([[True] * 7 + [False, False] + [True] * 7])
    mask = sample_span_mask(valid, 0.5, 3, torch.Generator().manual_seed(4))
    assert not mask[~valid].any()
    assert int(mask.sum()) >= 1
    assert mask[:, 9:].any()


def test_shuffled_tensor_and_cluster_target_alignment(tmp_path) -> None:
    dimensions = {"body": 50, "right_hand": 42, "left_hand": 42, "face": 74}
    arrays = {}
    generator = np.random.default_rng(4)
    for name, width in dimensions.items():
        arrays[f"{name}_mean"] = np.zeros(width, dtype=np.float32)
        arrays[f"{name}_scale"] = np.ones(width, dtype=np.float32)
        arrays[f"{name}_centroids"] = generator.normal(size=(5, width)).astype(np.float32)
    path = tmp_path / "codebooks.npz"
    np.savez(path, **arrays)
    batch = ContextBatchCollator("shuffled", 16, 1, 11)(
        [item(index) for index in range(4)]
    )
    streams = {name: batch[name] for name in ("body", "hands", "face")}
    observed = {
        name: batch[f"{name}_observed"] for name in ("body", "hands", "face")
    }
    model = SpatialSHuBERT(
        SpatialSHuBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_frames=32,
            graph_channels=8,
            graph_layers=1,
            clusters_per_stream=5,
        )
    )
    loss, metrics, _ = masked_cluster_objective(
        model,
        streams,
        observed,
        batch["valid"],
        {"mask_probability": 0.4, "mean_span_length": 4},
        SkeletonCodebooks(path, torch.device("cpu")),
        torch.Generator().manual_seed(3),
    )
    assert torch.isfinite(loss)
    assert metrics["cluster_accuracy"] >= 0
