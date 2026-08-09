from __future__ import annotations

import torch

from sign_semantics.shape_model import (
    BODY_EDGES,
    HAND_EDGES,
    ShapeAwareSkeletonBert,
    ShapeAwareSkeletonBertConfig,
    normalized_adjacency,
)


def test_explicit_hand_and_body_graphs_are_connected() -> None:
    hand = normalized_adjacency(21, HAND_EDGES)
    body = normalized_adjacency(25, BODY_EDGES)
    assert hand[0, 1] > 0
    assert hand[3, 4] > 0
    assert hand[5, 9] > 0
    assert body[11, 13] > 0
    assert body[13, 15] > 0
    assert body[11, 12] > 0


def test_shape_aware_model_preserves_training_interface() -> None:
    config = ShapeAwareSkeletonBertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_frames=8,
        graph_channels=8,
        graph_layers=1,
        graph_temporal_kernel=3,
    )
    model = ShapeAwareSkeletonBert(config)
    streams = {
        "body": torch.randn(2, 8, 25, 2),
        "hands": torch.randn(2, 8, 42, 2),
        "face": torch.randn(2, 8, 37, 2),
    }
    valid = torch.tensor(
        [[True] * 8, [True] * 6 + [False] * 2], dtype=torch.bool
    )
    mask = torch.zeros_like(valid)
    mask[:, 2:4] = True
    layers = model.encode(streams, valid, mask, return_all_layers=True)
    assert isinstance(layers, list)
    assert len(layers) == 2
    assert layers[-1].shape == (2, 8, 32)
    reconstruction = model(streams, valid, mask)
    assert reconstruction["body"].shape == (2, 8, 25, 2)
    assert reconstruction["hands"].shape == (2, 8, 42, 2)
    assert reconstruction["face"].shape == (2, 8, 37, 2)
    assert config.to_dict()["architecture"] == "part_stgcn_temporal_bert"
