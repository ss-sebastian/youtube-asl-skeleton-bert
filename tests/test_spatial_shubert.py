from __future__ import annotations

import numpy as np
import torch

from sign_semantics.skeleton_codebooks import PARTS, SkeletonCodebooks
from sign_semantics.spatial_shubert_model import SpatialSHuBERT, SpatialSHuBERTConfig
from sign_semantics.train import masked_cluster_objective


def tiny_model() -> SpatialSHuBERT:
    return SpatialSHuBERT(
        SpatialSHuBERTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_frames=8,
            graph_channels=8,
            graph_layers=1,
            clusters_per_stream=5,
        )
    )


def batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    streams = {
        "body": torch.randn(2, 8, 25, 2),
        "hands": torch.randn(2, 8, 42, 2),
        "face": torch.randn(2, 8, 37, 2),
    }
    observed = {
        "body": torch.ones(2, 8, 25, dtype=torch.bool),
        "hands": torch.ones(2, 8, 42, dtype=torch.bool),
        "face": torch.ones(2, 8, 37, dtype=torch.bool),
    }
    valid = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])
    return streams, observed, valid


def test_spatial_frontend_and_layerwise_interface() -> None:
    model = tiny_model()
    streams, _, valid = batch()
    layers = model.encode(streams, valid, return_all_layers=True)
    assert isinstance(layers, list)
    assert len(layers) == 2
    assert layers[-1].shape == (2, 8, 32)
    logits = model.predict_clusters(layers[-1])
    assert set(logits) == set(PARTS)
    assert all(value.shape == (2, 8, 5) for value in logits.values())
    assert not any("temporal" in name for name, _ in model.named_modules())


def test_masked_cluster_loss_uses_all_four_streams(tmp_path) -> None:
    dimensions = {"body": 50, "right_hand": 42, "left_hand": 42, "face": 74}
    arrays = {}
    generator = np.random.default_rng(0)
    for name, width in dimensions.items():
        arrays[f"{name}_mean"] = np.zeros(width, dtype=np.float32)
        arrays[f"{name}_scale"] = np.ones(width, dtype=np.float32)
        arrays[f"{name}_centroids"] = generator.normal(size=(5, width)).astype(np.float32)
    path = tmp_path / "codebooks.npz"
    np.savez(path, **arrays)
    codebooks = SkeletonCodebooks(path, torch.device("cpu"))
    streams, observed, valid = batch()
    loss, metrics, statistics = masked_cluster_objective(
        tiny_model(),
        streams,
        observed,
        valid,
        {"mask_probability": 0.5, "mean_span_length": 2},
        codebooks,
        torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(loss)
    assert statistics == {}
    assert "cluster_accuracy" in metrics
    assert all(f"{name}_loss" in metrics for name in PARTS)
