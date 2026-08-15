from __future__ import annotations

from typing import Any

import torch

from .model import SkeletonBert, SkeletonBertConfig
from .shape_model import ShapeAwareSkeletonBert, ShapeAwareSkeletonBertConfig
from .spatial_shubert_model import SpatialSHuBERT, SpatialSHuBERTConfig


SignModel = SkeletonBert | ShapeAwareSkeletonBert | SpatialSHuBERT


def model_from_checkpoint(checkpoint: dict[str, Any]) -> SignModel:
    """Instantiate the exact architecture recorded in a training checkpoint."""
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint must contain model and model_config")
    values = dict(checkpoint["model_config"])
    architecture = values.get("architecture", "flat_projection_temporal_bert")
    if architecture == "part_stgcn_temporal_bert":
        model: SignModel = ShapeAwareSkeletonBert(
            ShapeAwareSkeletonBertConfig(**values)
        )
    elif architecture == "multistream_spatial_gcn_shubert":
        model = SpatialSHuBERT(SpatialSHuBERTConfig(**values))
    elif architecture == "flat_projection_temporal_bert":
        values.pop("architecture", None)
        model = SkeletonBert(SkeletonBertConfig(**values))
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
    model.load_state_dict(checkpoint["model"])
    return model


def load_trained_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[SignModel, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = model_from_checkpoint(checkpoint).to(device)
    model.eval()
    return model, checkpoint
