from __future__ import annotations

"""Train the skeleton-only multi-stream Spatial-GCN SHuBERT experiment."""

from . import train as shared_training
from .spatial_shubert_model import SpatialSHuBERT, SpatialSHuBERTConfig


def main() -> None:
    shared_training.SkeletonBert = SpatialSHuBERT
    shared_training.SkeletonBertConfig = SpatialSHuBERTConfig
    shared_training.main()


if __name__ == "__main__":
    main()
