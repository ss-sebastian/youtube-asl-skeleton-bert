from __future__ import annotations

"""Train the shape-aware model through the unchanged objective/metrics loop."""

from . import train as shared_training
from .shape_model import ShapeAwareSkeletonBert, ShapeAwareSkeletonBertConfig


def main() -> None:
    # The shared loop owns the existing data, contrastive objective, validation,
    # checkpoint, progress, and metrics behavior. Only its model constructor is
    # replaced for this explicitly separate experiment.
    shared_training.SkeletonBert = ShapeAwareSkeletonBert
    shared_training.SkeletonBertConfig = ShapeAwareSkeletonBertConfig
    shared_training.main()


if __name__ == "__main__":
    main()
