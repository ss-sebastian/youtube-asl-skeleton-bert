from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features import STREAM_JOINTS, cluster_features_torch


class ClusterTargeter:
    def __init__(self, centers_path: str | Path, device: torch.device) -> None:
        with np.load(centers_path) as payload:
            self.centers = {
                name: torch.from_numpy(payload[name]).to(device=device, dtype=torch.float32)
                for name in STREAM_JOINTS
            }

    @property
    def cluster_sizes(self) -> dict[str, int]:
        return {name: value.shape[0] for name, value in self.centers.items()}

    @torch.no_grad()
    def assign(self, streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assignments: dict[str, torch.Tensor] = {}
        for name in STREAM_JOINTS:
            features = cluster_features_torch(streams[name]).float()
            batch, time, dimension = features.shape
            centers = self.centers[name]
            if centers.shape[1] != dimension:
                raise ValueError(
                    f"Cluster dimension mismatch for {name}: {centers.shape[1]} != {dimension}"
                )
            flat = features.reshape(-1, dimension)
            # Squared Euclidean distance without materializing (N, K, D).
            distances = (
                flat.square().sum(dim=1, keepdim=True)
                - 2 * flat @ centers.T
                + centers.square().sum(dim=1).unsqueeze(0)
            )
            assignments[name] = distances.argmin(dim=1).reshape(batch, time)
        return assignments

