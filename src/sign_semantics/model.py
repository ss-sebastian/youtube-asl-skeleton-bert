from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import STREAM_JOINTS


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.1
    max_frames: int = 256


class StreamProjection(nn.Module):
    def __init__(self, joints: int, d_model: int, dropout: float) -> None:
        super().__init__()
        input_dim = joints * 3
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, stream: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        projected = self.network(stream.flatten(start_dim=2))
        if mask is not None:
            projected = torch.where(mask.unsqueeze(-1), self.mask_token, projected)
        return projected


class MultiStreamSignTransformer(nn.Module):
    """SHuBERT-style body/hands/face masked-cluster Transformer.

    The implementation is intentionally compact and exposes every Transformer layer,
    making layer-wise sentence representations available for RSA.
    """

    def __init__(self, config: ModelConfig, cluster_sizes: dict[str, int]) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.projections = nn.ModuleDict(
            {
                name: StreamProjection(joints, config.d_model, config.dropout)
                for name, joints in STREAM_JOINTS.items()
            }
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * len(STREAM_JOINTS), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )
        self.position = nn.Parameter(torch.zeros(1, config.max_frames, config.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_model * config.ff_multiplier,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(config.d_model, cluster_sizes[name]) for name in STREAM_JOINTS}
        )

    def encode(
        self,
        streams: dict[str, torch.Tensor],
        valid: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        time = valid.shape[1]
        if time > self.config.max_frames:
            raise ValueError(f"Sequence length {time} exceeds max_frames={self.config.max_frames}")
        projected = [self.projections[name](streams[name], mask) for name in STREAM_JOINTS]
        hidden = self.fusion(torch.cat(projected, dim=-1)) + self.position[:, :time]
        states: list[torch.Tensor] = []
        padding = ~valid
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding)
            if return_all_layers:
                states.append(self.final_norm(hidden))
        hidden = self.final_norm(hidden)
        return states if return_all_layers else hidden

    def forward(
        self,
        streams: dict[str, torch.Tensor],
        valid: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(streams, valid, mask)
        assert isinstance(hidden, torch.Tensor)
        return {name: self.heads[name](hidden) for name in STREAM_JOINTS}


def masked_mean(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

