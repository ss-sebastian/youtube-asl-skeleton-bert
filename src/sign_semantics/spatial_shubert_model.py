from __future__ import annotations

"""Skeleton-only, multi-stream SHuBERT-style encoder.

The graph front-end is deliberately spatial-only.  It learns anatomical shape
within each frame; all temporal/contextual modelling is left to BERT so that
the real-versus-shuffled-context intervention has a clean interpretation.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers import BertConfig, BertModel

from .features import COORDINATE_DIM
from .shape_model import BODY_EDGES, HAND_EDGES, normalized_adjacency


class SpatialGraphBlock(nn.Module):
    """One residual spatial message-passing block with no temporal operation."""

    def __init__(self, channels: int, adjacency: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.message = nn.Linear(channels, channels, bias=False)
        self.update = nn.Linear(channels * 2, channels)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        neighbours = torch.einsum("vw,btwc->btvc", self.adjacency, values)
        update = self.update(torch.cat([values, self.message(neighbours)], dim=-1))
        return self.norm(values + self.dropout(self.activation(update)))


class PartSpatialGCN(nn.Module):
    """Encode anatomical shape independently at every video frame."""

    def __init__(
        self,
        joints: int,
        edges: tuple[tuple[int, int], ...] | None,
        channels: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        adjacency = normalized_adjacency(joints, edges)
        self.register_buffer("adjacency", adjacency)
        # Coordinates plus displacement from the graph-neighbour mean.  There
        # is intentionally no velocity or temporal convolution in this module.
        self.input_projection = nn.Linear(COORDINATE_DIM * 2, channels)
        self.joint_embedding = nn.Parameter(torch.empty(1, 1, joints, channels))
        nn.init.normal_(self.joint_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            SpatialGraphBlock(channels, adjacency, dropout) for _ in range(layers)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        present = coordinates.abs().sum(dim=-1, keepdim=True) > 0
        neighbour_mean = torch.einsum("vw,btwc->btvc", self.adjacency, coordinates)
        shape = coordinates - neighbour_mean
        values = self.input_projection(torch.cat([coordinates, shape], dim=-1))
        values = (values + self.joint_embedding) * present.to(values.dtype)
        for block in self.blocks:
            values = block(values) * present.to(values.dtype)
        weights = present.to(values.dtype)
        return (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


@dataclass(frozen=True)
class SpatialSHuBERTConfig:
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_frames: int = 256
    causal_attention: bool = False
    graph_channels: int = 48
    graph_layers: int = 2
    clusters_per_stream: int = 100
    architecture: str = "multistream_spatial_gcn_shubert"

    def to_dict(self) -> dict:
        return asdict(self)


class SpatialSHuBERT(nn.Module):
    """Part-specific spatial GCN followed by a contextual Transformer."""

    stream_names = ("body", "right_hand", "left_hand", "face")

    def __init__(self, config: SpatialSHuBERTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.architecture != "multistream_spatial_gcn_shubert":
            raise ValueError(f"Unsupported architecture: {config.architecture}")
        self.config = config
        kwargs = {
            "channels": config.graph_channels,
            "layers": config.graph_layers,
            "dropout": config.hidden_dropout_prob,
        }
        self.body_encoder = PartSpatialGCN(25, BODY_EDGES, **kwargs)
        # Weight sharing encourages the same handshape to have a related code
        # on either side; side embeddings preserve handedness when needed.
        self.hand_encoder = PartSpatialGCN(21, HAND_EDGES, **kwargs)
        self.face_encoder = PartSpatialGCN(37, None, **kwargs)
        self.side_embedding = nn.Parameter(torch.empty(2, config.graph_channels))
        nn.init.normal_(self.side_embedding, std=0.02)

        self.part_norms = nn.ModuleDict(
            {name: nn.LayerNorm(config.graph_channels) for name in self.stream_names}
        )
        self.part_fusion = nn.Sequential(
            nn.LayerNorm(config.graph_channels * len(self.stream_names)),
            nn.Linear(config.graph_channels * len(self.stream_names), config.hidden_size),
            nn.GELU(),
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        bert_config = BertConfig(
            vocab_size=1,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_frames,
            type_vocab_size=1,
            pad_token_id=0,
            is_decoder=False,
            use_cache=False,
        )
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.cluster_heads = nn.ModuleDict(
            {
                name: nn.Linear(config.hidden_size, config.clusters_per_stream)
                for name in self.stream_names
            }
        )

    def part_embeddings(self, streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hands = streams["hands"]
        parts = {
            "body": self.body_encoder(streams["body"]),
            "right_hand": self.hand_encoder(hands[:, :, :21]) + self.side_embedding[0],
            "left_hand": self.hand_encoder(hands[:, :, 21:]) + self.side_embedding[1],
            "face": self.face_encoder(streams["face"]),
        }
        return {name: self.part_norms[name](value) for name, value in parts.items()}

    def shape_embeddings(self, streams: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = self.part_embeddings(streams)
        return self.part_fusion(torch.cat([parts[name] for name in self.stream_names], dim=-1))

    def encode(
        self,
        streams: dict[str, torch.Tensor],
        valid: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        return_all_layers: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        if causal:
            raise ValueError("SpatialSHuBERT is a bidirectional masked-prediction model")
        embeddings = self.shape_embeddings(streams)
        if embeddings.shape[1] > self.config.max_frames:
            raise ValueError(
                f"Sequence length {embeddings.shape[1]} exceeds max_frames={self.config.max_frames}"
            )
        if mask is not None:
            embeddings = torch.where(mask.unsqueeze(-1), self.mask_token, embeddings)
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=valid.to(dtype=torch.long),
            output_hidden_states=return_all_layers,
            return_dict=True,
        )
        if return_all_layers:
            assert outputs.hidden_states is not None
            return list(outputs.hidden_states[1:])
        return outputs.last_hidden_state

    def predict_clusters(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(hidden) for name, head in self.cluster_heads.items()}
