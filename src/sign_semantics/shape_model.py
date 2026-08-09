from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers import BertConfig, BertModel

from .features import COORDINATE_DIM, STREAM_JOINTS


# MediaPipe hand topology. The palm cross-links keep the metacarpal bases from
# behaving like five unrelated chains.
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

# The first 25 MediaPipe pose landmarks retained by the released YouTube-ASL
# keypoints: face anchors, shoulders, arms, hands, and hips.
BODY_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
)


def normalized_adjacency(
    joints: int,
    edges: tuple[tuple[int, int], ...] | None,
) -> torch.Tensor:
    """Return symmetric degree-normalized adjacency with self loops."""
    adjacency = torch.eye(joints, dtype=torch.float32)
    if edges is None:
        # The public input retains only a sparse 37-point face subset rather
        # than the full MediaPipe mesh. A dense within-face graph avoids
        # inventing anatomical edges that are absent from the release.
        adjacency.fill_(1.0)
    else:
        for source, target in edges:
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class SpatialTemporalGraphBlock(nn.Module):
    """One fixed-graph spatial message pass followed by local temporal conv."""

    def __init__(
        self,
        channels: int,
        adjacency: torch.Tensor,
        temporal_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Linear(channels, channels, bias=False)
        self.temporal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
            groups=channels,
            bias=False,
        )
        self.channel_mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = torch.einsum("vw,btwc->btvc", self.adjacency, values)
        values = self.spatial(values)
        values = values.permute(0, 3, 1, 2)
        values = self.channel_mix(self.temporal(values))
        values = values.permute(0, 2, 3, 1)
        return self.dropout(self.activation(self.norm(values + residual)))


class PartSTGCN(nn.Module):
    """Encode one anatomical part while retaining its explicit joint graph."""

    def __init__(
        self,
        joints: int,
        edges: tuple[tuple[int, int], ...] | None,
        channels: int,
        layers: int,
        temporal_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        adjacency = normalized_adjacency(joints, edges)
        self.register_buffer("adjacency", adjacency)
        # xy + first temporal difference + displacement from graph-neighbour mean.
        self.input_projection = nn.Linear(COORDINATE_DIM * 3, channels)
        self.joint_embedding = nn.Parameter(torch.empty(1, 1, joints, channels))
        nn.init.normal_(self.joint_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            SpatialTemporalGraphBlock(
                channels, adjacency, temporal_kernel, dropout
            )
            for _ in range(layers)
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        present = coordinates.abs().sum(dim=-1, keepdim=True) > 0
        velocity = torch.diff(coordinates, dim=1, prepend=coordinates[:, :1])
        neighbour_mean = torch.einsum(
            "vw,btwc->btvc", self.adjacency, coordinates
        )
        relative_shape = coordinates - neighbour_mean
        features = torch.cat([coordinates, velocity, relative_shape], dim=-1)
        values = self.input_projection(features) + self.joint_embedding
        values = values * present.to(values.dtype)
        for block in self.blocks:
            values = block(values)

        weights = present.to(values.dtype)
        return (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


@dataclass(frozen=True)
class ShapeAwareSkeletonBertConfig:
    """Graph-fronted temporal BERT, trained from scratch without language input."""

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
    graph_temporal_kernel: int = 5
    architecture: str = "part_stgcn_temporal_bert"

    def to_dict(self) -> dict:
        return asdict(self)


class ShapeAwareSkeletonBert(nn.Module):
    """Part-specific ST-GCN shape encoder plus the original temporal BERT."""

    def __init__(self, config: ShapeAwareSkeletonBertConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.architecture != "part_stgcn_temporal_bert":
            raise ValueError(f"Unsupported architecture: {config.architecture}")
        self.config = config
        part = {
            "channels": config.graph_channels,
            "layers": config.graph_layers,
            "temporal_kernel": config.graph_temporal_kernel,
            "dropout": config.hidden_dropout_prob,
        }
        self.body_encoder = PartSTGCN(25, BODY_EDGES, **part)
        self.right_hand_encoder = PartSTGCN(21, HAND_EDGES, **part)
        self.left_hand_encoder = PartSTGCN(21, HAND_EDGES, **part)
        self.face_encoder = PartSTGCN(37, None, **part)
        self.part_fusion = nn.Sequential(
            nn.LayerNorm(config.graph_channels * 4),
            nn.Linear(config.graph_channels * 4, config.hidden_size),
            nn.GELU(),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
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
            is_decoder=config.causal_attention,
            use_cache=False,
        )
        self.bert = BertModel(bert_config, add_pooling_layer=False)

        self.stream_slices: dict[str, slice] = {}
        offset = 0
        for name, joints in STREAM_JOINTS.items():
            width = joints * COORDINATE_DIM
            self.stream_slices[name] = slice(offset, offset + width)
            offset += width
        self.reconstruction_head = nn.Linear(config.hidden_size, offset)

    def shape_embeddings(self, streams: dict[str, torch.Tensor]) -> torch.Tensor:
        hands = streams["hands"]
        parts = (
            self.body_encoder(streams["body"]),
            self.right_hand_encoder(hands[:, :, :21]),
            self.left_hand_encoder(hands[:, :, 21:]),
            self.face_encoder(streams["face"]),
        )
        return self.part_fusion(torch.cat(parts, dim=-1))

    def encode(
        self,
        streams: dict[str, torch.Tensor],
        valid: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        return_all_layers: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        embeddings = self.shape_embeddings(streams)
        if embeddings.shape[1] > self.config.max_frames:
            raise ValueError(
                f"Sequence length {embeddings.shape[1]} exceeds "
                f"max_frames={self.config.max_frames}"
            )
        if mask is not None:
            embeddings = torch.where(mask.unsqueeze(-1), self.mask_token, embeddings)
        if causal != self.config.causal_attention:
            raise ValueError(
                "The requested attention direction does not match the model configuration"
            )
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=valid.to(dtype=torch.long),
            use_cache=False,
            output_hidden_states=return_all_layers,
            return_dict=True,
        )
        if return_all_layers:
            assert outputs.hidden_states is not None
            return list(outputs.hidden_states[1:])
        return outputs.last_hidden_state

    def split_reconstruction(
        self, reconstruction: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, joints in STREAM_JOINTS.items():
            values = reconstruction[..., self.stream_slices[name]]
            result[name] = values.unflatten(-1, (joints, COORDINATE_DIM))
        return result

    def forward(
        self,
        streams: dict[str, torch.Tensor],
        valid: torch.Tensor,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(streams, valid, mask, causal=causal)
        assert isinstance(hidden, torch.Tensor)
        return self.split_reconstruction(self.reconstruction_head(hidden))
