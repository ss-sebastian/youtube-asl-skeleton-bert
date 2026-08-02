from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers import BertConfig, BertModel

from .features import COORDINATE_DIM, STREAM_JOINTS


@dataclass(frozen=True)
class SkeletonBertConfig:
    """Configuration for a small BERT trained from scratch on skeleton frames."""

    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_frames: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


class SkeletonBert(nn.Module):
    """A single-stream temporal BERT with masked skeleton reconstruction.

    BERT is randomly initialized. It receives continuous frame embeddings through
    ``inputs_embeds`` and never receives text, glosses, or pretrained language weights.
    """

    def __init__(self, config: SkeletonBertConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = config
        self.stream_slices: dict[str, slice] = {}
        offset = 0
        for name, joints in STREAM_JOINTS.items():
            width = joints * COORDINATE_DIM
            self.stream_slices[name] = slice(offset, offset + width)
            offset += width
        self.input_dim = offset

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(self.input_dim, config.hidden_size)
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
        )
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.reconstruction_head = nn.Linear(config.hidden_size, self.input_dim)

    def flatten_streams(self, streams: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([streams[name].flatten(start_dim=2) for name in STREAM_JOINTS], dim=-1)

    def split_reconstruction(self, reconstruction: torch.Tensor) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, joints in STREAM_JOINTS.items():
            values = reconstruction[..., self.stream_slices[name]]
            result[name] = values.unflatten(-1, (joints, COORDINATE_DIM))
        return result

    def encode(
        self,
        streams: dict[str, torch.Tensor],
        valid: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        frames = self.flatten_streams(streams)
        if frames.shape[1] > self.config.max_frames:
            raise ValueError(
                f"Sequence length {frames.shape[1]} exceeds max_frames={self.config.max_frames}"
            )
        embeddings = self.input_projection(self.input_norm(frames))
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
            # The first state is the embedding output; RSA uses encoder-layer outputs only.
            return list(outputs.hidden_states[1:])
        return outputs.last_hidden_state

    def forward(
        self,
        streams: dict[str, torch.Tensor],
        valid: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(streams, valid, mask)
        assert isinstance(hidden, torch.Tensor)
        return self.split_reconstruction(self.reconstruction_head(hidden))


def masked_mean(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
