from __future__ import annotations

import torch


def sample_span_mask(
    valid: torch.Tensor,
    probability: float,
    mean_span_length: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample contiguous temporal masks without touching padding.

    At least one valid frame is masked for every non-empty sequence.
    """
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean tensor with shape (batch, time)")
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    if mean_span_length < 1:
        raise ValueError("mean_span_length must be positive")

    batch, time = valid.shape
    mask = torch.zeros_like(valid)
    for row in range(batch):
        length = int(valid[row].sum().item())
        if length == 0:
            continue
        target = max(1, round(length * probability))
        attempts = 0
        while int(mask[row].sum().item()) < target and attempts < time * 4:
            start = int(torch.randint(length, (1,), generator=generator).item())
            # Geometric-like span variability centered near mean_span_length.
            low = max(1, mean_span_length // 2)
            high = max(low + 1, mean_span_length * 3 // 2 + 1)
            span = int(torch.randint(low, high, (1,), generator=generator).item())
            mask[row, start : min(length, start + span)] = True
            attempts += 1
        if not mask[row].any():
            mask[row, 0] = True
    return mask & valid

