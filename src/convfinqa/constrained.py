"""Constrained decoding: logits processors that restrict what the model may write."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor


class AllowedTokens(LogitsProcessor):
    """Allow only the given token ids. Every other logit becomes -inf, so its probability is exactly 0."""

    def __init__(self, allowed_ids):
        self.allowed = torch.as_tensor(list(allowed_ids), dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed.to(scores.device)] = 0.0
        return scores + mask
