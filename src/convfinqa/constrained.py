"""Constrained decoding: logits processors that restrict what the model may write."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor


class AllowedTokens(LogitsProcessor):
    """A LogitsProcessor that allows only the given token ids.
    
    Every other logit becomes -inf, so its probability is exactly 0.
    """

    def __init__(self, allowed_ids):
        """Initialize with the set of allowed token ids."""
        self.allowed = torch.as_tensor(list(allowed_ids), dtype=torch.long)
        """The set of allowed token ids."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Restrict the logits to only the allowed token ids.
        Args:
            input_ids: The input token ids generated so far.
            scores: The logits for the next token.
        
        Returns:
            The modified logits, where invalid tokens have all logits set to `-inf`, i.e. zero
            probability.
        """
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed.to(scores.device)] = 0.0
        return scores + mask
