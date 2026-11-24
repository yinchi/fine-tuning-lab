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


DSL_GRAMMAR = r'''
root      ::= number | step (", " step)*
step      ::= operation "(" argument ", " argument ")"
operation ::= "add" | "subtract" | "multiply" | "divide" | "exp" | "greater"
argument  ::= number | "#" [0-9]+ | constant
number    ::= "-"? ([0-9]+ ("." [0-9]+)? | "." [0-9]+) "%"?
constant  ::= "const_" ("m1" | [1-9] | "10" | "100" | "1000" | "10000" | "100000" | "1000000" | "10000000" | "1000000000")
'''
"""The syntax of a ConvFinQA program, in xgrammar's EBNF.

It follows the grammar of the DSL chapter and the form of the gold programs: `", "` between arguments and
between steps, numbers with an optional sign, a decimal part (or a leading point) and a percent sign, `#k`
references of any index, and the 18 constants of the dataset. It accepts every gold program of train and dev
(8,981 of them). It does not check that `#k` refers to an earlier step; that needs a grammar that counts steps.
"""


def compile_grammar(tok, grammar: str, vocab_size: int):
    """Compile a grammar for a tokenizer.

    The end-of-sequence token (`<|im_end|>` for Qwen's chat models) is what the model emits when the grammar
    is complete, and the only token allowed then.

    Args:
        tok: The tokenizer.
        grammar: An EBNF grammar in xgrammar's syntax, with a `root` rule.
        vocab_size: The size of the model's output layer, which can be larger than the tokenizer's
            vocabulary (Qwen3-4B pads it from 151,669 to 151,936).

    Returns:
        An `xgrammar.CompiledGrammar`.
    """
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size, stop_token_ids=[tok.eos_token_id])
    return xgr.GrammarCompiler(info).compile_grammar(grammar)


def grammar_accepts(compiled, text: str) -> bool:
    """Whether `text` is, in full, a string of the grammar."""
    import xgrammar as xgr

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return matcher.accept_string(text) and matcher.is_terminated()
