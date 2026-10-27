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


def generate_constrained(
    model, tok, chats: list[list[dict]], compiled, batch_size: int = 4, max_new_tokens: int = 96
) -> list[str]:
    """Greedy answers that are forced to follow a grammar.

    Like `probing.generate_greedy` (prompts sorted by length once, fixed batches, left padding), with
    xgrammar's logits processor added. Each batch gets a processor of its own, because it keeps the
    state of the parse.

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        chats: Each chat ends with a user message.
        compiled: A grammar compiled with `compile_grammar` for this tokenizer and model.
        batch_size: Prompts per batch.
        max_new_tokens: Generation limit; an answer that reaches it may be cut off before the grammar
            is complete.

    Returns:
        The generated text for each chat, without special tokens.
    """
    from transformers import LogitsProcessorList
    from xgrammar.contrib.hf import LogitsProcessor

    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [""] * len(texts)
    tok.padding_side = "left"
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([LogitsProcessor(compiled)]),
            )
        for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[i] = tok.decode(g, skip_special_tokens=True)
    return out
