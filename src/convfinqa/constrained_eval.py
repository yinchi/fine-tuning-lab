"""Generation with a grammar over a whole evaluation split (§3.6).

The same prompts and the same greedy decoding, one prompt at a time, as `evaluate.make_generator`; the only
difference is the logits processor that restricts the next token to those the grammar allows. A run with a
grammar is therefore paired with the run without one on the same turns.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Callable

from .constrained import compile_grammar


def make_constrained_generator(model, tok, compiled, max_new_tokens: int) -> Callable[[list[dict]], dict]:
    """`evaluate.make_generator` with a grammar.

    Each call gets a processor of its own, because the processor keeps the state of the parse.

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        compiled: A grammar compiled with `constrained.compile_grammar` for this tokenizer and model.
        max_new_tokens: Generation limit; an output that reaches it has `finish == "length"` and may be cut
            off before the grammar is complete.

    Returns:
        A function from a chat (ending with a user message) to `output`, `new_tokens`, `finish`,
        `prompt_tokens` and `seconds`, as `evaluate.make_generator` returns.
    """
    import torch
    from transformers import LogitsProcessorList
    from xgrammar.contrib.hf import LogitsProcessor

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def generate(chat: list[dict]) -> dict:
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([LogitsProcessor(compiled)]),
            )
        sync()
        new = out[0, enc["input_ids"].shape[1]:].tolist()
        return {
            "output": tok.decode(new, skip_special_tokens=True), "new_tokens": len(new),
            "finish": "stop" if new and new[-1] in eos else "length",
            "prompt_tokens": int(enc["input_ids"].shape[1]), "seconds": time.perf_counter() - start,
        }

    return generate


def generate_with_grammar(
    cfg: dict, convs: list[dict], examples: list[dict], *, system: str, grammar: str, max_new_tokens: int,
    mode: str, adapter: Path | None = None, sink: Callable[[dict], None] | None = None,
) -> list[dict]:
    """One greedy, grammar-constrained answer for every turn of `convs`.

    The conversations and examples are used as they are: for Python output, pass them through
    `python_eval.python_conversation` first. The base weights stay bf16 and an adapter, if given, fp32.

    Args:
        cfg: The settings (`model`, `memory_cap_gb`).
        convs: The conversations to answer.
        examples: The example conversations shown first (none for zero-shot).
        system: The system message.
        grammar: An EBNF grammar with a `root` rule (see `grammars`).
        max_new_tokens: Generation limit.
        mode: `gold` or `pred` (see `evaluate.run_generation`).
        adapter: The directory `finetune.train_lora` wrote the adapter to, or `None` for the untuned model.
        sink: Called with each row as soon as it exists.

    Returns:
        One row per turn, as `evaluate.run_generation` returns them.
    """
    import torch

    from .evaluate import load_model, run_generation

    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    generate = make_constrained_generator(model, tok, compile_grammar(tok, grammar, vocab_size), max_new_tokens)
    rows = run_generation(convs, generate, system=system, examples=examples, mode=mode, sink=sink)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
