"""Generation with the adapter merged into the weights, to compare with the adapter kept separate (§3.4)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

PRECISIONS = ("bf16", "fp32")
"""The precision of the whole model, merge included: the merge is rounded to bf16, or the model stays in fp32."""


def generate_merged(
    cfg: dict, adapter: Path, convs: list[dict], mode: str, precision: str,
    sink: Callable[[dict], None] | None = None,
) -> list[dict]:
    """One greedy answer for every turn of `convs` from the base model with the adapter merged into its weights.

    The prompt, the decoding and the history are those of `tuned_eval.generate_tuned`, so only the weights differ
    from the adapter kept separate (a bf16 base with an fp32 adapter).

    Args:
        cfg: The settings of the run (`model`, `memory_cap_gb`).
        adapter: The directory `finetune.train_lora` wrote the adapter to.
        convs: The conversations to answer.
        mode: `gold` or `pred` (the history of the earlier turns).
        precision: `bf16`, the merge rounded into bf16 weights, or `fp32`, the whole model kept in fp32.
        sink: Called with each row as soon as it exists.

    Returns:
        One row per turn, as `evaluate.run_generation` returns them.

    Raises:
        ValueError: If `precision` is not in `PRECISIONS`.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .evaluate import make_generator, run_generation
    from .formatting import SYSTEM_PROMPT
    from .tuned_eval import MAX_NEW_TOKENS

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["memory_cap_gb"] / total_gb))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype, device_map="cuda").eval()
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()
    generate = make_generator(model, tok, MAX_NEW_TOKENS)
    rows = run_generation(convs, generate, system=SYSTEM_PROMPT, examples=[], mode=mode, sink=sink)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
