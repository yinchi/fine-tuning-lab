"""Generation with a trained LoRA adapter over a whole evaluation split (the model of §3.3, zero-shot)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

MAX_NEW_TOKENS = 96  # the same limit as the untuned baseline (configs/baseline.json), so outputs are comparable


def split_conversations(split: str, val: list[dict], dev: list[dict], freeze_file: Path | None = None) -> list[dict]:
    """The conversations of an evaluation split, refusing the dev split before the freeze.

    Args:
        split: `val` or `dev`.
        val: The validation conversations.
        dev: The development conversations.
        freeze_file: The file whose existence marks the freeze (default: `configs/FROZEN`).

    Returns:
        The conversations of `split`.

    Raises:
        ValueError: If `split` is neither `val` nor `dev`.
        NotFrozenError: If `split` is `dev` and `configs/FROZEN` does not exist.
    """
    from .evaluate import FREEZE_FILE, require_freeze

    if split not in ("val", "dev"):
        raise ValueError(f"split must be 'val' or 'dev', not {split!r}")
    require_freeze(split, freeze_file or FREEZE_FILE)
    return val if split == "val" else dev


def generate_tuned(
    cfg: dict, adapter: Path, convs: list[dict], mode: str, sink: Callable[[dict], None] | None = None
) -> list[dict]:
    """One greedy answer for every turn of `convs` from the base model with the adapter attached.

    The prompt is the training prompt (the one-line system message, no examples), one prompt at a time, as in the
    baseline, so only the model differs. The base weights stay bf16 and the adapter fp32, as in training.

    Args:
        cfg: The settings of the run (`model`, `memory_cap_gb`).
        adapter: The directory `finetune.train_lora` wrote the adapter to.
        convs: The conversations to answer.
        mode: `gold` or `pred` (the history of the earlier turns; see `evaluate.run_generation`).
        sink: Called with each row as soon as it exists.

    Returns:
        One row per turn, as `evaluate.run_generation` returns them.
    """
    import torch
    from peft import PeftModel

    from .evaluate import load_model, make_generator, run_generation
    from .formatting import SYSTEM_PROMPT

    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    generate = make_generator(model, tok, MAX_NEW_TOKENS)
    rows = run_generation(convs, generate, system=SYSTEM_PROMPT, examples=[], mode=mode, sink=sink)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
