"""The fine-tuned model asked with one of the baseline's prompts: does a prompt still help after fine-tuning?"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from .tuned_eval import MAX_NEW_TOKENS


def baseline_prompt(baseline: dict, sets: dict, pool: dict[str, dict], variant: str, shots: int) -> tuple[str, list[dict]]:
    """The system message and the example conversations of a baseline prompt variant.

    Args:
        baseline: `configs/baseline.json`.
        sets: The `example_sets` artifact's `sets.json`.
        pool: The training conversations by id (the examples come from there, never from val or dev).
        variant: A variant of `prompts.VARIANTS`, such as `e3p`.
        shots: The example set (a seed of `baseline["example_seeds"]`); only 0 for a variant without examples.

    Returns:
        The system message, and the example conversations in prompt order.

    Raises:
        ValueError: If the variant is unknown, or has no examples and a set other than 0 is asked for.
    """
    from .prompts import parse_variant

    n_examples, kind = parse_variant(variant)
    if n_examples == 0 and shots != 0:
        raise ValueError("a variant with no examples has no example set: use set 0")
    examples = [pool[i] for i in sets["sets"][str(shots)][:n_examples]] if n_examples else []
    return baseline["system"][kind], examples


def generate_tuned_prompted(
    cfg: dict, adapter: Path, convs: list[dict], system: str, examples: list[dict], mode: str,
    sink: Callable[[dict], None] | None = None,
) -> list[dict]:
    """One greedy answer for every turn of `convs` from the base model with the adapter, with a given prompt.

    As `tuned_eval.generate_tuned`, but the system message and the examples are the caller's.

    Args:
        cfg: The settings of the run (`model`, `memory_cap_gb`).
        adapter: The directory `finetune.train_lora` wrote the adapter to.
        convs: The conversations to answer.
        system: The system message.
        examples: The example conversations shown first.
        mode: `gold` or `pred`.
        sink: Called with each row as soon as it exists.

    Returns:
        One row per turn, as `evaluate.run_generation` returns them.
    """
    import torch
    from peft import PeftModel

    from .evaluate import load_model, make_generator, run_generation

    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    generate = make_generator(model, tok, MAX_NEW_TOKENS)
    rows = run_generation(convs, generate, system=system, examples=examples, mode=mode, sink=sink)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
