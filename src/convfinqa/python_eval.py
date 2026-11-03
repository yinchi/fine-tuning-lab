"""Evaluating Python as the output format: the prompt, the generation and the scoring.

The prompt has three rules and nothing else (no structure, no syntax description). The conversations are
rewritten with Python as the gold programs, so that `evaluate.run_generation` builds the chats unchanged (the
examples, the gold history and the model's own history all read as Python). The scoring is separate from
`evaluate.score_rows`, which executes DSL programs; the scores have the same fields, so `evaluate.summarize`
and the paired bootstrap work on them.
"""

from __future__ import annotations

import copy
import gc
import re
from typing import Callable

from .dsl import executor, python_format
from .probing import CORRECT, WRONG_ANSWER_SAME_SHAPE

SYSTEM_PYTHON = (
    "Answer with a valid Python program. Use only numbers that appear on the page. "
    "Assign the final answer to ans."
)
MAX_NEW_TOKENS = 256  # more than the DSL's 96: free-form Python is longer, and the limit must not decide the result


def python_conversation(conv: dict) -> dict:
    """The conversation with every gold program translated to Python (`python_format.to_python`)."""
    out = copy.deepcopy(conv)
    out["annotation"]["turn_program"] = [python_format.to_python(p) for p in conv["annotation"]["turn_program"]]
    return out


_FENCE = re.compile(r"\A\s*```[A-Za-z]*\n(.*?)\n?```\s*\Z", re.S)


def strip_fence(output: str) -> str:
    """The code inside a markdown fence that wraps the whole output, or the output itself if it is not fenced."""
    m = _FENCE.match(output)
    return m.group(1) if m else output


def score_python_rows(rows: list[dict], turn_rows: list[dict], strip_fences: bool = False) -> list[dict]:
    """Execute and score every Python prediction against the gold turn.

    Args:
        rows: Predictions (`id`, `turn`, `output`, `finish`, `new_tokens`).
        turn_rows: The gold turns, with their DSL programs and answers (`data.tables.turn_rows`).
        strip_fences: Run the code inside a fence that wraps the output, instead of counting the fence as invalid.

    Returns:
        One dict per turn with the fields of `evaluate.score_rows`: `kind` is `correct`, `wrong_answer`, or the
        error kind of `python_format` (`code_fence`, `syntax`, `outside_subset`, ...).

    Raises:
        ValueError: If the predictions and the gold turns do not cover the same turns.
    """
    gold = {(r["id"], r["turn"]): r for r in turn_rows}
    predicted = {(r["id"], r["turn"]): r for r in rows}
    if len(predicted) != len(rows) or predicted.keys() != gold.keys():
        raise ValueError("predictions and gold turns differ")
    scores = []
    for key, g in gold.items():
        row = predicted[key]
        code = strip_fence(row["output"]) if strip_fences else row["output"]
        value, error = python_format.try_execute_python(code)
        if error is not None:
            kind = error
        else:
            kind = CORRECT if executor.answers_match(value, g["answer"]) else WRONG_ANSWER_SAME_SHAPE
        scores.append({
            "id": g["id"], "turn": g["turn"], "correct": kind == CORRECT, "kind": kind, "type": g["type"],
            "part": g["part"], "number_turn": g["number_turn"], "finish": row.get("finish"),
            "new_tokens": row.get("new_tokens"),
        })
    return scores


def generate_untuned_python(
    cfg: dict, convs: list[dict], examples: list[dict], mode: str, sink: Callable[[dict], None] | None = None
) -> list[dict]:
    """One greedy Python answer for every turn of `convs` from the untuned model.

    Args:
        cfg: The settings (`model`, `memory_cap_gb`).
        convs: The conversations to answer, with their DSL gold programs (translated here).
        examples: The example conversations, also with DSL gold programs.
        mode: `gold` or `pred` (see `evaluate.run_generation`).
        sink: Called with each row as soon as it exists.

    Returns:
        One row per turn, as `evaluate.run_generation` returns them.
    """
    import torch

    from .evaluate import load_model, make_generator, run_generation

    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    generate = make_generator(model, tok, MAX_NEW_TOKENS)
    rows = run_generation(
        [python_conversation(c) for c in convs], generate, system=SYSTEM_PYTHON,
        examples=[python_conversation(c) for c in examples], mode=mode, sink=sink,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
