"""Run a prompt over a split of ConvFinQA, and score the predictions.

Generation and scoring are separate: `run_generation` writes what the model said, `score_rows` reads it. The
model call is passed in as a function (`make_generator` builds it from a model), so the order of the prompts
and the filling of the history can be tested without a model.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from .data import ROOT, turns
from .probing import CORRECT, WRONG_ANSWER_SAME_SHAPE, WRONG_NUMBERS, WRONG_OPERATIONS, classify_output
from .prompts import prompt_chat

RUNS = {CORRECT, WRONG_OPERATIONS, WRONG_NUMBERS, WRONG_ANSWER_SAME_SHAPE}
"""The outcomes of an output that is a program and runs; any other outcome is an invalid output."""

FREEZE_FILE = ROOT / "configs" / "FROZEN"
"""Its existence means that the decisions of the project are frozen, and dev may be evaluated (PLAN.md, data protocol)."""


class NotFrozenError(RuntimeError):
    """Dev was asked for before the freeze."""


def require_freeze(split: str, freeze_file: Path = FREEZE_FILE) -> None:
    """Refuse to evaluate dev before the freeze: only the existence of `configs/FROZEN` lets it through.

    Args:
        split: `val` or `dev`.
        freeze_file: The file whose existence marks the freeze.

    Raises:
        NotFrozenError: If `split` is `dev` and the freeze file does not exist.
    """
    if split == "dev" and not freeze_file.exists():
        raise NotFrozenError(
            "dev is evaluated only after the freeze entry of PLAN.md's data protocol; "
            f"create {freeze_file} when the decisions are frozen"
        )


def load_model(model_name: str, memory_cap_gb: float):
    """Load a model in bf16 on the GPU, in evaluation mode, under a cap on the GPU memory the process may use.

    The GPU of the GB10 shares its memory with the desktop, and a run that takes all of it freezes the machine;
    over the cap, an allocation raises an error instead.

    Args:
        model_name: Hugging Face model id.
        memory_cap_gb: GPU memory the process may allocate.

    Returns:
        `(model, tokenizer)`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb / total_gb))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def make_generator(model, tok, max_new_tokens: int) -> Callable[[list[dict]], dict]:
    """A function that answers one chat greedily, one prompt at a time (no batching, no padding).

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        max_new_tokens: Generation limit; an output that reaches it has `finish == "length"`.

    Returns:
        A function from a chat (ending with a user message) to `output` (the text, special tokens
        removed, exactly as generated), `new_tokens`, `finish` (`stop` or `length`), `prompt_tokens` and
        `seconds`.
    """
    import torch

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def generate(chat: list[dict]) -> dict:
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
            )
        torch.cuda.synchronize()
        new = out[0, enc["input_ids"].shape[1]:].tolist()
        return {
            "output": tok.decode(new, skip_special_tokens=True), "new_tokens": len(new),
            "finish": "stop" if new and new[-1] in eos else "length",
            "prompt_tokens": int(enc["input_ids"].shape[1]), "seconds": time.perf_counter() - start,
        }

    return generate


def run_generation(
    convs: list[dict], generate: Callable[[list[dict]], dict], *, system: str, examples: list[dict], mode: str,
    sink: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Ask `generate` for the answer to every turn of every conversation.

    - `gold`: every turn is its own prompt, with the gold programs as the earlier answers; conversations in
      the order given, turns in order.
    - `pred`: one round per turn index. Round `k` asks every conversation that reaches turn `k`, in the order
      given, with the model's own answers to turns `0..k-1` as the earlier answers. An answer that is not a valid
      program stays in the history exactly as generated.

    Args:
        convs: The conversations of the split.
        generate: From a chat to a dict with at least `output` (see `make_generator`).
        system: The system message.
        examples: The example conversations shown first (none for zero-shot).
        mode: `gold` or `pred`.
        sink: Called with each row as soon as it exists (to write a file while the run goes on).

    Returns:
        One row per turn: `id`, `turn`, and what `generate` returned, in the order the prompts were run.
    """
    if mode not in ("gold", "pred"):
        raise ValueError(f"mode must be 'gold' or 'pred', not {mode!r}")
    rows = []

    def ask(conv, k, history):
        row = {"id": conv["id"], "turn": k, **generate(prompt_chat(conv, k, system, examples, history))}
        rows.append(row)
        if sink:
            sink(row)
        return row

    if mode == "gold":
        for conv in convs:
            for k in range(len(turns(conv))):
                ask(conv, k, None)
    else:
        history = {conv["id"]: [] for conv in convs}
        for k in range(max(len(turns(c)) for c in convs)):
            for conv in convs:
                if len(turns(conv)) > k:
                    history[conv["id"]].append(ask(conv, k, history[conv["id"]])["output"])
    return rows


def score_rows(rows: list[dict], turn_rows: list[dict]) -> list[dict]:
    """Execute and score every prediction against the gold turn.

    Execution accuracy is exact: the executed result, rounded to 5 decimals, equals the gold answer, and an
    output that does not run counts as wrong (`probing.classify_output` gives the kind of failure).

    Args:
        rows: Predictions, from `run_generation`.
        turn_rows: The gold turns of the split (`data.tables.turn_rows`).

    Returns:
        One dict per turn: `id`, `turn`, `correct`, `kind`, `type`, `part`, `number_turn`, `finish`, `new_tokens`.

    Raises:
        ValueError: If the predictions and the gold turns do not cover the same turns.
    """
    gold = {(r["id"], r["turn"]): r for r in turn_rows}
    predicted = {(r["id"], r["turn"]): r for r in rows}
    if len(predicted) != len(rows) or predicted.keys() != gold.keys():
        raise ValueError(
            f"predictions and gold turns differ: {len(gold.keys() - predicted.keys())} missing, "
            f"{len(predicted.keys() - gold.keys())} extra, {len(rows) - len(predicted)} repeated"
        )
    scores = []
    for key, g in gold.items():
        row = predicted[key]
        kind = classify_output(row["output"], g["program"], g["answer"])
        scores.append({
            "id": g["id"], "turn": g["turn"], "correct": kind == CORRECT, "kind": kind, "type": g["type"],
            "part": g["part"], "number_turn": g["number_turn"], "finish": row.get("finish"),
            "new_tokens": row.get("new_tokens"),
        })
    return scores


def summarize(scores: list[dict]) -> dict:
    """The headline numbers of a run: accuracy, invalid rate, and breakdowns.

    Args:
        scores: From `score_rows`.

    Returns:
        `n`, `accuracy`, `invalid_rate` (outputs that are not a program that runs), `truncated` (generation
        reached the token limit), `by_kind` (counts), and `by_turn`, `by_type`, `by_number_turn`, each a dict
        of `n` and `accuracy` per group (`by_turn` is keyed by the turn number from 1).
    """
    def accuracy(rows):
        return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows)}

    def grouped(key):
        groups = defaultdict(list)
        for r in scores:
            groups[key(r)].append(r)
        return {str(k): accuracy(v) for k, v in sorted(groups.items())}

    return {
        "n": len(scores),
        "accuracy": sum(r["correct"] for r in scores) / len(scores),
        "invalid_rate": sum(r["kind"] not in RUNS for r in scores) / len(scores),
        "truncated": sum(r["finish"] == "length" for r in scores),
        "by_kind": dict(Counter(r["kind"] for r in scores)),
        "by_turn": grouped(lambda r: r["turn"] + 1),
        "by_type": grouped(lambda r: r["type"]),
        "by_number_turn": grouped(lambda r: "number" if r["number_turn"] else "program"),
    }
