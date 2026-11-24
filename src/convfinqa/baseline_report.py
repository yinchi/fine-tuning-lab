"""Reading the baseline's validation runs: nested calls, the comparison table, and the choice of a prompt.

Kept apart from `evaluate` because the heavy artifacts import that module: changing it would make every run stale.
"""

from __future__ import annotations

import math

import numpy as np

from .evaluate import RUNS
from .prompts import parse_variant
from .stats import bootstrap_accuracy, paired_difference

KIND_ORDER = {"plain": 0, "description": 1, "flat": 2}


def has_nested_call(program: str) -> bool:
    """Whether an operation is written inside the arguments of another, as in `divide(subtract(5, 3), 3)`.

    The DSL has flat steps (`subtract(5, 3), divide(#0, 3)`), so a nested call does not run. The test is the depth
    of parentheses, and does not look at whether the rest of the text is a program.
    """
    depth = 0
    for ch in program:
        if ch == "(":
            depth += 1
            if depth > 1:
                return True
        elif ch == ")":
            depth = max(0, depth - 1)
    return False


def invalid_outputs(scores: list[dict]) -> list[dict]:
    """The scored turns whose output is not a program that runs."""
    return [s for s in scores if s["kind"] not in RUNS]


def nested_among_invalid(scores: list[dict], rows: list[dict]) -> int:
    """How many of the invalid outputs contain a nested call.

    Args:
        scores: From `evaluate.score_rows`.
        rows: The predictions the scores were made from (they hold the output text).
    """
    text = {(r["id"], r["turn"]): r["output"] for r in rows}
    return sum(has_nested_call(text[(s["id"], s["turn"])]) for s in invalid_outputs(scores))


def choose_variant(scores: dict[str, list[dict]], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Apply the rule for choosing a prompt, written down before any variant was run.

    Take the variant with the highest execution accuracy. Any variant whose accuracy the best one does not beat
    clearly (the paired bootstrap interval of the lead includes 0) is as good as the best: among those, take the
    one with the fewest examples, then the plain system message, then the description.

    Args:
        scores: Scored turns of every variant, keyed by variant name (`e3p`, ...), all on the same turns.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        `best` (highest accuracy), `tied` (the variants the best one does not clearly beat, the best included),
        `chosen`, and `leads` (the paired difference of the best over each other variant).
    """
    accuracy = {v: sum(s["correct"] for s in sc) / len(sc) for v, sc in scores.items()}
    best = max(accuracy, key=lambda v: (accuracy[v], v))
    leads = {v: paired_difference(scores[best], scores[v], n_boot, seed) for v in scores if v != best}
    tied = [best] + [v for v, d in leads.items() if not d["excludes_zero"]]
    key = lambda v: (parse_variant(v)[0], KIND_ORDER[parse_variant(v)[1]])
    return {"best": best, "tied": sorted(tied, key=key), "chosen": min(tied, key=key), "leads": leads}


def comparison_rows(scores: dict[str, list[dict]], predictions: dict[str, list[dict]]) -> list[dict]:
    """One row per variant: accuracy with its interval, invalid outputs, nested calls, and prompt length.

    Args:
        scores: Scored turns, keyed by variant name.
        predictions: The predictions of each variant (rows with `output`, `prompt_tokens`, `new_tokens`).
    """
    out = []
    for v, sc in scores.items():
        acc = bootstrap_accuracy(sc)
        invalid = invalid_outputs(sc)
        nested = nested_among_invalid(sc, predictions[v])
        out.append({
            "variant": v, "accuracy": acc["accuracy"], "low": acc["low"], "high": acc["high"],
            "invalid": len(invalid), "nested": nested, "nested_share": nested / len(invalid) if invalid else 0.0,
            "prompt_tokens": sum(r["prompt_tokens"] for r in predictions[v]) / len(predictions[v]),
        })
    return out


def turn_level_paired(scores_a: list[dict], scores_b: list[dict], z: float = 1.96) -> dict:
    """The paired comparison of two systems if every turn were an independent item (a paired t-test, normal approximation).

    This is the test `scipy.stats.ttest_rel` makes on the per-turn differences. It allows the two systems' results on a
    turn to be correlated, but treats different turns as independent, which turns of one conversation are not. It
    is here to show how much that assumption matters, not to be used: `stats.paired_difference` resamples pages.

    Args:
        scores_a: Scores of the first system.
        scores_b: Scores of the second system, on the same turns.
        z: The multiplier of the standard error for the interval (1.96 for 95%).

    Returns:
        `difference` (accuracy of A minus B), `half_width` of its interval, `low`, `high`, `excludes_zero` and `p`.

    Raises:
        ValueError: If the two score lists do not cover the same turns.
    """
    a = {(s["id"], s["turn"]): bool(s["correct"]) for s in scores_a}
    b = {(s["id"], s["turn"]): bool(s["correct"]) for s in scores_b}
    if a.keys() != b.keys():
        raise ValueError("the two systems were not scored on the same turns")
    d = np.array([a[k] - b[k] for k in sorted(a)], dtype=float)
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    return {
        "difference": mean, "half_width": z * se, "low": mean - z * se, "high": mean + z * se,
        "excludes_zero": abs(mean) > z * se, "p": math.erfc(abs(mean / se) / math.sqrt(2)),
    }
