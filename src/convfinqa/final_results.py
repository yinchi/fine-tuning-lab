"""The numbers of the final-results chapter: the share of a gap that a prompt closes, and val against dev.

Kept apart from `stats` because the chapters of the earlier parts import that module, and from `evaluate` because the
heavy artifacts import that one: changing either would make results stale. Both functions resample whole
conversations, as `stats` does.
"""

from __future__ import annotations

import numpy as np

from .evaluate import RUNS
from .stats import _by_conversation, _resample


PUBLISHED = (
    {"label": "Crowd workers", "accuracy": 46.9, "split": "200 questions", "key": "chen2022convfinqa", "human": True},
    {"label": "GPT-3, few-shot", "accuracy": 45.2, "split": "test, gold facts", "key": "chen2022convfinqa"},
    {"label": "FinQANet", "accuracy": 68.9, "split": "split not stated", "key": "chen2022convfinqa"},
    {"label": "FinQANet, gold facts", "accuracy": 77.3, "split": "split not stated", "key": "chen2022convfinqa"},
    {"label": "Human experts", "accuracy": 89.4, "split": "200 questions", "key": "chen2022convfinqa", "human": True},
    {"label": "GPT-4, zero-shot", "accuracy": 76.5, "split": "dev", "key": "li2023gpt"},
    {"label": "PoT-GPT-4", "accuracy": 74.8, "split": "dev", "key": "khatuya2025finder"},
    {"label": "ENCORE", "accuracy": 76.0, "split": "dev", "key": "khatuya2025finder"},
    {"label": "FINDER", "accuracy": 82.0, "split": "dev", "key": "khatuya2025finder"},
)
"""Published execution accuracies on ConvFinQA, in percent, checked against the papers (2026-11-23).

`split` is what the paper says about the evaluation set: the ConvFinQA paper does not say which split its Table 3 uses
(FinQANet, the crowd and expert rows), reports its few-shot GPT-3 on the test set with the gold facts as input
(Table 5), and gives its human rows on a sample of 200 questions. The GPT-4 figure is from Li et al. (2023), Table 8,
and the other three from Khatuya et al. (2025), Table 2, both on the dev set. Fin-R1 is left out: its accuracy is
judged by a language model, not computed from an executed program, so it does not measure the same thing.
"""


def outcome_counts(summary: dict) -> dict[str, int]:
    """The turns of a scored run by outcome: right, not a program that runs, or one of three kinds of wrong.

    Args:
        summary: The `summary.json` of a scores artifact (its `by_kind` counts).

    Returns:
        Counts that add up to the number of turns.
    """
    counts = summary["by_kind"]
    out = {
        "correct": counts["correct"],
        "invalid output": sum(v for k, v in counts.items() if k not in RUNS),
        "wrong operations": counts.get("wrong_operations", 0),
        "wrong numbers": counts.get("wrong_numbers", 0),
        "wrong answer, right shape": counts.get("wrong_answer", 0),
    }
    if sum(out.values()) != summary["n"]:
        raise ValueError("the outcomes do not add up to the number of turns")
    return out


def _percentile_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def share_of_gap_closed(
    low_scores: list[dict],
    mid_scores: list[dict],
    high_scores: list[dict],
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """The share of the distance from a floor to a ceiling that a middle system covers.

    With `low` the untuned model with the training prompt, `high` the fine-tuned model and `mid` the untuned model
    with the best prompt, the share is `(mid - low) / (high - low)`: how much of what fine-tuning adds the prompt
    already gets. All three are scored on the same turns, and the three are resampled together.

    Args:
        low_scores: Scores of the floor.
        mid_scores: Scores of the system in between.
        high_scores: Scores of the ceiling.
        n_boot: Number of resamples.
        seed: Seed of the resampling.
        level: Coverage of the interval.

    Returns:
        `share`, `low`, `high` (the interval), and `gap` (the accuracy of the ceiling minus the floor).

    Raises:
        ValueError: If the score lists do not cover the same turns, or the ceiling is not above the floor.
    """
    parts = [_by_conversation(s) for s in (low_scores, mid_scores, high_scores)]
    if not parts[0][3] == parts[1][3] == parts[2][3]:
        raise ValueError("the systems were not scored on the same turns")
    (_, n, k_low, _), (_, _, k_mid, _), (_, _, k_high, _) = parts
    if k_high.sum() <= k_low.sum():
        raise ValueError("the ceiling must score above the floor")
    idx = _resample(len(n), n_boot, seed)
    gap = k_high[idx].sum(1) - k_low[idx].sum(1)
    covered = k_mid[idx].sum(1) - k_low[idx].sum(1)
    # a resample in which the ceiling does not beat the floor has no share; with a real gap this does not occur
    ok = gap > 0
    share = covered[ok] / gap[ok]
    low, high = _percentile_interval(share, level)
    return {
        "share": float((k_mid.sum() - k_low.sum()) / (k_high.sum() - k_low.sum())),
        "low": low,
        "high": high,
        "gap": float((k_high.sum() - k_low.sum()) / n.sum()),
        "resamples_used": int(ok.sum()),
    }


def split_difference(
    scores_a: list[dict], scores_b: list[dict], n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> dict:
    """Accuracy on one split minus accuracy on another, for the same system, with an interval.

    The two splits hold different conversations, so the comparison is *unpaired*: each is resampled on its own.

    Args:
        scores_a: Scores on the first split.
        scores_b: Scores on the second split.
        n_boot: Number of resamples.
        seed: Seed of the resampling (the second split uses `seed + 1`).
        level: Coverage of the interval.

    Returns:
        `difference`, `low`, `high`, and `excludes_zero` (the interval does not contain 0).
    """
    _, n_a, k_a, _ = _by_conversation(scores_a)
    _, n_b, k_b, _ = _by_conversation(scores_b)
    ia = _resample(len(n_a), n_boot, seed)
    ib = _resample(len(n_b), n_boot, seed + 1)
    diff = k_a[ia].sum(1) / n_a[ia].sum(1) - k_b[ib].sum(1) / n_b[ib].sum(1)
    low, high = _percentile_interval(diff, level)
    return {
        "difference": float(k_a.sum() / n_a.sum() - k_b.sum() / n_b.sum()),
        "low": low,
        "high": high,
        "excludes_zero": low > 0 or high < 0,
    }
