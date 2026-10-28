"""Bootstrap intervals for accuracies, resampling whole conversations.

The turns of one conversation are not independent (they share a page and earlier answers), so the unit that is
resampled is the conversation. A comparison of two systems is *paired*: both are scored on the same resample.
"""

from __future__ import annotations

import numpy as np


def _by_conversation(scores: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Per conversation (sorted by id): the number of turns and of correct turns, and the turn keys."""
    per = {}
    for s in scores:
        n, k = per.get(s["id"], (0, 0))
        per[s["id"]] = (n + 1, k + bool(s["correct"]))
    ids = sorted(per)
    return ids, np.array([per[i][0] for i in ids]), np.array([per[i][1] for i in ids]), {(s["id"], s["turn"]) for s in scores}


def _resample(n_conv: int, n_boot: int, seed: int) -> np.ndarray:
    """Indices of the conversations in each resample: shape (n_boot, n_conv)."""
    return np.random.default_rng(seed).integers(0, n_conv, size=(n_boot, n_conv))


def bootstrap_accuracy(scores: list[dict], n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Accuracy with a percentile interval from resampling conversations.

    Args:
        scores: From `evaluate.score_rows` (each has `id`, `turn`, `correct`).
        n_boot: Number of resamples.
        seed: Seed of the resampling.
        level: Coverage of the interval.

    Returns:
        `accuracy`, `low`, `high`.
    """
    _, n, k, _ = _by_conversation(scores)
    idx = _resample(len(n), n_boot, seed)
    acc = k[idx].sum(1) / n[idx].sum(1)
    tail = (1 - level) / 2 * 100
    return {"accuracy": k.sum() / n.sum(), "low": float(np.percentile(acc, tail)), "high": float(np.percentile(acc, 100 - tail))}


def paired_difference(
    scores_a: list[dict], scores_b: list[dict], n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> dict:
    """Accuracy of A minus accuracy of B, with an interval and a p-value from the same resamples.

    Both systems must have been scored on exactly the same turns. The p-value is two-sided and approximate:
    twice the smaller share of resampled differences on either side of zero.

    Args:
        scores_a: Scores of the first system.
        scores_b: Scores of the second system.
        n_boot: Number of resamples.
        seed: Seed of the resampling.
        level: Coverage of the interval.

    Returns:
        `difference`, `low`, `high`, `p`, and `excludes_zero` (the interval does not contain 0).

    Raises:
        ValueError: If the two score lists do not cover the same turns.
    """
    ids_a, n_a, k_a, keys_a = _by_conversation(scores_a)
    ids_b, n_b, k_b, keys_b = _by_conversation(scores_b)
    if keys_a != keys_b:
        raise ValueError("the two systems were not scored on the same turns")
    idx = _resample(len(ids_a), n_boot, seed)
    diff = (k_a[idx].sum(1) - k_b[idx].sum(1)) / n_a[idx].sum(1)
    tail = (1 - level) / 2 * 100
    low, high = float(np.percentile(diff, tail)), float(np.percentile(diff, 100 - tail))
    p = min(1.0, 2 * min(float((diff <= 0).mean()), float((diff >= 0).mean())))
    return {
        "difference": float((k_a.sum() - k_b.sum()) / n_a.sum()), "low": low, "high": high, "p": p,
        "excludes_zero": low > 0 or high < 0,
    }


def turn_level_interval(scores: list[dict], n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """The interval of an accuracy if turns were resampled one by one (ignoring conversations).

    Only for comparison with `bootstrap_accuracy`: when outcomes within a conversation are correlated this
    interval is too narrow.
    """
    correct = np.array([bool(s["correct"]) for s in scores])
    idx = np.random.default_rng(seed).integers(0, len(correct), size=(n_boot, len(correct)))
    acc = correct[idx].mean(1)
    tail = (1 - level) / 2 * 100
    return {"accuracy": float(correct.mean()), "low": float(np.percentile(acc, tail)), "high": float(np.percentile(acc, 100 - tail))}
