"""The wrong-scale check: an answer that is the gold answer times a power of ten.

It works on the executed answers, so no extra generation is needed and it applies to any output format that can
be run. A factor of exactly 100 (a ratio where a percent was asked for, or the reverse) is reported on its own;
the other powers of ten are unit scaling (thousands, millions, billions).
"""

from __future__ import annotations

PERCENT = "factor_100"
OTHER_POWER = "other_power"
POWERS = tuple(range(1, 10))
"""The exponents of the factors looked for, in both directions."""
MIN_MAGNITUDE = 1e-3
"""Answers are rounded to 5 decimals, so a value much smaller than this says nothing about the scale."""


def scale_kind(predicted, gold) -> str | None:
    """Whether the predicted answer is the gold answer times (or over) a power of ten.

    The gold answers are rounded to 5 decimals, and a factor `f` magnifies that rounding, so the comparison allows
    `0.5e-5 * (1 + f)`, and answers below `MIN_MAGNITUDE` are not compared.

    Args:
        predicted: The executed answer of the output (a number, `yes`/`no`, or `None` if it did not run).
        gold: The gold answer.

    Returns:
        `factor_100`, `other_power` (any other power of ten, up to 10^9), or `None` if the answers are not a power of ten apart (or either is not a
        number, or the gold is 0).
    """
    if not isinstance(predicted, (int, float)) or not isinstance(gold, (int, float)) or gold == 0:
        return None
    for power in POWERS:
        for factor in (10.0**power, 10.0**-power):
            scaled = gold * factor
            if min(abs(predicted), abs(scaled)) >= MIN_MAGNITUDE and abs(predicted - scaled) <= 0.5e-5 * (1 + factor):
                return PERCENT if power == 2 else OTHER_POWER
    return None


def percent_style(question: str) -> bool:
    """Whether a question asks for a percent, as a proportion or a percentage."""
    return "percent" in question.lower()
