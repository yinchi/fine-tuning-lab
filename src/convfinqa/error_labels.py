"""One label for every wrong turn: the first cause, in a fixed order, that explains it.

The order puts the causes that are cheapest to see first, so that a turn with two problems counts once, under the
one a reader would fix first: an output that is not a program, an error that only exists because of an earlier
mistake (the predicted history), a wrong scale, a negated answer (a dropped sign, or reversed operands of a
subtraction), a number that is not from the page, other numbers, a wrong operation, and the rest.
"""

from __future__ import annotations

import re

from .dsl import executor, steps
from .evaluate import RUNS
from .program_match import equal_program

INVALID = "not a program that runs"
INHERITED = "right with the gold history"
SCALE = "wrong scale"
SIGN = "sign dropped"
REVERSED = "operands reversed"
NOT_FROM_PAGE = "number not from the page"
NUMBERS = "other numbers"
OPERATION = "wrong operation"
OTHER = "other"

ORDER = (INVALID, INHERITED, SCALE, SIGN, REVERSED, NOT_FROM_PAGE, NUMBERS, OPERATION, OTHER)

_NEGATIVE_LITERAL = re.compile(r"(?<![\w.])-\d")


def _negated(output: str, gold_answer) -> bool:
    """Whether the output runs to exactly the negative of the gold answer."""
    value, error = executor.try_execute(output)
    return (
        error is None and isinstance(gold_answer, (int, float)) and isinstance(value, float)
        and gold_answer != 0 and executor.answers_match(value, -gold_answer)
    )


def label_turn(
    kind: str, check: dict, output: str, gold_program: str, gold_answer, right_with_gold_history: bool | None = None
) -> str:
    """The label of one wrong turn.

    Args:
        kind: The outcome of the output (`probing.classify_output`, in the scores).
        check: The turn's row of `output_checks.check_rows` (`scale`, `lit_grounded`).
        output: The model's text.
        gold_program: The gold program.
        gold_answer: The gold answer.
        right_with_gold_history: Whether the same model answered this turn correctly when the earlier answers were
            the gold ones; `None` if that is not known (the run has a gold history already).

    Returns:
        One of `ORDER`.
    """
    if kind not in RUNS:
        return INVALID
    if right_with_gold_history:
        return INHERITED
    if check["scale"] is not None:
        return SCALE
    if _negated(output, gold_answer):
        if _NEGATIVE_LITERAL.search(gold_program):
            return SIGN
        if equal_program(output, gold_program, order_free=True):
            return REVERSED
    if not check["lit_grounded"]:
        return NOT_FROM_PAGE
    if kind == "wrong_numbers" or (steps.is_number_turn(gold_program) and steps.is_number_turn(output)):
        return NUMBERS   # the same operations with other numbers, or another number where a number was asked for
    return OPERATION if kind == "wrong_operations" else OTHER


def label_turns(
    scores: list[dict], checks: list[dict], rows: list[dict], turn_rows: list[dict],
    gold_history_scores: list[dict] | None = None,
) -> dict[tuple[str, int], str]:
    """The label of every wrong turn of a run, keyed by `(id, turn)`.

    Args:
        scores: From `evaluate.score_rows`.
        checks: From `output_checks.check_rows`.
        rows: The predictions.
        turn_rows: The gold turns.
        gold_history_scores: The scores of the same model with the gold history, if this run has a predicted one.

    Returns:
        A dict with one entry per turn the run got wrong.
    """
    check = {(c["id"], c["turn"]): c for c in checks}
    output = {(r["id"], r["turn"]): r["output"] for r in rows}
    gold = {(t["id"], t["turn"]): t for t in turn_rows}
    with_gold = {(s["id"], s["turn"]): s["correct"] for s in gold_history_scores or []}
    labels = {}
    for s in scores:
        key = (s["id"], s["turn"])
        if not s["correct"]:
            labels[key] = label_turn(
                s["kind"], check[key], output[key], gold[key]["program"], gold[key]["answer"], with_gold.get(key)
            )
    return labels
