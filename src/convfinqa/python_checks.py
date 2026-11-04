"""The literal check and the scale check for Python output (see `literals`, `scale_check` and `output_checks`).

The checks of `output_checks` execute the DSL. These run the output as Python and compare its literals with the
translated gold program's. One thing differs from the DSL check: Python has no named constants, so a program that
needs `100` writes `100`, and the numbers a DSL constant stands for (1 to 9, the powers of ten up to 10^9, and -1)
count as written without being on the page.
"""

from __future__ import annotations

import re

from .data.render import context_text
from .dsl import python_format, steps
from .literals import literal_flags, literals, page_numbers, value
from .scale_check import percent_style, scale_kind

CONSTANT_VALUES = frozenset(value(c) for c in steps.CONSTANTS)
"""The values of the DSL's named constants."""

_PERCENT_ENDING = re.compile(r"multiply\((?:#\d+, const_100|const_100, #\d+)\)$")


def grounded(output: str, page: set[str]) -> bool:
    """Whether every number written in `output` is on the page or is the value of a DSL constant.

    A sign and a percent sign are not part of a number (as in `literals.literal_flags`).
    """
    return all(
        x.lstrip("-").rstrip("%") in page or abs(value(x)) in CONSTANT_VALUES or value(x) in CONSTANT_VALUES
        for x in literals(output)
    )


def check_python_rows(rows: list[dict], turn_rows: list[dict], convs: list[dict]) -> list[dict]:
    """Run both checks over the Python predictions of a run.

    Args:
        rows: The predictions (`id`, `turn`, `output`).
        turn_rows: The gold turns of the split (`id`, `turn`, `program`, `answer`, `question`); the programs are
            DSL, translated here.
        convs: The conversations of the split, for the numbers of each page.

    Returns:
        One dict per prediction: `id`, `turn`, `runs` (the output is Python that executes), `n_literals`,
        `lit_text` and `lit_value` (against the translated gold program), `lit_grounded` (`grounded`), `scale`
        (`scale_check.scale_kind` of the executed answer), `percent_style` and `gold_percent`, as in
        `output_checks.check_rows`.
    """
    page = {
        c["id"]: page_numbers(context_text(c) + " " + " ".join(c["annotation"]["dialogue_break"])) for c in convs
    }
    gold = {(t["id"], t["turn"]): t for t in turn_rows}
    checks = []
    for r in rows:
        t = gold[(r["id"], r["turn"])]
        answer, error = python_format.try_execute_python(r["output"])
        flags = literal_flags(r["output"], python_format.to_python(t["program"]), page[r["id"]])
        flags["lit_grounded"] = grounded(r["output"], page[r["id"]])
        checks.append({
            "id": r["id"], "turn": r["turn"], "runs": error is None, **flags,
            "scale": scale_kind(answer, t["answer"]),
            "percent_style": percent_style(t["question"]),
            "gold_percent": bool(_PERCENT_ENDING.search(t["program"])),
        })
    return checks
