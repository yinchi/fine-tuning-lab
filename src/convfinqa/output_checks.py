"""The literal check and the scale check of every turn of a run (see `literals` and `scale_check`)."""

from __future__ import annotations

import re

from .data.render import context_text
from .dsl import executor
from .literals import literal_flags, page_numbers
from .scale_check import percent_style, scale_kind

_PERCENT_ENDING = re.compile(r"multiply\((?:#\d+, const_100|const_100, #\d+)\)$")


def check_rows(rows: list[dict], turn_rows: list[dict], convs: list[dict]) -> list[dict]:
    """Run both checks over a run's predictions.

    Args:
        rows: The predictions (`id`, `turn`, `output`).
        turn_rows: The gold turns of the split (`id`, `turn`, `program`, `answer`, `question`).
        convs: The conversations of the split, for the numbers of each page.

    Returns:
        One dict per prediction: `id`, `turn`, `runs` (the output is a program that executes), the measures of
        `literals.literal_flags`, `scale` (`scale_check.scale_kind`), `percent_style` (the question asks for a
        percent) and `gold_percent` (the gold program ends by multiplying by `const_100`, in either order of the arguments).
    """
    page = {
        c["id"]: page_numbers(context_text(c) + " " + " ".join(c["annotation"]["dialogue_break"])) for c in convs
    }
    gold = {(t["id"], t["turn"]): t for t in turn_rows}
    checks = []
    for r in rows:
        t = gold[(r["id"], r["turn"])]
        answer, error = executor.try_execute(r["output"])
        checks.append({
            "id": r["id"], "turn": r["turn"], "runs": error is None,
            **literal_flags(r["output"], t["program"], page[r["id"]]),
            "scale": scale_kind(answer, t["answer"]),
            "percent_style": percent_style(t["question"]),
            "gold_percent": bool(_PERCENT_ENDING.search(t["program"])),
        })
    return checks
