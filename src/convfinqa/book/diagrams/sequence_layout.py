"""One training sequence per conversation, against one per turn: what the model reads and what is trained."""

from __future__ import annotations

from .svg import Box, Svg

CONTEXT = ("#eeeeee", "#999999")
TRAINED = ("#d5e8d4", "#82b366")
GREY = "#555555"

# Widths in pixels, not to scale: the page is the long part.
SYSTEM, PAGE, QUESTION, ANSWER = 46, 240, 30, 26


def build() -> str:
    svg = Svg(780, 330)
    left, h = 190, 28

    def bar(y, parts, label, note):
        """A row of segments; `parts` is a list of (kind, text, width, trained)."""
        svg.text((left - 12, y + 18), label, size=11.5, bold=True, align="end")
        x = left
        for text, width, trained in parts:
            svg.box(Box(x, y, width, h), *(TRAINED if trained else CONTEXT), radius=4)
            svg.text((x + width / 2, y + 18), text, size=10)
            x += width
        svg.text((x + 12, y + 18), note, size=10.5, colour=GREY, align="start")

    def answer(k):
        return (f"A{k}", ANSWER, True)

    def turn(k):
        return [(f"Q{k}", QUESTION, False), (f"A{k}", ANSWER, True)]

    first = [("system", SYSTEM, False), ("page + Q1", PAGE, False)]
    svg.text((20, 24), "One sequence per conversation (what we train on)", size=13, bold=True, align="start")
    bar(44, first + [answer(1)] + turn(2) + turn(3), "one sequence", "")
    svg.text((left, 92), "1 sequence, 3 answers trained, the page read once", size=10.5, colour=GREY, align="start")

    svg.text((20, 124), "One sequence per turn (the alternative)", size=13, bold=True, align="start")
    one = first + [(f"A1", ANSWER, True)]
    two = first + [("A1", ANSWER, False)] + turn(2)
    three = first + [("A1", ANSWER, False)] + [("Q2", QUESTION, False), ("A2", ANSWER, False)] + turn(3)
    bar(144, one, "turn 1", "")
    bar(180, two, "turn 2", "")
    bar(216, three, "turn 3", "")
    svg.text((left + 345, 160), ["3 sequences, 3 answers trained,", "the page and the earlier answers read 3 times"], size=10.5, colour=GREY, align="start", line_height=13)

    svg.box(Box(20, 276, 16, 12), *TRAINED, radius=3)
    svg.text((44, 286), "tokens the loss counts (the assistant's answers, each ending with its end marker)", size=11, align="start")
    svg.box(Box(20, 298, 16, 12), *CONTEXT, radius=3)
    svg.text((44, 308), "tokens the model reads but is not trained to produce (system prompt, page, questions, earlier answers in the per-turn layout)", size=11, align="start")
    return svg.to_string()
