"""The baseline's prompt, and the run that uses one example set."""

from __future__ import annotations

from .svg import Box, Svg

CONTEXT = ("#eeeeee", "#999999")
EXAMPLE = ("#dae8fc", "#6c8ebf")
TARGET = ("#fff2cc", "#d6b656")
ANSWER = ("#d5e8d4", "#82b366")
GREY = "#555555"


def build() -> str:
    svg = Svg(900, 290)
    svg.text((20, 26), "One prompt: turn k of a target conversation", size=13, bold=True, align="start")

    y, h = 66, 56
    parts = [
        (Box(20, y, 90, h), CONTEXT, ["system", "message"], False),
        (Box(116, y, 112, h), EXAMPLE, ["example", "conversation A"], False),
        (Box(234, y, 112, h), EXAMPLE, ["example", "conversation B"], False),
        (Box(352, y, 128, h), TARGET, ["target conversation", "Q1 A1 … Qk"], False),
        (Box(486, y, 80, h), ANSWER, ["the model", "writes Ak"], True),
    ]
    for box, style, lines, dashed in parts:
        svg.box(box, *style, radius=6, dashed=dashed)
        svg.text((box.x + box.w / 2, y + 24), lines, size=11, line_height=14)

    # What varies, above the bars.
    svg.text((65, y - 12), "one line, or + DSL", size=10, colour=GREY)
    svg.text((231, y - 12), "0, 2 or 3 train-fit conversations: one set", size=10, colour=GREY)
    # Brackets below: what is fixed and what changes.
    svg.route((20, y + h + 10), "d8 r326 u8", arrow=False)
    svg.text((183, y + h + 38), "fixed: identical in every prompt of a run", size=10.5, colour=GREY)
    svg.route((352, y + h + 10), "d8 r214 u8", arrow=False)
    svg.text((459, y + h + 38), "changes with every turn", size=10.5, colour=GREY)

    svg.text((20, 186), [
        "The system message is a single line, or the same line plus a compact description of the DSL and the rule that",
        "an answer is a program and nothing else. With no examples the prompt has no blue part.",
    ], size=10.5, colour=GREY, align="start", line_height=14)

    # Right: one run.
    svg.text((620, 26), "The baseline: one run", size=13, bold=True, align="start")
    for i, name in enumerate(("set 1",)):
        ry = 56 + i * 52
        svg.box(Box(620, ry, 70, 36), *EXAMPLE, radius=6)
        svg.text((655, ry + 22), name, size=11.5, bold=True)
        svg.route((690, ry + 18), "r26")
        svg.box(Box(720, ry, 110, 36), *CONTEXT, radius=6)
        svg.text((775, ry + 15), "pass over every", size=10)
        svg.text((775, ry + 28), "turn of the split", size=10)
        svg.route((830, ry + 18), "r12")
        svg.text((848, ry + 22), "accuracy", size=10.5, align="start")
    svg.text((620, 126), "Report the result with its interval.", size=11, bold=True, align="start")

    svg.text((20, 258), "A set is fixed by its seed, so the same seed gives the same examples whichever split is evaluated.", size=10.5, colour=GREY, align="start")
    svg.text((20, 274), "Blue: train-fit conversations (never val or dev). Yellow: the conversation being evaluated.", size=10.5, colour=GREY, align="start")
    return svg.to_string()
