"""Which tokens count in the loss: each position is scored on the *next* token, and only answer tokens count."""

from __future__ import annotations

from .svg import Box, Svg

GREY_BOX = ("#eeeeee", "#999999")
GREEN = ("#d5e8d4", "#82b366")
RED = ("#f8cecc", "#b85450")
GREY = "#555555"


def build() -> str:
    svg = Svg(900, 300)
    tokens = ["…", "<|im_start|>", "assistant", "\\n", "1", "8", "9", "<|im_end|>"]
    answer = {4, 5, 6, 7}  # the tokens kept in `labels`
    width, gap, left, h = 90, 6, 110, 40
    rows = {"reads": 70, "scored": 150, "counts": 230}

    for name, label in [("reads", ["input_ids:", "the token read"]), ("scored", ["the token it", "must predict"]), ("counts", ["in the loss?", ""])]:
        svg.text((left - 14, rows[name] + 18), label, size=11.5, colour=GREY, align="end", line_height=14)

    for i, token in enumerate(tokens):
        x = left + i * (width + gap)
        cx = x + width / 2
        target = tokens[i + 1] if i + 1 < len(tokens) else "(nothing)"
        counted = i + 1 in answer

        svg.box(Box(x, rows["reads"], width, h), *(GREEN if i in answer else GREY_BOX))
        svg.text((cx, rows["reads"] + 25), token, size=12)

        svg.box(Box(x, rows["scored"], width, h), *(GREEN if counted else GREY_BOX), dashed=i + 1 >= len(tokens))
        svg.text((cx, rows["scored"] + 25), target, size=12)
        svg.route((cx, rows["reads"] + h), f"d{rows['scored'] - rows['reads'] - h:g}")

        svg.box(Box(x, rows["counts"], width, 34), *(GREEN if counted else RED), radius=17)
        svg.text((cx, rows["counts"] + 22), "yes" if counted else "no", size=12, bold=True)

    svg.text(
        (20, 30),
        "The model reads the first row. At each position it predicts the next token; only answer tokens are held against it.",
        size=12, align="start",
    )
    svg.text(
        (20, 284),
        "In `labels` the green tokens keep their ids and every other position is -100; the model shifts the labels by one itself.",
        size=11, colour=GREY, align="start",
    )
    return svg.to_string()
