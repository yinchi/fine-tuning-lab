"""Gold history: every turn is a separate prompt. Predicted history: one round per turn index."""

from __future__ import annotations

from .svg import Box, Svg

TURN = ("#fff2cc", "#d6b656")
GREY = "#555555"
LENGTHS = [4, 2, 3, 1]  # turns of four example conversations


def build() -> str:
    svg = Svg(900, 300)
    w, h, gap, row_gap = 56, 28, 16, 10

    def grid(x0, title):
        svg.text((x0, 26), title, size=13, bold=True, align="start")
        boxes = {}
        for r, n in enumerate(LENGTHS):
            y = 70 + r * (h + row_gap)
            svg.text((x0 + 56, y + 19), f"conv {r + 1}", size=11, colour=GREY, align="end")
            for k in range(n):
                box = Box(x0 + 70 + k * (w + gap), y, w, h)
                boxes[(r, k)] = box
                svg.box(box, *TURN, radius=5)
                svg.text((box.x + w / 2, y + 18), f"Q{k + 1}", size=11)
        return boxes

    # Left: gold history, every turn independent.
    left = grid(20, "Gold history: every turn is its own prompt")
    svg.text((20, 232), "The earlier answers in each prompt are the gold programs,", size=10.5, colour=GREY, align="start")
    svg.text((20, 248), "so all 10 prompts can be built and run in any order.", size=10.5, colour=GREY, align="start")
    svg.box(Box(88, 58, 4 * (w + gap) - gap + 24, 4 * (h + row_gap) + 6), "none", "#888888", radius=8, dashed=True)

    # Right: predicted history, one round per turn index; each answer feeds the next prompt.
    x0 = 480
    right = grid(x0, "Predicted history: one round per turn index")
    for k in range(4):
        col = [right[(r, k)] for r in range(len(LENGTHS)) if (r, k) in right]
        top = min(b.y for b in col) - 6
        bottom = max(b.y + b.h for b in col) + 6
        svg.box(Box(col[0].x - 6, top, w + 12, bottom - top), "none", "#6c8ebf", radius=8, dashed=True)
        svg.text((col[0].x + w / 2, 56), f"round {k + 1}", size=10.5, colour="#3b6ea5")
    for (r, k), box in right.items():
        if (r, k + 1) in right:
            svg.route(box.right, f"r{gap}")
    svg.text((x0, 232), "Round k asks every conversation that reaches turn k, with the model's own", size=10.5, colour=GREY, align="start")
    svg.text((x0, 248), "answers to the earlier turns in the prompt (an invalid one stays as written).", size=10.5, colour=GREY, align="start")
    svg.text((x0, 264), "Arrows: an answer enters the next prompt of its conversation.", size=10.5, colour=GREY, align="start")
    return svg.to_string()
