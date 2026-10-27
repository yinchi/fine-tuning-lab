"""One weight matrix with a LoRA adapter: the same update as a sum of matrices and as a computation.

Drawn for Qwen3-4B's `gate_proj` (2,560 -> 9,728) at rank 16. The thin matrices are drawn wider than
their true proportion (16 against 2,560 would be a line).
"""

from __future__ import annotations

from .svg import Box, Svg

FROZEN = ("#eeeeee", "#999999")
TRAINED = ("#d5e8d4", "#82b366")
PLAIN = ("#ffffff", "#444444")
GREY = "#555555"


def build() -> str:
    svg = Svg(900, 312)
    svg.text((20, 28), "As matrices (what merging produces)", size=13, bold=True, align="start")
    svg.text((500, 28), "As computation (what training runs)", size=13, bold=True, align="start")

    # Left: W' = W + (alpha / r) B A. Tall boxes, since the weight is (out, in) = (9,728, 2,560).
    top, height = 70, 150
    merged, frozen = Box(30, top, 60, height), Box(130, top, 60, height)
    b, a = Box(284, top, 16, height), Box(332, top + height / 2 - 8, 60, 16)
    svg.box(merged, *PLAIN)
    svg.text((merged.x + 30, top + height / 2 + 6), "W′", size=16, bold=True)
    svg.box(frozen, *FROZEN)
    svg.text((frozen.x + 30, top + height / 2 - 2), "W", size=16, bold=True)
    svg.text((frozen.x + 30, top + height / 2 + 16), "frozen", size=10.5, colour=GREY)
    svg.box(b, *TRAINED, radius=4)
    svg.text((b.x + 8, top - 8), "B", size=14, bold=True)
    svg.box(a, *TRAINED, radius=4)
    svg.text((a.x + 30, a.y - 8), "A", size=14, bold=True)
    for x, symbol in [(110, "="), (212, "+"), (316, "·")]:
        svg.text((x, top + height / 2 + 6), symbol, size=20)
    svg.text((252, top + height / 2 + 5), "(α/r)", size=12.5)

    for x, label in [(60, "9,728 × 2,560"), (160, "9,728 × 2,560"), (292, "9,728 × r"), (362, "r × 2,560")]:
        svg.text((x, 242), label, size=10.5)
    svg.text((60, 262), ["same shape", "as W"], size=10.5, colour=GREY, line_height=13)
    svg.text((160, 262), ["24.9 M weights,", "not trained"], size=10.5, colour=GREY, line_height=13)
    svg.text((335, 262), ["r = 16: 196,608", "weights, trained"], size=10.5, colour=GREY, line_height=13)

    # Right: y = W x + (alpha / r) B (A x). The activations pass through A and then B.
    fork = (535, 145)
    w, a2, b2 = Box(580, 85, 120, 40), Box(560, 185, 90, 40), Box(690, 185, 90, 40)
    total = (840, 145)
    svg.text((498, 150), "x", size=15, bold=True)
    svg.route((508, 145), "r27", arrow=False)
    svg.dot(fork)
    svg.box(w, *FROZEN)
    svg.text((w.x + 60, w.y + 18), "W", size=14, bold=True)
    svg.text((w.x + 60, w.y + 33), "frozen, 2,560 → 9,728", size=10, colour=GREY)
    for box, name, shape in [(a2, "A", "2,560 → r"), (b2, "B", "r → 9,728")]:
        svg.box(box, *TRAINED)
        svg.text((box.x + 45, box.y + 18), name, size=14, bold=True)
        svg.text((box.x + 45, box.y + 33), shape, size=10, colour=GREY)
    svg.circle(total, 11)
    svg.text((total[0], total[1] + 5), "+", size=16)
    svg.text((882, 150), "y", size=15, bold=True)

    svg.route(fork, "u40 r45")
    svg.route(fork, "d60 r25")
    svg.route(w.right, "r140 d29")
    svg.route(a2.right, "r40")
    svg.route(b2.right, "r60 u49")
    svg.route((total[0] + 11, total[1]), "r18", arrow=False)
    svg.text((670, 242), "r = 16 numbers", size=10, colour=GREY)
    svg.text((826, 224), "scaled by α/r", size=10.5, colour=GREY)

    svg.box(Box(548, 172, 244, 80), "none", "#888888", radius=10, dashed=True)
    svg.text((670, 276), "the 9,728 × 2,560 product B·A", size=11, bold=True)
    svg.text((670, 292), "is never built while training", size=11, bold=True)
    return svg.to_string()
