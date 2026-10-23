"""Inside the feed-forward block (SwiGLU) of Qwen3, with the shape of every tensor."""

from __future__ import annotations

from .svg import Box, Svg

WEIGHTS = ("#ffffff", "#444444")
PLAIN = ("#f1f1f1", "#888888")
GREY = "#555555"


def build() -> str:
    svg = Svg(440, 560)
    cx, gate_x, up_x = 220, 130, 310

    down = Box(cx - 90, 75, 180, 42)
    silu = Box(gate_x - 50, 215, 100, 36)
    w_gate = Box(gate_x - 52, 300, 104, 42)
    w_up = Box(up_x - 52, 300, 104, 42)
    product = (cx, 165)
    r = 13

    for box, style in [(down, WEIGHTS), (silu, PLAIN), (w_gate, WEIGHTS), (w_up, WEIGHTS)]:
        svg.box(box, *style)
    svg.circle(product, r)
    svg.text((product[0], product[1] + 5), "×", size=17)

    svg.text((cx, down.y + 18), "W_down", size=12.5)
    svg.text((cx, down.y + 33), "back to hidden size", size=10, colour=GREY)
    svg.text((gate_x, silu.y + 22), "SiLU", size=12.5)
    svg.text((gate_x, w_gate.y + 18), "W_gate", size=12.5)
    svg.text((gate_x, w_gate.y + 33), "widen: h → i", size=10, colour=GREY)
    svg.text((up_x, w_up.y + 18), "W_up", size=12.5)
    svg.text((up_x, w_up.y + 33), "widen: h → i", size=10, colour=GREY)

    # Input, the fork, and the two branches.
    svg.text((cx, 500), "input x  (b, s, h)")
    svg.route((cx, 486), "u25")
    fork = (cx, 461)
    svg.dot(fork)
    svg.route(fork, f"l{cx - gate_x} u{fork[1] - (w_gate.y + w_gate.h):g}")
    svg.route(fork, f"r{up_x - cx} u{fork[1] - (w_up.y + w_up.h):g}")
    svg.route(w_gate.top, f"u{w_gate.y - silu.y - silu.h:g}")
    svg.route(silu.top, f"u{silu.y - product[1]:g} r{cx - r - gate_x:g}")  # up, then right into the product
    svg.route(w_up.top, f"u{w_up.y - product[1]:g} l{up_x - cx - r:g}")
    svg.route((cx, product[1] - r), f"u{product[1] - r - down.y - down.h:g}")
    svg.route((cx, down.y), f"u{down.y - 50:g}")
    svg.text((cx, 36), "output  (b, s, h)")

    shape = lambda x, y, s: svg.text((x, y), s, size=10, colour=GREY, align="start")
    shape(gate_x + 6, 282, "(b, s, i)")
    shape(gate_x + 6, 203, "(b, s, i)")
    shape(up_x + 6, 282, "(b, s, i)")
    shape(cx + 6, 138, "(b, s, i)")
    svg.text((cx, 542), "b: batch · s: sequence length · h: hidden_size · i: intermediate_size", size=10, colour=GREY)
    return svg.to_string()
