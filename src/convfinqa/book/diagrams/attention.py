"""Draws one causal self-attention block of Qwen3, with the shape of every tensor."""

from __future__ import annotations

from .svg import Box, Svg

# Color codes for different types of boxes in the diagram.
# Pairs represent (fill color, border color) for the boxes in the diagram.

WEIGHTS = ("#ffffff", "#444444")  # a learned matrix
NORM = ("#d5e8d4", "#82b366")  # RMSNorm
PLAIN = ("#f1f1f1", "#888888")  # an operation with no learned weights
MASK = ("#f8cecc", "#b85450")  # the causal mask and softmax
GREY = "#555555"


def _label(svg: Svg, box: Box, title: str, note: str | None = None, bold: bool = False) -> None:
    """Label a box in the SVG diagram.

    Args:
        svg: The SVG object to draw on.
        box: The box to label.
        title: The main title of the label.
        note: An optional note to display below the title.
        bold: Whether to render the title in bold.
    """
    cx = box.x + box.w / 2
    if note is None:
        # Place the title in the center of the box if there is no note, offset slightly for
        # visual alignment.
        svg.text((cx, box.y + box.h / 2 + 4.5), title, size=12.5, bold=bold)
    else:
        # Place the title near the top of the box and the note below it.
        # The constants 17 and 32 are specific to the visual layout and box contents for
        # this diagram.
        svg.text((cx, box.y + 17), title, size=12.5, bold=bold)
        svg.text((cx, box.y + 32), note, size=10, colour=GREY)


def build() -> str:
    svg = Svg(590, 975)
    q_x, k_x, v_x = 215, 345, 495  # the three columns
    mid = (q_x + k_x) / 2  # 280

    # Define the box data structures for each component in the attention block.
    wq, wk, wv = (Box(x - 52, 815, 104, 42) for x in (q_x, k_x, v_x))
    rq, rk, rv = (Box(x - 52, 740, 104, 34) for x in (q_x, k_x, v_x))  # reshape into heads
    norm = Box(150, 650, 260, 40)
    rope = Box(150, 570, 260, 40)
    positions = Box(8, 570, 110, 40)
    repeat = Box(285, 490, 270, 40)
    scores = Box(150, 410, 260, 40)
    softmax = Box(mid - 105, 330, 210, 40)
    weighted = Box(215, 250, 340, 40)
    concat = Box(295, 170, 180, 36)
    out_proj = Box(295, 95, 180, 40)
    out_x = concat.x + concat.w / 2  # 385

    # For each defined box, draw it with the appropriate style.
    for box, style in [
        (wq, WEIGHTS), (wk, WEIGHTS), (wv, WEIGHTS), (rq, PLAIN), (rk, PLAIN), (rv, PLAIN),
        (norm, NORM), (rope, PLAIN), (positions, PLAIN), (repeat, PLAIN), (scores, PLAIN),
        (softmax, MASK), (weighted, PLAIN), (concat, PLAIN), (out_proj, WEIGHTS)
    ]:
        svg.box(box, *style)

    # Label each box with its corresponding description.
    _label(svg, wq, "W_Q", "matrix multiplication")
    _label(svg, wk, "W_K", "matrix multiplication")
    _label(svg, wv, "W_V", "matrix multiplication")
    for box in (rq, rk, rv):
        _label(svg, box, "split into heads")
    _label(svg, norm, "RMSNorm on every query and key vector")
    _label(svg, rope, "rotate by position (RoPE)")
    _label(svg, positions, "token positions")
    _label(svg, repeat, "repeat each key/value head n_Q / n_KV times")
    _label(svg, scores, "scores = Q·Kᵀ / √d")
    _label(svg, softmax, "add causal mask M, softmax")
    _label(svg, weighted, "weights × V")
    _label(svg, concat, "concatenate the heads")
    _label(svg, out_proj, "W_O", "back to hidden size")

    # Input and the three projections.
    split_y = wq.y + wq.h + 59  # where the input forks
    svg.text((k_x, split_y + 39), "input X  (b, s, h)")
    svg.route((k_x, split_y + 25), "u25")
    svg.dot((k_x, split_y))
    svg.route((k_x, split_y), f"l{k_x - q_x:g} u{split_y - wq.y - wq.h:g}")
    svg.route((k_x, split_y), f"u{split_y - wk.y - wk.h:g}")
    svg.route((k_x, split_y), f"r{v_x - k_x:g} u{split_y - wv.y - wv.h:g}")

    # Matrix multiplication -> reshape into heads.
    for w, r_ in [(wq, rq), (wk, rk), (wv, rv)]:
        svg.route(w.top, f"u{w.y - r_.y - r_.h:g}")

    # Up the three columns.
    svg.route(rq.top, f"u{rq.y - norm.y - norm.h:g}")
    svg.route(rk.top, f"u{rk.y - norm.y - norm.h:g}")
    svg.route(rv.top, f"u{rv.y - repeat.y - repeat.h:g}")  # values skip the norm and the rotation
    svg.route((q_x, norm.y), f"u{norm.y - rope.y - rope.h:g}")
    svg.route((k_x, norm.y), f"u{norm.y - rope.y - rope.h:g}")
    svg.route((q_x, rope.y), f"u{rope.y - scores.y - scores.h:g}")  # queries pass beside the repeat box
    svg.route((k_x, rope.y), f"u{rope.y - repeat.y - repeat.h:g}")
    svg.route((k_x, repeat.y), f"u{repeat.y - scores.y - scores.h:g}")
    svg.route((v_x, repeat.y), f"u{repeat.y - weighted.y - weighted.h:g}")
    svg.route((mid, scores.y), f"u{scores.y - softmax.y - softmax.h:g}")
    svg.route((mid, softmax.y), f"u{softmax.y - weighted.y - weighted.h:g}")
    svg.route((out_x, weighted.y), f"u{weighted.y - concat.y - concat.h:g}")
    svg.route((out_x, concat.y), f"u{concat.y - out_proj.y - out_proj.h:g}")
    svg.route((out_x, out_proj.y), f"u{out_proj.y - 48:g}")
    svg.text((out_x, 36), "output  (b, s, h)")
    svg.route(positions.right, f"r{rope.x - positions.x - positions.w:g}")

    # Tensor shapes beside the arrows.
    shape = lambda x, y, s: svg.text((x, y), s, size=10, colour=GREY, align="start")
    shape(q_x + 6, 802, "(b, s, n_Q·d)")
    shape(k_x + 6, 802, "(b, s, n_KV·d)")
    shape(v_x + 6, 802, "(b, s, n_KV·d)")
    shape(q_x + 6, 722, "(b, n_Q, s, d)")
    shape(k_x + 6, 722, "(b, n_KV, s, d)")
    shape(v_x + 6, 722, "(b, n_KV, s, d)")
    shape(k_x + 6, 470, "(b, n_Q, s, d)")
    shape(v_x + 6, 470, "(b, n_Q, s, d)")
    shape(mid + 6, 394, "(b, n_Q, s, s)")
    shape(mid + 6, 314, "(b, n_Q, s, s)")
    shape(out_x + 6, 232, "(b, n_Q, s, d)")
    shape(out_x + 6, 156, "(b, s, n_Q·d)")

    # Legend and notation.
    for i, (label, (fill, stroke)) in enumerate([
        ("learned matrix", WEIGHTS), ("RMSNorm", NORM),
        ("no learned weights", PLAIN), ("causal mask, softmax", MASK)
    ]):
        y = 14 + i * 20
        svg.box(Box(8, y, 16, 12), fill, stroke, radius=3)
        svg.text((30, y + 10), label, size=10.5, align="start", colour=GREY)

    svg.text(
        (8, 124),
        [
            "b: batch · s: sequence length",
            "h: hidden_size · d: head_dim",
            "n_Q: query heads · n_KV: key/value heads"
        ],
        size=10.5, align="start", colour=GREY, line_height=14
    )
    return svg.to_string()
