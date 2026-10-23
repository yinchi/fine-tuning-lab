"""Draws the decoder-only stack of Qwen3-4B.

Includes embeddings, 36 identical layers, a final norm, and the output head.
"""

from __future__ import annotations

from .svg import Box, Svg

# Color codes for different types of boxes in the diagram.
# Pairs represent (fill color, border color) for the boxes in the diagram.

NORM = ("#d5e8d4", "#82b366")  # green
ATTENTION = ("#f8cecc", "#b85450")  # red: the self-attention is causal ("masked")
MLP = ("#fff2cc", "#d6b656")  # yellow
PLAIN = ("#ffffff", "#444444")


def build() -> str:
    svg = Svg(500, 790)
    cx = 240

    # Boxes, from the top of the page to the bottom.
    linear = Box(cx - 80, 100, 160, 40)
    final_norm = Box(cx - 60, 165, 120, 30)
    mlp = Box(cx - 100, 300, 200, 60)
    norm2 = Box(cx - 60, 385, 120, 30)
    attn = Box(cx - 100, 505, 200, 62)
    norm1 = Box(cx - 60, 595, 120, 30)
    emb = Box(cx - 90, 690, 180, 44)
    layer = Box(cx - 125, 232, 275, 446)
    positions = Box(8, attn.y + attn.h / 2 - 26, 96, 52)  # level with the attention block
    add2, add1, r = (cx, 265), (cx, 475), 11
    split1, split2 = (cx, 655), (cx, 440)  # where the residual stream forks around a block

    svg.box(layer, "none", "#888888", radius=16, dashed=True)
    svg.text((layer.x - 8, 452), ["36 ×", "layers"], align="end", size=14, bold=True)

    for box, (fill, stroke) in [(linear, PLAIN), (final_norm, NORM), (mlp, MLP), (norm2, NORM),
                                (attn, ATTENTION), (norm1, NORM), (emb, PLAIN)]:
        svg.box(box, fill, stroke)
    svg.text((cx, final_norm.y + 20), "RMSNorm")
    svg.text((cx, norm2.y + 20), "RMSNorm")
    svg.text((cx, norm1.y + 20), "RMSNorm")
    svg.text((cx, linear.y + 17), "Linear (output head)", size=13)
    svg.text((cx, linear.y + 33), "tied to the embeddings", size=10.5, colour="#555555")
    svg.text((cx, emb.y + 19), "Embeddings")
    svg.text((cx, emb.y + 35), "one learned vector per token", size=10.5, colour="#555555")
    svg.text((cx, attn.y + 20), "Causal self-attention", bold=True)
    svg.text((cx, attn.y + 37), "32 query heads, 8 key/value heads", size=10.5, colour="#555555")
    svg.text(
        (cx, attn.y + 51), "rotary positions on queries and keys", size=10.5, colour="#555555"
    )
    svg.text((cx, mlp.y + 22), "Feed-forward (SwiGLU)", bold=True)
    svg.text((cx, mlp.y + 40), "2,560 → 9,728 → 2,560", size=10.5, colour="#555555")

    svg.box(positions, "#ffffff", "#444444")
    svg.text(
        (positions.x + positions.w / 2, positions.y + 21),
        "token positions",
        size=11.5
    )
    svg.text(
        (positions.x + positions.w / 2, positions.y + 37),
        "rotary cos, sin",
        size=10.5, colour="#555555"
    )
    svg.text(
        (positions.x + positions.w / 2, positions.y + positions.h + 15),
        "shared by all layers",
        size=10, colour="#555555"
    )
    svg.route(positions.right, f"r{attn.x - positions.x - positions.w:g}")

    for centre in (add1, add2):
        svg.circle(centre, r)
        svg.text((centre[0], centre[1] + 5), "+", size=16)

    # The main path, bottom to top.
    svg.text((cx, 770), "token ids")
    svg.route((cx, 758), "u24")
    svg.route(emb.top, f"u{emb.y - norm1.y - norm1.h:g}")  # the dot at the first fork is drawn on top of it
    svg.route(norm1.top, f"u{norm1.y - attn.y - attn.h:g}")
    svg.route(attn.top, f"u{attn.y - add1[1] - r:g}")
    svg.route((cx, add1[1] - r), f"u{add1[1] - r - norm2.y - norm2.h:g}")  # fork dot drawn on top
    svg.route(norm2.top, f"u{norm2.y - mlp.y - mlp.h:g}")
    svg.route(mlp.top, f"u{mlp.y - add2[1] - r:g}")
    svg.route((cx, add2[1] - r), f"u{add2[1] - r - final_norm.y - final_norm.h:g}")
    svg.route(final_norm.top, f"u{final_norm.y - linear.y - linear.h:g}")
    svg.route(linear.top, "u22")
    svg.text((cx, 62), "softmax: next-token probabilities")

    # Residual connections: fork the stream before a block, and add it back after.
    loop = 125
    svg.dot(split1)
    svg.dot(split2)
    svg.route(split1, f"r{loop} u{split1[1] - add1[1]:g} l{loop - r:g}")
    svg.route(split2, f"r{loop} u{split2[1] - add2[1]:g} l{loop - r:g}")
    svg.text((layer.x + layer.w + 8, 560), ["residual", "connection"], size=10.5, colour="#555555", align="start")
    return svg.to_string()
