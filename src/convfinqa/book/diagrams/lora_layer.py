"""One decoder layer of Qwen3-4B with the seven linear layers that LoRA adapts marked."""

from __future__ import annotations

from .svg import Box, Svg

ADAPTED = ("#d5e8d4", "#82b366")
PLAIN = ("#eeeeee", "#999999")
GREY = "#555555"


def build() -> str:
    cx, loop, r = 310, 200, 11
    width = 620

    # Positions, from the top of the page down.
    head = Box(cx - 110, 62, 220, 36)
    layer_top = 132
    y = layer_top + 22
    add2 = (cx, y + r)
    y += 2 * r + 22
    down = Box(cx - 55, y, 110, 36)
    y += 36 + 22
    act = Box(cx - 100, y, 200, 34)
    y += 34 + 22
    gate, up = Box(cx - 120, y, 90, 40), Box(cx + 30, y, 90, 40)
    y += 40 + 22
    norm2 = Box(cx - 60, y, 120, 28)
    y += 28 + 14
    fork2 = (cx, y)
    y += 12
    add1 = (cx, y + r)
    y += 2 * r + 22
    o = Box(cx - 55, y, 110, 36)
    y += 36 + 22
    mix = Box(cx - 180, y, 360, 34)
    y += 34 + 22
    q, k, v = (Box(cx + dx - 45, y, 90, 40) for dx in (-135, 0, 135))
    y += 40 + 22
    norm1 = Box(cx - 60, y, 120, 28)
    y += 28 + 14
    fork1 = (cx, y)
    layer = Box(70, layer_top, 460, y + 14 - layer_top)
    emb = Box(cx - 80, layer.y + layer.h + 30, 160, 36)

    svg = Svg(width, emb.y + emb.h + 38)

    # Legend.
    for row, (style, text) in enumerate([
        (ADAPTED, "linear layer with a LoRA adapter (36 layers × 7 = 252 in all)"),
        (PLAIN, "no adapter: embeddings, norms, the output head, and the parts with no weights"),
    ]):
        svg.box(Box(14, 14 + row * 20, 16, 12), *style, radius=3)
        svg.text((38, 24 + row * 20), text, size=11, align="start")

    svg.box(layer, "none", "#888888", radius=16, dashed=True)
    svg.text((layer.x - 8, layer.y + layer.h / 2), ["36 ×", "layers"], align="end", size=14, bold=True)

    def block(box, title, sub, style):
        svg.box(box, *style)
        svg.text((box.x + box.w / 2, box.y + 17), title, size=12.5, bold=style is ADAPTED)
        if sub:
            svg.text((box.x + box.w / 2, box.y + 31), sub, size=9.5, colour=GREY)

    block(head, "Output head", "tied to the embeddings", PLAIN)
    block(emb, "Embeddings", None, PLAIN)
    for box, name, shape in [
        (q, "q_proj", "2,560 → 4,096"), (k, "k_proj", "2,560 → 1,024"), (v, "v_proj", "2,560 → 1,024"),
        (o, "o_proj", "4,096 → 2,560"), (gate, "gate_proj", "2,560 → 9,728"),
        (up, "up_proj", "2,560 → 9,728"), (down, "down_proj", "9,728 → 2,560"),
    ]:
        block(box, name, shape, ADAPTED)
    block(mix, "attention: scores, softmax, weighted sum", None, PLAIN)
    block(act, "SiLU(gate) × up", None, PLAIN)
    block(norm1, "RMSNorm", None, PLAIN)
    block(norm2, "RMSNorm", None, PLAIN)
    for centre in (add1, add2):
        svg.circle(centre, r)
        svg.text((centre[0], centre[1] + 5), "+", size=16)

    # The main path, bottom to top.
    svg.text((cx, svg.height - 12), "token ids")
    svg.route((cx, svg.height - 26), "u" + f"{svg.height - 26 - (emb.y + emb.h):g}")
    svg.route(emb.top, f"u{emb.y - norm1.y - norm1.h:g}")
    svg.dot(fork1)
    for box in (q, k, v):                      # norm -> q, k, v
        dx = box.x + box.w / 2 - cx
        moves = f"u22" if dx == 0 else f"u11 {'r' if dx > 0 else 'l'}{abs(dx):g} u11"
        svg.route(norm1.top, moves)
    svg.dot((cx, norm1.y - 11))
    for box in (q, k, v):                      # q, k, v -> attention
        svg.route(box.top, f"u{box.y - mix.y - mix.h:g}")
    svg.route(mix.top, f"u{mix.y - o.y - o.h:g}")
    svg.route(o.top, f"u{o.y - add1[1] - r:g}")
    svg.route((cx, add1[1] - r), f"u{add1[1] - r - norm2.y - norm2.h:g}")
    svg.dot(fork2)
    for box, dx in ((gate, -75), (up, 75)):    # norm -> gate, up
        svg.route(norm2.top, f"u11 {'r' if dx > 0 else 'l'}{abs(dx):g} u11")
    svg.dot((cx, norm2.y - 11))
    svg.route(gate.top, f"u{gate.y - act.y - act.h:g}")
    svg.route(up.top, f"u{up.y - act.y - act.h:g}")
    svg.route(act.top, f"u{act.y - down.y - down.h:g}")
    svg.route(down.top, f"u{down.y - add2[1] - r:g}")
    svg.route((cx, add2[1] - r), f"u{add2[1] - r - head.y - head.h:g}")

    # Residual connections.
    svg.route(fork1, f"r{loop} u{fork1[1] - add1[1]:g} l{loop - r:g}")
    svg.route(fork2, f"r{loop} u{fork2[1] - add2[1]:g} l{loop - r:g}")
    svg.text((layer.x + layer.w + 6, add1[1] + 40), ["residual", "connection"], size=10, colour=GREY, align="start")
    return svg.to_string()
