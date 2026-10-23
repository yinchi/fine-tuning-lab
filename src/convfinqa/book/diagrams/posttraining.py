"""How Qwen3 models were post-trained, based on Figure 1 of the Qwen3 technical report."""

from __future__ import annotations

from .svg import Box, Svg

# Color codes for different types of boxes in the diagram.
# Pairs represent (fill color, border color) for the boxes in the diagram.

SFT = ("#dae8fc", "#6c8ebf")  # supervised fine-tuning
RL = ("#fff2cc", "#d6b656")  # reinforcement learning
DISTIL = ("#d5e8d4", "#82b366")  # distillation
PLAIN = ("#ffffff", "#444444")
GREY = "#555555"


def build() -> str:
    svg = Svg(590, 560)
    left_x, right_x, w = 150, 440, 240  # centres of the two routes, and the box width

    base = Box(215, 18, 160, 40)
    stages = [
        (SFT, "1  Long-CoT cold start", "supervised fine-tuning"),
        (RL, "2  Reasoning RL", "reinforcement learning"),
        (SFT, "3  Thinking mode fusion", "supervised fine-tuning"),
        (RL, "4  General RL", "reinforcement learning"),
    ]
    boxes = [Box(left_x - w / 2, 110 + i * 78, w, 50) for i in range(len(stages))]
    distil = Box(right_x - w / 2, 110, w, 62)
    flagship = Box(left_x - w / 2, 110 + 4 * 78, w, 56)
    light = Box(right_x - w / 2, 110 + 4 * 78, w, 56)

    svg.box(base, *PLAIN)
    svg.text((base.x + base.w / 2, base.y + 25), "Base models", size=13.5, bold=True)
    for box, (style, title, kind) in zip(boxes, stages):
        svg.box(box, *style)
        svg.text((left_x, box.y + 21), title, size=13, bold=True)
        svg.text((left_x, box.y + 38), kind, size=10.5, colour=GREY)
    svg.box(distil, *DISTIL)
    svg.text((right_x, distil.y + 23), "Strong-to-weak distillation", size=13, bold=True)
    svg.text((right_x, distil.y + 41), "off-policy, then on-policy", size=10.5, colour=GREY)
    svg.box(flagship, *PLAIN)
    svg.text((left_x, flagship.y + 23), "Flagship models", size=13, bold=True)
    svg.text((left_x, flagship.y + 41), "Qwen3-235B-A22B, Qwen3-32B", size=10.5, colour=GREY)
    svg.box(light, *PLAIN)
    svg.text((right_x, light.y + 23), "Lightweight models", size=13, bold=True)
    svg.text(
        (right_x, light.y + 41), "Qwen3-30B-A3B, 14B, 8B, 4B, 1.7B, 0.6B", size=10.5, colour=GREY
    )

    # Arrows: base -> the two routes, down the stages, and out to the results.
    split_y = base.y + base.h + 20
    svg.dot((base.x + base.w / 2, split_y))
    svg.route((base.x + base.w / 2, base.y + base.h), "d20")
    svg.route(
        (base.x + base.w / 2, split_y),
        f"l{base.x + base.w / 2 - left_x:g} d{boxes[0].y - split_y:g}"
    )
    svg.route(
        (base.x + base.w / 2, split_y),
        f"r{right_x - base.x - base.w / 2:g} d{distil.y - split_y:g}"
    )
    for upper, lower in zip(boxes, boxes[1:]):
        svg.route((left_x, upper.y + upper.h), f"d{lower.y - upper.y - upper.h:g}")
    svg.route(
        (left_x, boxes[-1].y + boxes[-1].h),
        f"d{flagship.y - boxes[-1].y - boxes[-1].h:g}"
    )
    svg.route(
        (right_x, distil.y + distil.h),
        f"d{light.y - distil.y - distil.h:g}"
    )
    # The flagship models are the teachers of the distillation: from their box, up the gap between the routes.
    gap = (left_x + w / 2 + (right_x - w / 2)) / 2 - (left_x + w / 2)  # half the gap between the columns
    svg.route(
        flagship.right,
        f"r{gap:g} u{flagship.y + flagship.h / 2 - distil.y - distil.h / 2:g} r{gap:g}"
    )
    svg.text(
        (left_x + w / 2 + gap + 6, 300),
        ["teachers"], size=10.5, colour=GREY, align="start"
    )
    svg.text(
        (right_x + 8, 270),
        [
            "the report: about a tenth of the",
            "GPU hours of the four stages",
            "(preliminary experiments)"
        ],
        size=10.5, colour=GREY, align="start", line_height=14
    )

    for i, (label, (fill, stroke)) in enumerate([
        ("supervised fine-tuning", SFT),
        ("reinforcement learning", RL),
        ("distillation", DISTIL)
    ]):
        x = 10 + i * 190
        svg.box(Box(x, 530, 16, 12), fill, stroke, radius=3)
        svg.text((x + 22, 540), label, size=10.5, align="start", colour=GREY)

    return svg.to_string()
