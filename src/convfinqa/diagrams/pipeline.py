"""From a base model to our fine-tuned model: the three stages in the lab's story."""

from __future__ import annotations

from .svg import Box, Svg

BLUE = ("#dae8fc", "#6c8ebf")
GREEN = ("#d5e8d4", "#82b366")
YELLOW = ("#fff2cc", "#d6b656")
GREY = "#555555"


def build() -> str:
    svg = Svg(640, 190)
    stages = [
        (BLUE, "Pretraining", ["next-token prediction", "on very large text corpora"], "base model"),
        (GREEN, "Post-training", ["follow instructions,", "answer in a chat format"], "instruct model"),
        (YELLOW, "Our fine-tuning", ["LoRA on ConvFinQA:", "write DSL programs"], "our model"),
    ]
    width, gap, top = 170, 50, 40
    for i, (style, title, lines, result) in enumerate(stages):
        box = Box(10 + i * (width + gap), top, width, 90)
        svg.box(box, *style, radius=10)
        cx = box.x + width / 2
        svg.text((cx, top + 26), title, size=14, bold=True)
        svg.text((cx, top + 48), lines, size=11.5, colour=GREY, line_height=15)
        svg.text((cx, top + 118), result, size=12.5, bold=True)
        if i < len(stages) - 1:
            svg.route(box.right, f"r{gap:g}")
    svg.text((10, 20), "Qwen3-4B-Instruct-2507 is the result of the first two stages", size=11, colour=GREY, align="start")
    return svg.to_string()
