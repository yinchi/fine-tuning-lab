"""One training step: batch, forward pass, loss, backward pass, optimizer step, and what each keeps."""

from __future__ import annotations

from .svg import Box, Svg

BLUE = ("#dae8fc", "#6c8ebf")
RED = ("#f8cecc", "#b85450")
YELLOW = ("#fff2cc", "#d6b656")
GREEN = ("#d5e8d4", "#82b366")
PLAIN = ("#ffffff", "#444444")
GREY = "#555555"


def build() -> str:
    svg = Svg(780, 330)
    width, gap, left, top, height = 120, 30, 25, 70, 62

    steps = [
        (BLUE, "Batch", ["input_ids, labels", "from the training data"]),
        (BLUE, "Forward pass", ["logits at every", "position"]),
        (YELLOW, "Loss", ["cross-entropy on the", "unmasked tokens"]),
        (RED, "Backward pass", ["gradient of the loss", "for every weight"]),
        (GREEN, "Optimizer step", ["AdamW changes", "the weights"]),
    ]
    boxes = [Box(left + i * (width + gap), top, width, height) for i in range(len(steps))]
    for i, (box, (style, title, lines)) in enumerate(zip(boxes, steps)):
        svg.box(box, *style, radius=10)
        cx = box.x + width / 2
        svg.text((cx, top + 24), title, size=13.5, bold=True)
        svg.text((cx, top + 42), lines, size=10.5, colour=GREY, line_height=13)
        if i < len(steps) - 1:
            svg.route(box.right, f"r{gap:g}")

    # The loop: after the weights change, the next batch is read.
    first, last = boxes[0], boxes[-1]
    svg.route(last.top, f"u36 l{last.top[0] - first.top[0]:g} d36")
    svg.text(((first.top[0] + last.top[0]) / 2, top - 46), "repeat with the next batch", size=11.5, colour=GREY)

    # What lives in memory between and during steps.
    state_top = 205
    state_h = 70
    weights = Box(boxes[1].x - 10, state_top, width + 20, state_h)
    grads = Box(boxes[3].x - 10, state_top, width + 20, state_h)
    adam = Box(boxes[4].x - 10, state_top, width + 20, state_h)
    for box, title, lines in [
        (weights, "Weights", ["one number per", "parameter"]),
        (grads, "Gradients", ["one number per", "trainable weight"]),
        (adam, "Optimizer state", ["two more per trainable", "weight (AdamW: m and v)"]),
    ]:
        svg.box(box, *PLAIN)
        cx = box.x + box.w / 2
        svg.text((cx, box.y + 24), title, size=13, bold=True)
        svg.text((cx, box.y + 42), lines, size=10.5, colour=GREY, line_height=13)

    svg.route(weights.top, f"u{weights.y - boxes[1].y - boxes[1].h:g}")  # read by the forward pass
    svg.route(boxes[3].bottom, f"d{grads.y - boxes[3].y - boxes[3].h:g}")  # written by the backward pass
    svg.route(adam.top, f"u{adam.y - boxes[4].y - boxes[4].h:g}", arrow=False)
    svg.text((20, 313), "Memory holds the weights, the gradients and the optimizer state at the same time; activations are kept for the backward pass.", size=10.5, colour=GREY, align="start")
    svg.text((20, 30), "One training step", size=13.5, bold=True, align="start")
    return svg.to_string()
