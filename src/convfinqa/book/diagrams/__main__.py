""" Main entry point for generating book diagrams as SVG files.

Syntax:
    python -m convfinqa.book.diagrams [diagram_name ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import attention, baseline_prompt, decoder, feedforward, history_rounds, label_shift, lora_layer, lora_matrix, pipeline, posttraining, sequence_layout, training_step

# Directory to output generated SVG figures
FIGURES = Path(__file__).resolve().parents[4] / "docs" / "figures"

# Mapping of diagram names to their corresponding build functions
DIAGRAMS = {
    "decoder": decoder.build,
    "attention": attention.build,
    "feedforward": feedforward.build,
    "pipeline": pipeline.build,
    "posttraining": posttraining.build,
    "training_step": training_step.build,
    "label_shift": label_shift.build,
    "lora_matrix": lora_matrix.build,
    "lora_layer": lora_layer.build,
    "sequence_layout": sequence_layout.build,
    "baseline_prompt": baseline_prompt.build,
    "history_rounds": history_rounds.build,
}


def main(argv: list[str]) -> int:
    """Generate specified book diagrams as SVG files.

    Args:
        argv: List of diagram names to generate. If empty, all diagrams in
            `DIAGRAMS` are generated.

    Returns:
        Exit code (0 for success, 1 for unknown diagrams).
    """
    names = argv or list(DIAGRAMS)
    unknown = [n for n in names if n not in DIAGRAMS]
    if unknown:
        print(f"unknown diagram(s): {unknown}; available: {sorted(DIAGRAMS)}", file=sys.stderr)
        return 1
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in names:
        out = FIGURES / f"{name}.svg"
        out.write_text(DIAGRAMS[name]())
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
