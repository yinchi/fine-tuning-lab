"""``python -m convfinqa.book.diagrams decoder`` writes docs/figures/decoder.svg."""

from __future__ import annotations

import sys
from pathlib import Path

from . import attention, decoder, feedforward, pipeline, posttraining

FIGURES = Path(__file__).resolve().parents[4] / "docs" / "figures"
DIAGRAMS = {"decoder": decoder.build, "attention": attention.build, "feedforward": feedforward.build, "pipeline": pipeline.build, "posttraining": posttraining.build}


def main(argv: list[str]) -> int:
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
