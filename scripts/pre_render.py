#!/usr/bin/env python3
"""Quarto pre-render hook: mark chapters that used outdated artifacts for re-execution.

Quarto runs this under the system python3, so it only starts the project venv's Python.
"""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
python = root / ".venv" / "bin" / "python"
if not python.exists():
    sys.exit(f"pre-render: {python} not found; run `uv sync` first")
sys.exit(subprocess.run([str(python), "-m", "convfinqa.artifacts", "refresh"], cwd=root).returncode)
