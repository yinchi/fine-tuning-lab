"""Download and load the ConvFinQA splits, and read a conversation's turns."""

from __future__ import annotations

import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
REPO_URL = "https://github.com/czyssrs/ConvFinQA.git"

# The official test split ships without labels, so dev is our test set.
_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test_private.json",
    "train_turn": "train_turn.json",
    "dev_turn": "dev_turn.json",
    "test_turn": "test_turn_private.json",
}


def download_data(raw_dir: Path = RAW_DIR) -> Path:
    """Clone the ConvFinQA repo and unzip its data into ``raw_dir/data``. Skips if already done."""
    data_dir = raw_dir / "data"
    if (data_dir / "train.json").exists():
        return data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo = raw_dir / "ConvFinQA"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    with zipfile.ZipFile(repo / "data.zip") as z:
        z.extractall(raw_dir)
    return data_dir


def load_split(name: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Load one split: ``train``, ``dev``, ``test`` (conversations) or ``*_turn`` (one row per turn)."""
    return json.loads((raw_dir / "data" / _FILES[name]).read_text())


@dataclass(frozen=True)
class Turn:
    index: int
    question: str
    program: str  # DSL; a bare number for a number-selection turn
    answer: float
    part: int  # 0 or 1: which of the two merged FinQA questions (always 0 for Type I)


def conversation_type(conv: dict) -> str:
    """``"I"`` (one FinQA question decomposed) or ``"II"`` (two questions merged)."""
    return "II" if conv["id"].startswith("Double_") else "I"


def turns(conv: dict) -> list[Turn]:
    a = conv["annotation"]
    return [
        Turn(i, q, p, ans, part)
        for i, (q, p, ans, part) in enumerate(
            zip(a["dialogue_break"], a["turn_program"], a["exe_ans_list"], a["qa_split"])
        )
    ]
