"""Load ConvFinQA and turn raw conversations into readable pieces."""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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


def is_number_turn(program: str) -> bool:
    """A number-selection turn's program is just a number, with no operation."""
    return "(" not in program


def split_steps(program: str) -> list[str]:
    """Split ``"subtract(a, b), divide(#0, c)"`` into its steps."""
    return re.findall(r"[a-z_]+\([^)]*\)", program)


# The constants the authors' generator may emit (code/finqanet_generator/constant_list.txt).
CONSTANTS = (
    "const_1", "const_2", "const_3", "const_4", "const_5", "const_6", "const_7", "const_8", "const_9",
    "const_10", "const_100", "const_1000", "const_10000", "const_100000", "const_1000000",
    "const_10000000", "const_1000000000", "const_m1",
)


def is_copyable_number(token: str) -> bool:
    """The paper's rule for "this token is a number the generator may copy from the page".

    Mirrors ``str_to_num`` in the authors' ``finqa_utils.py``: after removing commas, the token parses as
    a number, or it is a non-empty string ending in ``%``. The page text is pre-tokenized, so symbols and
    punctuation around a number (``$ 103102``, ``june 30 , 2009``) are separate tokens. A ``%`` written
    attached to its number (``19%``) stays one token; ``19 %`` splits into ``19`` and ``%``.
    """
    token = token.replace(",", "")
    try:
        float(token)
        return True
    except ValueError:
        return token.endswith("%")


def copyable_numbers(text: str) -> list[str]:
    """Every space-separated token of ``text`` that passes :func:`is_copyable_number`."""
    return [tok for tok in text.split(" ") if is_copyable_number(tok)]


def table_to_markdown(table: list[list[str]]) -> str:
    """Render a table (header row first) as a markdown table."""
    width = max(len(r) for r in table)
    rows = [list(r) + [""] * (width - len(r)) for r in table]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def context_text(conv: dict) -> str:
    """The page the questions refer to: text before the table, the table, text after."""
    return "\n\n".join(
        [" ".join(conv["pre_text"]), table_to_markdown(conv["table"]), " ".join(conv["post_text"])]
    )


def format_conversation(conv: dict) -> str:
    """Pretty-print a conversation's turns as ``question -> program -> answer``."""
    lines = [f"{conv['id']}  (Type {conversation_type(conv)})"]
    for t in turns(conv):
        lines.append(f"  [{t.index + 1}] Q: {t.question}")
        lines.append(f"      program: {t.program}")
        lines.append(f"      answer:  {t.answer:g}" + (f"   (part {t.part + 1})" if conv["id"].startswith("Double_") else ""))
    return "\n".join(lines)
