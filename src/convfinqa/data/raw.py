"""Download and load the ConvFinQA splits, and read a conversation's turns."""

from __future__ import annotations

import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
"""The root directory of the ConvFinQA repository."""

RAW_DIR = ROOT / "data" / "raw"
"""The directory where the raw ConvFinQA data is stored."""

REPO_URL = "https://github.com/czyssrs/ConvFinQA.git"
"""The URL of the ConvFinQA GitHub repository."""

# The official test split ships without labels, so dev is our test set.
_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test_private.json",
    "train_turn": "train_turn.json",
    "dev_turn": "dev_turn.json",
    "test_turn": "test_turn_private.json",
}
"""Mapping from split names to their corresponding JSON files.

Files with `*_turn` contain one row per turn, while the others contain full conversations.
Since the official `test` split ships without labels, we use the `dev` split as our test set.
"""



def download_data(raw_dir: Path = RAW_DIR) -> Path:
    """Clone the ConvFinQA repo and unzip its data into `raw_dir/data`. Skips if already done.
    
    Args:
        raw_dir: The directory where the raw ConvFinQA data should be stored. Defaults to `RAW_DIR`.
    
    Returns:
        The path to the directory containing the raw ConvFinQA data, i.e. `raw_dir/data`.
    
    Raises:
        subprocess.CalledProcessError: If the git clone command fails.
        zipfile.BadZipFile: If the data.zip file is not a valid zip file.
    """
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
    """Load one split: `train`, `dev`, `test` (conversations) or `*_turn` (one row per turn).
    
    Args:
        name: The name of the split to load. One of `train`, `dev`, `test`, or `*_turn`.
        raw_dir: The directory where the raw ConvFinQA data is stored. Defaults to `RAW_DIR`.

    Returns:
        A list of dictionaries representing the conversations or turns in the specified split.
    """
    return json.loads((raw_dir / "data" / _FILES[name]).read_text())


@dataclass(frozen=True)
class Turn:
    """A single turn in a conversation."""
    index: int
    """The index of the turn within the conversation."""
    question: str
    """The question asked in this turn."""
    program: str  # DSL; a bare number for a number-selection turn
    """The program (in DSL) executed for this turn."""
    answer: float
    """The answer obtained for this turn."""
    part: int
    """The part of the FinQA question this turn belongs to.
    
    Always 0 for Type I conversations, which involve only one FinQA question.
    For Type II conversations, consisting of two merged FinQA questions, either 0 or 1.
    """


def conversation_type(conv: dict) -> str:
    """Determine the type of conversation.

    Args:
        conv: A dictionary representing the conversation.
    
    Returns:
        A string indicating the type of conversation: "I" or "II".
        A Type I conversation involves one FinQA question decomposed into multiple turns.
        A Type II conversation involves two FinQA questions merged into one conversation before
        decomposition into turns.
    """
    return "II" if conv["id"].startswith("Double_") else "I"


def turns(conv: dict) -> list[Turn]:
    """Extract the turns from a conversation.

    Extracts the components of each turn from the conversation's annotation data, merging them into
    `Turn` objects.

    Args:
        conv: A dictionary representing the conversation.

    Returns:
        A list of `Turn` objects corresponding to the turns in the conversation.
    """
    a = conv["annotation"]
    return [
        Turn(i, q, p, ans, part)
        for i, (q, p, ans, part) in enumerate(
            zip(a["dialogue_break"], a["turn_program"], a["exe_ans_list"], a["qa_split"])
        )
    ]
