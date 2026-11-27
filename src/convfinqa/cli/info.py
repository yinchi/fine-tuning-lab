"""What an adapter directory says about itself: `train_info.json`, and the checks `chat` makes before using one.

An adapter is only useful with the prompt it was trained on, and with the base model it was trained from. The
directory `PeftModel.save_pretrained` writes names the base model but not the prompt, so `train-adapter` adds a file
with both, and `chat` refuses an adapter whose prompt or base model is not the one it would use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..formatting import SYSTEM_PROMPT, conversation_messages

INFO_FILE = "train_info.json"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

_PROBE = {
    "id": "Single_X/2000/page_1.pdf-0",
    "pre_text": ["before ."],
    "table": [["", "2000"], ["sales", "$ 5"]],
    "post_text": ["after ."],
    "annotation": {
        "dialogue_break": ["what was the sales?", "and twice that?"],
        "turn_program": ["5", "multiply(5, const_2)"],
        "exe_ans_list": [5, 10],
        "qa_split": [0, 0],
    },
}


def format_fingerprint() -> str:
    """A short hash of the prompt format: the system message, and the messages of a fixed small conversation.

    It changes when `formatting` changes how a conversation is turned into chat messages, so an adapter trained with
    one format is not used with another without notice. (The tokenizer's chat template is not part of it.)
    """
    blob = json.dumps({"system": SYSTEM_PROMPT, "messages": conversation_messages(_PROBE)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class AdapterError(ValueError):
    """An adapter directory that cannot be used."""


@dataclass(frozen=True)
class AdapterCheck:
    """What `check_adapter` found."""

    base_model: str
    warnings: list[str] = field(default_factory=list)


def write_info(directory: str | Path, info: dict) -> Path:
    """Write `train_info.json` into `directory`."""
    path = Path(directory) / INFO_FILE
    path.write_text(json.dumps(info, indent=2) + "\n")
    return path


def check_adapter(directory: str | Path) -> AdapterCheck:
    """Check that `directory` holds an adapter `chat` can use, and find its base model.

    The base model is read from `adapter_config.json`. If `train_info.json` is there, its format fingerprint must be the
    current one and its base model that of the adapter. An adapter without it (one of the book's `artifacts/lora_*`)
    is accepted with a warning.

    Raises:
        AdapterError: If a file is missing, or the format or the base model does not match.
    """
    directory = Path(directory)
    missing = [name for name in ADAPTER_FILES if not (directory / name).is_file()]
    if missing:
        raise AdapterError(f"{directory} is not an adapter directory: missing {', '.join(missing)}")
    try:
        base = json.loads((directory / "adapter_config.json").read_text())["base_model_name_or_path"]
    except (ValueError, KeyError) as e:
        raise AdapterError(f"{directory / 'adapter_config.json'} does not name a base model") from e
    info_path = directory / INFO_FILE
    if not info_path.is_file():
        return AdapterCheck(base, [f"no {INFO_FILE} in {directory}: cannot check that the prompt format matches"])
    try:
        info = json.loads(info_path.read_text())
    except ValueError as e:
        raise AdapterError(f"{info_path} is not valid JSON") from e
    if info.get("format") != format_fingerprint():
        raise AdapterError(
            f"{directory} was trained with another prompt format (format {info.get('format')}, "
            f"this code {format_fingerprint()}); it would run but misbehave"
        )
    if info.get("base_model") != base:
        raise AdapterError(f"{INFO_FILE} says the base model is {info.get('base_model')}, the adapter says {base}")
    return AdapterCheck(base)
