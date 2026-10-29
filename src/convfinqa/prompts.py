"""Prompt variants of the baseline: how many examples, and which system message."""

from __future__ import annotations

import re

from .formatting import SYSTEM_PROMPT, prompt_messages

VARIANT = re.compile(r"e([023])([pdf])")
"""A variant name: `e` + the number of examples (0, 2 or 3) + `p` (plain system message), `d` (description) or `f` (description and the flat-steps rule; added after reading the validation errors of the first six)."""

VARIANTS = ("e0p", "e0d", "e2p", "e2d", "e3p", "e3d", "e3f")


def parse_variant(name: str) -> tuple[int, str]:
    """The number of examples and the kind of system message (`plain`, `description` or `flat`) of a variant."""
    m = VARIANT.fullmatch(name)
    if not m:
        raise ValueError(f"unknown prompt variant {name!r}; expected one of {VARIANTS}")
    return int(m.group(1)), {"p": "plain", "d": "description", "f": "flat"}[m.group(2)]


def prompt_chat(
    conv: dict, k: int, system: str, examples: list[dict] = (), history: list[str] | None = None
) -> list[dict]:
    """The chat for turn `k` of `conv`: `formatting.prompt_messages` with another system message.

    Args:
        conv: The conversation asked about.
        k: Index of the turn to answer.
        system: The system message.
        examples: Complete conversations shown first, with their gold programs.
        history: Programs for turns `0..k-1`; the gold ones when omitted.
    """
    chat = prompt_messages(conv, k, examples, history)
    if system != SYSTEM_PROMPT:
        chat[0] = {"role": "system", "content": system}
    return chat
