"""Reading DSL programs as text: steps, number-selection turns, and numbers the generator may copy."""

from __future__ import annotations

import re


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
