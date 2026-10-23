"""Reading DSL programs as text: steps, number-selection turns, and numbers the generator may copy."""

from __future__ import annotations

import re


def is_number_turn(program: str) -> bool:
    """Check if the program is a bare number, i.e. contains no operations."""
    return "(" not in program


def split_steps(program: str) -> list[str]:
    """Split a program into its steps.
    
    Each step is expected to be in the form `op(arg1, arg2)`.
    """
    return re.findall(r"[a-z_]+\([^)]*\)", program)


CONSTANTS = (
    "const_1", "const_2", "const_3", "const_4", "const_5", "const_6", "const_7", "const_8", "const_9",
    "const_10", "const_100", "const_1000", "const_10000", "const_100000", "const_1000000",
    "const_10000000", "const_1000000000", "const_m1",
)
"""The constants the authors' generator may emit (`code/finqanet_generator/constant_list.txt`)."""


def is_copyable_number(token: str) -> bool:
    """Checks if a token is a number the generator may copy from the page.

    A token is considered copyable if it can be interpreted as a number after removing commas,
    or if it is a non-empty string ending with a percent sign ("%"). A malformed token such as
    "foo%" would be considered copyable since it ends with a percent sign, but would be caught
    by the executor as an invalid number.
    """
    token = token.replace(",", "")
    try:
        float(token)
        return True
    except ValueError:
        return token.endswith("%")


def copyable_numbers(text: str) -> list[str]:
    """Every space-separated token of `text` that passes `is_copyable_number`."""
    return [tok for tok in text.split(" ") if is_copyable_number(tok)]
