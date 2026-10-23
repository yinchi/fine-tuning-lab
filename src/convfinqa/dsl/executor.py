"""Parse and execute ConvFinQA DSL programs, following the official evaluation code.

The reference is ``eval_program`` and ``str_to_num`` in the authors' ``code/utils/general_utils.py``.
Differences are deliberate and listed in the 1.3 chapter: errors are reported by kind, and ``table_*``
operations (unused by every ConvFinQA gold program) are rejected as unknown operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OPERATIONS = ("add", "subtract", "multiply", "divide", "exp", "greater")
ROUND_DIGITS = 5  # the official script rounds every result to 5 decimal places

# Error kinds, used for the invalid-output breakdown in the evaluation chapter.
EMPTY = "empty"
SYNTAX = "syntax"
UNKNOWN_OPERATOR = "unknown_operator"
BAD_NUMBER = "bad_number"
BAD_REFERENCE = "bad_reference"
DIVISION_BY_ZERO = "division_by_zero"
MATH_ERROR = "math_error"

Value = float | str  # a number, or "yes"/"no" for ``greater``


class ProgramError(Exception):
    """A program that cannot be parsed or executed. ``kind`` is one of the constants above."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Step:
    op: str
    args: tuple[str, str]


_STEP = re.compile(r"\s*([A-Za-z_]\w*)\(([^(),]*),([^(),]*)\)\s*")


def parse_number(token: str) -> float:
    """A number literal, a ``%`` literal (divided by 100) or a ``const_`` token, as in the official code."""
    text = token.strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        pass
    if "%" in text:
        try:
            return float(text.replace("%", "")) / 100.0
        except ValueError:
            raise ProgramError(BAD_NUMBER, f"not a number: {token!r}") from None
    if text.startswith("const_"):
        name = text[len("const_"):]
        try:
            return float("-1" if name == "m1" else name)
        except ValueError:
            pass
    raise ProgramError(BAD_NUMBER, f"not a number: {token!r}")


def parse_program(program: str) -> list[Step]:
    """Split ``"op(a, b), op(#0, c)"`` into steps. A bare number returns an empty list."""
    text = program.strip()
    if not text:
        raise ProgramError(EMPTY, "empty program")
    if "(" not in text and ")" not in text:
        return []
    steps, pos = [], 0
    while pos < len(text):
        m = _STEP.match(text, pos)
        if not m:
            raise ProgramError(SYNTAX, f"cannot parse {text[pos:]!r}")
        op, a, b = m.group(1), m.group(2).strip(), m.group(3).strip()
        if not a or not b:
            raise ProgramError(SYNTAX, f"missing argument in {m.group(0).strip()!r}")
        if op not in OPERATIONS:
            raise ProgramError(UNKNOWN_OPERATOR, f"unknown operator {op!r}")
        steps.append(Step(op, (a, b)))
        pos = m.end()
        if pos < len(text):
            if text[pos] != ",":
                raise ProgramError(SYNTAX, f"expected ',' before {text[pos:]!r}")
            pos += 1
    return steps


def _resolve(arg: str, results: list[Value], index: int) -> float:
    if arg.startswith("#"):
        try:
            k = int(arg[1:])
        except ValueError:
            raise ProgramError(BAD_REFERENCE, f"bad step reference {arg!r}") from None
        if not 0 <= k < index:
            raise ProgramError(BAD_REFERENCE, f"step {index} cannot use {arg} (only #0..#{index - 1})")
        value = results[k]
        if isinstance(value, str):
            raise ProgramError(BAD_REFERENCE, f"{arg} is a yes/no result and cannot be used as a number")
        return value
    return parse_number(arg)


def trace(program: str) -> list[Value]:
    """The unrounded result of every step, in order. A bare number gives a one-element list."""
    steps = parse_program(program)
    if not steps:
        return [parse_number(program)]
    results: list[Value] = []
    for i, step in enumerate(steps):
        a, b = (_resolve(x, results, i) for x in step.args)
        try:
            if step.op == "add":
                r: Value = a + b
            elif step.op == "subtract":
                r = a - b
            elif step.op == "multiply":
                r = a * b
            elif step.op == "divide":
                r = a / b
            elif step.op == "exp":
                r = a**b
            else:
                r = "yes" if a > b else "no"
        except ZeroDivisionError:
            raise ProgramError(DIVISION_BY_ZERO, f"division by zero in step {i}") from None
        except OverflowError:
            raise ProgramError(MATH_ERROR, f"overflow in step {i}") from None
        if isinstance(r, complex):  # a negative base with a fractional exponent
            raise ProgramError(MATH_ERROR, f"complex result in step {i}")
        results.append(r)
    return results


def execute(program: str) -> Value:
    """Run a program. Returns a float rounded to 5 places, or ``"yes"``/``"no"`` if the last step is ``greater``."""
    last = trace(program)[-1]
    return last if isinstance(last, str) else round(last, ROUND_DIGITS)


def try_execute(program: str) -> tuple[Value | None, str | None]:
    """``(value, None)`` on success, ``(None, error_kind)`` on failure. Never raises for a bad program."""
    try:
        return execute(program), None
    except ProgramError as e:
        return None, e.kind


def answers_match(predicted: Value | None, gold: Value) -> bool:
    """The official comparison: exact equality of the rounded results (``"yes"``/``"no"`` compare as strings)."""
    return predicted is not None and predicted == gold
