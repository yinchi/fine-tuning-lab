"""Program accuracy: does the generated program say the same thing as the gold program?

The official ConvFinQA script (`equal_program`) compares programs symbolically with sympy: `add(a, b)` equals
`add(b, a)`, and `#k` references are expanded, so how the steps are ordered or split does not matter. The
arguments are compared as exact text, so `2.70` is not `2.7` and `17%` is not `0.17`, and a program that
computes the right answer from other numbers is not the gold program.

This lives apart from `evaluate` so that adding a measure does not change the code hash of the heavy artifacts
that generate the predictions.
"""

from __future__ import annotations

import re
from collections import Counter

import sympy

from .dsl import executor


def _combine(op: str, a, b, order_free: bool = False):
    """The sympy expression of one step. Add and multiply are commutative, the rest keep their order.

    With `order_free`, subtract and divide ignore the order of their arguments too (`subtract(a, b)` and
    `subtract(b, a)` become the same expression), to find programs that differ only by reversed operands.
    """
    if order_free and op in ("subtract", "divide"):
        return sympy.Function(op)(*sorted((a, b), key=sympy.default_sort_key))
    if op == "add":
        return sympy.Add(a, b)
    if op == "subtract":
        return sympy.Add(a, sympy.Mul(-1, b))
    if op == "multiply":
        return sympy.Mul(a, b)
    if op == "divide":
        return sympy.Mul(a, sympy.Pow(b, -1))
    if op == "exp":
        return sympy.Pow(a, b)
    return sympy.Function(op)(a, b)  # greater: its arguments are not interchangeable


def program_expression(program: str, order_free: bool = False) -> tuple[sympy.Expr, Counter]:
    """A program as a symbolic expression of its arguments, and the arguments it uses.

    The expression is the last step with every `#k` replaced by the expression of step `k`. Each argument is a
    symbol named by its exact text. A bare number is the symbol of its text.

    Args:
        program: A DSL program.
        order_free: Treat subtract and divide as symmetric in their arguments (see `_combine`).

    Returns:
        The expression, and a count of the argument texts in the steps the result depends on. The count guards
        against the cancellation that sympy does on its own: `subtract(5, 5)` and `subtract(7, 7)` are both 0,
        but they do not use the same numbers.

    Raises:
        executor.ProgramError: If the program cannot be parsed or has a `#k` that does not name an earlier step.
    """
    steps = executor.parse_program(program)
    if not steps:
        text = program.strip()
        return sympy.Symbol(text), Counter([text])
    results: list[tuple[sympy.Expr, Counter]] = []
    for i, step in enumerate(steps):
        parts = []
        for arg in step.args:
            if arg.startswith("#"):
                if not arg[1:].isdigit() or int(arg[1:]) >= i:
                    raise executor.ProgramError(executor.BAD_REFERENCE, f"bad reference {arg!r} in step {i}")
                parts.append(results[int(arg[1:])])
            else:
                parts.append((sympy.Symbol(arg), Counter([arg])))
        results.append((_combine(step.op, parts[0][0], parts[1][0], order_free), parts[0][1] + parts[1][1]))
    return results[-1]


def equal_program(predicted: str, gold: str, order_free: bool = False) -> bool:
    """Whether two programs are the same up to the order of a sum or a product, in the sense of the official script.

    A predicted program that cannot be parsed is not equal to anything.

    Args:
        predicted: The model's output.
        gold: The gold program.
        order_free: Also ignore the order of the arguments of subtract and divide.

    Returns:
        True if the expressions are equal and they use the same arguments.
    """
    try:
        expr_p, used_p = program_expression(predicted, order_free)
    except executor.ProgramError:
        return False
    expr_g, used_g = program_expression(gold, order_free)
    return expr_p == expr_g and used_p == used_g


def equal_up_to_constants(predicted: str, gold: str) -> bool:
    """Whether the programs are equal once `const_100` is read as `100` and `const_m1` as `-1`.

    The gold programs write the number 100 as `const_100` in some conversations and as `100` in others, and
    `equal_program` counts the two spellings as different arguments.

    Args:
        predicted: The model's output.
        gold: The gold program.

    Returns:
        True if `equal_program` holds after the constants are spelled as plain numbers.
    """
    plain = lambda p: re.sub(r"const_(m?)(\d+)", lambda m: ("-" if m.group(1) else "") + m.group(2), p)
    return equal_program(plain(predicted), plain(gold))


def same_text(predicted: str, gold: str) -> bool:
    """Whether the two programs are the same step for step, ignoring only the white space between tokens."""
    try:
        steps_p = executor.parse_program(predicted)
    except executor.ProgramError:
        return False
    steps_g = executor.parse_program(gold)
    if not steps_g or not steps_p:
        return predicted.strip() == gold.strip()
    return steps_p == steps_g


def add_program_match(scores: list[dict], rows: list[dict], turn_rows: list[dict]) -> list[dict]:
    """Add `program_correct` and `text_match` to every score.

    Args:
        scores: From `evaluate.score_rows`.
        rows: The predictions they were computed from (`id`, `turn`, `output`).
        turn_rows: The gold turns (`id`, `turn`, `program`).

    Returns:
        The scores with two more keys: `program_correct` (`equal_program`) and `text_match` (the same steps,
        written the same way).
    """
    gold = {(r["id"], r["turn"]): r["program"] for r in turn_rows}
    output = {(r["id"], r["turn"]): r["output"] for r in rows}
    return [
        {
            **s,
            "program_correct": equal_program(output[(s["id"], s["turn"])], gold[(s["id"], s["turn"])]),
            "text_match": same_text(output[(s["id"], s["turn"])], gold[(s["id"], s["turn"])]),
        }
        for s in scores
    ]
