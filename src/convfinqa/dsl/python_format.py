"""Python as an output format: translating DSL programs to Python, and running Python programs safely.

A program is straight-line Python that assigns the answer to `ans`. The translation writes the natural form:
a step used once is inlined into the expression that uses it, a step used more than once gets a name, and
`greater` is a comparison. Running a program never calls `exec` or `eval`: the code is parsed with `ast` and
evaluated by walking a whitelist of nodes (assignments; arithmetic and comparisons on numbers; names;
`round`, `abs`, `min`, `max` and `sum`). Code that is valid Python but uses anything else is reported as such,
so the limit of the evaluator is visible and not mistaken for a mistake of the model.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

from . import executor

# Error kinds, for the invalid-output breakdown.
EMPTY = "empty"
FENCED = "code_fence"
SYNTAX = "syntax"
OUTSIDE_SUBSET = "outside_subset"  # valid Python that the evaluator does not run
RUNTIME = "runtime"  # an undefined name, a division by zero, an overflow, a complex result
NO_ANS = "no_ans"
NOT_A_NUMBER = "not_a_number"  # `ans` is neither a number nor a comparison

ROUND_DIGITS = executor.ROUND_DIGITS
FUNCTIONS = {"round": round, "abs": abs, "min": min, "max": max, "sum": sum}
_BINARY = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a**b, ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
_COMPARE = {
    ast.Gt: lambda a, b: a > b, ast.Lt: lambda a, b: a < b, ast.GtE: lambda a, b: a >= b,
    ast.LtE: lambda a, b: a <= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}


class PythonError(Exception):
    """A program that cannot be parsed or run. `kind` is one of the constants above."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# --- translation -------------------------------------------------------------------------------------------

_OPERATORS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "**", "greater": ">"}
_PRECEDENCE = {">": 0, "+": 1, "-": 1, "*": 2, "/": 2, "**": 3}


@dataclass(frozen=True)
class _Node:
    text: str  # the source of an atom, or the operator of a binary node
    left: "_Node | None" = None
    right: "_Node | None" = None
    precedence: float = 9  # an atom binds tighter than any operator


def _literal(token: str) -> _Node:
    """A DSL argument as a Python operand: a number as written, a percentage as a quotient, a constant as a number."""
    text = token.strip().replace(",", "")
    if text.startswith("const_"):
        text = "-1" if text == "const_m1" else text[len("const_"):]
    elif text.endswith("%"):
        return _Node("/", _literal(text[:-1]), _Node("100"), _PRECEDENCE["/"])
    executor.parse_number(text)  # raises ProgramError for anything that is not a number
    if re.fullmatch(r"-?\d+", text):
        text = str(int(text))
    return _Node(text, precedence=0.5 if text.startswith("-") else 9)


def _render(node: _Node, parent: float = -1, side: str = "", parent_op: str = "") -> str:
    """The source of a node, with the parentheses that keep the structure of the DSL program."""
    if node.left is None:
        text = node.text
    else:
        text = f"{_render(node.left, node.precedence, 'left', node.text)} {node.text} " \
               f"{_render(node.right, node.precedence, 'right', node.text)}"
    needs = node.precedence < parent or (
        node.precedence == parent and (side == "right" or parent_op == "**")
    )
    return f"({text})" if needs and node.precedence < 9 else text


def to_python(program: str) -> str:
    """Translate a DSL program to Python.

    A step that is used once is inlined; one that is used more than once is assigned to `v<k>` first. Steps the
    last step does not depend on are dropped. The last line is `ans = ...`. The structure is kept exactly, so the
    arithmetic is the same operations in the same order as the DSL's.

    Args:
        program: A DSL program.

    Returns:
        The Python source, one assignment per line.

    Raises:
        executor.ProgramError: If the program cannot be parsed.
    """
    steps = executor.parse_program(program)
    if not steps:
        return f"ans = {_render(_literal(program))}"
    needed = {len(steps) - 1}
    for i in reversed(range(len(steps))):
        if i in needed:
            needed |= {int(a[1:]) for a in steps[i].args if a.startswith("#")}
    uses = {k: 0 for k in needed}
    for i in needed:
        for a in steps[i].args:
            if a.startswith("#"):
                uses[int(a[1:])] += 1
    nodes: dict[int, _Node] = {}
    lines = []
    for i in sorted(needed):
        op = _OPERATORS[steps[i].op]
        operands = [nodes[int(a[1:])] if a.startswith("#") else _literal(a) for a in steps[i].args]
        node = _Node(op, operands[0], operands[1], _PRECEDENCE[op])
        if uses[i] >= 2:
            lines.append(f"v{i} = {_render(node)}")
            node = _Node(f"v{i}")
        nodes[i] = node
    lines.append(f"ans = {_render(nodes[len(steps) - 1])}")
    return "\n".join(lines)


# --- safe evaluation ---------------------------------------------------------------------------------------

def _number(value, what: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PythonError(RUNTIME, f"{what} needs a number")
    return value


def _evaluate(node: ast.AST, names: dict):
    """The value of an expression node, by the whitelist. Numbers are floats, so no integer can grow without bound."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        try:
            return float(node.value)
        except OverflowError:
            raise PythonError(RUNTIME, "number too large") from None
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise PythonError(RUNTIME, f"name {node.id!r} is not defined")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _number(_evaluate(node.operand, names), "a sign")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        a = _number(_evaluate(node.left, names), "arithmetic")
        b = _number(_evaluate(node.right, names), "arithmetic")
        try:
            result = _BINARY[type(node.op)](a, b)
        except ZeroDivisionError:
            raise PythonError(RUNTIME, "division by zero") from None
        except OverflowError:
            raise PythonError(RUNTIME, "overflow") from None
        if isinstance(result, complex):
            raise PythonError(RUNTIME, "complex result")
        return result
    if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
        left = _number(_evaluate(node.left, names), "a comparison")
        for op, right_node in zip(node.ops, node.comparators):
            right = _number(_evaluate(right_node, names), "a comparison")
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_number(_evaluate(e, names), "a list") for e in node.elts]
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        args = [_evaluate(a, names) for a in node.args]
        if node.func.id == "round" and len(args) == 2 and isinstance(args[1], float) and args[1].is_integer():
            args[1] = int(args[1])  # every number is a float here, and `round` wants an int for the digits
        try:
            return FUNCTIONS[node.func.id](*args)
        except (TypeError, ValueError, OverflowError) as e:
            raise PythonError(RUNTIME, f"{node.func.id}: {e}") from None
    raise PythonError(OUTSIDE_SUBSET, f"{type(node).__name__} is not in the evaluator's subset")


def execute_python(code: str) -> executor.Value:
    """Run a Python program and return `ans`: a number rounded to 5 places, or `"yes"`/`"no"` for a comparison.

    Args:
        code: The program, exactly as the model wrote it (a fenced block is an error).

    Returns:
        The value of `ans`.

    Raises:
        PythonError: If the program cannot be parsed or run, or does not assign a number or a comparison to `ans`.
    """
    if not code.strip():
        raise PythonError(EMPTY, "empty program")
    if "```" in code:
        raise PythonError(FENCED, "the program is wrapped in a code fence")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as e:
        raise PythonError(SYNTAX, str(e)) from None
    names: dict = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            names[stmt.targets[0].id] = _evaluate(stmt.value, names)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and type(stmt.op) in _BINARY:
            names[stmt.target.id] = _evaluate(ast.BinOp(ast.Name(stmt.target.id), stmt.op, stmt.value), names)
        else:
            raise PythonError(OUTSIDE_SUBSET, f"{type(stmt).__name__} is not in the evaluator's subset")
    if "ans" not in names:
        raise PythonError(NO_ANS, "the program does not assign ans")
    value = names["ans"]
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, (int, float)):
        raise PythonError(NOT_A_NUMBER, "ans is not a number")
    if not math.isfinite(value):
        raise PythonError(RUNTIME, "ans is not finite")
    return round(float(value), ROUND_DIGITS)


def try_execute_python(code: str) -> tuple[executor.Value | None, str | None]:
    """`execute_python` that returns `(value, None)` or `(None, error_kind)` instead of raising."""
    try:
        return execute_python(code), None
    except PythonError as e:
        return None, e.kind
