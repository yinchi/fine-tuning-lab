"""The literal check: the numbers a generated program is written with, against the gold program's and the page's.

A *literal* is a number written in the output, as text (`2.70`, `17%`, `const_100`). The check is made on the text
of an output, so it works for any output format (DSL arguments or the constants of Python code), and whether the
output runs does not matter. Three measures per turn:

- `text`: the literals, as exact text, are the same as the gold program's (in any order, with repeats counted);
- `value`: the literals, as values, are the same (`2.70` is `2.7`, `17%` is `0.17`, `const_100` is `100`);
- `grounded`: every literal that is not a named constant comes from the page (it is a number of the page, as text).

The gap between the first two measures is formatting (trailing zeros, percent forms, constants); the third says
whether the numbers were copied from the page. A named constant (`const_100`) is not on the page by design, so it
is left out of the third measure. A sign and a percent sign are not part of the number for that measure, because the
gold writes a loss as a negative number where the page shows it in parentheses.
"""

from __future__ import annotations

import re
from collections import Counter

_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_LITERAL = re.compile(rf"const_m?\d+|(?<![\w.]){'-?'}{_NUMBER}%?")
_PAGE_NUMBER = re.compile(rf"(?<![\w.]){_NUMBER}")


def literals(output: str) -> list[str]:
    """The numbers written in an output, as text, without thousands separators.

    `#k` references are not literals. A number inside a word (`x1`) is not a literal.

    A comma followed by exactly three digits is a thousands separator, as on the page (`206,588`). The
    DSL separates arguments with `, ` (comma and space), which is not read that way; an output that writes
    `add(5,100)` without the space is read as the one number `5100`. No stored output has that form.

    Args:
        output: The model's text for one turn.

    Returns:
        The literals in order of appearance, such as `["206588", "const_100", "2.70", "17%"]`.
    """
    return [m.replace(",", "") for m in _LITERAL.findall(re.sub(r"#\d+", " ", output))]


def value(literal: str) -> float:
    """The value of a literal: a percent divided by 100, a constant as its number."""
    if literal.startswith("const_"):
        name = literal[len("const_"):]
        return -1.0 if name == "m1" else float(name)
    if literal.endswith("%"):
        return float(literal[:-1]) / 100
    return float(literal)


def page_numbers(page: str) -> set[str]:
    """The numbers of a page (its text, table and questions), as text without thousands separators."""
    return {m.replace(",", "") for m in _PAGE_NUMBER.findall(page)}


def _magnitude(literal: str) -> str:
    return literal.lstrip("-").rstrip("%")


def literal_flags(output: str, gold_program: str, page: set[str]) -> dict:
    """The three measures of the literal check for one turn.

    Args:
        output: The model's text.
        gold_program: The gold program.
        page: The numbers of the page, from `page_numbers`.

    Returns:
        `n_literals` (in the output), `lit_text`, `lit_value` and `lit_grounded` (see the module). An output with
        no literal is grounded; it has the same literals as a gold program only if that has none either.
    """
    mine, gold = literals(output), literals(gold_program)
    return {
        "n_literals": len(mine),
        "lit_text": Counter(mine) == Counter(gold),
        "lit_value": Counter(round(value(x), 9) for x in mine) == Counter(round(value(x), 9) for x in gold),
        "lit_grounded": all(_magnitude(x) in page for x in mine if not x.startswith("const_")),
    }
