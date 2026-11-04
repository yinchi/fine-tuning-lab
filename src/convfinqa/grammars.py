"""Grammars for constrained decoding, in xgrammar's EBNF: the DSL (unbounded or bounded) and the Python subset.

`constrained.DSL_GRAMMAR` is the unbounded DSL grammar and stays where it is (the earlier chapters import it).
The grammars here are for the evaluation of §3.6. A grammar guarantees syntax and not termination: with an
unbounded number rule a model can keep writing digits up to the generation limit. The *bounded* variants limit a
number to the digits that the gold programs use.
"""

from __future__ import annotations

from .constrained import DSL_GRAMMAR

MAX_INTEGER_DIGITS = 10
MAX_DECIMAL_DIGITS = 4
"""The most digits before and after the point in a number of a gold program (checked in `tests/test_grammars.py`)."""

_DSL_NUMBER = 'number    ::= "-"? ([0-9]+ ("." [0-9]+)? | "." [0-9]+) "%"?'
_BOUNDED_NUMBER = (
    f'number    ::= "-"? ([0-9]{{1,{MAX_INTEGER_DIGITS}}} ("." [0-9]{{1,{MAX_DECIMAL_DIGITS}}})? '
    f'| "." [0-9]{{1,{MAX_DECIMAL_DIGITS}}}) "%"?'
)


def dsl_grammar(bounded: bool = False) -> str:
    """The DSL grammar, with the number rule bounded to the digits of the gold programs if `bounded`."""
    if not bounded:
        return DSL_GRAMMAR
    assert _DSL_NUMBER in DSL_GRAMMAR
    return DSL_GRAMMAR.replace(_DSL_NUMBER, _BOUNDED_NUMBER)


PYTHON_GRAMMAR = r'''
root       ::= (assignment "\n")* final
assignment ::= "v" [0-9]+ " = " comparison
final      ::= "ans = " comparison
comparison ::= sum (" > " sum)?
sum        ::= product ((" + " | " - ") product)*
product    ::= power ((" * " | " / ") power)*
power      ::= atom (" ** " atom)?
atom       ::= number | "-" number | "v" [0-9]+ | "(" comparison ")" | "(-" number ")"
number     ::= [0-9]+ ("." [0-9]+)? | "." [0-9]+
'''
"""The Python that `python_format.to_python` writes: assignments to `v<k>`, then `ans = …`, with numbers, names,
`+ - * / ** >`, parentheses and negative numbers. It accepts every translated gold program of train and dev; it is narrower than the
safe evaluator (no calls, no other names), so a grammar-constrained model cannot leave the evaluator's subset."""

_PY_NUMBER = 'number     ::= [0-9]+ ("." [0-9]+)? | "." [0-9]+'
_PY_BOUNDED = (
    f'number     ::= [0-9]{{1,{MAX_INTEGER_DIGITS}}} ("." [0-9]{{1,{MAX_DECIMAL_DIGITS}}})? '
    f'| "." [0-9]{{1,{MAX_DECIMAL_DIGITS}}}'
)


def python_grammar(bounded: bool = False) -> str:
    """The Python grammar, with the number rule bounded to the digits of the gold programs if `bounded`."""
    return PYTHON_GRAMMAR.replace(_PY_NUMBER, _PY_BOUNDED) if bounded else PYTHON_GRAMMAR
