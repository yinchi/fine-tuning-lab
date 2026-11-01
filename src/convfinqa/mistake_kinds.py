"""Which kinds of difference separate a generated program from the gold program, and how few explain it.

Four kinds of difference are tested, each by a normalisation applied to both programs:

- `constants`: `const_100` is read as `100` and `const_m1` as `-1`;
- `zeros`: trailing zeros of a number are dropped (`100.00` is `100`);
- `signs`: minus signs on numbers are dropped;
- `order`: the arguments of subtract and divide may come in either order.

A program that is equal to the gold program after a set of normalisations has differences only of those kinds.
A turn is explained by a set if that set makes the programs equal (`program_match.equal_program`), and the
*minimal* sets are those with no smaller set that also works. This says how the programs differ, not why the model
wrote what it wrote.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

from .dsl import executor
from .program_match import equal_program

KINDS = ("constants", "zeros", "signs", "order")


def _constants(program: str) -> str:
    return re.sub(r"const_(m?)(\d+)", lambda m: ("-" if m.group(1) else "") + m.group(2), program)


def _zeros(program: str) -> str:
    return re.sub(r"\d+\.\d+", lambda m: m.group(0).rstrip("0").rstrip("."), program)


def _signs(program: str) -> str:
    return re.sub(r"(?<![\w.])-(?=\d)", "", program)


NORMALISERS = {"constants": _constants, "zeros": _zeros, "signs": _signs}
"""The text normalisations; `order` is not a text rewrite and is passed to `equal_program` instead."""


def explained_by(predicted: str, gold: str, kinds: frozenset[str]) -> bool:
    """Whether the programs are equal once the normalisations named in `kinds` are applied to both."""
    p, g = predicted, gold
    for name, fix in NORMALISERS.items():
        if name in kinds:
            p, g = fix(p), fix(g)
    return equal_program(p, g, order_free="order" in kinds)


def minimal_explanations(predicted: str, gold: str) -> list[frozenset[str]]:
    """The smallest sets of kinds that make the generated program equal to the gold one.

    Args:
        predicted: The model's output.
        gold: The gold program.

    Returns:
        The minimal sets (none that contains another), smallest first; `[frozenset()]` if the programs are
        already equal, and `[]` if no combination of the four kinds makes them equal (another kind of difference,
        or an output that is not a program).
    """
    working: list[frozenset[str]] = []
    for size in range(len(KINDS) + 1):
        for combo in combinations(KINDS, size):
            kinds = frozenset(combo)
            if not any(w <= kinds for w in working) and explained_by(predicted, gold, kinds):
                working.append(kinds)
    return working


def count_minimal_sets(pairs: list[tuple[str, str]]) -> dict:
    """Count how many programs each combination of kinds explains, and no smaller combination does.

    Args:
        pairs: `(output, gold program)` for each turn.

    Returns:
        `equal` (programs that already match), `by_set` (a count for each of the combinations that is a minimal
        explanation of a turn, keyed by the sorted tuple of kinds; a turn with several minimal sets counts under
        each), `multiple` (turns with more than one minimal set), `unexplained` (no combination works), and `n`.
    """
    by_set: Counter = Counter()
    equal = multiple = unexplained = 0
    for output, gold in pairs:
        sets = minimal_explanations(output, gold)
        if sets == [frozenset()]:
            equal += 1
        elif not sets:
            unexplained += 1
        else:
            multiple += len(sets) > 1
            for s in sets:
                by_set[tuple(k for k in KINDS if k in s)] += 1
    return {"n": len(pairs), "equal": equal, "by_set": dict(by_set), "multiple": multiple, "unexplained": unexplained}
