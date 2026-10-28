"""Few-shot example sets: which training conversations are shown in the prompt."""

from __future__ import annotations

import random

from .splits import page_group


def conversation_type_of(conv_id: str) -> str:
    """`II` for a conversation made of two merged questions (its id starts with `Double_`), else `I`."""
    return "II" if conv_id.startswith("Double_") else "I"


def draw_example_sets(ids: list[str], seeds: list[int], types: list[str]) -> dict[str, list[str]]:
    """Draw one example set per seed from a pool of conversation ids.

    Each set is built position by position from a seeded shuffle of the pool: the position takes the first
    conversation in the shuffle that has the type asked for there. So a set of 2 is the first 2 of the set of 3
    with the same seed. No two examples, within a set or across sets, come from the same report page: several
    conversations are written from one page, and two of them in a prompt would show the same table twice. The
    draw depends only on the pool, the seeds and the types, never on the data that will be evaluated.

    Args:
        ids: The pool (the training conversations), in any order; it is sorted first.
        seeds: One seed per set.
        types: The conversation type (`I` or `II`) at each position of a set, in prompt order.

    Returns:
        The ids of each set, keyed by the seed as a string (JSON-friendly), in prompt order.
    """
    pool = sorted(ids)
    taken_pages: set[str] = set()
    sets = {}
    for seed in seeds:
        shuffled = list(pool)
        random.Random(seed).shuffle(shuffled)
        chosen = []
        for wanted in types:
            pick = next((i for i in shuffled if conversation_type_of(i) == wanted and page_group(i) not in taken_pages), None)
            assert pick is not None, "the pool is too small for these sets"
            chosen.append(pick)
            taken_pages.add(page_group(pick))
        sets[str(seed)] = chosen
    return sets
