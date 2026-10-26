"""The train-fit / val split of `train.json`, grouped by report page."""

from __future__ import annotations

import random
import re


def page_group(conv_id: str) -> str:
    """The report page a conversation was written from.

    The `Single_`/`Double_` prefix and the `-N` suffix are removed, so
    `Single_JKHY/2009/page_28.pdf-3` becomes `JKHY/2009/page_28.pdf`.
    """
    return re.sub(r"-\d+$", "", re.sub(r"^(Single|Double)_", "", conv_id))


def make_val_split(
    convs: list[dict], fraction: float = 0.1, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Split conversations into `(train_fit, val)`, keeping every page on one side.

    About `fraction` of the *pages* (not of the conversations) are held out, chosen with a
    generator seeded by `seed`. The input order is kept inside each part.

    Args:
        convs: Conversations, each with an `id`.
        fraction: Share of pages to hold out.
        seed: Seed of the page draw.

    Returns:
        The training conversations and the validation conversations.
    """
    pages = sorted({page_group(c["id"]) for c in convs})
    held = set(random.Random(seed).sample(pages, round(fraction * len(pages))))
    val = [c for c in convs if page_group(c["id"]) in held]
    fit = [c for c in convs if page_group(c["id"]) not in held]
    return fit, val
