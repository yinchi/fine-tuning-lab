"""The page a conversation is about: the three fields of a dataset entry that the model reads."""

from __future__ import annotations

import json
from pathlib import Path

PAGE_KEYS = ("pre_text", "table", "post_text")
"""The fields a page needs; every other field of a dataset entry is ignored."""


class PageError(ValueError):
    """A page that does not have the shape the model was trained on."""


def _problems(obj) -> list[str]:
    if not isinstance(obj, dict):
        return [f"the page must be a JSON object with the keys {', '.join(PAGE_KEYS)}, not {type(obj).__name__}"]
    problems = [f"missing `{key}`" for key in PAGE_KEYS if key not in obj]
    for key in ("pre_text", "post_text"):
        if key in obj and not (isinstance(obj[key], list) and all(isinstance(s, str) for s in obj[key])):
            problems.append(f"`{key}` must be a list of strings")
    table = obj.get("table")
    if "table" in obj:
        if not isinstance(table, list) or not table:
            problems.append("`table` must be a non-empty list of rows (the header row first)")
        else:
            bad = [i for i, row in enumerate(table) if not (isinstance(row, list) and all(isinstance(c, str) for c in row))]
            if bad:
                problems.append(f"`table` rows must be lists of strings; row {bad[0]} is not (of {len(bad)} such rows)")
    return problems


def validate_page(obj) -> dict:
    """The page held in `obj`: a JSON object with `pre_text`, `table` and `post_text`.

    Extra fields (`id`, `annotation`, `table_ori`, ...) are allowed and dropped, so a dataset entry is a valid page.
    The text is not cleaned: the model has seen the dataset's own conventions and no others.

    Args:
        obj: The parsed JSON.

    Returns:
        A new dict with the three fields only.

    Raises:
        PageError: With every problem found, if the shape is wrong.
    """
    problems = _problems(obj)
    if problems:
        raise PageError("; ".join(problems))
    return {key: obj[key] for key in PAGE_KEYS}


def load_page(path: str | Path) -> dict:
    """Read a page from a JSON file (see `validate_page`).

    Raises:
        PageError: If the file cannot be read, is not JSON, or has the wrong shape.
    """
    try:
        obj = json.loads(Path(path).read_text())
    except OSError as e:
        raise PageError(f"cannot read {path}: {e.strerror or e}") from e
    except json.JSONDecodeError as e:
        raise PageError(f"{path} is not valid JSON: {e}") from e
    try:
        return validate_page(obj)
    except PageError as e:
        raise PageError(f"{path}: {e}") from e
