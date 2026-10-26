"""The book's cross-references: labelled chapters, resolvable references, and no plan numbers in the prose.

Quarto numbers chapters straight through the book, not by part, so a chapter's rendered number is not its
`PCC` file code or its `P.C` identifier in `PLAN.md`. The prose therefore refers to chapters with `@sec-`
labels, which Quarto turns into the right number, and to chapters that are not written yet in words.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
CHAPTERS = sorted(p for p in DOCS.glob("*.qmd") if re.match(r"\d{3}_", p.name))
REF = re.compile(r"(?<![\w@])@((?:sec|fig|tbl|eq)-[\w-]+)")
LABEL = re.compile(r"\{#((?:sec|fig|tbl|eq)-[\w-]+)")
CELL_LABEL = re.compile(r"^#\|\s*label:\s*((?:sec|fig|tbl|eq)-[\w-]+)", re.M)
PLAN_NUMBER = re.compile(r"\b[Cc]hapters? \d+\.\d+|§\s*\d|\bSection \d+\.\d+\b")


def prose(path: Path):
    """The (line number, text) of every line outside fenced code blocks, inline code removed."""
    fence = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        marker = re.match(r"\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
            continue
        if fence is None:
            yield number, re.sub(r"`[^`]*`", "", line)


def labels(path: Path) -> list[str]:
    text = path.read_text()
    return LABEL.findall(text) + CELL_LABEL.findall(text)


def test_chapters_exist():
    assert len(CHAPTERS) >= 8


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda p: p.name)
def test_every_chapter_opens_with_a_labelled_heading(chapter):
    first = chapter.read_text().splitlines()[0]
    assert re.fullmatch(r"# .+ \{#sec-[\w-]+\}", first) or chapter.name.startswith("9"), (
        f"{chapter.name}: a chapter's title must be `# Title {{#sec-...}}` (a YAML title cannot carry a label)"
    )


def test_labels_are_unique():
    seen = {}
    for chapter in CHAPTERS:
        for label in labels(chapter):
            assert label not in seen, f"{label} is defined in {seen[label]} and {chapter.name}"
            seen[label] = chapter.name


def test_every_reference_resolves():
    defined = {label for chapter in CHAPTERS for label in labels(chapter)}
    problems = [
        f"{chapter.name}:{number}: @{ref}"
        for chapter in CHAPTERS
        for number, line in prose(chapter)
        for ref in REF.findall(line)
        if ref not in defined
    ]
    assert not problems, "unresolved references:\n" + "\n".join(problems)


def test_no_plan_numbers_in_the_prose():
    """`chapter 2.4` and `§3.3` are plan identifiers; the book's own numbers differ (see the module docstring)."""
    problems = [
        f"{chapter.name}:{number}: {match.group(0)}"
        for chapter in CHAPTERS
        for number, line in prose(chapter)
        for match in PLAN_NUMBER.finditer(line)
    ]
    assert not problems, "use an @sec- reference, or words for a chapter that does not exist yet:\n" + "\n".join(problems)


def test_every_chapter_is_in_the_book():
    config = (DOCS / "_quarto.yml").read_text()
    missing = [p.name for p in CHAPTERS if p.name not in config]
    assert not missing, f"not listed in _quarto.yml: {missing}"
