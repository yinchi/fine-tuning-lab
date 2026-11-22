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


GLOSSARY = DOCS / "901_glossary.qmd"
GLOSSARY_LINK = re.compile(r"901_glossary\.qmd#(gloss-[\w-]+)")
ENTRY = re.compile(r"^\[([^\]]+)\]\{#(gloss-[\w-]+)\}\n:\s+(.+?)(?=\n\n|\Z)", re.M | re.S)


def glossary_entries() -> list[tuple[str, str, str]]:
    """The (term, id, definition) of every glossary entry, in file order."""
    return ENTRY.findall(GLOSSARY.read_text())


def test_glossary_entries_are_well_formed():
    text = GLOSSARY.read_text()
    entries = glossary_entries()
    assert len(entries) == len(re.findall(r"\{#gloss-", text)), "an entry is not `[term]{#gloss-id}` then `:   definition`"
    ids = [i for _, i, _ in entries]
    assert len(ids) == len(set(ids)), "duplicate glossary id"
    for term, _, definition in entries:
        assert re.search(r"See\s+@sec-[\w-]+", definition), f"{term}: an entry ends with the chapter that develops the term"


def test_glossary_is_alphabetical():
    terms = [t.lower() for t, _, _ in glossary_entries()]
    assert terms == sorted(terms)


def test_every_glossary_link_resolves_and_every_entry_is_linked():
    """A chapter links a term where the book introduces it, so an entry nothing links to is dead."""
    defined = {i for _, i, _ in glossary_entries()}
    linked = set()
    problems = []
    for chapter in CHAPTERS:
        if chapter == GLOSSARY:
            continue
        for number, line in enumerate(chapter.read_text().splitlines(), 1):
            for target in GLOSSARY_LINK.findall(line):
                linked.add(target)
                if target not in defined:
                    problems.append(f"{chapter.name}:{number}: {target}")
    assert not problems, "links to missing glossary entries:\n" + "\n".join(problems)
    assert defined <= linked, f"glossary entries no chapter links to: {sorted(defined - linked)}"
