"""Dev numbers appear in one chapter only (PLAN.md, "Where dev appears: one chapter")."""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
FINAL = "402_the_final.qmd"  # the final-results chapter, the only one that may ask for dev artifacts
# get("...") or get(f"...") whose name has "dev" as a word of its own: scores_tuned_dev_main_0_pred, ...
DEV_GET = re.compile(r"""\bget\(\s*f?(["'])(?:[^"']*_)?dev(?:_[^"']*)?\1""")
CELL = re.compile(r"^```\{python\}.*?^```", re.S | re.M)


def chapters_before_final():
    return [p for p in sorted(DOCS.glob("[0-9]*.qmd")) if p.name != FINAL]


def test_the_pattern_finds_dev_names_and_only_those():
    assert DEV_GET.search('x = get("scores_tuned_dev_main_0_pred")')
    assert DEV_GET.search("x = get(f'tuned_dev_{run}_0_pred')")
    assert not DEV_GET.search('get("scores_tuned_val_main_0_pred")')
    assert not DEV_GET.search('get("device_map")')
    assert not DEV_GET.search('get("development")')


def test_no_chapter_before_the_final_one_asks_for_a_dev_artifact():
    offenders = []
    for path in chapters_before_final():
        for cell in CELL.findall(path.read_text()):
            offenders += [f"{path.name}: {m.group(0)}" for m in DEV_GET.finditer(cell)]
    assert not offenders, offenders


# any string with `_dev_` in it, as the dev artifacts' names have: it catches a name that is built, or kept in a
# variable, before it reaches `get` (the pattern above sees only a literal inside the call)
DEV_NAME = re.compile(r"""["'][^"'\n]*_dev_[^"'\n]*["']""")


def test_no_chapter_before_the_final_one_writes_a_dev_artifact_name_anywhere():
    assert DEV_NAME.search('name = f"scores_tuned_dev_{run}_0_pred"')
    assert not DEV_NAME.search('open("dev_turn.json") or "dev.jsonl" or "device_map"')
    offenders = []
    for path in chapters_before_final():
        for cell in CELL.findall(path.read_text()):
            offenders += [f"{path.name}: {m.group(0)}" for m in DEV_NAME.finditer(cell)]
    assert not offenders, offenders
