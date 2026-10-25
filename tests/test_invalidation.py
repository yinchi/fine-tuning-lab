"""What makes an artifact stale, and what must not.

Only the code of a builder's own function (and what it imports), its inputs and its parameters
count. Editing the file a builder lives in must not invalidate it.
"""

import importlib
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import pytest

from convfinqa import artifacts as A
from convfinqa.artifacts import hashing
from convfinqa.artifacts.policy import Policy

FENCE = "```"

DATA = '''@artifact("data")
def build_data(out: Path, deps: Deps):
    (out / "d.txt").write_text("v1")
'''
TRAIN = '''@artifact("train", deps=["data"], heavy=True)
def build_train(out: Path, deps: Deps):
    (out / "m.txt").write_text("model of " + deps["data"].file("d.txt").read_text())
'''
TABLE = '''@artifact("table", deps=["train"])
def build_table(out: Path, deps: Deps):
    (out / "t.txt").write_text("table of " + deps["train"].file("m.txt").read_text())
'''
OTHER = '''@artifact("other")
def build_other(out: Path, deps: Deps):
    (out / "o.txt").write_text("o")
'''


def chapter(*builders, extra="", prose="Prose."):
    code = "\n".join(builders)
    return (
        f"---\ntitle: t\n---\n\n{prose}\n\n{FENCE}{{python}}\nfrom pathlib import Path\n"
        f"from convfinqa.artifacts import artifact, get, Deps\n\n{code}\n{FENCE}\n\n{extra}"
    )


@pytest.fixture
def book(tmp_path, monkeypatch):
    for name in ("RUN_HEAVY", "ALLOW_STALE", "QUARTO_DOCUMENT_FILE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "docs").mkdir()

    class Book:
        root = tmp_path

        def write(self, name, text):
            (tmp_path / "docs" / f"{name}.qmd").write_text(text)

        def read(self, name):
            return (tmp_path / "docs" / f"{name}.qmd").read_text()

        def registry(self, modules=()):
            reg = A.Registry(
                tmp_path / "artifacts", root=tmp_path, chapters_dir=tmp_path / "docs",
                modules=list(modules), extra_paths=[str(tmp_path)],
            )
            reg.catalog.load()
            return reg

    return Book()


def kinds(reg, *names):
    return {n: reg.status(n).kind for n in names}


BUILD_ALL = Policy(run_heavy=True)


# --- editing the chapter file --------------------------------------------------------------------

BASE_EXTRA = f"{FENCE}{{python}}\nplot_title = 'results'\n{FENCE}\n"


@pytest.mark.parametrize("what, change", [
    ("prose", lambda t: t.replace("Prose.", "Completely different prose, and a lot more of it.")),
    ("a plain cell", lambda t: t.replace("plot_title = 'results'", "plot_title = 'new'\nprint(plot_title)")),
    ("a comment and a constant next to the builders", lambda t: t.replace(
        "from pathlib import Path\n", "from pathlib import Path  # new\nimport math\nLIMIT = 3\n")),
    ("an unrelated builder added in the same cell", lambda t: t.replace(
        f"\n{FENCE}\n\n{FENCE}{{python}}\nplot",
        f'\n@artifact("extra")\ndef build_extra(out: Path, deps: Deps):\n    pass\n{FENCE}\n\n{FENCE}{{python}}\nplot')),
    ("a get call added", lambda t: t + f"\n{FENCE}{{python}}\nd = get('data')\n{FENCE}\n"),
])
def test_editing_the_chapter_file_invalidates_nothing(book, what, change):
    book.write("301_a", chapter(DATA, TRAIN, OTHER, extra=BASE_EXTRA))
    reg = book.registry()
    reg.get("train", policy=BUILD_ALL)
    reg.get("other")
    book.write("301_a", change(book.read("301_a")))
    assert kinds(book.registry(), "data", "train", "other") == {n: "fresh" for n in ("data", "train", "other")}


def test_moving_a_builder_to_another_chapter_invalidates_nothing(book):
    book.write("301_a", chapter(DATA, TRAIN, OTHER))
    reg = book.registry()
    reg.get("train", policy=BUILD_ALL)
    reg.get("other")
    book.write("301_a", chapter(TRAIN, OTHER))
    book.write("302_b", chapter(DATA))
    assert kinds(book.registry(), "data", "train", "other") == {n: "fresh" for n in ("data", "train", "other")}


def test_editing_a_builder_invalidates_it_and_what_is_built_from_it_but_not_its_neighbours(book):
    book.write("301_a", chapter(DATA, TRAIN, OTHER))
    book.registry().get("train", policy=BUILD_ALL)
    book.registry().get("other")
    book.write("301_a", book.read("301_a").replace('write_text("o")', 'write_text("changed")'))
    assert kinds(book.registry(), "data", "train", "other") == {"data": "fresh", "train": "fresh", "other": "stale"}


# --- builders in modules --------------------------------------------------------------------------

MODULE = '''
from pathlib import Path
from convfinqa.artifacts import artifact, Deps

@artifact("a")
def build_a(out: Path, deps: Deps):
    (out / "a.txt").write_text("a")

@artifact("b")
def build_b(out: Path, deps: Deps):
    (out / "b.txt").write_text("b")
'''


@pytest.fixture
def module_book(book, monkeypatch):
    name = f"inv_mod_{abs(hash(str(book.root)))}"
    path = book.root / f"{name}.py"
    path.write_text(MODULE)
    monkeypatch.syspath_prepend(str(book.root))
    importlib.invalidate_caches()

    def edit(old, new):
        path.write_text(path.read_text().replace(old, new))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()

    book.module_name, book.edit_module, book.module_path = name, edit, path
    yield book
    sys.modules.pop(name, None)


def test_editing_one_module_builder_leaves_the_others_in_the_file_fresh(module_book):
    reg = module_book.registry([module_book.module_name])
    reg.get("a"), reg.get("b")
    module_book.edit_module('write_text("b")', 'write_text("B changed")')
    reg = module_book.registry([module_book.module_name])
    assert kinds(reg, "a", "b") == {"a": "fresh", "b": "stale"}


def test_other_code_in_the_module_file_invalidates_nothing(module_book):
    reg = module_book.registry([module_book.module_name])
    reg.get("a"), reg.get("b")
    module_book.edit_module("from pathlib import Path\n", "from pathlib import Path\nimport math\nSETTING = 1\n\n"
                            "def helper():\n    return SETTING\n")
    assert kinds(module_book.registry([module_book.module_name]), "a", "b") == {"a": "fresh", "b": "fresh"}


def test_a_module_builder_must_be_self_contained(module_book):
    module_book.edit_module("@artifact(\"a\")", "LIMIT = 3\n\n@artifact(\"a\")")
    module_book.edit_module('write_text("a")', 'write_text(str(LIMIT))')
    with pytest.raises(A.ArtifactError, match=rf"{module_book.module_name}\.py:\d+: builder 'build_a' uses \['LIMIT'\]"):
        module_book.registry([module_book.module_name])


def test_a_module_builder_may_not_use_a_module_level_helper(module_book):
    module_book.edit_module("@artifact(\"a\")", "def helper():\n    return 'a'\n\n@artifact(\"a\")")
    module_book.edit_module('write_text("a")', "write_text(helper())")
    with pytest.raises(A.ArtifactError, match=r"uses \['helper'\]"):
        module_book.registry([module_book.module_name])


def test_a_module_builder_may_import_inside_its_body(module_book):
    module_book.edit_module('write_text("a")', 'write_text(__import__("os").sep)')
    module_book.edit_module('(out / "a.txt")', 'import json\n    (out / "a.txt")')
    assert module_book.registry([module_book.module_name]).get("a").file("a.txt").read_text()


def test_a_relative_import_inside_a_module_builder_is_rejected(module_book):
    module_book.edit_module('(out / "a.txt")', 'from . import sibling\n    (out / "a.txt")')
    with pytest.raises(A.ArtifactError, match="relative import"):
        module_book.registry([module_book.module_name])


def test_the_location_of_a_module_builder_is_its_def_line(module_book):
    spec = module_book.registry([module_book.module_name]).catalog.specs["b"]
    expected = module_book.module_path.read_text().splitlines().index("def build_b(out: Path, deps: Deps):") + 1
    assert spec.location == f"{module_book.module_path}:{expected}"


def test_only_the_modules_a_builder_imports_enter_its_hash(monkeypatch):
    def uses_steps(out: Path, deps: A.Deps):
        from convfinqa.dsl import steps

        return steps

    digests = defaultdict(lambda: "same")
    monkeypatch.setattr(hashing, "module_digest", lambda m: digests[m])
    from convfinqa.artifacts.spec import make_spec

    spec = make_spec("m", uses_steps, deps=(), files=(), heavy=False, version=1, seed=0)
    before = spec.code_hash()
    digests["convfinqa.data.raw"] = "edited"  # a module this builder does not import
    assert spec.code_hash() == before
    digests["convfinqa.dsl.steps"] = "edited"  # one it does
    assert spec.code_hash() != before


# --- light inputs of heavy artifacts --------------------------------------------------------------


def refactor_data(text):
    return text.replace('(out / "d.txt").write_text("v1")', 'value = "v1"\n    (out / "d.txt").write_text(value)')


def test_a_refactored_light_input_does_not_make_the_heavy_artifact_stale(book):
    book.write("301_a", chapter(DATA, TRAIN))
    book.registry().get("train", policy=BUILD_ALL)
    book.write("301_a", refactor_data(book.read("301_a")))  # same output, new code
    reg = book.registry()
    assert reg.status("data").kind == "stale" and reg.status("train").kind == "stale"  # not yet judged
    h = reg.get("train", policy=Policy())  # no RUN_HEAVY: it must not need one
    assert not h.stale and reg.status("data").fresh and reg.status("train").fresh


def test_a_light_input_that_changes_its_output_makes_the_heavy_artifact_stale(book):
    book.write("301_a", chapter(DATA, TRAIN))
    book.registry().get("train", policy=BUILD_ALL)
    book.write("301_a", book.read("301_a").replace('write_text("v1")', 'write_text("v2")'))
    reg = book.registry()
    with pytest.raises(A.StaleArtifactError, match="dependency data was rebuilt"):
        reg.get("train", policy=Policy())
    assert reg.status("data").fresh  # the light input was refreshed on the way
    assert reg.get("train", policy=BUILD_ALL).file("m.txt").read_text() == "model of v2"


def test_a_missing_heavy_artifact_is_not_built_by_refreshing_light_inputs(book):
    book.write("301_a", chapter(DATA, TRAIN))
    reg = book.registry()
    with pytest.raises(A.StaleArtifactError):
        reg.get("train", policy=Policy())
    assert reg.status("data").fresh  # the light input was built
    assert reg.status("train").kind == "missing"  # the heavy one was not


# --- light artifacts built from heavy ones, under ALLOW_STALE --------------------------------------


def stale_train(book):
    book.write("301_a", chapter(DATA, TRAIN, TABLE))
    book.registry().get("table", policy=BUILD_ALL)
    book.write("301_a", book.read("301_a").replace('"model of "', '"retrained model of "'))
    return book.registry()


def test_allow_stale_loads_a_light_artifact_whose_heavy_input_is_stale(book):
    reg = stale_train(book)
    assert reg.status("table").kind == "stale"
    h = reg.get("table", policy=Policy(allow_stale=True))
    assert h.stale and h.file("t.txt").read_text() == "table of model of v1"
    assert reg.status("train").kind == "stale"  # nothing was rebuilt behind our back


def test_without_allow_stale_a_light_artifact_with_a_stale_heavy_input_is_refused(book):
    reg = stale_train(book)
    with pytest.raises(A.StaleArtifactError, match="Heavy artifact 'train' is stale"):
        reg.get("table", policy=Policy())


def test_allow_stale_cannot_load_a_light_artifact_that_was_never_built(book):
    book.write("301_a", chapter(DATA, TRAIN, TABLE))
    reg = book.registry()
    reg.get("train", policy=BUILD_ALL)  # train exists; table does not
    book.write("301_a", book.read("301_a").replace('"model of "', '"retrained model of "'))
    with pytest.raises(A.StaleArtifactError):
        book.registry().get("table", policy=Policy(allow_stale=True))


def test_run_heavy_wins_over_allow_stale(book):
    reg = stale_train(book)
    h = reg.get("table", policy=Policy(run_heavy=True, allow_stale=True))
    assert not h.stale and h.file("t.txt").read_text() == "table of retrained model of v1"


def test_a_stale_light_result_is_recorded_so_the_chapter_is_re_executed(book, monkeypatch):
    reg = stale_train(book)
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "302_b.qmd")
    reg.get("table", policy=Policy(allow_stale=True))
    record = json.loads((reg.dir / "_chapters" / "302_b.json").read_text())
    assert record["stale"] == ["table"]
    (book.root / "docs" / "_freeze" / "302_b").mkdir(parents=True)
    assert reg.records.refresh(book.root / "docs" / "_freeze", reg.status) == ["302_b"]
