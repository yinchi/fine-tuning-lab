"""Artifacts defined in chapter code cells: scanning, purity, line numbers, hashing, and the name rules."""

import textwrap
import traceback
from pathlib import Path

import pytest

from convfinqa import artifacts as A
from convfinqa.artifacts import chapters
from convfinqa.artifacts.registry import LoadState

FENCE = "```"


def qmd(*cells: str, prose: str = "Some prose.") -> str:
    out = ["---", "title: t", "---", "", prose, ""]
    for cell in cells:
        out += [FENCE + "{python}", textwrap.dedent(cell).strip("\n"), FENCE, ""]
    return "\n".join(out)


SETUP = """
    from pathlib import Path
    from convfinqa.artifacts import artifact, get, Deps, Files
"""

SQUARE = """
    @artifact("sq_{n}")
    def build_sq(out: Path, deps: Deps, *, n: int):
        (out / "v.txt").write_text(str(n * n))

    sq = get("sq_4")
"""


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_HEAVY", raising=False)
    monkeypatch.delenv("ALLOW_STALE", raising=False)
    monkeypatch.delenv("QUARTO_DOCUMENT_FILE", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()

    class Book:
        root = tmp_path

        def write(self, name, *cells, prose="Some prose."):
            path = docs / f"{name}.qmd"
            path.write_text(qmd(*cells, prose=prose))
            return path

        def registry(self):
            return A.Registry(tmp_path / "artifacts", root=tmp_path, chapters_dir=docs).load()

    return Book()


def line_of(path: Path, text: str) -> int:
    return next(i for i, l in enumerate(path.read_text().splitlines(), 1) if text in l)


# --- scanning ---------------------------------------------------------------------------------------


def test_scan_finds_builders_and_literal_gets(book):
    path = book.write("301_a", SETUP, SQUARE, """
        d = {}
        d.get("not_an_artifact")
        other = get(f"sq_{3}")
    """)
    scan = chapters.scan_chapters(path.parent)
    [b] = scan.builders
    assert (b.chapter, b.template, b.node.name) == ("301_a", "sq_{n}", "build_sq")
    assert b.line == line_of(path, "def build_sq")
    assert b.node.decorator_list == []
    assert [name for name, _ in scan.gets] == ["sq_4"]
    assert scan.gets[0][1] == f"{path}:{line_of(path, 'get(\"sq_4\")')}"


def test_scan_reads_decorator_options_as_literals(book):
    path = book.write("301_a", SETUP, """
        @artifact("m_{v}", deps=["sq_{n}"], files=["cfg.json"], heavy=True, version=3, seed=7)
        def build_m(out: Path, deps: Deps, files: Files, *, v: str):
            pass
    """)
    [b] = chapters.scan_chapters(path.parent).builders
    assert b.options == {"deps": ["sq_{n}"], "files": ["cfg.json"], "heavy": True, "version": 3, "seed": 7}


@pytest.mark.parametrize("decorator, message", [
    ('@artifact("a_b", colour="red")', "unexpected argument"),
    ("@artifact(name)", "string literal"),
    ('@artifact("a_b", heavy=flag)', "must be a literal"),
])
def test_scan_rejects_bad_decorators(book, decorator, message):
    book.write("301_a", SETUP, f"""
        {decorator}
        def build(out: Path, deps: Deps):
            pass
    """)
    with pytest.raises(A.ArtifactError, match=message):
        chapters.scan_chapters(book.root / "docs")


def test_cells_that_do_not_parse_are_reported_not_fatal(book):
    book.write("301_a", """
        %matplotlib inline
        x = get("sq_4")
    """)
    scan = chapters.scan_chapters(book.root / "docs")
    assert len(scan.unparsed) == 1 and scan.gets == []


# --- purity -----------------------------------------------------------------------------------------


def test_builder_using_a_chapter_global_is_rejected(book):
    book.write("301_a", SETUP, """
        SCALE = 3

        @artifact("sq_{n}")
        def build_sq(out: Path, deps: Deps, *, n: int):
            (out / "v.txt").write_text(str(n * SCALE))
    """)
    with pytest.raises(A.ArtifactError, match=r"301_a\.qmd:\d+: builder 'build_sq' uses \['SCALE'\]"):
        book.registry()


def test_builder_may_use_builtins_and_its_own_imports(book):
    path = book.write("301_a", SETUP, """
        @artifact("sq_{n}")
        def build_sq(out: Path, deps: Deps, *, n: int):
            import json
            values = [i * i for i in range(n)]
            (out / "v.json").write_text(json.dumps(sorted(values)))
    """)
    reg = book.registry()
    assert reg.get("sq_3").json("v.json") == [0, 1, 4]


# --- running chapter builders -----------------------------------------------------------------------


def test_light_chapter_builder_runs_in_process(book):
    book.write("301_a", SETUP, SQUARE)
    reg = book.registry()
    assert reg.targets() == ["sq_4"]
    assert reg.get("sq_4").file("v.txt").read_text() == "16"
    assert reg.get("sq_5").file("v.txt").read_text() == "25"  # any name matching the template works


def test_heavy_chapter_builder_runs_in_a_subprocess(book, monkeypatch):
    book.write("301_a", SETUP, """
        @artifact("h_{n}", heavy=True)
        def build_h(out: Path, deps: Deps, *, n: int):
            print("hello from the chapter builder")
            (out / "v.txt").write_text(str(n))
    """)
    reg = book.registry()
    with pytest.raises(A.StaleArtifactError):
        reg.get("h_2")
    monkeypatch.setenv("RUN_HEAVY", "1")
    h = reg.get("h_2")
    assert h.file("v.txt").read_text() == "2"
    assert "hello from the chapter builder" in (h.path / "build.log").read_text()


def test_chapter_builder_depends_on_another_chapters_artifact(book):
    book.write("301_a", SETUP, SQUARE)
    book.write("302_b", SETUP, """
        @artifact("plus_{n}", deps=["sq_{n}"])
        def build_plus(out: Path, deps: Deps, *, n: int):
            value = int(deps[f"sq_{n}"].file("v.txt").read_text())
            (out / "v.txt").write_text(str(value + 1))

        p = get("plus_3")
    """)
    reg = book.registry()
    assert reg.targets() == ["plus_3", "sq_4"]
    assert reg.get("plus_3").file("v.txt").read_text() == "10"
    assert reg.status("sq_3").fresh  # built as a dependency


def test_tracebacks_point_into_the_qmd_file(book):
    path = book.write("301_a", SETUP, """
        @artifact("bad_{n}")
        def build_bad(out: Path, deps: Deps, *, n: int):
            raise ValueError("boom")
    """)
    reg = book.registry()
    with pytest.raises(ValueError) as info:
        reg.get("bad_1")
    frames = [f for f in traceback.extract_tb(info.value.__traceback__) if f.filename == str(path)]
    assert [(f.lineno, f.line) for f in frames] == [(line_of(path, "raise ValueError"), 'raise ValueError("boom")')]


def test_decorator_in_the_running_chapter_is_a_no_op_after_the_scan(book):
    book.write("301_a", SETUP, SQUARE)
    reg = book.registry()
    scanned = reg.specs["sq_{n}"].fn

    def kernel_version(out, deps, *, n):  # what executing the cell would define
        raise AssertionError("the scanned definition is the registered one")

    assert reg.artifact("sq_{n}")(kernel_version) is kernel_version
    assert reg.specs["sq_{n}"].fn is scanned


def test_the_same_template_in_two_chapters_is_an_error(book):
    book.write("301_a", SETUP, SQUARE)
    book.write("302_b", SETUP, SQUARE)
    with pytest.raises(A.ArtifactError, match="registered twice"):
        book.registry()


# --- freshness of chapter builders ------------------------------------------------------------------

TWO = """
    @artifact("one")
    def build_one(out: Path, deps: Deps):
        (out / "v.txt").write_text("1")

    @artifact("two")
    def build_two(out: Path, deps: Deps):
        (out / "v.txt").write_text("2")
"""


def test_only_the_edited_builder_goes_stale(book):
    path = book.write("301_a", SETUP, TWO)
    reg = book.registry()
    reg.get("one"), reg.get("two")

    path.write_text(path.read_text().replace("Some prose.", "Different prose."))
    reg = book.registry()
    assert reg.status("one").fresh and reg.status("two").fresh  # prose and layout don't matter

    path.write_text(path.read_text().replace('write_text("2")', 'write_text("22")'))
    reg = book.registry()
    assert reg.status("one").fresh
    assert reg.status("two").kind == "stale" and reg.status("two").reason == "code changed"


def test_cosmetic_edits_to_a_builder_keep_it_fresh(book):
    path = book.write("301_a", SETUP, TWO)
    reg = book.registry()
    reg.get("one")
    text = path.read_text().replace(
        '(out / "v.txt").write_text("1")', '(out / "v.txt").write_text(  "1"  )  # a comment'
    )
    path.write_text(text)
    assert book.registry().status("one").fresh


def test_a_long_lived_registry_notices_edited_chapters(book):
    path = book.write("301_a", SETUP, TWO)
    reg = book.registry()  # stays alive, like a Jupyter kernel that Quarto keeps between renders
    reg.get("one"), reg.get("two")
    assert reg.status("two").fresh and reg.targets() == []

    path.write_text(path.read_text().replace('write_text("2")', 'write_text("22")') + qmd(
        'x = get("two")'
    ))
    assert reg.status("two").kind == "stale"  # the edit was picked up without a new registry
    assert reg.targets() == ["two"]

    path.write_text(qmd(SETUP))  # both builders deleted
    with pytest.raises(A.ArtifactError, match="no artifact is registered"):
        reg.status("two")


# --- targets and error locations --------------------------------------------------------------------


def test_unknown_target_error_names_the_chapter_line(book):
    path = book.write("301_a", SETUP, SQUARE, """
        x = get("nope")
    """)
    reg = book.registry()
    with pytest.raises(A.ArtifactError, match=rf"301_a\.qmd:{line_of(path, 'get(\"nope\")')}: .*nope"):
        reg.validate(reg.targets())


# --- name rules -------------------------------------------------------------------------------------


@pytest.mark.parametrize("template", ["a-b", "a b", "a_{x:>3}", "a_{x!r}", "a_{x}{y}", "a_{1}", "a/{x}"])
def test_bad_name_templates_are_rejected(template):
    reg = A.Registry(modules=[])

    def f(out: Path, deps: A.Deps, **kw):
        pass

    with pytest.raises(A.ArtifactError, match="bad name template"):
        reg.artifact(template)(f)


@pytest.mark.parametrize("name", ["../x", "sq-4", "sq_04", "sq_x", "sq_", "sq_4_5", "sq 4", ""])
def test_names_that_do_not_fit_are_rejected(book, name):
    book.write("301_a", SETUP, SQUARE)
    with pytest.raises(A.ArtifactError):
        book.registry().resolve(name)


def test_string_fields_cannot_contain_the_separator(book):
    book.write("301_a", SETUP, """
        @artifact("m_{v}_{n}")
        def build_m(out: Path, deps: Deps, *, v: str, n: int):
            (out / "v.txt").write_text(f"{v}/{n}")
    """)
    reg = book.registry()
    assert reg.get("m_dsl_3").file("v.txt").read_text() == "dsl/3"
    for name in ("m_dsl_v2_3", "m_dsl-v2_3"):
        with pytest.raises(A.ArtifactError):
            reg.resolve(name)


def test_float_fields_are_not_allowed():
    reg = A.Registry(modules=[])

    def f(out: Path, deps: A.Deps, *, x: float):
        pass

    with pytest.raises(A.ArtifactError, match="str or int"):
        reg.artifact("a_{x}")(f)


# --- load state -------------------------------------------------------------------------------------

IMPURE = """
    LIMIT = 3

    @artifact("sq_{n}")
    def build_sq(out: Path, deps: Deps, *, n: int):
        (out / "v.txt").write_text(str(n * LIMIT))
"""


def unloaded_registry(book):
    return A.Registry(book.root / "artifacts", root=book.root, chapters_dir=book.root / "docs")


def test_a_failed_first_load_can_be_retried(book):
    path = book.write("301_a", SETUP, IMPURE)
    reg = unloaded_registry(book)
    assert reg.load_state is LoadState.UNLOADED
    for _ in range(2):  # the same error each time, until the chapter is fixed
        with pytest.raises(A.ArtifactError, match="LIMIT"):
            reg.status("sq_3")
        assert reg.load_state is LoadState.UNLOADED
    path.write_text(qmd(SETUP, SQUARE))
    assert reg.status("sq_3").kind == "missing"
    assert reg.load_state is LoadState.LOADED


def test_a_failed_rescan_leaves_the_registry_loaded_and_is_retried(book):
    path = book.write("301_a", SETUP, SQUARE)
    reg = unloaded_registry(book)
    reg.get("sq_3")
    path.write_text(qmd(SETUP, IMPURE))  # an edit that breaks the chapter
    with pytest.raises(A.ArtifactError, match="LIMIT"):
        reg.status("sq_3")
    assert reg.load_state is LoadState.LOADED
    with pytest.raises(A.ArtifactError, match="LIMIT"):  # still broken, so still reported
        reg.status("sq_3")
    path.write_text(qmd(SETUP, SQUARE))
    assert reg.status("sq_3").fresh


def test_an_invalid_graph_keeps_being_reported(book):
    book.write("301_a", SETUP, """
        @artifact("needs_{n}", deps=["nothing_{n}"])
        def build_needs(out: Path, deps: Deps, *, n: int):
            pass
    """)
    reg = unloaded_registry(book)
    for _ in range(2):
        with pytest.raises(A.ArtifactError, match="does not resolve"):
            reg.resolve("needs_1")
        assert reg.load_state is LoadState.UNLOADED


def test_the_state_is_loading_while_builders_register(book):
    book.write("301_a", SETUP, SQUARE)
    reg = unloaded_registry(book)
    seen = []
    original = reg._register

    def spy(*args, **kwargs):
        seen.append(reg.load_state)
        return original(*args, **kwargs)

    reg._register = spy
    reg.load()
    assert seen == [LoadState.LOADING] and reg.load_state is LoadState.LOADED
