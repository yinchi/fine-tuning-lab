"""Chapter records: what a chapter used, and which chapters `refresh` marks, without a registry."""

import json
from pathlib import Path

import pytest

from convfinqa.artifacts import ArtifactError, Handle, Status
from convfinqa.artifacts.records import ChapterRecord, ChapterRecords


def handle(name, hash_="h1", stale=False):
    return Handle(name, Path("unused"), hash_, stale)


def fresh(hash_):
    return Status("fresh", output_hash=hash_)


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.delenv("QUARTO_DOCUMENT_FILE", raising=False)
    return ChapterRecords(tmp_path / "_chapters")


def read(records, chapter):
    return json.loads((records.directory / f"{chapter}.json").read_text())


def test_nothing_is_recorded_outside_a_chapter(records):
    records.note_use(handle("a"))
    assert not records.directory.exists()


def test_the_chapter_comes_from_quarto_or_the_argument(records, monkeypatch):
    records.note_use(handle("a"), document="docs/201_x.qmd")
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "202_y.qmd")
    records.note_use(handle("b"))
    assert sorted(p.stem for p in records.directory.glob("*.json")) == ["201_x", "202_y"]


def test_each_chapter_records_only_its_own_uses(records):
    records.note_use(handle("a", "h1"), document="201_x.qmd")
    records.note_use(handle("b", "h2"), document="202_y.qmd")
    records.note_use(handle("c", "h3", stale=True), document="201_x.qmd")
    assert read(records, "201_x") == {"artifacts": {"a": "h1", "c": "h3"}, "stale": ["c"]}
    assert read(records, "202_y") == {"artifacts": {"b": "h2"}, "stale": []}


def test_a_record_loads_back_as_written(records):
    records.note_use(handle("a", "h1", stale=True), document="c.qmd")
    loaded = ChapterRecord.load(records.directory / "c.json")
    assert (loaded.chapter, loaded.artifacts, loaded.stale) == ("c", {"a": "h1"}, {"a"})


def test_a_record_removed_by_refresh_starts_over_when_the_chapter_runs_again(records, tmp_path):
    records.note_use(handle("a", "h1"), document="c.qmd")
    records.refresh(tmp_path / "_freeze", lambda name: Status("stale", "code changed"))
    records.note_use(handle("b", "h2"), document="c.qmd")  # the chapter re-executed, using only b
    assert read(records, "c")["artifacts"] == {"b": "h2"}


def test_a_record_accumulates_this_processs_uses(records):
    records.note_use(handle("a", "h1"), document="c.qmd")
    records.note_use(handle("b", "h2", stale=True), document="c.qmd")
    assert read(records, "c") == {"artifacts": {"a": "h1", "b": "h2"}, "stale": ["b"]}


def test_a_new_process_starts_a_fresh_record(tmp_path):
    first = ChapterRecords(tmp_path)
    first.note_use(handle("old"), document="c.qmd")
    second = ChapterRecords(tmp_path)  # the chapter no longer uses `old`
    second.note_use(handle("new"), document="c.qmd")
    assert read(second, "c")["artifacts"] == {"new": "h1"}


def test_refresh_keeps_chapters_whose_artifacts_are_unchanged(records, tmp_path):
    records.note_use(handle("a", "h1"), document="c.qmd")
    freeze = tmp_path / "_freeze"
    (freeze / "c").mkdir(parents=True)
    assert records.refresh(freeze, lambda name: fresh("h1")) == []
    assert (freeze / "c").exists() and (records.directory / "c.json").exists()


@pytest.mark.parametrize("current", [
    Status("fresh", output_hash="other"),   # rebuilt: different output
    Status("stale", "code changed"),
    Status("missing", "not built"),
])
def test_refresh_marks_chapters_whose_artifacts_changed(records, tmp_path, current):
    records.note_use(handle("a", "h1"), document="c.qmd")
    freeze = tmp_path / "_freeze"
    (freeze / "c").mkdir(parents=True)
    assert records.refresh(freeze, lambda name: current) == ["c"]
    assert not (freeze / "c").exists()
    assert not (records.directory / "c.json").exists()  # re-recorded when the chapter re-executes


def test_refresh_marks_chapters_using_an_artifact_that_no_longer_exists(records, tmp_path):
    records.note_use(handle("gone"), document="c.qmd")

    def status(name):
        raise ArtifactError("no artifact is registered")

    assert records.refresh(tmp_path / "_freeze", status) == ["c"]


def test_refresh_always_marks_a_chapter_that_used_a_stale_result(records, tmp_path):
    records.note_use(handle("a", "h1", stale=True), document="c.qmd")
    # even if the artifact looks current now, the chapter showed an old result
    assert records.refresh(tmp_path / "_freeze", lambda name: fresh("h1")) == ["c"]


def test_refresh_only_touches_the_outdated_chapters(records, tmp_path):
    records.note_use(handle("a", "h1"), document="keep.qmd")
    records.note_use(handle("b", "h2"), document="drop.qmd")
    freeze = tmp_path / "_freeze"
    for chapter in ("keep", "drop", "untracked"):
        (freeze / chapter).mkdir(parents=True)
    hashes = {"a": "h1", "b": "changed"}
    assert records.refresh(freeze, lambda name: fresh(hashes[name])) == ["drop"]
    assert (freeze / "keep").exists() and (freeze / "untracked").exists() and not (freeze / "drop").exists()
