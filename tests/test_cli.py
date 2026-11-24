"""The command line: build, status and refresh, through `cli.main`."""

import json
import sys

import pytest

from convfinqa.artifacts import cli

FENCE = "```"
CHAPTER = f"""---
title: t
---

{FENCE}{{python}}
from pathlib import Path
from convfinqa.artifacts import artifact, get, Deps

@artifact("one")
def build_one(out: Path, deps: Deps):
    (out / "v.txt").write_text("1")

@artifact("two", deps=["one"])
def build_two(out: Path, deps: Deps):
    (out / "v.txt").write_text("2" + deps["one"].file("v.txt").read_text())

@artifact("slow", heavy=True)
def build_slow(out: Path, deps: Deps):
    print("heavy builder output")
    (out / "v.txt").write_text("slow")

a = get("two")
b = get("slow")
{FENCE}
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "301_a.qmd").write_text(CHAPTER)

    class Project:
        root = tmp_path
        artifacts = tmp_path / "artifacts"

        def run(self, *args):
            return cli.main(["--artifacts-dir", str(self.artifacts), "--root", str(tmp_path), "--import=", *args])

    return Project()


def test_build_all_builds_what_the_chapters_ask_for_in_dependency_order(project, capsys):
    assert project.run("build", "--all") == 0
    assert "[artifacts] 2 target(s) fresh" in capsys.readouterr().out
    for name in ("one", "two", "slow"):
        assert (project.artifacts / name / "node.json").exists()
    assert (project.artifacts / "two" / "v.txt").read_text() == "21"


def test_build_by_name_builds_that_artifact_and_what_it_needs_only(project):
    assert project.run("build", "two") == 0
    assert (project.artifacts / "one" / "node.json").exists() and (project.artifacts / "two" / "node.json").exists()
    assert not (project.artifacts / "slow").exists()


def test_build_force_rebuilds_a_fresh_artifact(project):
    project.run("build", "one")
    first = json.loads((project.artifacts / "one" / "node.json").read_text())["built_at"]
    import time

    time.sleep(1.1)
    project.run("build", "one", "--force")
    assert json.loads((project.artifacts / "one" / "node.json").read_text())["built_at"] != first


def test_build_needs_names_or_all(project, capsys):
    with pytest.raises(SystemExit) as info:
        project.run("build")
    assert info.value.code == 2 and "give artifact names or --all" in capsys.readouterr().err


def test_build_with_names_and_all_is_a_mistake_not_a_silent_choice(project, capsys):
    with pytest.raises(SystemExit) as info:
        project.run("build", "two", "--all")
    assert info.value.code == 2 and "not both" in capsys.readouterr().err
    assert not project.artifacts.exists()  # nothing was built


def test_an_unknown_artifact_is_an_error_with_exit_code_1(project, capsys):
    assert project.run("build", "nonesuch") == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_status_shows_a_tree_and_exits_0_by_default(project, capsys):
    assert project.run("status") == 0
    out = capsys.readouterr().out
    assert "two  [missing: not built]" in out and "  one  [missing: not built]" in out  # one is shown under two
    assert "slow  [missing: not built]" in out


def test_status_check_exits_1_unless_everything_shown_is_fresh(project, capsys):
    assert project.run("status", "--check") == 1
    project.run("build", "--all")
    capsys.readouterr()
    assert project.run("status", "--check") == 0
    assert "[fresh]" in capsys.readouterr().out


def test_status_of_named_artifacts_only(project, capsys):
    project.run("build", "one")
    capsys.readouterr()
    assert project.run("status", "one", "--check") == 0  # the others being missing does not matter
    assert project.run("status", "two", "--check") == 1


def test_status_with_no_chapters_has_no_targets(tmp_path, capsys):
    code = cli.main(["--artifacts-dir", str(tmp_path / "a"), "--root", str(tmp_path), "--chapters-dir=", "--import=", "status"])
    assert code == 0 and "(no targets)" in capsys.readouterr().out


def test_refresh_marks_a_chapter_whose_artifact_changed(project, capsys, monkeypatch):
    project.run("build", "--all")
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "301_a.qmd")
    from convfinqa.artifacts import Registry

    reg = Registry(project.artifacts, root=project.root, chapters_dir=project.root / "docs")
    reg.get("two")  # the chapter's run records what it used
    monkeypatch.delenv("QUARTO_DOCUMENT_FILE")
    (project.root / "docs" / "_freeze" / "301_a").mkdir(parents=True)
    capsys.readouterr()
    assert project.run("refresh") == 0 and capsys.readouterr().out == ""  # nothing changed: silent
    # change what "one" produces, rebuild it, and the chapter's record no longer matches
    qmd = project.root / "docs" / "301_a.qmd"
    qmd.write_text(qmd.read_text().replace('write_text("1")', 'write_text("changed")'))
    project.run("build", "one")
    capsys.readouterr()
    assert project.run("refresh") == 0
    assert "chapters marked for re-execution: 301_a" in capsys.readouterr().out
    assert not (project.root / "docs" / "_freeze" / "301_a").exists()


def test_a_heavy_build_runs_in_a_subprocess_and_logs_there(project):
    assert project.run("build", "slow") == 0
    assert "heavy builder output" in (project.artifacts / "slow" / "build.log").read_text()


def test_the_help_lists_the_commands_and_hides_the_internal_one(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "build" in out and "status" in out and "refresh" in out and "_run" not in out


def test_the_subprocess_gets_no_modules_when_the_parent_has_none(project):
    # the build subprocess is told `--import=`, so it registers the same (empty) set of modules
    from convfinqa.artifacts import subprocess_build as sb

    cmd = sb.child_command("x", artifacts_dir=project.artifacts, root=project.root, chapters_dir=None, modules=[], force=False)
    assert "--import=" in cmd and "--import" not in cmd


def test_refresh_drops_the_freeze_of_a_chapter_that_has_no_record(project, capsys):
    """A clone has the committed freeze but not `artifacts/`: its saved results are not known to match."""
    project.run("build", "--all")
    freeze = project.root / "docs" / "_freeze" / "301_a"
    freeze.mkdir(parents=True)
    capsys.readouterr()
    assert project.run("refresh") == 0
    assert "chapters marked for re-execution: 301_a" in capsys.readouterr().out
    assert not freeze.exists()


def test_refresh_changes_nothing_when_the_chapters_do_not_load(project, capsys):
    """A chapter mid-edit must not make every chapter lose its saved results."""
    freeze = project.root / "docs" / "_freeze" / "301_a"
    freeze.mkdir(parents=True)
    records = project.artifacts / "_chapters"
    records.mkdir(parents=True)
    (records / "301_a.json").write_text(json.dumps({"artifacts": {"one": "h"}, "stale": []}))
    (project.root / "docs" / "302_b.qmd").write_text(CHAPTER)  # defines "one" a second time
    assert project.run("refresh") == 1
    assert "error:" in capsys.readouterr().err
    assert freeze.exists() and (records / "301_a.json").exists()
