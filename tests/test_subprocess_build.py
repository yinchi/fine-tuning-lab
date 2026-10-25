"""The pieces of a subprocess build: its command, its environment, and reporting a failure."""

import sys
from pathlib import Path

import pytest

from convfinqa.artifacts import BuildError
from convfinqa.artifacts import subprocess_build as sb


def command(**kw):
    args = dict(artifacts_dir=Path("/a"), root=Path("/r"), chapters_dir=Path("/r/docs"), modules=["m1", "m2"], force=False)
    return sb.child_command("sft_dsl_0", **{**args, **kw})


def test_the_command_repeats_the_callers_setup_and_names_the_artifact():
    cmd = command()
    assert cmd[:3] == [sys.executable, "-m", "convfinqa.artifacts"]
    assert cmd[cmd.index("--artifacts-dir") + 1] == "/a" and cmd[cmd.index("--root") + 1] == "/r"
    assert "--chapters-dir=/r/docs" in cmd
    assert [cmd[i + 1] for i, c in enumerate(cmd) if c == "--import"] == ["m1", "m2"]
    assert cmd[-2:] == ["_run", "sft_dsl_0"]


def test_force_is_passed_after_the_artifact_name():
    assert command(force=True)[-3:] == ["_run", "sft_dsl_0", "--force"]


def test_no_chapters_directory_is_an_empty_value_in_the_equals_form():
    cmd = command(chapters_dir=None)
    assert "--chapters-dir=" in cmd and "--chapters-dir" not in cmd


def test_the_environment_pins_the_seed_and_cublas(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    env = sb.child_environment(7, ["/extra1", "/extra2"])
    assert env["PYTHONHASHSEED"] == "7"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["PYTHONPATH"].split(":") == ["/extra1", "/extra2", "/existing"]


def test_the_environment_leaves_pythonpath_alone_without_extra_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    assert sb.child_environment(0)["PYTHONPATH"] == "/existing"


def test_a_failed_build_reports_the_exit_code_and_the_end_of_the_log(tmp_path):
    log = tmp_path / "build.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)))
    with pytest.raises(BuildError, match="exit code 3") as info:
        sb.run("x", [sys.executable, "-c", "import sys; sys.exit(3)"], {}, cwd=tmp_path, log=log)
    message = str(info.value)
    assert "line 99" in message and f"line {100 - sb.LOG_TAIL_LINES}" in message and "line 69\n" not in message


def test_a_failed_build_without_a_log_still_raises(tmp_path):
    with pytest.raises(BuildError, match="exit code 1"):
        sb.run("x", [sys.executable, "-c", "import sys; sys.exit(1)"], {}, cwd=tmp_path, log=tmp_path / "nope.log")


def test_a_successful_build_prints_two_progress_lines(tmp_path, capsys):
    sb.run("x", [sys.executable, "-c", "pass"], {}, cwd=tmp_path, log=tmp_path / "build.log")
    err = capsys.readouterr().err
    assert "building x in a subprocess" in err and "built x in" in err
