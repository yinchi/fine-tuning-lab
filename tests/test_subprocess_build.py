"""The pieces of a subprocess build: its command, its environment, and reporting a failure."""

import os
import subprocess
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


def test_no_modules_is_an_empty_import_not_a_missing_one():
    cmd = command(modules=[])
    assert "--import=" in cmd and "--import" not in cmd  # otherwise the child would use the default modules


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


# --- seeds, paths and failures --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 5, 2**32 - 1, 2**32, 5_000_000_000, 2**63, -1])
def test_every_seed_gives_an_interpreter_that_can_start(seed):
    env = sb.child_environment(seed)
    assert 0 <= int(env["PYTHONHASHSEED"]) < 2**32
    result = subprocess.run([sys.executable, "-c", "print('started')"], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "started"


def test_small_seeds_are_used_as_they_are():
    assert sb.child_environment(7)["PYTHONHASHSEED"] == "7"


def test_no_pythonpath_set_gives_no_empty_entry(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert sb.child_environment(0, ["/extra1", "/extra2"])["PYTHONPATH"] == "/extra1:/extra2"
    monkeypatch.setenv("PYTHONPATH", "")
    assert sb.child_environment(0, ["/extra1"])["PYTHONPATH"] == "/extra1"


def test_the_error_output_of_a_child_that_fails_early_is_in_the_message(tmp_path):
    cmd = [sys.executable, "-c", "import no_such_module_anywhere"]
    with pytest.raises(BuildError) as info:
        sb.run("x", cmd, dict(os.environ), cwd=tmp_path, log=tmp_path / "never_created.log")
    message = str(info.value)
    assert "failed (exit code 1)" in message and "--- error output before the log started ---" in message
    assert "ModuleNotFoundError: No module named 'no_such_module_anywhere'" in message
    assert "--- end of the log ---" not in message  # there is no log to show


def test_the_message_has_both_the_log_and_the_early_output_when_both_exist(tmp_path):
    log = tmp_path / "build.log"
    log.write_text("training started\nstep 1\n")
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('early warning\\n'); sys.exit(2)"]
    with pytest.raises(BuildError) as info:
        sb.run("x", cmd, dict(os.environ), cwd=tmp_path, log=log)
    message = str(info.value)
    assert "exit code 2" in message and "--- end of the log ---\ntraining started\nstep 1" in message
    assert "early warning" in message


def test_an_empty_log_adds_no_empty_section(tmp_path):
    log = tmp_path / "build.log"
    log.write_text("")
    with pytest.raises(BuildError) as info:
        sb.run("x", [sys.executable, "-c", "import sys; sys.exit(1)"], dict(os.environ), cwd=tmp_path, log=log)
    assert "--- end of the log ---" not in str(info.value)


def test_a_child_killed_by_a_signal_says_so(tmp_path):
    cmd = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"]
    with pytest.raises(BuildError, match=r"was killed by SIGKILL \(often the out-of-memory killer\)") as info:
        sb.run("x", cmd, dict(os.environ), cwd=tmp_path, log=tmp_path / "build.log")
    assert "exit code" not in str(info.value)


def test_another_signal_is_named_without_the_memory_hint(tmp_path):
    cmd = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"]
    with pytest.raises(BuildError, match="was killed by SIGTERM; log:") as info:
        sb.run("x", cmd, dict(os.environ), cwd=tmp_path, log=tmp_path / "build.log")
    assert "memory" not in str(info.value).splitlines()[0]


def test_early_output_of_a_successful_build_is_still_shown(tmp_path, capsys):
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('a harmless warning\\n')"]
    sb.run("x", cmd, dict(os.environ), cwd=tmp_path, log=tmp_path / "build.log")
    assert "a harmless warning" in capsys.readouterr().err
