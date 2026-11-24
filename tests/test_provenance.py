"""What is recorded about a build's environment, and noticing drift."""

import subprocess
import sys

import pytest

from convfinqa.artifacts import provenance as pv


def test_library_versions_covers_the_tracked_libraries():
    versions = pv.library_versions()
    assert tuple(versions) == pv.TRACKED_LIBRARIES
    assert all(v is None or isinstance(v, str) for v in versions.values())


def test_python_minor_is_major_dot_minor():
    assert pv.python_minor() == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_provenance_has_the_recorded_fields(tmp_path):
    record = pv.provenance(tmp_path)  # not a git repository
    assert set(record) == {"git_commit", "git_dirty", "library_versions", "python", "inputs"}
    assert record["inputs"]["dataset_commit"] is None and set(record["inputs"]["models"]) == set(pv.MODELS)
    assert record["git_commit"] is None and record["git_dirty"] is None
    assert record["python"] == pv.python_minor()


def test_no_drift_when_the_environment_is_unchanged(tmp_path):
    assert pv.drift_notes(pv.provenance(tmp_path)) == []


def test_drift_notes_name_the_library_and_python_differences(tmp_path):
    node = pv.provenance(tmp_path)
    node["library_versions"] = {**node["library_versions"], "torch": "0.0.1"}
    node["python"] = "2.7"
    notes = pv.drift_notes(node)
    assert any(n.startswith("torch was 0.0.1 when built") for n in notes)
    assert "built with Python 2.7" in notes


def test_a_node_without_recorded_versions_reports_installed_libraries_as_drift():
    installed = [lib for lib, v in pv.library_versions().items() if v is not None]
    notes = pv.drift_notes({"python": pv.python_minor()})
    assert len(notes) == len(installed)


def test_the_tracked_libraries_cover_the_pipeline():
    for lib in ("torch", "transformers", "tokenizers", "datasets", "accelerate", "peft", "trl", "xgrammar",
                "sympy", "numpy"):
        assert lib in pv.TRACKED_LIBRARIES
    for lib in ("tensorboard", "pandas", "matplotlib"):  # they only report
        assert lib not in pv.TRACKED_LIBRARIES


def test_a_library_missing_then_or_now_is_described_as_not_installed():
    node = {"python": pv.python_minor(), "library_versions": {"torch": None, "peft": "9.9.9"}}
    notes = pv.drift_notes(node)
    assert "torch was not installed when built, now " + pv.library_versions()["torch"] in notes
    assert f"peft was 9.9.9 when built, now {pv.library_versions()['peft']}" in notes


# --- git ---------------------------------------------------------------------------------------------


def run_git(path, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=path, check=True,
        capture_output=True,
    )


def test_git_is_unknown_outside_a_repository_not_clean(tmp_path):
    assert pv.git(tmp_path, "rev-parse", "HEAD") is None
    assert pv.git_dirty(tmp_path) is None  # not False: "clean" would claim something we do not know


def test_git_in_a_repository_reports_the_commit_and_whether_the_tree_is_dirty(tmp_path):
    run_git(tmp_path, "init", "-q")
    assert pv.git_dirty(tmp_path) is False
    (tmp_path / "a.txt").write_text("a")
    assert pv.git_dirty(tmp_path) is True  # an untracked file counts
    run_git(tmp_path, "add", "a.txt")
    run_git(tmp_path, "commit", "-q", "-m", "first")
    commit = pv.git(tmp_path, "rev-parse", "HEAD")
    assert len(commit) == 40 and pv.git_dirty(tmp_path) is False
    (tmp_path / "a.txt").write_text("changed")
    assert pv.git_dirty(tmp_path) is True
    assert pv.provenance(tmp_path)["git_commit"] == commit


def test_git_is_called_without_optional_locks_and_with_a_timeout(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="out\n", stderr="")

    monkeypatch.setattr(pv.subprocess, "run", fake_run)
    assert pv.git(tmp_path, "status", "--porcelain") == "out"
    [(cmd, kwargs)] = calls
    assert cmd == ["git", "--no-optional-locks", "status", "--porcelain"]
    assert kwargs["timeout"] == pv.GIT_TIMEOUT_SECONDS and kwargs["cwd"] == tmp_path


@pytest.mark.parametrize("error", [
    subprocess.TimeoutExpired("git", 10), subprocess.CalledProcessError(128, "git"), FileNotFoundError("git"),
])
def test_a_git_that_hangs_fails_or_is_missing_gives_unknown(tmp_path, monkeypatch, error):
    def fake_run(cmd, **kwargs):
        raise error

    monkeypatch.setattr(pv.subprocess, "run", fake_run)
    assert pv.git(tmp_path, "rev-parse", "HEAD") is None
    assert pv.git_dirty(tmp_path) is None


def test_model_revisions_come_from_the_cache_and_a_changed_one_is_noted(tmp_path, monkeypatch):
    ref = tmp_path / "models--Qwen--Qwen3-4B-Instruct-2507" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert pv.model_revisions()["Qwen/Qwen3-4B-Instruct-2507"] == "abc123"
    assert pv.model_revisions()["Qwen/Qwen2.5-0.5B-Instruct"] is None  # not cached
    node = pv.provenance(tmp_path)
    assert pv.drift_notes(node) == []
    ref.write_text("def456\n")
    assert any("Qwen3-4B-Instruct-2507 was revision abc123 when built, now def456" in n for n in pv.drift_notes(node))
