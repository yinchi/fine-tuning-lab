"""What is recorded about how and where an artifact was built, and noticing when that has drifted."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

TRACKED_LIBRARIES = (
    "torch", "transformers", "tokenizers", "datasets", "accelerate",
    "peft", "trl", "xgrammar", "sympy", "numpy",
)
"""Libraries whose versions are recorded: the ones that shape the results of the pipeline.

`tokenizers` decides the token ids and `numpy` the random streams, so they count though they are
installed as dependencies of the others. Libraries that only report (tensorboard, pandas,
matplotlib) are left out. A change warns; it does not make an artifact stale.
"""

GIT_TIMEOUT_SECONDS = 10
"""How long to wait for a git command before giving up and recording the answer as unknown."""


def library_versions() -> dict[str, str | None]:
    """The installed versions of the tracked libraries (`None` for one that is not installed).

    These are distribution versions, as `pip show` prints them, not a module's `__version__`.
    """
    versions: dict[str, str | None] = {}
    for lib in TRACKED_LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def python_minor() -> str:
    """The running Python's major.minor version, e.g. `"3.14"`."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def git(root: Path, *args: str) -> str | None:
    """Run a git command in `root` and return its output, or `None` if it cannot be answered.

    That is when git or the repository is missing, the command fails, or it takes longer than
    `GIT_TIMEOUT_SECONDS`. `--no-optional-locks` keeps `git status` from refreshing the index, so a
    build never contends with a `git commit` or an editor's git integration running at the same time.
    """
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=root, capture_output=True, text=True, check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty(root: Path) -> bool | None:
    """Whether the working tree has uncommitted changes (untracked files count), or `None` if unknown."""
    status = git(root, "status", "--porcelain")
    return None if status is None else bool(status)


def provenance(root: Path) -> dict:
    """Where this build is running: the keys `git_commit`, `git_dirty`, `library_versions`, `python`.

    Recorded in `node.json` so a result can be traced to the code and environment that produced it.
    It never affects freshness. `git_commit` and `git_dirty` are `None` when git cannot say.
    """
    return {
        "git_commit": git(root, "rev-parse", "HEAD"),
        "git_dirty": git_dirty(root),
        "library_versions": library_versions(),
        "python": python_minor(),
    }


def drift_notes(node: dict) -> list[str]:
    """Differences between the environment recorded in `node` and the current one, as warnings.

    A difference never makes an artifact stale; `get` shows these as warnings.
    """
    notes = []
    built = node.get("library_versions", {})
    for lib, version in library_versions().items():
        if built.get(lib) != version:
            then = built.get(lib) or "not installed"
            now = version or "not installed"
            notes.append(f"{lib} was {then} when built, now {now}")
    if node.get("python") != python_minor():
        notes.append(f"built with Python {node.get('python')}")
    return notes
