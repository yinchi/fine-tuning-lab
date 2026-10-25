"""What is recorded about how and where an artifact was built, and noticing when that has drifted."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

TRACKED_LIBRARIES = ("torch", "transformers", "peft", "trl")
"""Libraries whose versions are recorded. A change warns; it does not make an artifact stale."""


def library_versions() -> dict[str, str | None]:
    """The installed versions of the tracked libraries (`None` for one that is not installed)."""
    versions: dict[str, str | None] = {}
    for lib in TRACKED_LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def python_minor() -> str:
    """The running Python's major.minor version, e.g. `"3.14"`."""
    return sys.version.split()[0].rpartition(".")[0]


def git(root: Path, *args: str) -> str | None:
    """Run a git command in `root` and return its output, or `None` if git or the repository is missing."""
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty(root: Path) -> bool | None:
    """Whether the working tree has uncommitted changes, or `None` if that cannot be determined."""
    status = git(root, "status", "--porcelain")
    return None if status is None else bool(status)


def provenance(root: Path) -> dict:
    """Where this build is running: git commit and cleanliness, library versions, Python version.

    Recorded in `node.json` so a result can be traced to the code and environment that produced it. It
    never affects freshness.
    """
    return {
        "git_commit": git(root, "rev-parse", "HEAD"),
        "git_dirty": git_dirty(root),
        "library_versions": library_versions(),
        "python": python_minor(),
    }


def drift_notes(node: dict) -> list[str]:
    """Differences between the environment recorded in `node` and the current one, as warnings."""
    notes = []
    built = node.get("library_versions", {})
    for lib, version in library_versions().items():
        if built.get(lib) != version:
            notes.append(f"{lib} was {built.get(lib)} when built, now {version}")
    if node.get("python") != python_minor():
        notes.append(f"built with Python {node.get('python')}")
    return notes
