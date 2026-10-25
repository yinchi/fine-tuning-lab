"""The artifacts directory: where each artifact's files, `node.json`, log and lock live.

Layout, for an artifact `name`:

    <dir>/name/            the finished artifact: its outputs, `build.log`, and `node.json` (written last)
    <dir>/name.tmp/        a build in progress, or a failed one (its log is kept for inspection)
    <dir>/name.old/        the previous version, for the instant of the swap
    <dir>/.locks/name.lock one builder at a time
    <dir>/_chapters/       the chapters' records (see `records.py`)

A build goes `lock` -> `scratch` -> (the builder fills the scratch directory) -> `commit`. `node.json`
is what makes an artifact exist: it is written last, so a build that stops at any earlier point leaves
nothing that looks finished. `node.json` and `build.log` are reserved names in an artifact's directory.

POSIX only: the lock uses `fcntl.flock`.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
from collections.abc import Generator
from pathlib import Path, PurePosixPath

from .errors import ArtifactError
from .handles import Handle
from .hashing import LOG_NAME, hash_outputs, sha256_json


class Store:
    """The artifacts directory, as methods: paths, reading what was built, and building safely.

    An artifact's name must be a plain file name (no `/`, not `.` or `..`, not empty), because it
    becomes a directory name and `scratch` deletes directories. The catalog only resolves names made
    of letters, digits and underscores; the store checks again, so a bad name cannot reach the disk
    through it. Several processes can use one store at once: `lock` serialises builds of the same
    artifact, and `commit` swaps a finished build in with renames.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        """The artifacts directory."""

    @staticmethod
    def _checked(name: str) -> str:
        """Return `name` if it is a plain file name, else raise `ArtifactError`."""
        if not name or name in (".", "..") or PurePosixPath(name).name != name or "\0" in name:
            raise ArtifactError(f"{name!r} is not a plain artifact name and cannot be used as a directory")
        return name

    def path(self, name: str) -> Path:
        """Where the finished artifact `name` lives."""
        return self.directory / self._checked(name)

    def scratch_path(self, name: str) -> Path:
        """Where a build of `name` is carried out before it is swapped into place."""
        return self.directory / f"{self._checked(name)}.tmp"

    def old_path(self, name: str) -> Path:
        """Where the previous version waits during the swap, and is deleted straight after."""
        return self.directory / f"{self._checked(name)}.old"

    def log_path(self, name: str) -> Path:
        """The log of the build in progress (or the last failed one) for `name`."""
        return self.scratch_path(name) / LOG_NAME

    def read_node(self, name: str) -> dict | None:
        """The parsed `node.json` of `name`, or `None` if the artifact counts as not built.

        That includes a file that is missing, cannot be read, is not text or not JSON, or is JSON that
        is not a node (it needs `output_hash` and an `outputs` mapping). Treating a damaged node as
        "not built" is the safe direction: the artifact is rebuilt instead of trusted.
        """
        try:
            node = json.loads((self.path(name) / "node.json").read_text())
        except (OSError, ValueError):  # missing, unreadable, not UTF-8, not JSON
            return None
        valid = isinstance(node, dict) and "output_hash" in node and isinstance(node.get("outputs"), dict)
        return node if valid else None

    def handle(self, name: str, *, stale: bool = False) -> Handle:
        """A handle to the built artifact `name`; raise `ArtifactError` if it has not been built.

        `stale` marks a result that is being used although it is out of date (`ALLOW_STALE`); the store
        does not judge freshness, so it is whatever the caller says.
        """
        if (node := self.read_node(name)) is None:
            raise ArtifactError(f"artifact {name!r} has not been built")
        return Handle(name, self.path(name), node["output_hash"], stale, frozenset(node["outputs"]))

    @contextlib.contextmanager
    def lock(self, name: str) -> Generator[None, None, None]:
        """Hold the exclusive lock for building `name`, waiting for as long as another process has it.

        The lock is advisory (`flock`): it only keeps out code that also calls `lock`. The operating
        system releases it when the process ends, however it ends, so a crashed build never leaves a
        stale lock. It is not re-entrant: taking it again for the same name, in the same process,
        waits for itself.
        """
        lock_name = f"{self._checked(name)}.lock"  # before anything is created on disk
        locks = self.directory / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / lock_name).open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield

    def scratch(self, name: str) -> Path:
        """A new, empty directory to build `name` in, after clearing leftovers of earlier attempts.

        That deletes the previous attempt's directory, log included, and any `name.old`.
        """
        scratch = self.scratch_path(name)
        for leftover in (scratch, self.old_path(name)):
            shutil.rmtree(leftover, ignore_errors=True)
        scratch.mkdir(parents=True)
        return scratch

    def commit(self, name: str, record: dict) -> None:
        """Finish a build: add the outputs' hashes to `record`, write it as `node.json`, and swap in.

        `record` becomes `node.json` with `outputs` (file path to SHA-256, see `hashing.is_output` for
        what is left out) and `output_hash` added. It is written last and atomically, then the scratch
        directory replaces the old artifact with two renames: the old one is moved aside, the new one
        moved into place. If the second rename fails the old artifact is put back, and the error is
        raised; the finished build stays in the scratch directory.

        A build that stops before this point leaves no `node.json`, so it counts as not built. A crash
        in the instant between the two renames leaves the artifact missing (the old version in
        `name.old`, the new one in `name.tmp`), which the next build repairs by rebuilding.
        """
        scratch, final, old = self.scratch_path(name), self.path(name), self.old_path(name)
        if not scratch.is_dir():
            raise ArtifactError(f"no build of {name!r} is in progress: there is no {scratch.name} directory")
        outputs = hash_outputs(scratch)
        node = {**record, "outputs": outputs, "output_hash": sha256_json(outputs)}
        part = scratch / "node.json.part"
        part.write_text(json.dumps(node, indent=2))
        os.replace(part, scratch / "node.json")
        shutil.rmtree(old, ignore_errors=True)  # a leftover would make the first rename fail
        if final.exists():
            os.replace(final, old)
        try:
            os.replace(scratch, final)
        except OSError:
            if old.exists() and not final.exists():
                os.replace(old, final)  # roll back: keep the previous version
            raise
        shutil.rmtree(old, ignore_errors=True)
