"""The artifacts directory: where each artifact's files, `node.json`, log and lock live.

Layout, for an artifact `name`:

    <dir>/name/            the finished artifact: its outputs, `build.log`, and `node.json` (written last)
    <dir>/name.tmp/        a build in progress, or a failed one (its log is kept for inspection)
    <dir>/name.old/        the previous version, for the instant of the swap
    <dir>/.locks/name.lock one builder at a time
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
from collections.abc import Generator
from pathlib import Path

from .errors import ArtifactError
from .handles import Handle
from .hashing import LOG_NAME, hash_outputs, sha256_json


class Store:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path(self, name: str) -> Path:
        """Where the finished artifact `name` lives."""
        return self.directory / name

    def scratch_path(self, name: str) -> Path:
        """Where a build of `name` is carried out before it is swapped into place."""
        return self.directory / f"{name}.tmp"

    def log_path(self, name: str) -> Path:
        """The log of the build in progress (or the last failed one) for `name`."""
        return self.scratch_path(name) / LOG_NAME

    def read_node(self, name: str) -> dict | None:
        """The parsed `node.json` of `name`, or `None` if it has not been built (or the file is unreadable)."""
        try:
            return json.loads((self.path(name) / "node.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def handle(self, name: str, *, stale: bool = False) -> Handle:
        """A handle to the built artifact `name`; raise if it has not been built."""
        if (node := self.read_node(name)) is None:
            raise ArtifactError(f"artifact {name!r} has not been built")
        return Handle(name, self.path(name), node["output_hash"], stale, frozenset(node["outputs"]))

    @contextlib.contextmanager
    def lock(self, name: str) -> Generator[None, None, None]:
        """Hold the exclusive lock for building `name`, waiting if another process has it."""
        locks = self.directory / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / f"{name}.lock").open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield

    def scratch(self, name: str) -> Path:
        """A new, empty directory to build `name` in, after clearing leftovers of earlier attempts."""
        scratch = self.scratch_path(name)
        for leftover in (scratch, self.directory / f"{name}.old"):
            shutil.rmtree(leftover, ignore_errors=True)
        scratch.mkdir(parents=True)
        return scratch

    def commit(self, name: str, record: dict) -> None:
        """Finish a build: add the outputs' hashes to `record`, write it as `node.json`, and swap in.

        `node.json` is written last and atomically, then the scratch directory replaces the old
        artifact. A build that stops before this point leaves no `node.json`, so it counts as missing.
        """
        scratch, final, old = self.scratch_path(name), self.path(name), self.directory / f"{name}.old"
        outputs = hash_outputs(scratch)
        node = {**record, "outputs": outputs, "output_hash": sha256_json(outputs)}
        part = scratch / "node.json.part"
        part.write_text(json.dumps(node, indent=2))
        os.replace(part, scratch / "node.json")
        if final.exists():
            os.replace(final, old)
        os.replace(scratch, final)
        shutil.rmtree(old, ignore_errors=True)
