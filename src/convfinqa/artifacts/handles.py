"""What builders and chapters receive: handles to artifacts and input files, and freshness status."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal


@dataclass(frozen=True)
class Handle:
    """One resolved artifact. `stale` is True only for a stale result loaded under ALLOW_STALE."""

    name: str
    """The name of the artifact."""
    path: Path
    """The artifact's directory. Unlike `file()`, it also reaches files that are not recorded
    outputs, such as `build.log` and TensorBoard logs."""
    hash: str
    """The artifact's output hash: a hash of the file hashes of its recorded outputs (logs and
    TensorBoard files excluded). A rebuild with identical outputs has the same hash, which keeps
    what was built from it fresh."""
    stale: bool = False
    """True only when this handle is a stale result that was loaded on purpose (`ALLOW_STALE=1`). A
    fresh artifact is never marked stale, and a stale one that was not loaded has no handle."""
    _outputs: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)
    """The recorded outputs as relative paths with forward slashes, e.g. `"sub/w.txt"`. Set by the
    store; not part of equality or `repr`."""

    def file(self, rel: str) -> Path:
        """Path of a recorded output (logs and TensorBoard files are not outputs).

        Args:
            rel: The output's path within the artifact directory, written as recorded: relative,
                with forward slashes and no leading `./`.

        Returns:
            The full path to the output file.

        Raises:
            ValueError: If the path is absolute or attempts to escape the artifact directory.
            FileNotFoundError: If the path is not a recorded output of the artifact.
        """
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"{rel!r}: path must be relative and stay inside the artifact")
        if rel not in self._outputs:
            raise FileNotFoundError(f"{rel!r} is not a recorded output of artifact {self.name!r}")
        return self.path / rel

    def json(self, rel: str) -> Any:
        """Load a recorded output as JSON.

        Args:
            rel: The relative path of the output file within the artifact directory.

        Returns:
            The parsed JSON object.

        Raises:
            ValueError: If the path is absolute or attempts to escape the artifact directory.
            FileNotFoundError: If the path is not a recorded output of the artifact.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        return json.loads(self.file(rel).read_text())

    def jsonl(self, rel: str) -> Iterator[dict]:
        """Load a recorded output as JSON Lines.

        Args:
            rel: The relative path of the output file within the artifact directory.

        Yields:
            Parsed JSON objects, one per non-empty line.

        Raises:
            ValueError: If the path is absolute or attempts to escape the artifact directory.
            FileNotFoundError: If the path is not a recorded output of the artifact.
        """
        with self.file(rel).open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


@dataclass(frozen=True)
class InputFile:
    """One declared static input file."""

    path: Path
    """The path to the input file."""
    hash: str
    """The SHA-256 of the file's content when the build started. It is the hash recorded in the
    artifact's config, so a change to the file makes the artifact stale."""

    def read_text(self) -> str:
        """Read the contents of the input file as text."""
        return self.path.read_text()

    def json(self) -> Any:
        """Load the input file as JSON."""
        return json.loads(self.path.read_text())

    def jsonl(self) -> Iterator[dict]:
        """Load the input file as JSON Lines.

        Yields:
            Parsed JSON objects, one per non-empty line.
        """
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


class _ReadOnlyMapping(Mapping):
    """A read-only mapping from names to entries, with an error message that lists what was declared.

    It is a `Mapping`, which has no way to add or remove entries, and it keeps its own copy of the
    dict it is given, so changing that dict later has no effect. A `frozendict` would do the same if
    the project's minimum Python version ever has one.
    """

    _kind = "entry"
    """What the entries are called, singular, in error messages."""

    _kinds = "entries"
    """What the entries are called, plural, in error messages."""

    def __init__(self, items: dict[str, Any]):
        self._items = dict(items)

    def __getitem__(self, key: str) -> Any:
        """Retrieve an item from the mapping by key, or raise a KeyError if not found."""
        try:
            return self._items[key]
        except KeyError:
            declared = sorted(self._items)
            raise KeyError(f"{key!r} was not declared; declared {self._kinds}: {declared}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class Deps(_ReadOnlyMapping):
    """Handles of the declared dependencies, keyed by concrete artifact name."""

    _kind = "dependency"
    _kinds = "dependencies"


class Files(_ReadOnlyMapping):
    """Declared input files, keyed by the path as declared (relative to the project root)."""

    _kind = "file"
    _kinds = "files"


@dataclass(frozen=True)
class Status:
    """Whether an artifact is fresh, stale (and why), or missing."""

    kind: Literal["fresh", "stale", "missing"]
    """Fresh: built and up to date. Stale: built, but something it depends on changed. Missing
    never built."""

    reason: str = ""
    """Why the artifact is stale (`"code changed"`, `"dependency x was rebuilt"`, ...), or
    `"not built"` if it is missing. Empty when fresh."""

    output_hash: str | None = None
    """The recorded output hash. Set only when fresh; `None` for stale and missing."""

    notes: tuple[str, ...] = ()
    """Warnings about a fresh artifact: the libraries or Python it was built with differ from the
    current ones. They never make it stale."""

    @property
    def fresh(self) -> bool:
        """Return True if the status is 'fresh', else False."""
        return self.kind == "fresh"
