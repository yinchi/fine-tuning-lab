"""What builders and chapters receive: handles to artifacts and input files, and freshness status."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class Handle:
    """One resolved artifact. `stale` is True only for a stale result loaded under ALLOW_STALE."""

    name: str
    """The name of the artifact."""
    path: Path
    """The path to the artifact's directory."""
    hash: str
    """The hash of the artifact's contents."""
    stale: bool = False
    """Whether the artifact is stale."""
    _outputs: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)
    """The set of recorded output files."""

    def file(self, rel: str) -> Path:
        """Path of a recorded output (logs and TensorBoard files are not outputs).

        Args:
            rel: The relative path of the output file within the artifact directory.

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
    """The hash of the input file."""

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
    """Wrapper for a Mapping that enforces read-only access and provides custom error messages.
    
    Prevents access to the underlying mapping's `__setitem__` and `__delitem__` methods,
    enforcing read-only behavior (except if directly accessing the underlying mapping via
    `self._items`).

    Python 3.15 has `frozendict`, but we declare Python 3.14 compatibility for our project.
    """

    _kind = "entry"

    def __init__(self, items: dict[str, Any]):
        self._items = dict(items)

    def __getitem__(self, key: str) -> Any:
        """Retrieve an item from the mapping by key, or raise a KeyError if not found."""
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(f"{key!r} was not declared; declared {self._kind}s: {sorted(self._items)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class Deps(_ReadOnlyMapping):
    """Handles of the declared dependencies, keyed by concrete artifact name."""

    _kind = "dependency"


class Files(_ReadOnlyMapping):
    """Declared input files, keyed by the path as declared (relative to the project root)."""

    _kind = "file"


@dataclass(frozen=True)
class Status:
    """Represents the status of an artifact, indicating whether it is fresh, stale, or missing."""
    kind: str  # "fresh" | "stale" | "missing"
    """The status kind, one of "fresh", "stale", or "missing"."""
    
    reason: str = ""
    """The reason for the status, if any."""
    
    output_hash: str | None = None
    """The hash of the output, if available."""

    notes: tuple[str, ...] = ()
    """Additional notes about the status."""

    @property
    def fresh(self) -> bool:
        """Return True if the status is 'fresh', else False."""
        return self.kind == "fresh"

    @property
    def stale(self) -> bool:
        """Return True if the status is 'stale', else False."""
        return self.kind == "stale"

    @property
    def missing(self) -> bool:
        """Return True if the status is 'missing', else False."""
        return self.kind == "missing"
