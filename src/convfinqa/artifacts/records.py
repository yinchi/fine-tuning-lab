"""Which artifacts each chapter used, so a chapter can be re-executed when they change.

Quarto's freeze cache watches only a chapter's `.qmd`, so a chapter whose artifact was rebuilt would
otherwise keep showing the old result. While a chapter runs, `ChapterRecord.note_use` writes
`artifacts/_chapters/<chapter>.json` listing the artifacts it loaded (with their output hashes) and
whether any were loaded stale. `ChapterRecords.refresh`, run before every render, deletes the freeze
folder of every chapter whose record no longer matches.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from .errors import ArtifactError
from .handles import Handle, Status


class ChapterRecord:
    """What one chapter used: artifact name to the output hash it had, and which were loaded stale."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.artifacts: dict[str, str] = {}
        self.stale: set[str] = set()

    @property
    def chapter(self) -> str:
        """The chapter's name: its `.qmd` file's stem."""
        return self.path.stem

    @classmethod
    def load(cls, path: Path) -> ChapterRecord:
        """Read a record file written by an earlier run."""
        data = json.loads(Path(path).read_text())
        record = cls(path)
        record.artifacts = dict(data["artifacts"])
        record.stale = set(data.get("stale", []))
        return record

    def note_use(self, handle: Handle) -> None:
        """Add an artifact the chapter loaded, and write the record."""
        self.artifacts[handle.name] = handle.hash
        if handle.stale:
            self.stale.add(handle.name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"artifacts": self.artifacts, "stale": sorted(self.stale)}, indent=2))

    def is_outdated(self, status: Callable[[str], Status]) -> bool:
        """Whether the chapter must be re-executed.

        True if it loaded a stale result, or if an artifact it used is no longer fresh, no longer
        exists, or has a different output hash than when the chapter ran. `status` maps an artifact
        name to its `Status`; it may raise `ArtifactError` for a name that no longer resolves.
        """
        if self.stale:
            return True
        for name, recorded_hash in self.artifacts.items():
            try:
                st = status(name)
            except ArtifactError:
                return True
            if not st.fresh or st.output_hash != recorded_hash:
                return True
        return False


class ChapterRecords:
    """The records of all chapters, in one directory, plus the ones this process has been writing."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        # Starts empty in each process, so an artifact a chapter no longer uses drops out of its record.
        self._records: dict[str, ChapterRecord] = {}

    def note_use(self, handle: Handle, document: str | None = None) -> None:
        """Record that the running chapter loaded `handle`.

        `document` is the chapter file; by default the `QUARTO_DOCUMENT_FILE` variable Quarto sets.
        Outside Quarto there is no chapter and nothing is recorded.
        """
        document = document or os.environ.get("QUARTO_DOCUMENT_FILE")
        if not document:
            return
        chapter = Path(document).stem
        if chapter not in self._records:
            self._records[chapter] = ChapterRecord(self.directory / f"{chapter}.json")
        self._records[chapter].note_use(handle)

    def refresh(self, freeze_dir: Path, status: Callable[[str], Status]) -> list[str]:
        """Delete the freeze folder of every chapter that used an artifact that is no longer current.

        Returns the chapters that were marked, and removes their records: a chapter writes a new one
        when it re-executes.
        """
        removed = []
        for path in sorted(self.directory.glob("*.json")):
            record = ChapterRecord.load(path)
            if record.is_outdated(status):
                shutil.rmtree(freeze_dir / record.chapter, ignore_errors=True)
                path.unlink()
                self._records.pop(record.chapter, None)
                removed.append(record.chapter)
        return removed
