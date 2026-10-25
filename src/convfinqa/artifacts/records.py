"""Which artifacts each chapter used, so a chapter can be re-executed when they change.

Quarto's freeze cache (`docs/_freeze/<chapter>/`) watches only a chapter's `.qmd` file, so a chapter
whose artifact was rebuilt would otherwise keep showing the old result. This module closes that gap:

- While a chapter runs, every `get` records what it loaded in `artifacts/_chapters/<chapter>.json`
  (`ChapterRecords.note_use`, which fills in one `ChapterRecord` per chapter).
- Before every render, `ChapterRecords.refresh` compares each record with the artifacts as they are
  now, and deletes the freeze folder and the record of every chapter that is out of date, so that
  Quarto executes it again.

Chapters sit directly in the docs directory, so a chapter's freeze folder is `<freeze dir>/<file stem>`.
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
    """The artifacts one chapter loaded the last time it ran, kept in a file `<chapter>.json`.

    The file looks like this:

        {"artifacts": {"split_v1": "89ab...", "sft_dsl_0": "ec8a..."}, "stale": ["sft_dsl_0"]}

    `artifacts` maps each artifact the chapter loaded to the output hash it had at that moment; if the
    artifact has changed since, what the chapter shows is out of date. `stale` lists the artifacts that
    were loaded as stale results (`ALLOW_STALE=1`): the chapter then showed an old result and is out of
    date until it runs again, however the artifacts look later.

    Each run of a chapter starts a new, empty record and rewrites the whole file on every `note_use`,
    so the record describes that run only, and an artifact the chapter stopped using drops out.
    `load` reads the record an earlier run left behind.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        """The record's file, `<records directory>/<chapter>.json`."""

        self.artifacts: dict[str, str] = {}
        """Artifact name to the output hash it had when the chapter loaded it."""

        self.stale: set[str] = set()
        """The names of the artifacts the chapter loaded as stale results."""

    @property
    def chapter(self) -> str:
        """The chapter's name: its `.qmd` file's stem."""
        return self.path.stem

    @classmethod
    def load(cls, path: Path) -> ChapterRecord:
        """Read a record file written by an earlier run.

        Raises `json.JSONDecodeError`, `KeyError` or `TypeError` if the file is not a valid record.
        """
        data = json.loads(Path(path).read_text())
        record = cls(path)
        record.artifacts = dict(data["artifacts"])
        record.stale = set(data.get("stale", []))
        return record

    def note_use(self, handle: Handle) -> None:
        """Add an artifact the chapter loaded, and write the record.

        The file is replaced atomically (written beside it, then renamed), so a reader never sees a
        half-written record.
        """
        self.artifacts[handle.name] = handle.hash
        if handle.stale:
            self.stale.add(handle.name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_name(self.path.name + ".part")
        part.write_text(json.dumps({"artifacts": self.artifacts, "stale": sorted(self.stale)}, indent=2))
        os.replace(part, self.path)

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
    """The records of all the chapters, which live together in one directory (`artifacts/_chapters`).

    It does two jobs, in different processes:

    - **Writing**, in the process that runs a chapter. Every `get` calls `note_use`, which updates the
      `ChapterRecord` of the chapter being rendered, creating it empty the first time this process
      sees that chapter. Each process has its own `ChapterRecords`, so what it writes starts from
      nothing and never inherits the previous run's list.
    - **Reading**, in the pre-render hook. `refresh` loads every record file from disk, asks each
      whether its chapter is out of date, and clears the freeze folders of those that are.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        """The directory holding one `<chapter>.json` per chapter."""

        self._records: dict[str, ChapterRecord] = {}
        # The records this process has been writing, by chapter name. Separate from the files on disk so
        # that a run starts from an empty record and an artifact it no longer uses drops out.

    def note_use(self, handle: Handle, document: str | None = None) -> None:
        """Record that the running chapter loaded `handle`, writing the chapter's file at once.

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

        For each record on disk whose chapter is out of date (`ChapterRecord.is_outdated`), this removes
        `freeze_dir/<chapter>` and the record itself: the chapter writes a new record when it runs
        again. A record that cannot be read (a leftover from a crash, say) counts as out of date, since
        nothing is known about what the chapter used.

        Returns the names of the chapters it marked, in file-name order, whether or not they had a
        freeze folder.
        """
        removed = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                outdated = ChapterRecord.load(path).is_outdated(status)
            except (json.JSONDecodeError, KeyError, TypeError):
                outdated = True
            if outdated:
                shutil.rmtree(freeze_dir / path.stem, ignore_errors=True)
                path.unlink()
                self._records.pop(path.stem, None)
                removed.append(path.stem)
        return removed
