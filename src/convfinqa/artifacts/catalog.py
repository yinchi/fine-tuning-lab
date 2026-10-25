"""The catalog: which artifacts exist.

Builders are declared with `@artifact`, either in modules (imported to register them) or in chapter
code cells (found by scanning the `.qmd` files). The catalog loads them, resolves artifact names to
their declarations, and validates the dependency graph. It never touches the artifacts on disk.
"""

from __future__ import annotations

import contextlib
import enum
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .chapters import builder_namespace, compile_builder, scan_chapters
from .errors import ArtifactError
from .spec import NAME, SAMPLE, Spec, make_spec

_active: Catalog | None = None


def active_catalog() -> Catalog | None:
    """The catalog that is currently loading builders, if any: where `@artifact` registers."""
    return _active


class LoadState(enum.StrEnum):
    """Whether a catalog has read its builders (from modules and chapters) yet."""

    UNLOADED = "unloaded"  # not loaded, or a load failed (so the next use tries again)
    LOADING = "loading"  # builders are being registered; `@artifact` must register, not look up
    LOADED = "loaded"


ChapterSignature = tuple[str, int, int]
"""Type alias for the signature of a chapter file.

Fields are the name, modification time (in nanoseconds), and size of the chapter file.
"""


class Catalog:
    def __init__(self, modules=(), chapters_dir: Path | None = None):
        self.modules = list(modules)
        """Default list of modules to import to register builders."""

        self.chapters_dir = Path(chapters_dir) if chapters_dir else None
        """The directory containing the .qmd files for the book's chapters, if defined.

        The catalog scans this directory's `.qmd` files to find builders and `get` calls.
        """

        self.specs: dict[str, Spec] = {}
        """Mapping from artifact template names to their specifications."""

        self.target_locations: dict[str, str] = {}
        """Mapping from concrete artifact names to their target locations as found in chapters.

        Each name requested in the chapters with `get("literal")` maps to the `<path>:<line>` of the
        first such call found.
        """

        self.unparsed_cells: list[str] = []
        """Chapter code cells that mention `artifact(` or `get(` but are not valid Python.

        Each cell is represented as the `<path>:<line>` of the cell's first code line
        (not of the error). These cells may contain a Jupyter magic or syntax errors. Builders in
        such a cell are not registered and `get` calls in it are not targets, so `resolve` mentions
        these cells when it cannot find a name.
        """

        self.load_state: LoadState = LoadState.UNLOADED
        """The current load state of the catalog."""

        self._scan_signature: tuple[ChapterSignature, ...] = ()
        """Signature of the chapter files last scanned.

        A signature is a tuple of (name, mtime, size) triples for each chapter file last scanned. It
        is empty if nothing has been scanned yet, `chapters_dir` is `None`, or it has no `.qmd` files.
        """

    def artifact(self, template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
        """Return a decorator that registers a builder under the name template `template`.

        The decorator returns the function unchanged. In a chapter cell that the scan already
        registered, the returned decorator does nothing: the scanned definition is the registered
        one.
        """

        # If the catalog is not currently loading, ensure it is loaded and check for
        # an existing registration for this template.
        if self.load_state is not LoadState.LOADING:
            self._ensure_loaded()
            existing = self.specs.get(template)
            if existing is not None and existing.chapter is not None:
                # Already registered in a chapter, skip re-registration.
                return lambda fn: fn

        # Otherwise, return a decorator that registers the builder function.
        def decorate(fn: Callable) -> Callable:
            self._register(
                template, fn, deps=deps, files=files, heavy=heavy, version=version, seed=seed
            )
            return fn

        return decorate

    def _register(
        self, template: str, fn: Callable, *, deps, files, heavy, version, seed,
        chapter: str | None = None, source: str | None = None, location: str = "",
    ) -> None:
        """Validate a builder declaration and add its `Spec` to the catalog.

        Registering the same function again (a module reload, or a chapter re-scan) replaces the
        spec; a different function for the same template raises `ArtifactError`.
        """
        spec = make_spec(
            template, fn, deps=deps, files=files, heavy=heavy, version=version, seed=seed,
            chapter=chapter, source=source, location=location,
        )

        # Check for an existing Spec for this template; if so, ensure it is not a conflicting
        # registration.
        old = self.specs.get(template)
        if old and (old.fn.__module__, old.fn.__qualname__) != (fn.__module__, fn.__qualname__):
            where = f" (also at {old.location})" if old.location else ""
            raise ArtifactError(f"artifact template {template!r} is registered twice{where}")

        # Register the new Spec for this template.
        self.specs[template] = spec

    @contextlib.contextmanager
    def _loading(self):
        """Make this the catalog that `@artifact` registers into, while builders are being loaded.

        On success the state becomes `LOADED`. If anything raises, the previous state is restored: a
        failed first load stays `UNLOADED` (the next use tries again, and raises the same error
        until it is fixed), and a failed re-scan leaves a catalog that was loaded still loaded.
        """
        global _active

        # Set the active catalog and load_state, and back up the previous values.
        _active, previous_active = self, _active
        self.load_state, previous_state = LoadState.LOADING, self.load_state

        try:
            yield
        except BaseException:
            # Restore the previous load state if an exception occurs.
            self.load_state = previous_state
            raise
        else:
            # If no exception occurs, mark the catalog as successfully loaded.
            self.load_state = LoadState.LOADED
        finally:
            # Restore the previous active catalog regardless of success or failure.
            _active = previous_active

    def load(self, modules=None) -> Catalog:
        """Register builders from the modules and the chapters' code cells, then validate the graph.

        Modules that are already imported are reloaded, which re-runs their `@artifact` decorators.

        Args:
            modules: Optional list of module names to load. If None, uses the catalog's default
                modules.
        """
        with self._loading():
            for m in self.modules if modules is None else modules:
                if m in sys.modules:
                    importlib.reload(sys.modules[m])
                else:
                    importlib.import_module(m)
            if self.chapters_dir is not None:
                self._load_chapters()
            self.validate()
        return self

    def _chapters_signature(self) -> tuple[ChapterSignature, ...]:
        """Compute signatures for each chapter file in `self.chapters_dir`."""
        if self.chapters_dir is None:
            return ()
        return tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(self.chapters_dir.glob("*.qmd"))
        )

    def _load_chapters(self) -> None:
        """Load artifact specifications from the chapter files in `self.chapters_dir`."""
        # Forget what an earlier scan registered, so builders removed from a chapter disappear.
        self.specs = {t: sp for t, sp in self.specs.items() if sp.chapter is None}
        self.target_locations = {}

        # Taken before scanning, so a file that changes during the scan is noticed next time; it is
        # only stored once the scan has succeeded, so a failed scan is retried.
        signature = self._chapters_signature()

        scan = scan_chapters(self.chapters_dir)

        self.unparsed_cells = scan.unparsed

        for name, where in scan.gets:
            # Use setdefault to ensure each target name maps to its first encountered location only.
            self.target_locations.setdefault(name, where)

        # Compile and register each builder from the scan (registering checks its purity)
        for b in scan.builders:
            fn = compile_builder(b, builder_namespace(b.chapter))
            self._register(
                b.template, fn, deps=b.options.get("deps", ()), files=b.options.get("files", ()),
                heavy=b.options.get("heavy", False), version=b.options.get("version", 1),
                seed=b.options.get("seed", 0), chapter=b.chapter, source=b.source,
                location=b.location,
            )

        self._scan_signature = signature

    def targets(self) -> list[str]:
        """Return the concrete artifact names requested by chapters, sorted.

        An artifact is requested via `get("literal")` in a chapter's code cell, where `"literal"` is
        the name of the artifact. A name built at run time (an f-string) is not seen. `build --all`
        in the CLI builds every artifact requested across all chapters.
        """
        self._ensure_loaded()
        return sorted(self.target_locations)

    def _ensure_loaded(self) -> None:
        """Make sure the catalog is loaded and up to date, so lookups never see stale declarations.

        Loads on first use. A catalog that is already loaded re-reads the chapters if their files
        changed (a Jupyter kernel that Quarto keeps alive between renders would otherwise serve an
        edited builder stale). Does nothing while a load is already running.
        """
        if self.load_state is LoadState.LOADING:
            return
        if self.load_state is LoadState.UNLOADED:
            self.load()
        elif self.chapters_dir is not None and self._scan_signature != self._chapters_signature():
            with self._loading():
                self._load_chapters()
                self.validate()

    # names ......................................................................................

    def resolve(self, name: str) -> tuple[Spec, dict[str, Any]]:
        """Resolve a concrete artifact name to the `Spec` it matches and its parameters.

        Returns `(spec, params)`, where `params` maps each field of the name to its value, converted
        to the field's type: `resolve("sft_dsl_0")` gives `{"variant": "dsl", "seed": 0}`. Raises
        `ArtifactError` if the name is not valid, matches no template, or matches several.
        """
        self._ensure_loaded()
        if not NAME.fullmatch(name):
            raise ArtifactError(f"{name!r} is not a valid artifact name (letters, digits and underscores only)")
        matches = []
        for spec in self.specs.values():
            if (m := spec.pattern.fullmatch(name)):
                try:
                    params = {f: spec.fields[f](v) for f, v in m.groupdict().items()}
                except ValueError:
                    continue
                matches.append((spec, params))
        if not matches:
            hint = ""
            if self.unparsed_cells:
                hint = (
                    "; these chapter cells could not be parsed, so builders in them are not registered: "
                    + ", ".join(self.unparsed_cells)
                )
            raise ArtifactError(f"no artifact is registered for the name {name!r}{hint}")
        if len(matches) > 1:
            raise ArtifactError(f"the name {name!r} matches several templates: {[s.template for s, _ in matches]}")
        return matches[0]

    def dependencies(self, name: str) -> list[str]:
        """List the concrete names of the artifacts that `name` is built from (direct dependencies only)."""
        spec, params = self.resolve(name)
        return spec.dep_names(params)

    def validate(self, targets=()) -> None:
        """Validate the artifact catalog; raise `ArtifactError` if it is inconsistent.

        Checks that every dependency template resolves to a registered artifact and that there are no
        cycles. Each name in `targets`, together with its whole chain of dependencies, must also
        resolve; a failure is prefixed with the `<path>:<line>` of the chapter's `get` call.
        """
        graph: dict[str, list[str]] = {}
        for tpl, spec in self.specs.items():
            sample = {f: SAMPLE[t] for f, t in spec.fields.items()}
            sample = {f: spec.fields[f](v) for f, v in sample.items()}
            edges = []
            for d in spec.deps:
                try:
                    dep_spec, _ = self.resolve(d.format(**sample))
                except (KeyError, ArtifactError) as e:
                    raise ArtifactError(f"{tpl!r} depends on {d!r}, which does not resolve: {e}") from None
                edges.append(dep_spec.template)
            graph[tpl] = edges
        state: dict[str, int] = {}

        def visit(node: str, trail: list[str]) -> None:
            if state.get(node) == 1:
                raise ArtifactError("dependency cycle: " + " -> ".join(trail + [node]))
            if state.get(node) == 2:
                return
            state[node] = 1
            for nxt in graph[node]:
                visit(nxt, trail + [node])
            state[node] = 2

        for node in graph:
            visit(node, [])
        for name in targets:
            try:
                self._walk(name)
            except ArtifactError as e:
                where = self.target_locations.get(name)
                raise ArtifactError(f"{where}: {e}" if where else str(e)) from None

    def _walk(self, name: str) -> list[str]:
        """List `name` and everything it depends on, dependencies first (concrete names)."""
        order: list[str] = []

        def visit(n: str) -> None:
            if n in order:
                return
            for d in self.dependencies(n):
                visit(d)
            order.append(n)

        visit(name)
        return order
