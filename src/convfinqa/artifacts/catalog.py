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

from .chapters import check_pure, compile_builder, scan_chapters
from .errors import ArtifactError
from .handles import Deps, Files
from .spec import NAME, SAMPLE, Spec, make_spec

_active: Catalog | None = None


def active_catalog() -> Catalog | None:
    """The catalog that is currently loading builders, if any: where `@artifact` registers."""
    return _active


class LoadState(enum.Enum):
    """Whether a catalog has read its builders (from modules and chapters) yet."""

    UNLOADED = "unloaded"  # not loaded, or a load failed (so the next use tries again)
    LOADING = "loading"  # builders are being registered; `@artifact` must register, not look up
    LOADED = "loaded"


class Catalog:
    def __init__(self, modules=(), chapters_dir: Path | None = None):
        self.modules = list(modules)
        """Modules to import to register builders."""

        self.chapters_dir = Path(chapters_dir) if chapters_dir else None
        """Where the chapters' `.qmd` files are scanned for builders and `get` calls, if anywhere."""

        self.specs: dict[str, Spec] = {}
        self.target_locations: dict[str, str] = {}  # literal get("name") calls found in chapters
        self.unparsed_cells: list[str] = []
        self.load_state = LoadState.UNLOADED
        self._scan_signature: tuple = ()  # (name, mtime, size) of the chapter files last scanned

    def artifact(self, template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
        """Decorator registering a builder.
        
        In a chapter cell that the scan already registered, it is a no-op.
        """
        if self.load_state is not LoadState.LOADING:
            self._ensure_loaded()
            existing = self.specs.get(template)
            if existing is not None and existing.chapter is not None:
                return lambda fn: fn  # the chapter's scanned definition is the registered one

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
        spec = make_spec(
            template, fn, deps=deps, files=files, heavy=heavy, version=version, seed=seed,
            chapter=chapter, source=source, location=location,
        )
        old = self.specs.get(template)
        if old and (old.fn.__module__, old.fn.__qualname__) != (fn.__module__, fn.__qualname__):
            where = f" (also at {old.location})" if old.location else ""
            raise ArtifactError(f"artifact template {template!r} is registered twice{where}")
        self.specs[template] = spec

    @contextlib.contextmanager
    def _loading(self):
        """Make this the catalog that `@artifact` registers into, while builders are being loaded.

        On success the state becomes `LOADED`. If anything raises, the previous state is restored: a
        failed first load stays `UNLOADED` (the next use tries again, and raises the same error until
        it is fixed), and a failed re-scan leaves a catalog that was loaded still loaded.
        """
        global _active
        previous_active, previous_state = _active, self.load_state
        _active, self.load_state = self, LoadState.LOADING
        try:
            yield
        except BaseException:
            self.load_state = previous_state
            raise
        else:
            self.load_state = LoadState.LOADED
        finally:
            _active = previous_active

    def load(self, modules=None) -> Catalog:
        """Register builders from the modules and the chapters' code cells, then validate the graph."""
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

    def _chapters_signature(self) -> tuple:
        if self.chapters_dir is None:
            return ()
        return tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(self.chapters_dir.glob("*.qmd"))
        )

    def _load_chapters(self) -> None:
        # Forget what an earlier scan registered, so builders removed from a chapter disappear.
        self.specs = {t: sp for t, sp in self.specs.items() if sp.chapter is None}
        self.target_locations = {}
        signature = self._chapters_signature()  # taken before scanning: a change during the scan is seen next time
        scan = scan_chapters(self.chapters_dir)
        self.unparsed_cells = scan.unparsed
        for name, where in scan.gets:
            self.target_locations.setdefault(name, where)
        for b in scan.builders:
            check_pure(b)
            namespace = {
                "__name__": f"convfinqa_chapters.{b.chapter}",
                "Path": Path,
                "Deps": Deps,
                "Files": Files
            }
            fn = compile_builder(b, namespace)
            self._register(
                b.template, fn, deps=b.options.get("deps", ()), files=b.options.get("files", ()),
                heavy=b.options.get("heavy", False), version=b.options.get("version", 1),
                seed=b.options.get("seed", 0), chapter=b.chapter, source=b.source,
                location=b.location,
            )
        self._scan_signature = signature  # only after success, so a failed scan is retried

    def targets(self) -> list[str]:
        """The artifact names chapters ask for with `get("literal")`: what `build --all` builds."""
        self._ensure_loaded()
        return sorted(self.target_locations)

    def _ensure_loaded(self) -> None:
        """Load on first use. A long-lived process (a Jupyter kernel that Quarto keeps alive between
        renders) also re-reads the chapters when their files change, so an edited builder is never
        served stale."""
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
            raise ArtifactError(f"no artifact is registered for the name {name!r}")
        if len(matches) > 1:
            raise ArtifactError(f"the name {name!r} matches several templates: {[s.template for s, _ in matches]}")
        return matches[0]

    def dependencies(self, name: str) -> list[str]:
        """The concrete names of the artifacts that the artifact `name` is built from."""
        spec, params = self.resolve(name)
        return spec.dep_names(params)

    def validate(self, targets=()) -> None:
        """Every dependency template resolves, there are no cycles; `targets` are resolved fully."""
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
        """`name` and everything it depends on, dependencies first (concrete names)."""
        order: list[str] = []

        def visit(n: str) -> None:
            if n in order:
                return
            for d in self.dependencies(n):
                visit(d)
            order.append(n)

        visit(name)
        return order
