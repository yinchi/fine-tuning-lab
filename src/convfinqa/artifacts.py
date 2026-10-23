"""Named, cached results ("artifacts") with recursive freshness.

An artifact is a directory ``artifacts/<name>/`` produced by a registered builder::

    @artifact("sft-{variant}-{seed}", deps=["tokenized-{variant}"], files=["configs/sft.json"],
              heavy=True, version=1)
    def build_sft(out: Path, deps: Deps, files: Files, *, variant: str, seed: int) -> None: ...

Builder purity: a builder receives only ``out`` (a scratch directory it fills), the handles of its
declared dependencies (``deps``), its declared static input files (``files``), and the fields of its
own name (its identity). Hyperparameters live in a declared file or a light config artifact.

An artifact is *fresh* when ``node.json`` exists and matches: the builder's ``version``, its config
(name fields, input-file hashes, seed), a hash of the normalized source of the modules it uses, and
the output hashes of its dependencies, each of which must itself be fresh. Bump ``version`` when a
change this cannot see (a library upgrade, a changed external resource) should invalidate results.

Heavy artifacts build in a subprocess, and only with ``RUN_HEAVY=1`` or the ``build`` command;
otherwise a stale or missing one raises :class:`StaleArtifactError`. ``ALLOW_STALE=1`` additionally
lets a stale-but-present heavy artifact load, with a visible warning.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
import typing
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(os.environ.get("CONVFINQA_ROOT") or Path(__file__).resolve().parents[2])
ARTIFACTS = Path(os.environ.get("CONVFINQA_ARTIFACTS") or ROOT / "artifacts")
DEFAULT_MODULES = ("convfinqa.builders",)
# Modules whose edits do not invalidate artifacts (the framework itself; bump `version` by hand).
_FRAMEWORK = {"convfinqa.artifacts", "convfinqa.targets"}
_TRACKED_LIBRARIES = ("torch", "transformers", "peft", "trl")
_LOG_NAME = "build.log"


class ArtifactError(Exception):
    """Base class for artifact-system errors."""


class StaleArtifactError(ArtifactError):
    """A heavy artifact is stale or missing and heavy builds are not enabled."""


class BuildError(ArtifactError):
    """A builder failed."""


# --- handles passed to builders and chapters ---------------------------------------------------


@dataclass(frozen=True)
class Handle:
    """One resolved artifact. ``stale`` is True only for a stale result loaded under ALLOW_STALE."""

    name: str
    path: Path
    hash: str
    stale: bool = False
    _outputs: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def file(self, rel: str) -> Path:
        """Path of a recorded output (logs and TensorBoard files are not outputs)."""
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"{rel!r}: path must be relative and stay inside the artifact")
        if rel not in self._outputs:
            raise FileNotFoundError(f"{rel!r} is not a recorded output of artifact {self.name!r}")
        return self.path / rel

    def json(self, rel: str) -> Any:
        return json.loads(self.file(rel).read_text())

    def jsonl(self, rel: str) -> Iterator[dict]:
        with self.file(rel).open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


@dataclass(frozen=True)
class InputFile:
    """One declared static input file."""

    path: Path
    hash: str

    def read_text(self) -> str:
        return self.path.read_text()

    def json(self) -> Any:
        return json.loads(self.path.read_text())

    def jsonl(self) -> Iterator[dict]:
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


class _ReadOnlyMapping(Mapping):
    _kind = "entry"

    def __init__(self, items: dict[str, Any]):
        self._items = dict(items)

    def __getitem__(self, key: str) -> Any:
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
    kind: str  # "fresh" | "stale" | "missing"
    reason: str = ""
    output_hash: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def fresh(self) -> bool:
        return self.kind == "fresh"


# --- hashing -----------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _is_output(rel: str) -> bool:
    """Logs and TensorBoard files are not significant outputs (they never invalidate dependents)."""
    p = PurePosixPath(rel)
    return not (rel == "node.json" or p.name == _LOG_NAME or p.suffix == ".log" or "tb" in p.parts)


def _hash_outputs(directory: Path) -> dict[str, str]:
    outputs = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            rel = full.relative_to(directory).as_posix()
            if _is_output(rel):
                outputs[rel] = _sha256_file(full)
    return outputs


class _Normalize(ast.NodeTransformer):
    """Drop what cannot change behaviour: docstrings and annotations on local variables."""

    def _strip_docstring(self, node):
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return self.generic_visit(node)

    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip_docstring

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        if node.value is None:
            return ast.Pass()
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


_norm_cache: dict[tuple[str, int, int], str] = {}


def normalized_source(source: str) -> str:
    """Source text with whitespace, comments, line continuations, docstrings and local annotations
    removed, so only behavioural edits change it."""
    tree = _Normalize().visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree))


def _module_path(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def _module_digest(module: str) -> str | None:
    path = _module_path(module)
    if path is None:
        return None
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _norm_cache:
        _norm_cache[key] = hashlib.sha256(normalized_source(path.read_text()).encode()).hexdigest()
    return _norm_cache[key]


def _local_imports(module: str) -> set[str]:
    """The ``convfinqa.*`` modules a module imports, anywhere in its source."""
    path = _module_path(module)
    if path is None:
        return set()
    spec = importlib.util.find_spec(module)
    package = module if spec.submodule_search_locations else module.rpartition(".")[0]
    found: set[str] = set()

    def add(name: str) -> None:
        if name == "convfinqa" or name.startswith("convfinqa."):
            found.add(name)

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            add(target)
            for alias in node.names:  # `from convfinqa import data` imports a submodule
                if _module_path(f"{target}.{alias.name}") is not None:
                    add(f"{target}.{alias.name}")
    return found


def module_closure(module: str) -> set[str]:
    """``module`` plus every ``convfinqa.*`` module it imports, transitively (framework excluded)."""
    seen: set[str] = set()
    todo = [module]
    while todo:
        m = todo.pop()
        if m in seen or m in _FRAMEWORK:
            continue
        seen.add(m)
        todo.extend(_local_imports(m))
    return seen


def code_hash(modules: set[str] | list[str]) -> str:
    return _sha256_json({m: _module_digest(m) for m in sorted(modules)})


def library_versions() -> dict[str, str | None]:
    out = {}
    for lib in _TRACKED_LIBRARIES:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            out[lib] = None
    return out


# --- seeding -----------------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) so a build does not depend on earlier RNG use."""
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed % (2**32))
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def _preserved_rng_state() -> Iterator[None]:
    """Restore the global RNG state afterwards, so an in-kernel build never perturbs a chapter."""
    py = random.getstate()
    np_state = sys.modules["numpy"].random.get_state() if "numpy" in sys.modules else None
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch else None
    cuda_state = torch.cuda.get_rng_state_all() if torch and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        if np_state is not None:
            sys.modules["numpy"].random.set_state(np_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# --- registry ----------------------------------------------------------------------------------


@dataclass
class Spec:
    template: str
    fn: Callable
    deps: list[str]
    files: list[str]
    heavy: bool
    version: int
    seed: int
    fields: dict[str, type]
    pattern: re.Pattern

    @property
    def module(self) -> str:
        return self.fn.__module__


_SAMPLE = {str: "x", int: "0", float: "0.0"}
_active: Registry | None = None


def artifact(template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
    """Register a builder with the active registry (the default one outside :meth:`Registry.load`)."""
    return (_active or DEFAULT).artifact(template, deps=deps, files=files, heavy=heavy, version=version, seed=seed)


class Registry:
    def __init__(self, artifacts_dir: Path = ARTIFACTS, *, root: Path = ROOT, modules=(), extra_paths=()):
        self.dir = Path(artifacts_dir)
        self.root = Path(root)
        self.modules = list(modules)
        self.extra_paths = [str(p) for p in extra_paths]
        self.specs: dict[str, Spec] = {}
        self._loaded = False
        self._chapter_seen: dict[str, str] = {}
        self._chapter_stale: set[str] = set()

    # registration ...............................................................................

    def artifact(self, template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
        def decorate(fn: Callable) -> Callable:
            fields = [f for _, f, _, _ in string.Formatter().parse(template) if f]
            hints = typing.get_type_hints(fn)
            spec_fields = {f: hints.get(f) for f in fields}
            self._check_signature(template, fn, spec_fields, bool(files))
            regex = "".join(
                re.escape(lit) + (f"(?P<{f}>.+?)" if f else "")
                for lit, f, _, _ in string.Formatter().parse(template)
            )
            spec = Spec(template, fn, list(deps), list(files), heavy, version, seed, spec_fields, re.compile(regex))
            old = self.specs.get(template)
            if old and (old.fn.__module__, old.fn.__qualname__) != (fn.__module__, fn.__qualname__):
                raise ArtifactError(f"artifact template {template!r} is registered twice")
            self.specs[template] = spec
            return fn

        return decorate

    @staticmethod
    def _check_signature(template, fn, fields, has_files):
        import inspect

        params = list(inspect.signature(fn).parameters.values())
        expected = ["out", "deps"] + (["files"] if has_files else [])
        names = [p.name for p in params]
        pos = [p for p in params if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        kw = {p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
        if [p.name for p in pos] != expected or kw != set(fields):
            raise ArtifactError(
                f"builder for {template!r} must have the signature "
                f"({', '.join(expected)}, *, {', '.join(sorted(fields))}); got ({', '.join(names)}). "
                "Builders take only out, deps, files and the fields of their own name."
            )
        for f, t in fields.items():
            if t not in _SAMPLE:
                raise ArtifactError(f"builder for {template!r}: field {f!r} needs a str, int or float annotation")

    def load(self, modules=None) -> Registry:
        """Import the builder modules, registering their artifacts here, then validate the graph."""
        global _active
        previous, _active = _active, self
        try:
            for m in self.modules if modules is None else modules:
                if m in sys.modules:
                    importlib.reload(sys.modules[m])
                else:
                    importlib.import_module(m)
        finally:
            _active = previous
        self._loaded = True
        self.validate()
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # names ......................................................................................

    def resolve(self, name: str) -> tuple[Spec, dict[str, Any]]:
        self._ensure_loaded()
        matches = []
        for spec in self.specs.values():
            m = spec.pattern.fullmatch(name)
            if m:
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

    def dep_names(self, spec: Spec, params: dict[str, Any]) -> list[str]:
        return [d.format(**params) for d in spec.deps]

    def validate(self, targets=()) -> None:
        """Every dependency template resolves, there are no cycles; ``targets`` are resolved fully."""
        graph: dict[str, list[str]] = {}
        for tpl, spec in self.specs.items():
            sample = {f: _SAMPLE[t] for f, t in spec.fields.items()}
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
            self._walk(name)

    def _walk(self, name: str) -> list[str]:
        """``name`` and everything it depends on, dependencies first (concrete names)."""
        order: list[str] = []

        def visit(n: str) -> None:
            if n in order:
                return
            spec, params = self.resolve(n)
            for d in self.dep_names(spec, params):
                visit(d)
            order.append(n)

        visit(name)
        return order

    # freshness ..................................................................................

    def _node_path(self, name: str) -> Path:
        return self.dir / name / "node.json"

    def _read_node(self, name: str) -> dict | None:
        try:
            return json.loads(self._node_path(name).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _seed(self, spec: Spec, params: dict[str, Any]) -> int:
        return int(params.get("seed", spec.seed))

    def _input_hashes(self, spec: Spec) -> dict[str, str | None]:
        out = {}
        for rel in spec.files:
            p = self.root / rel
            out[rel] = _sha256_file(p) if p.is_file() else None
        return out

    def _config(self, spec: Spec, params: dict[str, Any]) -> dict:
        return {"params": params, "seed": self._seed(spec, params), "files": self._input_hashes(spec)}

    def status(self, name: str) -> Status:
        spec, params = self.resolve(name)
        node = self._read_node(name)
        if node is None:
            return Status("missing", "not built")
        if node.get("version") != spec.version:
            return Status("stale", f"version changed ({node.get('version')} -> {spec.version})")
        config = self._config(spec, params)
        if any(h is None for h in config["files"].values()):
            missing = [f for f, h in config["files"].items() if h is None]
            return Status("stale", f"input file missing: {missing}")
        if node.get("config") != config:
            changed = [k for k in config if node.get("config", {}).get(k) != config[k]]
            return Status("stale", f"config changed ({', '.join(changed)})")
        modules = module_closure(spec.module) | set(node.get("dynamic_modules", []))
        if node.get("code_hash") != code_hash(modules):
            return Status("stale", "code changed")
        deps = self.dep_names(spec, params)
        if sorted(node.get("deps", {})) != sorted(deps):
            return Status("stale", "dependencies changed")
        for d in deps:
            ds = self.status(d)
            if not ds.fresh:
                return Status("stale", f"dependency {d} is {ds.kind}")
            if ds.output_hash != node["deps"][d]:
                return Status("stale", f"dependency {d} was rebuilt")
        notes = []
        built = node.get("library_versions", {})
        for lib, ver in library_versions().items():
            if built.get(lib) != ver:
                notes.append(f"{lib} was {built.get(lib)} when built, now {ver}")
        if node.get("python") != sys.version.split()[0].rpartition(".")[0]:
            notes.append(f"built with Python {node.get('python')}")
        return Status("fresh", output_hash=node["output_hash"], notes=tuple(notes))

    def handle(self, name: str, *, stale: bool = False) -> Handle:
        node = self._read_node(name)
        if node is None:
            raise ArtifactError(f"artifact {name!r} has not been built")
        return Handle(name, self.dir / name, node["output_hash"], stale, frozenset(node["outputs"]))

    # building ...................................................................................

    @staticmethod
    def build_command(name: str) -> str:
        return f"uv run python -m convfinqa.artifacts build {name}"

    def ensure(self, name: str, *, allow_heavy: bool, force: bool = False) -> Handle:
        """Make ``name`` fresh (building it and its stale dependencies) and return its handle."""
        spec, params = self.resolve(name)
        for dep in self.dep_names(spec, params):
            self.ensure(dep, allow_heavy=allow_heavy)
        st = self.status(name)
        if st.fresh and not force:
            return self.handle(name)
        if spec.heavy:
            if not allow_heavy:
                raise StaleArtifactError(self._stale_message(name, st))
            self._spawn(name, force=force)
        else:
            self.execute(name, force=force)
        return self.handle(name)

    @staticmethod
    def _stale_message(name: str, st: Status) -> str:
        detail = "is missing" if st.kind == "missing" else f"is stale ({st.reason})"
        msg = (f"Heavy artifact {name!r} {detail}. Build it with:\n    {Registry.build_command(name)}\n"
               "(or run ./build.sh, or render with RUN_HEAVY=1).")
        if st.kind == "stale":
            msg += " ALLOW_STALE=1 would load the old result with a warning."
        return msg

    def _spawn(self, name: str, *, force: bool) -> None:
        spec, params = self.resolve(name)
        cmd = [sys.executable, "-m", "convfinqa.artifacts", "--artifacts-dir", str(self.dir), "--root", str(self.root)]
        for m in self.modules:
            cmd += ["--import", m]
        cmd += ["_run", name] + (["--force"] if force else [])
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(self._seed(spec, params))
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if self.extra_paths:
            env["PYTHONPATH"] = os.pathsep.join(self.extra_paths + [env.get("PYTHONPATH", "")])
        log = self.dir / f"{name}.tmp" / _LOG_NAME
        print(f"[artifacts] building {name} in a subprocess (log: {log})", file=sys.stderr, flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, cwd=self.root)
        if proc.returncode != 0:
            tail = ""
            if log.exists():
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
            raise BuildError(f"building {name!r} failed (exit code {proc.returncode}); log: {log}\n{tail}")
        print(f"[artifacts] built {name} in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    def execute(self, name: str, *, force: bool = False, redirect_output: bool = False) -> None:
        """Run the builder in this process. Takes the lock, then re-checks freshness."""
        spec, params = self.resolve(name)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / ".locks").mkdir(exist_ok=True)
        with (self.dir / ".locks" / f"{name}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not force and self.status(name).fresh:
                return  # another process built it while we waited
            dep_handles = {}
            for d in self.dep_names(spec, params):
                if not self.status(d).fresh:
                    raise BuildError(f"cannot build {name!r}: dependency {d!r} is not fresh")
                dep_handles[d] = self.handle(d)
            tmp, final, old = (self.dir / f"{name}{s}" for s in (".tmp", "", ".old"))
            for p in (tmp, old):
                shutil.rmtree(p, ignore_errors=True)
            tmp.mkdir(parents=True)
            if redirect_output:
                fd = os.open(tmp / _LOG_NAME, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
                os.dup2(fd, 1)
                os.dup2(fd, 2)
            files = Files({
                rel: InputFile(self.root / rel, _sha256_file(self.root / rel)) for rel in spec.files
            })
            args = [tmp, Deps(dep_handles)] + ([files] if spec.files else [])
            config = self._config(spec, params)
            before = set(sys.modules)
            t0 = time.time()
            with _preserved_rng_state(), _chdir(tmp):
                seed_everything(config["seed"])
                spec.fn(*args, **params)
            dynamic = sorted(
                m for m in set(sys.modules) - before
                if (m == "convfinqa" or m.startswith("convfinqa.")) and m not in _FRAMEWORK
            )
            outputs = _hash_outputs(tmp)
            node = {
                "name": name,
                "version": spec.version,
                "config": config,
                "deps": {d: h.hash for d, h in dep_handles.items()},
                "outputs": outputs,
                "output_hash": _sha256_json(outputs),
                "code_hash": code_hash(module_closure(spec.module) | set(dynamic)),
                "dynamic_modules": dynamic,
                "git_commit": _git(self.root, "rev-parse", "HEAD"),
                "git_dirty": bool(_git(self.root, "status", "--porcelain")),
                "library_versions": library_versions(),
                "python": sys.version.split()[0].rpartition(".")[0],
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "duration_s": round(time.time() - t0, 1),
            }
            tmp_node = tmp / "node.json.part"
            tmp_node.write_text(json.dumps(node, indent=2))
            os.replace(tmp_node, tmp / "node.json")  # written last, atomically
            if final.exists():
                os.replace(final, old)
            os.replace(tmp, final)
            shutil.rmtree(old, ignore_errors=True)

    # what chapters call .........................................................................

    def get(self, name: str) -> Handle:
        """Return the artifact ``name``, building it if the rules allow (see the module docstring)."""
        spec, _ = self.resolve(name)
        st = self.status(name)
        if st.fresh:
            for note in st.notes:
                warnings.warn(f"artifact {name}: {note}", stacklevel=2)
            h = self.handle(name)
        elif not spec.heavy or os.environ.get("RUN_HEAVY") == "1":
            h = self.ensure(name, allow_heavy=os.environ.get("RUN_HEAVY") == "1")
        elif st.kind == "stale" and os.environ.get("ALLOW_STALE") == "1":
            h = self.handle(name, stale=True)
            _show_stale_warning(name, st.reason)
        else:
            raise StaleArtifactError(self._stale_message(name, st))
        self._record_use(h)
        return h

    def _record_use(self, h: Handle) -> None:
        document = os.environ.get("QUARTO_DOCUMENT_FILE")
        if not document:
            return
        self._chapter_seen[h.name] = h.hash
        if h.stale:
            self._chapter_stale.add(h.name)
        path = self.dir / "_chapters" / f"{Path(document).stem}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"artifacts": self._chapter_seen, "stale": sorted(self._chapter_stale)}, indent=2))

    # CLI actions ................................................................................

    def tree(self, names: list[str]) -> list[str]:
        lines: list[str] = []

        def show(n: str, depth: int) -> None:
            st = self.status(n)
            label = st.kind + (f": {st.reason}" if st.reason else "")
            lines.append(f"{'  ' * depth}{n}  [{label}]")
            for note in st.notes:
                lines.append(f"{'  ' * depth}  note: {note}")
            spec, params = self.resolve(n)
            for d in self.dep_names(spec, params):
                show(d, depth + 1)

        for n in names:
            show(n, 0)
        return lines

    def refresh(self, freeze_dir: Path) -> list[str]:
        """Delete the freeze folder of every chapter that used an artifact that is no longer current."""
        removed = []
        for rec in sorted((self.dir / "_chapters").glob("*.json")):
            data = json.loads(rec.read_text())
            outdated = bool(data.get("stale"))
            for name, h in data["artifacts"].items():
                try:
                    st = self.status(name)
                except ArtifactError:
                    outdated = True
                    continue
                outdated |= not st.fresh or st.output_hash != h
            if outdated:
                shutil.rmtree(freeze_dir / rec.stem, ignore_errors=True)
                rec.unlink()  # the chapter re-records its usage when it re-executes
                removed.append(rec.stem)
        return removed


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _show_stale_warning(name: str, reason: str) -> None:
    msg = f"STALE artifact {name!r} ({reason}). Showing the old result; rebuild with ./build.sh."
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None:
            display(HTML(
                '<div style="border:2px solid #c0392b;background:#fdecea;color:#7b241c;padding:6px 10px;'
                f'margin:4px 0;font-weight:600">&#9888; {msg}</div>'
            ))
            return
    except ImportError:
        pass
    print(f"WARNING: {msg}", file=sys.stderr)


DEFAULT = Registry(modules=DEFAULT_MODULES)


def get(name: str) -> Handle:
    """Chapter entry point: ``artifacts.get("split-v1")``."""
    return DEFAULT.get(name)


# --- command line ------------------------------------------------------------------------------


def _targets() -> list[str]:
    from convfinqa.targets import TARGETS

    return list(TARGETS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m convfinqa.artifacts", description=__doc__.splitlines()[0])
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--import", dest="imports", action="append", default=[], metavar="MODULE")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build artifacts (heavy ones in subprocesses) and anything stale they need")
    b.add_argument("names", nargs="*")
    b.add_argument("--all", action="store_true", help="build every target in targets.py")
    b.add_argument("--force", action="store_true", help="rebuild the named artifacts even if fresh")
    s = sub.add_parser("status", help="show a fresh/stale/missing tree")
    s.add_argument("names", nargs="*", help="default: every target")
    sub.add_parser("refresh", help="delete the freeze folders of chapters that used outdated artifacts")
    r = sub.add_parser("_run", help=argparse.SUPPRESS)  # build one artifact in this process
    r.add_argument("name")
    r.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    reg = Registry(args.artifacts_dir, root=args.root, modules=args.imports or DEFAULT_MODULES)
    try:
        if args.cmd == "_run":
            reg.execute(args.name, force=args.force, redirect_output=True)
        elif args.cmd == "build":
            names = _targets() if args.all else args.names
            if not names and not args.all:
                p.error("give artifact names or --all")
            reg.validate(names)
            for n in names:
                reg.ensure(n, allow_heavy=True, force=args.force)
            print(f"[artifacts] {len(names)} target(s) fresh")
        elif args.cmd == "status":
            print("\n".join(reg.tree(args.names or _targets())) or "(no targets)")
        elif args.cmd == "refresh":
            removed = reg.refresh(args.root / "docs" / "_freeze")
            if removed:
                print(f"[artifacts] chapters marked for re-execution: {', '.join(removed)}")
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Run as `-m`, this file is `__main__`; builders import `convfinqa.artifacts`, a second copy with its
    # own registry. Delegate to the canonical module so both see the same one.
    from convfinqa.artifacts import main as _main

    sys.exit(_main())
