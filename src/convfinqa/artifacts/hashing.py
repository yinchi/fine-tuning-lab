"""Hashing for freshness: files, outputs, and the normalized source of the modules a builder uses."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

"""Tracked libraries for hashing purposes.

If any of these libraries are updated, artifacts depending on them are declared stale.
"""

LOG_NAME = "build.log"


def is_framework(module: str) -> bool:
    """Check if a module is considered a framework module.
    
    The framework is the artifact package itself. Edits to it do not invalidate artifacts. (Bump a
    builder's `version` by hand if a framework change should rebuild things.)
    """
    return module == "convfinqa.artifacts" or module.startswith("convfinqa.artifacts.")


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file.

    Args:
        path: The path to the file to hash.

    Returns:
        The SHA-256 hash of the file as a hexadecimal string.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Compute the SHA-256 hash of a JSON-serializable object.

    The object is first serialized to JSON with sorted keys to ensure consistent hashing.

    Args:
        obj: The object to hash.

    Returns:
        The SHA-256 hash of the object as a hexadecimal string.
    """
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def is_output(rel: str) -> bool:
    """Check if a relative path corresponds to a significant output.

    Logs and TensorBoard files are not significant outputs (they never invalidate dependents).
    """
    p = PurePosixPath(rel)
    return not (rel == "node.json" or p.name == LOG_NAME or p.suffix == ".log" or "tb" in p.parts)


def hash_outputs(directory: Path) -> dict[str, str]:
    """Compute the SHA-256 hashes of all significant output files in a directory.

    Args:
        directory: The root directory containing output files.

    Returns:
        A dictionary mapping relative paths of significant output files to their SHA-256 hashes.
    """
    outputs = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            rel = full.relative_to(directory).as_posix()
            if is_output(rel):
                outputs[rel] = sha256_file(full)
    return outputs


class _Normalize(ast.NodeTransformer):
    """AST node transformer that normalizes source code.
    
    Drops what cannot change behaviour: docstrings and annotations on local variables.
    """

    def _strip_docstring(self, node):
        """Remove the docstring from a node if it exists."""
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return self.generic_visit(node)

    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip_docstring

    def visit_AnnAssign(self, node):
        """Normalize annotated assignments by removing the annotation if the value is present."""
        node = self.generic_visit(node)
        if node.value is None:
            return ast.Pass()
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


_norm_cache: dict[tuple[str, int, int], str] = {}
"""Cache for normalized source digests keyed by (file path, modification time, file size)."""


def normalized_source(source: str) -> str:
    """Source text with whitespace, comments, line continuations, docstrings and local annotations
    removed, so only behavioural edits change it."""
    tree = _Normalize().visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree))


def module_path(module: str) -> Path | None:
    """Compute the file path of a module if it exists and is a Python source file."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def module_digest(module: str) -> str | None:
    """Compute the SHA-256 digest of a module's normalized source code if it exists."""
    path = module_path(module)
    if path is None:
        return None
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _norm_cache:
        _norm_cache[key] = hashlib.sha256(normalized_source(path.read_text()).encode()).hexdigest()
    return _norm_cache[key]


def local_imports(source: str, package: str = "") -> set[str]:
    """The `convfinqa.*` modules that `source` imports, anywhere in it.

    `package` is the package the source lives in, used to resolve relative imports (empty for code
    that is not in a module, such as a function defined in a chapter cell).
    """
    found: set[str] = set()

    def add(name: str) -> None:
        if name == "convfinqa" or name.startswith("convfinqa."):
            found.add(name)

    for node in ast.walk(ast.parse(source)):
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
                if module_path(f"{target}.{alias.name}") is not None:
                    add(f"{target}.{alias.name}")
    return found


def _local_imports(module: str) -> set[str]:
    """The `convfinqa.*` modules a module imports, anywhere in its source."""
    path = module_path(module)
    if path is None:
        return set()
    spec = importlib.util.find_spec(module)
    package = module if spec.submodule_search_locations else module.rpartition(".")[0]
    return local_imports(path.read_text(), package)


def module_closure(module: str) -> set[str]:
    """`module` plus every `convfinqa.*` module it imports, transitively (framework excluded)."""
    seen: set[str] = set()

    # Starting with `module` itself, recursively walk through its local imports to build the
    # closure.
    todo = [module]
    while todo:
        m = todo.pop()
        if m in seen or is_framework(m):
            continue
        seen.add(m)
        todo.extend(_local_imports(m))
    return seen


def code_hash(modules: set[str] | list[str]) -> str:
    """Compute the SHA-256 hash of the normalized source code for a set of modules."""
    return sha256_json({m: module_digest(m) for m in sorted(modules)})


def closure_of_modules(modules: set[str] | list[str]) -> set[str]:
    """The union of `module_closure` over several modules."""
    out: set[str] = set()
    for m in modules:
        out |= module_closure(m)
    return out


def inline_code_hash(source: str, extra_modules: set[str] | list[str] = ()) -> str:
    """Hash for a builder defined by source text (a chapter cell) rather than in a module.

    Covers the function's normalized source and the `convfinqa.*` modules it imports (plus
    `extra_modules`, e.g. modules loaded dynamically during a build).
    """
    modules = closure_of_modules(local_imports(source)) | set(extra_modules)
    return sha256_json({"source": normalized_source(source), "modules": code_hash(modules)})
