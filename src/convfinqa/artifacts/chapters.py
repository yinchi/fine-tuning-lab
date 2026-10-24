"""Find artifact builders and `get` calls in the book's `.qmd` files, without running them.

A builder can be defined right in a chapter cell:

```python
@artifact("sft_{variant}_{seed}", deps=["tokenized_{variant}"], heavy=True)
def build_sft(out: Path, deps: Deps, *, variant: str, seed: int) -> None:
    from convfinqa.llm import train
    ...

sft = get("sft_dsl_0")
```

The scanner parses each Python cell with `ast`, so every builder and every `get("literal")` is known
before any chapter runs. That is what lets `build --all` and `status` work up front, and lets a
heavy build in a subprocess find a builder that lives in a chapter.

Builders defined this way must be pure: the function may use only its parameters, builtins and
imports made inside it. A reference to anything defined elsewhere in the cell is an error here.
"""

from __future__ import annotations

import __future__
import ast
import builtins
import re
import symtable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ArtifactError

_FENCE_OPEN = re.compile(r"^```\{python[^}]*\}\s*$")
"""Regex for the opening fence of a Python code block in a .qmd file.

Contains three backticks, followed by `{python` and optional attributes, and then a closing brace.
"""

_FENCE_CLOSE = re.compile(r"^```\s*$")
"""Regex for the closing fence of a Python code block in a .qmd file.

Contains three backticks and optional trailing whitespace.
"""


_OPTIONS = {"deps", "files", "heavy", "version", "seed"}
"""Recognized keyword options for the @artifact decorator."""

ANNOTATION_NAMES = ("Path", "Deps", "Files")
"""Names the builder's annotations may use; they are provided when the function is compiled."""


@dataclass
class ChapterBuilder:
    """Represents a builder defined in a chapter cell.
    
    An artifact builder corresponds to a function decorated with `@artifact`.
    """
    chapter: str
    """The .qmd file's stem, e.g., 'chapter1' for 'chapter1.qmd'."""
    path: Path
    """The path to the .qmd file containing the builder."""
    line: int
    """Line of the `def` in the .qmd file."""
    template: str
    """The artifact template string the builder produces."""
    options: dict[str, Any]
    """Keyword options passed to the @artifact decorator."""
    node: ast.FunctionDef
    """The AST node of the function definition, with the decorator removed and line numbers matching the .qmd file."""
    source: str
    """The function's source code, without the decorator."""

    @property
    def location(self) -> str:
        """The location of the builder in the .qmd file as 'path:line'."""
        return f"{self.path}:{self.line}"


@dataclass
class ChapterScan:
    """Represents the scan result of a chapter, including builders, get calls, and unparsed cells."""
    builders: list[ChapterBuilder] = field(default_factory=list)
    """The list of builders defined in the chapter."""
    
    gets: list[tuple[str, str]] = field(default_factory=list)  # (artifact name, location)
    """The list of literal `get` calls found in the chapter."""

    unparsed: list[str] = field(default_factory=list)
    """The list of unparsed cells that mention artifacts but could not be parsed."""


def _cells(path: Path) -> list[tuple[int, str]]:
    """The Python code cells of a .qmd file as (line number of the first code line, code)."""
    cells, start, buf = [], None, []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if start is None:
            if _FENCE_OPEN.match(line):
                start, buf = i + 1, []
        elif _FENCE_CLOSE.match(line):
            cells.append((start, "\n".join(buf)))
            start = None
        else:
            buf.append(line)
    return cells


def _is_artifact_decorator(d: ast.expr) -> bool:
    """Check if the given AST node is an @artifact decorator."""
    return isinstance(d, ast.Call) and (
        (isinstance(d.func, ast.Name) and d.func.id == "artifact")
        or (isinstance(d.func, ast.Attribute) and d.func.attr == "artifact")
    )


def _is_get_call(node: ast.AST) -> bool:
    """Check if the given AST node is a `get` call."""
    if not (isinstance(node, ast.Call) and node.args):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "get") or (
        isinstance(f, ast.Attribute) and f.attr == "get" and isinstance(f.value, ast.Name) and f.value.id == "artifacts"
    )


def _parse_decorator(d: ast.Call, where: str) -> tuple[str, dict[str, Any]]:
    """Parse an @artifact decorator, returning the name template and options.
    
    The name template is returned directly as `d.args[0].value`, while options pass through
    `ast.literal_eval` to ensure they are literal values.
    """
    if len(d.args) != 1 or not (isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str)):
        raise ArtifactError(f"{where}: @artifact needs the name template as one string literal")
    options = {}
    for kw in d.keywords:
        if kw.arg not in _OPTIONS:
            raise ArtifactError(f"{where}: @artifact got an unexpected argument {kw.arg!r}")
        try:
            options[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            raise ArtifactError(f"{where}: @artifact argument {kw.arg!r} must be a literal") from None
    return d.args[0].value, options


def scan_chapters(docs_dir: Path) -> ChapterScan:
    """Scan every `.qmd` file in `docs_dir` for chapter-defined builders and literal `get` calls.
    
    Returns the scan result as a `ChapterScan` data structure.
    """
    scan = ChapterScan()
    for path in sorted(Path(docs_dir).glob("*.qmd")):
        for first_line, code in _cells(path):
            if "artifact" not in code and "get(" not in code:
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                if "artifact(" in code or "get(" in code:
                    scan.unparsed.append(f"{path}:{first_line}")
                continue
            builders = []
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                decorators = [d for d in node.decorator_list if _is_artifact_decorator(d)]
                if not decorators:
                    continue
                where = f"{path}:{first_line + node.lineno - 1}"
                if len(decorators) > 1 or len(node.decorator_list) > 1:
                    raise ArtifactError(f"{where}: a builder takes exactly one decorator, @artifact(...)")
                template, options = _parse_decorator(decorators[0], where)
                source = ast.get_source_segment(code, node)  # before the line numbers are shifted
                builders.append((node, template, options, source))
            ast.increment_lineno(tree, first_line - 1)
            for node, template, options, source in builders:
                node.decorator_list = []
                scan.builders.append(
                    ChapterBuilder(path.stem, path, node.lineno, template, options, node, source)
                )
            for node in ast.walk(tree):
                if _is_get_call(node) and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    scan.gets.append((node.args[0].value, f"{path}:{node.lineno}"))
    return scan


def outside_names(source: str) -> set[str]:
    """Returns the names a function reads from outside itself (module globals), other than builtins."""
    outside: set[str] = set()

    def walk(table: symtable.SymbolTable) -> None:
        for sym in table.get_symbols():
            if sym.is_global() and sym.is_referenced() and not hasattr(builtins, sym.get_name()):
                outside.add(sym.get_name())
        for child in table.get_children():
            if child.get_type() != symtable.SymbolTableType.ANNOTATION:
                walk(child)

    # Python 3.14 gives the def's annotations a scope of their own, next to the function's. The build
    # never runs them (the registry resolves the names), so only the function's own scope is checked.
    for table in symtable.symtable(source, "<builder>", "exec").get_children():
        if table.get_type() != symtable.SymbolTableType.ANNOTATION:
            walk(table)
    return outside


def check_pure(b: ChapterBuilder) -> None:
    """Raise if the builder uses anything defined outside its own body.
    
    Artifacts may import modules on their own, but they should not rely on any other external
    definitions except builtins.
    """
    bad = outside_names(b.source)
    if bad:
        raise ArtifactError(
            f"{b.location}: builder {b.node.name!r} uses {sorted(bad)} from outside the function. A builder "
            "defined in a chapter may use only its parameters, builtins and imports made inside it."
        )


def compile_builder(b: ChapterBuilder, namespace: dict[str, Any]) -> Any:
    """Compile the builder with the .qmd file as its filename (so tracebacks point into the chapter)."""
    module = ast.Module(body=[b.node], type_ignores=[])
    code = compile(module, str(b.path), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)
    return namespace[b.node.name]
