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
imports made inside it. A reference to anything defined outside the function, in the cell or in a
module, is an error (raised when the builder is registered, by `spec.check_pure`).

What the scanner sees:

- Every `{python}` fenced cell except those whose leading options say `#| eval: false`: Quarto does
  not run those, so the scanner ignores them too (no builders, no `get` calls). A plain `python`
  fence, which Quarto only displays, is never a cell.
- A builder must be a plain `def` at the top level of its cell. `@artifact` on a function nested in
  an `if`, `with`, loop, function or class, or on an `async def`, is an error. (A `get("name")` may
  appear anywhere, including inside flow control.)
- A cell that does not parse (a Jupyter magic, say) is skipped and listed in `ChapterScan.unparsed`.
- Only cells at the top level of the file count. Fences follow CommonMark: a block opened with
  backticks or tildes ends at a fence of the same character and at least the same length, and
  everything inside it is text. A `{python}` fence inside a four-backtick or `~~~` block (an example
  of a cell shown to the reader) is therefore not a cell.
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
from .handles import Deps, Files

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
"""A fence line, as in CommonMark: up to three spaces, then three or more backticks or tildes, then
the rest of the line (the info string when it opens a block)."""

_CELL_INFO = re.compile(r"^\{python[^}]*\}$")
"""The info string of an executable Quarto cell: `{python}`, optionally with attributes
(`{python .class}`). Plain `python` fences, which Quarto only displays, do not match."""

_EVAL_FALSE = re.compile(r"^#\|\s*eval:\s*false\s*$", re.IGNORECASE)
"""A cell option line that turns execution off: `#| eval: false`."""

_OPTIONS = {"deps", "files", "heavy", "version", "seed"}
"""The keyword arguments `@artifact(...)` accepts in a chapter, each of which must be a literal.
They are the parameters of `artifact()` after the template."""

_ANNOTATION_NAMESPACE = {"Path": Path, "Deps": Deps, "Files": Files}
"""The names a builder's annotations may use, with what they refer to."""

ANNOTATION_NAMES = tuple(_ANNOTATION_NAMESPACE)
"""The names a builder's annotations may use: `Path`, `Deps` and `Files`."""


@dataclass
class ChapterBuilder:
    """A builder defined in a chapter cell: a top-level function decorated with `@artifact(...)`."""

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
    """The function definition's AST node, with the decorator removed and line numbers matching the
    .qmd file."""
    source: str
    """The function's source code, without the decorator."""

    @property
    def location(self) -> str:
        """The location of the builder in the .qmd file as 'path:line'."""
        return f"{self.path}:{self.line}"


@dataclass
class ChapterScan:
    """What a scan of the chapters found: builders, literal `get` calls, and cells that did not
    parse."""

    builders: list[ChapterBuilder] = field(default_factory=list)
    """Builders defined in chapter code cells, across all scanned chapters."""

    gets: list[tuple[str, str]] = field(default_factory=list)
    """Literal `get("name")` calls as `(artifact name, "<path>:<line>")` pairs, in source order
    (files by name, then top to bottom)."""

    unparsed: list[str] = field(default_factory=list)
    """Cells that mention `artifact(` or `get(` but are not valid Python, as `<path>:<line>` of the
    cell's first code line."""


def _cells(path: Path) -> list[tuple[int, str]]:
    """The executable Python cells of a .qmd file as (line number of the first code line, code).

    Fences follow CommonMark. A block opened by backticks or tildes ends at a fence of the same
    character with at least as many characters, and what is inside it is text. Only a top-level
    backtick fence whose info string is `{python}` (or with attributes) is a cell, so an example
    cell shown inside a longer or tilde fence is not one. A block that is never closed (the file
    ends first) is dropped.
    """
    cells: list[tuple[int, str]] = []
    block = None  # (fence character, fence length, is a cell, first line of its content)
    code: list[str] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        fence = _FENCE.match(line)
        if block is None:
            if fence is None:
                continue
            marker, info = fence.groups()
            if marker[0] == "`" and "`" in info:
                continue  # backticks in the info string: inline code, not a fence
            is_cell = marker[0] == "`" and bool(_CELL_INFO.match(info.strip()))
            block, code = (marker[0], len(marker), is_cell, number + 1), []
        else:
            character, length, is_cell, first_line = block
            closes = (
                fence is not None
                and fence.group(1)[0] == character
                and len(fence.group(1)) >= length
                and not fence.group(2).strip()
            )
            if closes:
                if is_cell:
                    cells.append((first_line, "\n".join(code)))
                block = None
            elif is_cell:
                code.append(line)
    return cells


def _is_disabled(code: str) -> bool:
    """Whether the cell's leading `#|` option lines include `eval: false`, so Quarto does not run it."""
    for line in code.splitlines():
        if not line.startswith("#|"):
            return False
        if _EVAL_FALSE.match(line):
            return True
    return False


def _is_artifact_decorator(d: ast.expr) -> bool:
    """Whether `d` is a call of something named `artifact`: `artifact(...)` or `x.artifact(...)`."""
    return isinstance(d, ast.Call) and (
        (isinstance(d.func, ast.Name) and d.func.id == "artifact")
        or (isinstance(d.func, ast.Attribute) and d.func.attr == "artifact")
    )


def _is_get_call(node: ast.AST) -> bool:
    """Whether `node` is a call, with at least one argument, to a bare `get` or to `artifacts.get`.

    Method calls such as `d.get("k")` do not count; any bare function named `get` does.
    """
    if not (isinstance(node, ast.Call) and node.args):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "get") or (
        isinstance(f, ast.Attribute)
        and f.attr == "get"
        and isinstance(f.value, ast.Name)
        and f.value.id == "artifacts"
    )


def _parse_decorator(d: ast.Call, where: str) -> tuple[str, dict[str, Any]]:
    """Return `(template, options)` from an `@artifact(...)` call.

    The template must be one string literal and every option a literal; otherwise raise
    `ArtifactError` naming `where`.
    """
    if len(d.args) != 1 or not (isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str)):
        raise ArtifactError(f"{where}: @artifact needs the name template as one string literal")
    options = {}
    for kw in d.keywords:
        if kw.arg is None:
            raise ArtifactError(
                f"{where}: @artifact does not accept **options; write each option out as a literal"
            )
        if kw.arg not in _OPTIONS:
            raise ArtifactError(f"{where}: @artifact got an unexpected argument {kw.arg!r}")
        try:
            options[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            raise ArtifactError(f"{where}: @artifact argument {kw.arg!r} must be a literal") from None
    return d.args[0].value, options


def scan_chapters(docs_dir: Path) -> ChapterScan:
    """Scan every `.qmd` file directly in `docs_dir` (not recursively, in name order) for builders
    and literal `get` calls.

    Cells marked `#| eval: false` are ignored. Only top-level `def`s in a cell are builders; an
    `@artifact` function anywhere else raises `ArtifactError`, as does a malformed `@artifact(...)`.
    A cell that does not parse is recorded in `unparsed` if it mentions `artifact(` or `get(`.
    """
    scan = ChapterScan()
    for path in sorted(Path(docs_dir).glob("*.qmd")):
        for first_line, code in _cells(path):
            if "artifact" not in code and "get(" not in code:
                continue
            if _is_disabled(code):
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
            accepted = {id(node) for node, *_ in builders}
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and id(node) not in accepted
                    and any(_is_artifact_decorator(d) for d in node.decorator_list)
                ):
                    raise ArtifactError(
                        f"{path}:{first_line + node.lineno - 1}: a builder must be a plain `def` at the "
                        "top level of a cell, not nested in another block and not an `async def`"
                    )
            ast.increment_lineno(tree, first_line - 1)
            for node, template, options, source in builders:
                node.decorator_list = []
                scan.builders.append(
                    ChapterBuilder(path.stem, path, node.lineno, template, options, node, source)
                )
            calls = [
                node for node in ast.walk(tree)
                if _is_get_call(node)
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ]
            for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):  # source order, not tree order
                scan.gets.append((node.args[0].value, f"{path}:{node.lineno}"))
    return scan


def outside_names(source: str) -> set[str]:
    """Return the names a function reads from outside itself (module globals), other than builtins."""
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


def builder_namespace(chapter: str) -> dict[str, Any]:
    """A fresh namespace to compile the builders of `chapter` in: its module name and the annotation names."""
    return {"__name__": f"convfinqa_chapters.{chapter}", **_ANNOTATION_NAMESPACE}


def compile_builder(b: ChapterBuilder, namespace: dict[str, Any]) -> Any:
    """Define the builder function in `namespace` and return it.

    Only the `def` statement is executed (its decorator was removed), so this defines the function
    without running it. It is compiled with the .qmd file as its filename and the cell's real line
    numbers, so tracebacks point into the chapter, and with deferred annotations: they stay strings
    and the registry resolves them from `namespace`, which must come from `builder_namespace`.
    """
    module = ast.Module(body=[b.node], type_ignores=[])
    code = compile(module, str(b.path), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)
    return namespace[b.node.name]
