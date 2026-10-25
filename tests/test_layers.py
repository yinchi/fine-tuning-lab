"""The package layers import downward only, and torch stays out of the pure layers."""

import importlib.util
import subprocess
import sys
from pathlib import Path

from convfinqa.artifacts import hashing

# module -> the only convfinqa packages it may depend on
ALLOWED = {
    "convfinqa.dsl": set(),
    "convfinqa.data": {"convfinqa.dsl"},
    "convfinqa.book": set(),
}


def _packages(module: str) -> set[str]:
    return {".".join(m.split(".")[:2]) for m in hashing.module_closure(module)} - {"convfinqa"}


def test_layers_import_only_downward():
    for package, allowed in ALLOWED.items():
        spec = importlib.util.find_spec(package)
        root = Path(spec.origin).parent
        names = [package] + [
            f"{package}.{'.'.join(p.relative_to(root).with_suffix('').parts)}"
            for p in root.rglob("*.py") if p.name not in ("__init__.py", "__main__.py")
        ]
        used = set()
        for name in names:
            used |= _packages(name)
        assert used - {package} <= allowed, f"{package} imports {sorted(used - {package} - allowed)}"


def test_pure_layers_do_not_import_torch():
    code = "import sys, convfinqa.dsl.executor, convfinqa.dsl.steps, convfinqa.data; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


# --- the artifact package: modules import only the modules they are allowed to ---------------------

# module -> the other `artifacts` modules it may import. `registry`, `cli`, `__init__` and `__main__`
# are the coordinators and may import any of them.
ARTIFACT_IMPORTS = {
    "errors": set(),
    "handles": set(),
    "hashing": set(),
    "seeding": set(),
    "provenance": set(),
    "chapters": {"errors", "handles"},
    "spec": {"chapters", "errors", "hashing"},
    "catalog": {"chapters", "errors", "handles", "spec"},  # what exists: no disk, no building
    "store": {"errors", "handles", "hashing"},  # the disk layout: no catalog, no building
    "records": {"errors", "handles"},
    "policy": {"handles"},
    "subprocess_build": {"errors"},
}


def test_artifact_modules_import_only_what_they_may():
    import convfinqa.artifacts as package

    directory = Path(package.__file__).parent
    for module, allowed in ARTIFACT_IMPORTS.items():
        imported = hashing.local_imports((directory / f"{module}.py").read_text(), "convfinqa.artifacts")
        used = {name.split(".")[2] for name in imported if name.startswith("convfinqa.artifacts.")}
        assert used <= allowed, f"{module} imports {sorted(used - allowed)}"
