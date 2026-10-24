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
