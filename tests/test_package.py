"""The package's front door: its exports, its import cost, and that its documentation example is valid."""

import re
import subprocess
import sys

import pytest

import convfinqa.artifacts as A
from convfinqa.artifacts.spec import template_parts


def test_everything_in_all_is_exported():
    assert sorted(A.__all__) == sorted(set(A.__all__))  # no duplicates
    for name in A.__all__:
        assert hasattr(A, name), name


def test_the_policy_that_get_accepts_is_exported():
    assert A.Policy().run_heavy is False and A.Policy(run_heavy=True).run_heavy is True


def test_what_chapters_import_is_exported():
    for name in ("artifact", "get", "Deps", "Files"):  # the first lines of every builder cell
        assert name in A.__all__


def test_importing_the_package_does_not_import_torch_or_numpy():
    code = "import sys, convfinqa.artifacts; print('torch' in sys.modules, 'numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]  # chapters import it first thing; status and refresh pay for it too


def documented_example():
    match = re.search(r'@artifact\("([^"]+)", deps=\[([^\]]*)\], files=\[([^\]]*)\]', A.__doc__)
    assert match, "the documentation example is gone"
    template, deps, files = match.groups()
    return template, re.findall(r'"([^"]+)"', deps), re.findall(r'"([^"]+)"', files)


def test_the_example_in_the_documentation_is_a_valid_declaration():
    template, deps, files = documented_example()
    fields = {f for _, f in template_parts(template) if f}
    assert fields == {"variant", "seed"}
    for dep in deps:  # dependency templates may use only the name's own fields
        assert {f for _, f in template_parts(dep) if f} <= fields
    assert files == ["configs/sft.json"]


def test_the_documented_name_resolves_to_the_documented_family():
    template, _, _ = documented_example()
    from convfinqa.artifacts.spec import NAME

    member = template.replace("{variant}", "dsl").replace("{seed}", "0")
    assert member == "sft_dsl_0" and NAME.fullmatch(member)
    assert "get(\"sft_dsl_0\")" in A.__doc__


@pytest.mark.parametrize("module", [
    "spec", "chapters", "catalog", "store", "records", "provenance", "seeding", "subprocess_build",
    "policy", "handles", "errors", "hashing", "registry", "cli",
])
def test_every_module_the_documentation_names_exists_and_none_is_missing(module):
    assert f"`{module}`" in A.__doc__
    __import__(f"convfinqa.artifacts.{module}")


def test_the_documentation_lists_every_module_there_is():
    from pathlib import Path

    named = set(re.findall(r"`([a-z_]+)`", A.__doc__.split("**The modules, by role**")[1]))
    present = {p.stem for p in Path(A.__file__).parent.glob("*.py") if p.stem not in ("__init__", "__main__")}
    assert present <= named, f"not in the module list: {sorted(present - named)}"
