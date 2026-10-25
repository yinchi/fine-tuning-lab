"""Hashing for freshness: normalized source, outputs, and the import scan."""

import pytest

from convfinqa.artifacts import hashing as h


def norm(source):
    return h.normalized_source(source)


# --- annotations: only those inside functions are noise --------------------------------------------

DATACLASS = """
from dataclasses import dataclass

@dataclass
class Config:
    lr: float = 1e-4
    rank: int = 16
"""


def test_adding_a_dataclass_field_changes_the_normalized_source():
    assert norm(DATACLASS) != norm(DATACLASS + "    alpha: int = 32\n")


def test_a_field_without_a_default_is_not_dropped():
    assert norm(DATACLASS) != norm(DATACLASS + "    alpha: int\n")


def test_turning_a_field_into_a_plain_attribute_changes_the_normalized_source():
    assert norm(DATACLASS) != norm(DATACLASS.replace("rank: int = 16", "rank = 16"))


def test_retyping_a_field_changes_the_normalized_source():
    assert norm(DATACLASS) != norm(DATACLASS.replace("rank: int", "rank: float"))


def test_module_level_annotations_are_kept():
    assert norm("LIMIT: int = 3") != norm("LIMIT = 3")


def test_annotations_inside_a_function_are_dropped():
    with_annotations = "def f(x):\n    y: int = x + 1\n    z: list[int] = []\n    return y, z\n"
    without = "def f(x):\n    y = x + 1\n    z = []\n    return y, z\n"
    assert norm(with_annotations) == norm(without)


def test_a_bare_annotation_inside_a_function_becomes_pass():
    assert norm("def f():\n    y: int\n    return 1\n") == norm("def f():\n    pass\n    return 1\n")


def test_a_class_nested_in_a_function_keeps_its_field_annotations():
    inner = "def f():\n    class C:\n        x: int = 0\n    return C\n"
    assert norm(inner) != norm(inner.replace("x: int = 0", "x = 0"))


def test_a_function_nested_in_a_class_drops_its_local_annotations():
    method = "class C:\n    def m(self):\n        y: int = 1\n        return y\n"
    assert norm(method) == norm(method.replace("y: int = 1", "y = 1"))


def test_docstrings_are_dropped_in_every_scope():
    documented = '"""M."""\nclass C:\n    """C."""\n    def m(self):\n        """m."""\n        return 1\n'
    assert norm(documented) == norm("class C:\n    def m(self):\n        return 1\n")


# --- outputs -------------------------------------------------------------------------------------


@pytest.mark.parametrize("rel, expected", [
    ("v.txt", True), ("sub/w.json", True), ("adapter/model.safetensors", True),
    ("node.json", False), ("build.log", False), ("sub/build.log", False), ("train.log", False),
    ("tb/events.out", False), ("runs/tb/events.out", False),
    ("tbx/data.txt", True),  # only a component named exactly `tb` counts
])
def test_is_output(rel, expected):
    assert h.is_output(rel) is expected


def test_hash_outputs_lists_files_with_forward_slashes_and_skips_non_outputs(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "w.txt").write_text("2")
    (tmp_path / "v.txt").write_text("1")
    (tmp_path / "build.log").write_text("noise")
    (tmp_path / "empty_dir").mkdir()
    outputs = h.hash_outputs(tmp_path)
    assert list(outputs) == ["v.txt", "sub/w.txt"]  # a directory's files, then its subdirectories
    assert h.sha256_json(outputs) == h.sha256_json(dict(reversed(outputs.items())))  # order is irrelevant
    assert outputs["v.txt"] == h.sha256_file(tmp_path / "v.txt")


def test_sha256_json_ignores_key_order():
    assert h.sha256_json({"a": 1, "b": [1, 2]}) == h.sha256_json({"b": [1, 2], "a": 1})
    assert h.sha256_json({"a": 1}) != h.sha256_json({"a": 2})


@pytest.mark.parametrize("module, expected", [
    ("convfinqa.artifacts", True), ("convfinqa.artifacts.store", True),
    ("convfinqa.artifactsx", False), ("convfinqa.data", False), ("convfinqa", False),
])
def test_is_framework(module, expected):
    assert h.is_framework(module) is expected


# --- imports -------------------------------------------------------------------------------------


def test_local_imports_finds_modules_wherever_the_import_is():
    source = "import os\nimport convfinqa.dsl.executor\ndef f():\n    from convfinqa.data import raw\n"
    assert h.local_imports(source) == {"convfinqa.dsl.executor", "convfinqa.data", "convfinqa.data.raw"}


def test_local_imports_resolves_relative_imports_against_the_package():
    assert h.local_imports("from .store import Store\nfrom . import records\n", "convfinqa.artifacts") == {
        "convfinqa.artifacts.store", "convfinqa.artifacts", "convfinqa.artifacts.records",
    }


def test_importing_a_submodule_does_not_include_its_parent_package():
    # `import convfinqa.data.raw` also runs convfinqa/data/__init__.py, which is not included
    assert h.local_imports("import convfinqa.data.raw") == {"convfinqa.data.raw"}


def test_module_closure_is_transitive_and_leaves_out_the_framework():
    closure = h.module_closure("convfinqa.formatting")
    assert {"convfinqa.formatting", "convfinqa.data"} <= closure
    assert not any(h.is_framework(m) for m in closure)


def test_a_module_without_source_changes_the_code_hash_when_it_appears(monkeypatch):
    monkeypatch.setattr(h, "module_digest", lambda m: None)
    missing = h.code_hash({"convfinqa.ghost"})
    monkeypatch.setattr(h, "module_digest", lambda m: "abc")
    assert h.code_hash({"convfinqa.ghost"}) != missing
