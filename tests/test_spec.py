"""The declaration rules in `artifacts/spec.py`, tested without a registry."""

from pathlib import Path

import pytest

from convfinqa import artifacts as A
from convfinqa.artifacts import spec as spec_module
from convfinqa.artifacts.spec import FIELD_TYPES, NAME, SAMPLE, make_spec, template_parts


@pytest.mark.parametrize("template, parts", [
    ("sft_{variant}_{seed}", [("sft_", "variant"), ("_", "seed")]),
    ("a_{x}_b", [("a_", "x"), ("_b", None)]),
    ("{x}_a", [("", "x"), ("_a", None)]),
    ("mid", [("mid", None)]),
    ("{x}", [("", "x")]),
])
def test_template_parts(template, parts):
    assert template_parts(template) == parts


def build(out: Path, deps: A.Deps, *, variant: str, n: int):
    pass


def spec_for(template, **kw):
    return make_spec(template, build, deps=kw.get("deps", ()), files=(), heavy=False, version=1, seed=0)


def test_make_spec_builds_a_matching_pattern():
    spec = spec_for("m_{variant}_{n}")
    assert spec.pattern.fullmatch("m_dsl_3").groupdict() == {"variant": "dsl", "n": "3"}
    assert spec.fields == {"variant": str, "n": int}
    assert spec.module == __name__
    assert spec.source is None and spec.chapter is None


def test_make_spec_rejects_a_dependency_using_an_unknown_field():
    with pytest.raises(A.ArtifactError, match="uses fields .* the name lacks"):
        spec_for("m_{variant}_{n}", deps=["split_{other}"])


def test_make_spec_rejects_a_builder_whose_fields_differ_from_the_name():
    with pytest.raises(A.ArtifactError, match="signature"):
        spec_for("m_{variant}")


@pytest.mark.parametrize("name, ok", [("sft_dsl_0", True), ("a", True), ("", False), ("a-b", False), ("../x", False)])
def test_name_rule(name, ok):
    assert bool(NAME.fullmatch(name)) is ok


def test_code_hash_depends_on_the_source_for_chapter_builders():
    spec = spec_for("m_{variant}_{n}")
    module_hash = spec.code_hash()
    spec.source = "def f(out, deps):\n    return 1"
    first = spec.code_hash()
    spec.source = "def f(out, deps):\n    return 2"
    assert len({module_hash, first, spec.code_hash()}) == 3


def test_empty_template_is_rejected():
    with pytest.raises(A.ArtifactError, match="must not be empty"):
        template_parts("")


def test_names_and_template_literals_allow_the_same_characters():
    chars = "abcXYZ019_"
    assert NAME.fullmatch(chars) and spec_module._LITERAL.fullmatch(chars)
    for bad in "-. /{}":
        assert not NAME.fullmatch("a" + bad) and not spec_module._LITERAL.fullmatch("a" + bad)
    assert spec_module._LITERAL.fullmatch("") and not NAME.fullmatch("")  # the one difference


def test_every_field_type_has_a_sample_value_that_its_pattern_accepts():
    assert set(SAMPLE) == set(FIELD_TYPES) == set(spec_module._FIELD_REGEX)
    for t, value in SAMPLE.items():
        assert __import__("re").fullmatch(spec_module._FIELD_REGEX[t], value)


def test_build_seed_is_the_name_field_or_the_declared_default():
    spec = spec_for("m_{variant}_{n}")
    spec.seed = 5
    assert spec.build_seed({"variant": "dsl", "n": 3}) == 5
    assert spec.build_seed({"variant": "dsl", "n": 3, "seed": 9}) == 9


def test_config_holds_the_fields_the_seed_and_the_input_file_hashes(tmp_path):
    spec = spec_for("m_{variant}_{n}")
    spec.files = ["cfg.json", "missing.json"]
    (tmp_path / "cfg.json").write_text("{}")
    config = spec.config({"variant": "dsl", "n": 3}, tmp_path)
    assert config["params"] == {"variant": "dsl", "n": 3} and config["seed"] == 0
    assert len(config["files"]["cfg.json"]) == 64 and config["files"]["missing.json"] is None
    (tmp_path / "cfg.json").write_text('{"changed": true}')
    assert spec.config({"variant": "dsl", "n": 3}, tmp_path) != config
