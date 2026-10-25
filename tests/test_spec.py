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
    assert spec.chapter is None  # defined in a module, not a chapter cell
    assert spec.source.startswith("def build(out: Path, deps: A.Deps, *, variant: str, n: int):")
    assert "@" not in spec.source  # the decorator is not part of the source
    assert spec.location.endswith(f"test_spec.py:{build.__code__.co_firstlineno}")


def test_make_spec_rejects_a_dependency_using_an_unknown_field():
    with pytest.raises(A.ArtifactError, match="uses fields .* the name lacks"):
        spec_for("m_{variant}_{n}", deps=["split_{other}"])


def test_make_spec_rejects_a_builder_whose_fields_differ_from_the_name():
    with pytest.raises(A.ArtifactError, match="signature"):
        spec_for("m_{variant}")


@pytest.mark.parametrize("name, ok", [("sft_dsl_0", True), ("a", True), ("", False), ("a-b", False), ("../x", False)])
def test_name_rule(name, ok):
    assert bool(NAME.fullmatch(name)) is ok


def test_code_hash_depends_on_the_builders_own_source():
    spec = spec_for("m_{variant}_{n}")
    own = spec.code_hash()
    spec.source = "def f(out, deps):\n    return 1"
    first = spec.code_hash()
    spec.source = "def f(out, deps):\n    return 2"
    second = spec.code_hash()
    assert len({own, first, second}) == 3
    spec.source = "def f(out, deps):\n    return 2  # only a comment changed\n"
    assert spec.code_hash() == second


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


# --- templates the formatter itself rejects --------------------------------------------------------


@pytest.mark.parametrize("template", ["a_{x", "a_}", "a_{x}}", "{}", "{0}", "a_{x.y}", "a_{x[0]}", "a_{x!r}"])
def test_malformed_templates_raise_artifact_errors_not_value_errors(template):
    with pytest.raises(A.ArtifactError, match="bad name template"):
        template_parts(template)


# --- the options @artifact takes -------------------------------------------------------------------


def build_with_files(out: Path, deps: A.Deps, files: A.Files, *, n: int):
    pass


def declare(fn=build, **options):
    template = "m_{n}" if fn is build_with_files else "m_{variant}_{n}"
    base = dict(deps=(), files=(), heavy=False, version=1, seed=0)
    return make_spec(template, fn, **{**base, **options})


@pytest.mark.parametrize("options, message", [
    (dict(files=["../outside.json"]), "relative path inside the project"),
    (dict(files=["a/../../outside.json"]), "relative path inside the project"),
    (dict(files=["/etc/passwd"]), "relative path inside the project"),
    (dict(files=[""]), "relative path inside the project"),
    (dict(files=[3]), "files must be a list of strings"),
    (dict(files="cfg.json"), "files must be a list of strings"),  # not eight one-letter files
    (dict(deps="cfg"), "deps must be a list of strings"),
    (dict(heavy="false"), "heavy must be True or False"),  # a non-empty string is truthy
    (dict(heavy=1), "heavy must be True or False"),
    (dict(version="2"), "version must be an integer"),
    (dict(version=True), "version must be an integer"),
    (dict(seed=1.5), "seed must be an integer"),
    (dict(seed=None), "seed must be an integer"),
])
def test_bad_options_are_rejected(options, message):
    fn = build_with_files if options.get("files") else build
    with pytest.raises(A.ArtifactError, match=message):
        declare(fn, **options)


def test_good_options_are_accepted_as_lists_or_tuples():
    spec = declare(build_with_files, files=("configs/sft.json", "data/split.json"), deps=["split_v1"],
                   heavy=True, version=3, seed=7)
    assert spec.files == ["configs/sft.json", "data/split.json"] and spec.deps == ["split_v1"]
    assert (spec.heavy, spec.version, spec.seed) == (True, 3, 7)
    assert declare(build, files=[], deps=[]).files == []
