"""Tests for the artifact system: freshness, build policy, seeding, code hashing, refresh."""

import importlib
import json
import os
import random
import re
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from convfinqa import artifacts as A
from convfinqa.artifacts import cli, hashing

TOY = '''
import os, random, time
from pathlib import Path
from convfinqa.artifacts import artifact, Deps, Files

CALLS = []
K = 1

@artifact("base_{n}")
def base(out: Path, deps: Deps, *, n: int):
    CALLS.append(f"base_{n}")
    (out / "x.txt").write_text(str(n * K))
    (out / "build.log").write_text(f"log {time.time()}")

@artifact("noisy_{n}")
def noisy(out: Path, deps: Deps, *, n: int):
    # the output depends on n only through a constant, but the build log differs every time
    (out / "x.txt").write_text("same")
    (out / "build.log").write_text(str(time.time()))

@artifact("mid", deps=["base_2"])
def mid(out: Path, deps: Deps):
    CALLS.append("mid")
    (out / "y.txt").write_text(deps["base_2"].file("x.txt").read_text() + "!")

@artifact("leaf", deps=["mid"])
def leaf(out: Path, deps: Deps):
    CALLS.append("leaf")
    (out / "z.txt").write_text(deps["mid"].file("y.txt").read_text() + "?")

@artifact("onnoisy", deps=["noisy_1"])
def onnoisy(out: Path, deps: Deps):
    CALLS.append("onnoisy")
    (out / "w.txt").write_text(deps["noisy_1"].file("x.txt").read_text())

@artifact("cfg", files=["cfg.json"])
def cfg(out: Path, deps: Deps, files: Files):
    (out / "v.json").write_text(files["cfg.json"].read_text())

@artifact("usecfg", deps=["cfg"])
def usecfg(out: Path, deps: Deps):
    CALLS.append("usecfg")
    (out / "u.txt").write_text(deps["cfg"].file("v.json").read_text())

@artifact("useuse", deps=["usecfg"])
def useuse(out: Path, deps: Deps):
    CALLS.append("useuse")
    (out / "uu.txt").write_text(deps["usecfg"].file("u.txt").read_text())

@artifact("heavy_{n}", heavy=True)
def heavy(out: Path, deps: Deps, *, n: int):
    print("hello from the heavy builder")
    (out / "h.txt").write_text(str(n))

@artifact("crash_heavy", heavy=True)
def crash_heavy(out: Path, deps: Deps):
    (out / "partial.txt").write_text("half")
    os._exit(3)

@artifact("boom")
def boom(out: Path, deps: Deps):
    (out / "partial.txt").write_text("half")
    raise RuntimeError("builder failed")

@artifact("draws_{seed}")
def draws(out: Path, deps: Deps, *, seed: int):
    import numpy as np
    import torch
    values = [random.random(), float(np.random.rand()), float(torch.rand(1))]
    (out / "d.json").write_text(__import__("json").dumps(values))

@artifact("slow")
def slow(out: Path, deps: Deps):
    CALLS.append("slow")
    time.sleep(0.3)
    (out / "s.txt").write_text("s")
'''


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_HEAVY", raising=False)
    monkeypatch.delenv("ALLOW_STALE", raising=False)
    monkeypatch.delenv("QUARTO_DOCUMENT_FILE", raising=False)
    name = f"toy_{abs(hash(str(tmp_path)))}"
    path = tmp_path / f"{name}.py"
    path.write_text(TOY)
    (tmp_path / "cfg.json").write_text('{"lr": 1}')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    class Toy:
        module_name = name
        file = path
        root = tmp_path

        def registry(self):
            reg = A.Registry(tmp_path / "artifacts", root=tmp_path, modules=[name], extra_paths=[tmp_path])
            reg.load()
            self.reg = reg
            return reg

        @property
        def calls(self):
            return sys.modules[name].CALLS

        def edit(self, old, new):
            text = path.read_text()
            assert old in text
            path.write_text(text.replace(old, new))
            future = time.time() + 5 * (1 + len(text) % 7)  # make sure mtime differs
            os.utime(path, (future, future))
            importlib.invalidate_caches()

    t = Toy()
    yield t
    sys.modules.pop(name, None)


# --- freshness -----------------------------------------------------------------------------------


def test_build_then_fresh_then_no_rebuild(toy):
    reg = toy.registry()
    assert reg.status("base_3").kind == "missing"
    h = reg.get("base_3")
    assert h.file("x.txt").read_text() == "3"
    assert reg.status("base_3").fresh
    n = len(toy.calls)
    reg.get("base_3")
    assert len(toy.calls) == n


def test_dependencies_are_built_first(toy):
    reg = toy.registry()
    reg.get("leaf")
    assert toy.calls == ["base_2", "mid", "leaf"]
    assert reg.handle("leaf").file("z.txt").read_text() == "2!?"


def test_missing_dependency_makes_dependent_stale(toy):
    reg = toy.registry()
    reg.get("mid")
    import shutil

    shutil.rmtree(reg.dir / "base_2")
    st = reg.status("mid")
    assert st.kind == "stale" and "base_2" in st.reason


def test_dependency_rebuilt_makes_dependent_stale(toy):
    reg = toy.registry()
    reg.get("usecfg")
    (toy.root / "cfg.json").write_text('{"lr": 2}')
    reg.ensure("cfg", allow_heavy=False)  # rebuild only the dependency
    st = reg.status("usecfg")
    assert st.kind == "stale" and "rebuilt" in st.reason


def test_stale_middle_node_makes_fresh_looking_leaf_stale(toy):
    reg = toy.registry()
    reg.get("useuse")
    assert reg.status("useuse").fresh
    (toy.root / "cfg.json").write_text('{"lr": 2}')  # makes cfg stale; nothing was rebuilt yet
    assert reg.status("usecfg").kind == "stale"
    st = reg.status("useuse")  # the leaf's own node.json still looks consistent
    assert st.kind == "stale" and "usecfg" in st.reason
    reg.get("useuse")
    assert '"lr": 2' in reg.handle("useuse").file("uu.txt").read_text()


def test_early_cutoff_when_rebuild_is_identical(toy):
    reg = toy.registry()
    reg.get("onnoisy")
    reg.ensure("noisy_1", allow_heavy=False, force=True)  # same output, new build.log
    assert reg.status("onnoisy").fresh
    toy.calls.clear()
    reg.get("onnoisy")
    assert toy.calls == []


def test_input_file_change_makes_artifact_stale(toy):
    reg = toy.registry()
    reg.get("cfg")
    assert reg.status("cfg").fresh
    (toy.root / "cfg.json").write_text('{"lr": 2}')
    assert reg.status("cfg").kind == "stale"
    reg.get("cfg")
    assert json.loads(reg.handle("cfg").file("v.json").read_text()) == {"lr": 2}


def test_version_bump_makes_artifact_stale(toy):
    reg = toy.registry()
    reg.get("base_1")
    toy.edit('@artifact("base_{n}")', '@artifact("base_{n}", version=2)')
    reg.load()
    st = reg.status("base_1")
    assert st.kind == "stale" and "version" in st.reason


# --- heavy policy --------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "0", ""])
def test_heavy_missing_raises_without_run_heavy(toy, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("RUN_HEAVY", value)
    reg = toy.registry()
    with pytest.raises(A.StaleArtifactError, match="python -m convfinqa.artifacts build heavy_1"):
        reg.get("heavy_1")
    assert not (reg.dir / "heavy_1").exists()


def test_run_heavy_builds_in_subprocess_with_log(toy, monkeypatch):
    monkeypatch.setenv("RUN_HEAVY", "1")
    reg = toy.registry()
    h = reg.get("heavy_4")
    assert h.file("h.txt").read_text() == "4"
    assert "hello from the heavy builder" in (h.path / "build.log").read_text()


def test_allow_stale_loads_old_result_but_still_raises_on_missing(toy, monkeypatch):
    monkeypatch.setenv("RUN_HEAVY", "1")
    reg = toy.registry()
    reg.get("heavy_5")
    monkeypatch.delenv("RUN_HEAVY")
    toy.edit('str(n))\n\n@artifact("crash', 'str(n + 1))\n\n@artifact("crash')
    reg.load()
    with pytest.raises(A.StaleArtifactError, match="ALLOW_STALE"):
        reg.get("heavy_5")
    monkeypatch.setenv("ALLOW_STALE", "1")
    h = reg.get("heavy_5")
    assert h.stale and h.file("h.txt").read_text() == "5"
    with pytest.raises(A.StaleArtifactError):
        reg.get("heavy_6")  # never built: ALLOW_STALE does not help


def test_build_command_builds_heavy_artifact(toy, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(toy.root))
    reg = toy.registry()
    argv = ["--artifacts-dir", str(reg.dir), "--root", str(toy.root), "--import", toy.module_name]
    assert cli.main(argv + ["build", "heavy_7"]) == 0
    assert reg.status("heavy_7").fresh
    assert cli.main(argv + ["status", "heavy_7"]) == 0


# --- interrupted builds -------------------------------------------------------------------------


def test_failing_builder_leaves_no_node(toy):
    reg = toy.registry()
    with pytest.raises(RuntimeError):
        reg.get("boom")
    assert not (reg.dir / "boom").exists()
    assert reg.status("boom").kind == "missing"


def test_killed_heavy_build_leaves_no_node(toy, monkeypatch):
    monkeypatch.setenv("RUN_HEAVY", "1")
    reg = toy.registry()
    with pytest.raises(A.BuildError, match="exit code 3"):
        reg.get("crash_heavy")
    assert not (reg.dir / "crash_heavy" / "node.json").exists()
    assert reg.status("crash_heavy").kind == "missing"


def test_old_result_survives_a_failed_rebuild(toy):
    reg = toy.registry()
    reg.get("base_1")
    toy.edit("K = 1", "K = 1\nraise_now = True")
    toy.edit('CALLS.append(f"base_{n}")', 'CALLS.append(f"base_{n}"); assert not raise_now')
    reg.load()
    with pytest.raises(AssertionError):
        reg.ensure("base_1", allow_heavy=False)
    assert (reg.dir / "base_1" / "node.json").exists()


# --- locking ------------------------------------------------------------------------------------


def test_lock_serialises_concurrent_builders(toy):
    reg = toy.registry()
    errors = []

    def run():
        try:
            reg.ensure("slow", allow_heavy=False)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert toy.calls.count("slow") == 1


# --- graph validation and builder purity --------------------------------------------------------


def test_dependencies_and_dep_names(toy):
    reg = toy.registry()
    assert reg.dependencies("leaf") == ["mid"]
    assert reg.dependencies("base_3") == []
    spec, params = reg.resolve("usecfg")
    assert spec.dep_names(params) == ["cfg"] == reg.dependencies("usecfg")


def test_default_registry_graph_is_valid():
    A.DEFAULT.load()
    A.DEFAULT.validate(A.DEFAULT.targets())


def test_missing_dependency_template_is_rejected():
    reg = A.Registry(modules=[])

    @reg.artifact("a", deps=["nope"])
    def a(out: Path, deps: A.Deps):
        pass

    with pytest.raises(A.ArtifactError, match="does not resolve"):
        reg.validate()


def test_cycle_is_rejected():
    reg = A.Registry(modules=[])

    @reg.artifact("a", deps=["b"])
    def a(out: Path, deps: A.Deps):
        pass

    @reg.artifact("b", deps=["a"])
    def b(out: Path, deps: A.Deps):
        pass

    with pytest.raises(A.ArtifactError, match="cycle"):
        reg.validate()


@pytest.mark.parametrize("source", [
    "def f(out: Path, deps: A.Deps, lr: float): pass",              # extra positional
    "def f(out: Path, deps: A.Deps, *, lr: float): pass",           # keyword that is not a name field
    "def f(out: Path): pass",                                       # no deps argument
    "def f(out: Path, deps: A.Deps, files: A.Files): pass",         # files without declaring any
])
def test_builder_signature_is_enforced(source):
    reg = A.Registry(modules=[])
    ns = {"Path": Path, "A": A}
    exec(source, ns)
    with pytest.raises(A.ArtifactError, match="signature"):
        reg.artifact("a")(ns["f"])


def test_name_fields_need_a_simple_annotation():
    reg = A.Registry(modules=[])
    with pytest.raises(A.ArtifactError, match="annotation"):
        @reg.artifact("a_{x}")
        def f(out: Path, deps: A.Deps, *, x): pass


def test_handles_and_deps_are_strict(toy):
    reg = toy.registry()
    h = reg.get("base_1")
    for bad in ("../x.txt", "/etc/passwd"):
        with pytest.raises(ValueError):
            h.file(bad)
    with pytest.raises(FileNotFoundError):
        h.file("build.log")  # logs are not outputs
    deps = A.Deps({"base_1": h})
    with pytest.raises(KeyError, match="was not declared"):
        deps["other"]


def test_ambiguous_and_unknown_names(toy):
    reg = toy.registry()
    with pytest.raises(A.ArtifactError, match="no artifact"):
        reg.resolve("nothing")
    with pytest.raises(A.ArtifactError):
        reg.resolve("base_notanint")


# --- seeding -------------------------------------------------------------------------------------


def test_builder_draws_do_not_depend_on_prior_random_state(toy):
    reg = toy.registry()
    import numpy as np
    import torch

    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    a = reg.get("draws_7").file("d.json").read_text()
    reg.ensure("draws_7", allow_heavy=False, force=True)
    for _ in range(5):
        random.random(); np.random.rand(); torch.rand(3)  # perturb the global state
    reg.ensure("draws_7", allow_heavy=False, force=True)
    assert reg.handle("draws_7").file("d.json").read_text() == a
    reg.ensure("draws_8", allow_heavy=False)
    assert reg.handle("draws_8").file("d.json").read_text() != a  # the seed matters


def test_in_kernel_build_restores_global_rng_state(toy):
    import numpy as np
    import torch

    reg = toy.registry()
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    reg.get("draws_1")
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


# --- code hash -----------------------------------------------------------------------------------

BASE = '''
def f(x):
    """Doc."""
    y: int = x + 1   # comment
    return y * 2
'''


def test_code_hash_ignores_cosmetic_edits():
    ref = hashing.normalized_source(BASE)
    cosmetic = '''
def f( x ):
    """A totally different docstring."""

    y = (x +
         1)
    # another comment
    return y * \\
        2
'''
    assert hashing.normalized_source(cosmetic) == ref
    assert hashing.normalized_source(BASE.replace('"Doc."', "'Doc.'")) == ref


@pytest.mark.parametrize("old, new", [("x + 1", "x + 2"), ("y * 2", "y / 2"), ("return y", "return -y")])
def test_code_hash_sees_behavioural_edits(old, new):
    assert hashing.normalized_source(BASE.replace(old, new)) != hashing.normalized_source(BASE)


def test_module_closure_follows_local_imports_and_skips_framework(tmp_path, monkeypatch):
    closure = hashing.module_closure("convfinqa.formatting")
    assert "convfinqa.formatting" in closure
    assert not any(hashing.is_framework(m) for m in closure)
    for m in closure:
        assert m == "convfinqa" or m.startswith("convfinqa.")


def test_code_hash_changes_when_a_helper_module_changes(monkeypatch):
    mods = {"convfinqa.a", "convfinqa.b"}
    digests = {"convfinqa.a": "1", "convfinqa.b": "1"}
    monkeypatch.setattr(hashing, "module_digest", lambda m: digests[m])
    before = hashing.code_hash(mods)
    digests["convfinqa.b"] = "2"
    assert hashing.code_hash(mods) != before


# --- chapter records and refresh ----------------------------------------------------------------


def test_chapter_records_and_refresh(toy, monkeypatch):
    reg = toy.registry()
    freeze = toy.root / "docs" / "_freeze"
    for ch in ("201_a", "202_b", "203_c"):
        (freeze / ch).mkdir(parents=True)
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "201_a.qmd")
    reg.get("base_1")
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "202_b.qmd")
    reg2 = toy.registry()
    reg2.get("leaf")  # uses a different chain
    monkeypatch.delenv("QUARTO_DOCUMENT_FILE")

    assert reg.refresh(freeze) == []  # everything fresh: nothing removed
    toy.edit("K = 1", "K = 3")
    reg.load()
    removed = reg.refresh(freeze)
    # base_1 and (through base_2) leaf are both stale now
    assert removed == ["201_a", "202_b"]
    assert not (freeze / "201_a").exists() and not (freeze / "202_b").exists()
    assert (freeze / "203_c").exists()  # a chapter without records is untouched
    assert not list((reg.dir / "_chapters").glob("*.json"))


def test_stale_use_is_recorded_and_always_refreshed(toy, monkeypatch):
    monkeypatch.setenv("RUN_HEAVY", "1")
    reg = toy.registry()
    reg.get("heavy_1")
    monkeypatch.delenv("RUN_HEAVY")
    toy.edit('str(n))\n\n@artifact("crash', 'str(n + 1))\n\n@artifact("crash')
    reg.load()
    monkeypatch.setenv("ALLOW_STALE", "1")
    monkeypatch.setenv("QUARTO_DOCUMENT_FILE", "204_d.qmd")
    reg.get("heavy_1")
    record = json.loads((reg.dir / "_chapters" / "204_d.json").read_text())
    assert record["stale"] == ["heavy_1"]
    freeze = toy.root / "docs" / "_freeze"
    (freeze / "204_d").mkdir(parents=True)
    assert reg.refresh(freeze) == ["204_d"]


def test_status_tree_lists_dependencies(toy):
    reg = toy.registry()
    lines = cli.status_tree(reg, ["leaf"])
    assert [l.strip().split()[0] for l in lines] == ["leaf", "mid", "base_2"]
    assert all("missing" in l for l in lines)


def test_seed_field_must_be_an_int():
    reg = A.Registry(modules=[])

    def as_text(out: Path, deps: A.Deps, *, seed: str):
        pass

    def as_number(out: Path, deps: A.Deps, *, seed: int):
        pass

    with pytest.raises(A.ArtifactError, match="seed.*must be an int"):
        reg.artifact("m_{seed}")(as_text)
    reg.artifact("m_{seed}")(as_number)  # fine
    assert reg.resolve("m_7")[1] == {"seed": 7}


def test_git_provenance_is_unknown_outside_a_repository_not_clean(tmp_path):
    from convfinqa.artifacts.registry import _git, _git_dirty

    assert _git(tmp_path, "rev-parse", "HEAD") is None
    assert _git_dirty(tmp_path) is None  # not False: "clean" would claim something we do not know

    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert _git_dirty(tmp_path) is False
    (tmp_path / "new.txt").write_text("x")
    assert _git_dirty(tmp_path) is True
