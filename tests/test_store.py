"""The artifacts directory: nodes, handles, scratch directories, the atomic commit, and the lock."""

import json
import threading
import time
from pathlib import Path

import pytest

from convfinqa import artifacts as A
from convfinqa.artifacts.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "artifacts")


def build(store, name, files, record=None):
    out = store.scratch(name)
    for rel, text in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
    store.commit(name, record or {"name": name})
    return store.path(name)


def test_an_artifact_that_was_not_built_has_no_node_and_no_handle(store):
    assert store.read_node("a") is None
    with pytest.raises(A.ArtifactError, match="has not been built"):
        store.handle("a")


def test_a_corrupt_node_counts_as_not_built(store):
    (store.path("a")).mkdir(parents=True)
    (store.path("a") / "node.json").write_text("{not json")
    assert store.read_node("a") is None


def test_paths_and_log_location(store):
    assert store.path("a") == store.directory / "a"
    assert store.scratch_path("a") == store.directory / "a.tmp"
    assert store.log_path("a") == store.directory / "a.tmp" / "build.log"


def test_scratch_is_new_and_empty_and_clears_leftovers(store):
    (store.directory / "a.tmp").mkdir(parents=True)
    (store.directory / "a.tmp" / "left.txt").write_text("x")
    (store.directory / "a.old").mkdir()
    out = store.scratch("a")
    assert out == store.scratch_path("a") and list(out.iterdir()) == []
    assert not (store.directory / "a.old").exists()


def test_commit_records_outputs_and_swaps_the_artifact_in(store):
    final = build(store, "a", {"v.txt": "1", "sub/w.txt": "2"}, {"name": "a", "version": 1})
    node = store.read_node("a")
    assert node["version"] == 1 and sorted(node["outputs"]) == ["sub/w.txt", "v.txt"]
    assert len(node["output_hash"]) == 64
    assert not store.scratch_path("a").exists() and (final / "v.txt").read_text() == "1"
    handle = store.handle("a")
    assert handle.hash == node["output_hash"] and handle.file("sub/w.txt").read_text() == "2"


def test_logs_and_tensorboard_files_are_not_outputs(store):
    build(store, "a", {"v.txt": "1", "build.log": "noise", "run.log": "noise", "tb/events": "noise"})
    assert list(store.read_node("a")["outputs"]) == ["v.txt"]
    with pytest.raises(FileNotFoundError):
        store.handle("a").file("build.log")


def test_a_rebuild_with_the_same_outputs_has_the_same_hash_despite_different_logs(store):
    build(store, "a", {"v.txt": "1", "build.log": "first"})
    first = store.read_node("a")["output_hash"]
    build(store, "a", {"v.txt": "1", "build.log": "second, longer log"})
    assert store.read_node("a")["output_hash"] == first
    build(store, "a", {"v.txt": "2"})
    assert store.read_node("a")["output_hash"] != first


def test_a_second_commit_replaces_the_first_and_leaves_no_old_directory(store):
    build(store, "a", {"old.txt": "1"})
    build(store, "a", {"new.txt": "2"})
    assert sorted(p.name for p in store.path("a").iterdir()) == ["new.txt", "node.json"]
    assert not (store.directory / "a.old").exists()


def test_a_build_that_never_commits_leaves_no_node(store):
    out = store.scratch("a")
    (out / "partial.txt").write_text("half")
    assert store.read_node("a") is None and not store.path("a").exists()
    assert not (out / "node.json").exists()


def test_a_failed_build_keeps_the_previous_version(store):
    build(store, "a", {"v.txt": "good"})
    out = store.scratch("a")  # a later attempt starts and then dies
    (out / "partial.txt").write_text("half")
    assert store.read_node("a") is not None and (store.path("a") / "v.txt").read_text() == "good"


def test_the_lock_serialises_holders(store):
    log = []

    def worker(tag):
        with store.lock("a"):
            log.append(f"start {tag}")
            time.sleep(0.15)
            log.append(f"end {tag}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    # never two starts in a row: each holder finishes before the next begins
    assert all(log[i].startswith("start") and log[i + 1].startswith("end") for i in range(0, 6, 2))
