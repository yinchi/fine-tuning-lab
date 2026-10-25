"""The artifacts directory: nodes, handles, scratch directories, the atomic commit, and the lock."""

import json
import os
import subprocess
import sys
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


# --- damaged or odd state on disk --------------------------------------------------------------------


@pytest.mark.parametrize("content", [
    b"\xff\xfe\x00",  # not text
    b"{not json",
    b"[1, 2]",  # JSON, but not a node
    b"3",
    b'{"name": "a"}',  # a dict without output_hash
    b'{"output_hash": "h", "outputs": ["v.txt"]}',  # outputs must be a mapping
    b"",
])
def test_a_node_that_cannot_be_trusted_counts_as_not_built(store, content):
    store.path("a").mkdir(parents=True)
    (store.path("a") / "node.json").write_bytes(content)
    assert store.read_node("a") is None
    with pytest.raises(A.ArtifactError, match="has not been built"):
        store.handle("a")


def test_an_artifact_path_that_is_a_file_counts_as_not_built(store):
    store.directory.mkdir(parents=True)
    store.path("a").write_text("not a directory")
    assert store.read_node("a") is None


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "a/b", "/abs", "a\0b"])
def test_names_that_are_not_plain_file_names_never_reach_the_disk(store, name):
    for call in (store.path, store.scratch_path, store.old_path, store.log_path, store.read_node,
                 store.handle, store.scratch, lambda n: store.commit(n, {}), lambda n: store.lock(n).__enter__()):
        with pytest.raises(A.ArtifactError, match="not a plain artifact name"):
            call(name)
    assert not store.directory.exists()


def test_commit_needs_a_build_in_progress(store):
    with pytest.raises(A.ArtifactError, match="no build of 'a' is in progress"):
        store.commit("a", {})


# --- the swap ------------------------------------------------------------------------------------------


def test_a_leftover_old_directory_does_not_lose_a_finished_build(store):
    build(store, "a", {"v.txt": "first"})
    out = store.scratch("a")
    (out / "v.txt").write_text("second")
    store.old_path("a").mkdir()  # something left behind by an earlier swap
    (store.old_path("a") / "junk").write_text("x")
    store.commit("a", {"name": "a"})
    assert (store.path("a") / "v.txt").read_text() == "second"
    assert not store.old_path("a").exists()


def test_a_failed_second_rename_keeps_the_previous_version_and_the_finished_build(store, monkeypatch):
    build(store, "a", {"v.txt": "first"})
    out = store.scratch("a")
    (out / "v.txt").write_text("second")
    real_replace = os.replace

    def fail_moving_into_place(src, dst):
        if Path(src) == store.scratch_path("a") and Path(dst) == store.path("a"):
            raise OSError("disk trouble")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_moving_into_place)
    with pytest.raises(OSError, match="disk trouble"):
        store.commit("a", {"name": "a"})
    assert (store.path("a") / "v.txt").read_text() == "first"  # the old version is back
    assert store.read_node("a") is not None
    assert (store.scratch_path("a") / "v.txt").read_text() == "second"  # the new one was not thrown away
    assert not store.old_path("a").exists()


def test_a_crash_between_the_two_renames_leaves_the_artifact_missing_and_the_next_build_recovers(store, monkeypatch):
    build(store, "a", {"v.txt": "first"})
    out = store.scratch("a")
    (out / "v.txt").write_text("second")
    real_replace = os.replace

    def die_between_the_renames(src, dst):
        if Path(src) == store.scratch_path("a") and Path(dst) == store.path("a"):
            raise SystemExit("killed")  # not an OSError: no rollback, as if the process had died
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", die_between_the_renames)
    with pytest.raises(SystemExit):
        store.commit("a", {"name": "a"})
    monkeypatch.setattr(os, "replace", real_replace)
    assert store.read_node("a") is None  # counts as not built, so it will be rebuilt
    assert (store.old_path("a") / "v.txt").read_text() == "first"
    assert (store.scratch_path("a") / "v.txt").read_text() == "second"
    build(store, "a", {"v.txt": "third"})  # the next build starts clean and succeeds
    assert (store.path("a") / "v.txt").read_text() == "third" and not store.old_path("a").exists()


# --- the lock across processes -------------------------------------------------------------------------

HOLD_THE_LOCK = """
import sys, time
from pathlib import Path
from convfinqa.artifacts.store import Store
with Store(Path(sys.argv[1])).lock("a"):
    print("locked", flush=True)
    time.sleep(60)
"""


def start_holder(store):
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLD_THE_LOCK, str(store.directory)], stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "locked"
    return proc


def acquire_in_a_thread(store):
    acquired = threading.Event()

    def take():
        with store.lock("a"):
            acquired.set()

    thread = threading.Thread(target=take, daemon=True)
    thread.start()
    return thread, acquired


def test_another_process_holding_the_lock_makes_us_wait(store):
    holder = start_holder(store)
    try:
        thread, acquired = acquire_in_a_thread(store)
        assert not acquired.wait(0.7)  # still waiting while the other process holds it
    finally:
        holder.kill()
        holder.wait()
    thread.join(5)
    assert acquired.is_set()


def test_a_process_killed_while_holding_the_lock_leaves_no_stale_lock(store):
    holder = start_holder(store)
    holder.kill()
    holder.wait()
    thread, acquired = acquire_in_a_thread(store)
    assert acquired.wait(5)
    thread.join(5)


def test_locks_for_different_artifacts_do_not_block_each_other(store):
    holder = start_holder(store)  # holds "a"
    try:
        with store.lock("b"):
            pass
    finally:
        holder.kill()
        holder.wait()
