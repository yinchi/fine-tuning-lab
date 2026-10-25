"""What is recorded about a build's environment, and noticing drift."""

import sys

from convfinqa.artifacts import provenance as pv


def test_library_versions_covers_the_tracked_libraries():
    versions = pv.library_versions()
    assert tuple(versions) == pv.TRACKED_LIBRARIES
    assert all(v is None or isinstance(v, str) for v in versions.values())


def test_python_minor_is_major_dot_minor():
    assert pv.python_minor() == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_provenance_has_the_recorded_fields(tmp_path):
    record = pv.provenance(tmp_path)  # not a git repository
    assert set(record) == {"git_commit", "git_dirty", "library_versions", "python"}
    assert record["git_commit"] is None and record["git_dirty"] is None
    assert record["python"] == pv.python_minor()


def test_no_drift_when_the_environment_is_unchanged(tmp_path):
    assert pv.drift_notes(pv.provenance(tmp_path)) == []


def test_drift_notes_name_the_library_and_python_differences(tmp_path):
    node = pv.provenance(tmp_path)
    node["library_versions"] = {**node["library_versions"], "torch": "0.0.1"}
    node["python"] = "2.7"
    notes = pv.drift_notes(node)
    assert any(n.startswith("torch was 0.0.1 when built") for n in notes)
    assert "built with Python 2.7" in notes


def test_a_node_without_recorded_versions_reports_installed_libraries_as_drift():
    installed = [lib for lib, v in pv.library_versions().items() if v is not None]
    notes = pv.drift_notes({"python": pv.python_minor()})
    assert len(notes) == len(installed)
