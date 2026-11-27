import json

import pytest

from convfinqa.cli import info
from convfinqa.cli.info import AdapterError, check_adapter, format_fingerprint, write_info


def adapter(tmp_path, base="org/model"):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return tmp_path


def test_the_fingerprint_is_stable_and_follows_the_format(monkeypatch):
    before = format_fingerprint()
    assert before == format_fingerprint()
    monkeypatch.setattr(info, "SYSTEM_PROMPT", "Another system message.")
    assert format_fingerprint() != before


def test_a_directory_without_adapter_files_is_refused(tmp_path):
    with pytest.raises(AdapterError, match="missing adapter_config.json, adapter_model.safetensors"):
        check_adapter(tmp_path)


def test_a_book_adapter_without_train_info_is_accepted_with_a_warning(tmp_path):
    checked = check_adapter(adapter(tmp_path))
    assert checked.base_model == "org/model"
    assert "train_info.json" in checked.warnings[0]


def test_a_matching_train_info_is_accepted_without_warning(tmp_path):
    write_info(adapter(tmp_path), {"format": format_fingerprint(), "base_model": "org/model"})
    checked = check_adapter(tmp_path)
    assert checked.base_model == "org/model" and checked.warnings == []


def test_another_prompt_format_is_refused(tmp_path):
    write_info(adapter(tmp_path), {"format": "0123456789abcdef", "base_model": "org/model"})
    with pytest.raises(AdapterError, match="another prompt format"):
        check_adapter(tmp_path)


def test_another_base_model_is_refused(tmp_path):
    write_info(adapter(tmp_path), {"format": format_fingerprint(), "base_model": "org/other"})
    with pytest.raises(AdapterError, match="org/other"):
        check_adapter(tmp_path)


def test_unreadable_files_are_refused(tmp_path):
    adapter(tmp_path)
    (tmp_path / "train_info.json").write_text("{")
    with pytest.raises(AdapterError, match="not valid JSON"):
        check_adapter(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(AdapterError, match="does not name a base model"):
        check_adapter(tmp_path)
