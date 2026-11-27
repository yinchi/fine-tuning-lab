import pytest

from convfinqa.cli import train


def test_an_empty_or_missing_output_directory_is_fine(tmp_path):
    train.check_output(tmp_path / "new", force=False)
    (tmp_path / "empty").mkdir()
    train.check_output(tmp_path / "empty", force=False)


def test_a_non_empty_output_directory_needs_force(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(FileExistsError, match="--force"):
        train.check_output(tmp_path, force=False)
    train.check_output(tmp_path, force=True)


def test_main_stops_before_touching_the_gpu_when_the_output_is_taken(tmp_path, capsys):
    (tmp_path / "x").write_text("")
    with pytest.raises(SystemExit) as stop:
        train.main(["--out", str(tmp_path)])
    assert stop.value.code == 2 and "--force" in capsys.readouterr().err


def test_main_refuses_an_unknown_run(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        train.main(["--out", str(tmp_path / "o"), "--run", "nosuchrun"])
    assert stop.value.code == 2 and "nosuchrun" in capsys.readouterr().err
