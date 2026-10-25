"""When `get` may build or load stale, decided without touching any artifact."""

import itertools

import pytest

from convfinqa.artifacts import Status
from convfinqa.artifacts.policy import Action, Policy, show_stale_warning, refusal_message

FRESH = Status("fresh", output_hash="h")
STALE = Status("stale", "code changed")
MISSING = Status("missing", "not built")


@pytest.mark.parametrize("status", [FRESH, STALE, MISSING])
@pytest.mark.parametrize("run_heavy, allow_stale", list(itertools.product([False, True], repeat=2)))
def test_a_fresh_artifact_is_always_used_and_a_light_one_is_built_when_not(status, run_heavy, allow_stale):
    policy = Policy(run_heavy, allow_stale)
    expected = Action.USE if status.fresh else Action.BUILD
    assert policy.decide(heavy=False, status=status) is expected


@pytest.mark.parametrize("policy, status, expected", [
    (Policy(), FRESH, Action.USE),
    (Policy(), STALE, Action.REFUSE),
    (Policy(), MISSING, Action.REFUSE),
    (Policy(allow_stale=True), STALE, Action.USE_STALE),
    (Policy(allow_stale=True), MISSING, Action.REFUSE),  # a missing artifact has nothing to load
    (Policy(run_heavy=True), STALE, Action.BUILD),
    (Policy(run_heavy=True), MISSING, Action.BUILD),
    (Policy(run_heavy=True, allow_stale=True), STALE, Action.BUILD),  # building wins over loading old
])
def test_heavy_artifacts(policy, status, expected):
    assert policy.decide(heavy=True, status=status) is expected


@pytest.mark.parametrize("run_heavy, allow_stale, expected", [
    (None, None, Policy()),
    ("1", None, Policy(run_heavy=True)),
    (None, "1", Policy(allow_stale=True)),
    ("1", "1", Policy(True, True)),
    ("0", "", Policy()),
    ("true", "yes", Policy()),  # only the exact value "1" switches a flag on
])
def test_from_env(monkeypatch, run_heavy, allow_stale, expected):
    for key, value in (("RUN_HEAVY", run_heavy), ("ALLOW_STALE", allow_stale)):
        monkeypatch.delenv(key, raising=False)
        if value is not None:
            monkeypatch.setenv(key, value)
    assert Policy.from_env() == expected


def test_the_refusal_message_names_the_build_command_and_the_reason():
    message = refusal_message("sft_dsl_0", STALE)
    assert "uv run python -m convfinqa.artifacts build sft_dsl_0" in message
    assert "stale (code changed)" in message and "ALLOW_STALE=1" in message


def test_the_missing_message_does_not_offer_to_load_the_old_result():
    message = refusal_message("sft_dsl_0", MISSING)
    assert "is missing" in message and "ALLOW_STALE" not in message


def test_the_stale_warning_goes_to_stderr_outside_a_notebook(capsys):
    show_stale_warning("sft_dsl_0", "code changed")
    err = capsys.readouterr().err
    assert "STALE artifact 'sft_dsl_0' (code changed)" in err


def test_the_notebook_warning_is_html_with_the_text_escaped(monkeypatch):
    import IPython
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj))
    show_stale_warning("sft_dsl_0", "input file missing: ['a<b&c.json']")
    [widget] = shown
    assert "STALE artifact 'sft_dsl_0'" in widget.data.replace("&#x27;", "'")
    assert "a&lt;b&amp;c.json" in widget.data and "a<b&c.json" not in widget.data
    assert widget.data.startswith('<div style="border:2px solid #c0392b')
