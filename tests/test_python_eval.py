"""The Python prompt, generation order and scoring, with a stub in place of the model."""

import pytest

from convfinqa.data.tables import turn_rows
from convfinqa.evaluate import RUNS, run_generation, summarize
from convfinqa.python_eval import SYSTEM_PYTHON, python_conversation, score_python_rows, strip_fence
from test_evaluate import B, Stub, conv, contents

A = conv("a", ["5", "add(5, 1)", "add(5, 1), multiply(#0, 2)"], [5, 6, 12])


def test_the_gold_programs_of_a_conversation_are_translated_and_the_original_is_untouched():
    py = python_conversation(A)
    assert py["annotation"]["turn_program"] == ["ans = 5", "ans = 5 + 1", "ans = (5 + 1) * 2"]
    assert A["annotation"]["turn_program"] == ["5", "add(5, 1)", "add(5, 1), multiply(#0, 2)"]


def test_the_chat_has_the_python_system_message_and_python_history_and_examples():
    stub = Stub()
    run_generation(
        [python_conversation(A)], stub, system=SYSTEM_PYTHON, examples=[python_conversation(B)], mode="gold"
    )
    chat = stub.chats[2]
    assert chat[0] == {"role": "system", "content": SYSTEM_PYTHON}
    assert contents(chat) == ["ans = 7", "ans = 7 - 2", "ans = 5", "ans = 5 + 1"]   # example B, then the history of A


def test_a_fence_is_stripped_only_when_it_wraps_the_whole_output():
    assert strip_fence("```python\nans = 1\n```") == "ans = 1"
    assert strip_fence("```\nans = 1\n```\n") == "ans = 1"
    assert strip_fence("ans = 1") == "ans = 1"
    assert strip_fence("Here:\n```python\nans = 1\n```") == "Here:\n```python\nans = 1\n```"


def rows_for(outputs):
    return [{"id": r["id"], "turn": r["turn"], "output": o, "finish": "stop", "new_tokens": 3}
            for r, o in zip(turn_rows([A]), outputs)]


def test_scores_have_the_fields_of_the_dsl_scores_and_kinds_by_cause():
    gold = turn_rows([A])    # answers 5, 6, 12
    scores = score_python_rows(rows_for(["ans = 5", "ans = 5 + 2", "```python\nans = 12\n```"]), gold)
    assert [s["kind"] for s in scores] == ["correct", "wrong_answer", "code_fence"]
    assert [s["correct"] for s in scores] == [True, False, False]
    assert {"id", "turn", "correct", "kind", "type", "part", "number_turn", "finish", "new_tokens"} <= scores[0].keys()
    stripped = score_python_rows(rows_for(["ans = 5", "ans = 5 + 2", "```python\nans = 12\n```"]), gold, True)
    assert [s["kind"] for s in stripped] == ["correct", "wrong_answer", "correct"]


def test_invalid_kinds_are_not_runs_so_the_summary_counts_them():
    gold = turn_rows([A])
    scores = score_python_rows(rows_for(["import math\nans = 5", "ans = (", "ans = 12"]), gold)
    assert [s["kind"] for s in scores] == ["outside_subset", "syntax", "correct"]
    summary = summarize(scores)
    assert summary["invalid_rate"] == pytest.approx(2 / 3) and summary["accuracy"] == pytest.approx(1 / 3)
    assert "wrong_answer" in RUNS


def test_predictions_and_gold_must_cover_the_same_turns():
    with pytest.raises(ValueError):
        score_python_rows(rows_for(["ans = 5"]), turn_rows([A]))
