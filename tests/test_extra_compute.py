"""Retry with feedback and self-consistency, with a fake model."""

import pytest

from convfinqa import extra_compute
from convfinqa.extra_compute import first_conversations, run_error, vote, with_retry, with_vote


def result(output, prompt=10, new=3):
    return {"output": output, "finish": "stop", "prompt_tokens": prompt, "new_tokens": new, "seconds": 0.5}


def scripted(outputs, chats=None):
    queue = iter(outputs)

    def generate(chat):
        if chats is not None:
            chats.append(chat)
        return result(next(queue))

    return generate


def test_a_program_that_runs_has_no_error_and_one_that_does_not_has_a_reason():
    assert run_error("add(1, 2)") is None
    assert run_error("```add(1, 2)```").startswith("syntax")
    assert run_error("add(1, #0)")


def test_a_valid_first_answer_is_not_retried():
    out = with_retry(scripted(["add(1, 2)"]), 2)([{"role": "user", "content": "q"}])
    assert out["output"] == "add(1, 2)" and out["generations"] == 1 and out["attempts"] == ["add(1, 2)"]


def test_an_invalid_answer_is_retried_with_the_error_up_to_the_limit():
    chats = []
    chat = [{"role": "user", "content": "q"}]
    out = with_retry(scripted(["oops", "bad(", "add(1, 2)"], chats), 2)(chat)
    assert out["output"] == "add(1, 2)" and out["generations"] == 3
    assert out["prompt_tokens"] == 30 and out["new_tokens"] == 9 and out["seconds"] == 1.5
    assert chats[0] == chat
    assert chats[1][:1] == chat and chats[1][1] == {"role": "assistant", "content": "oops"}
    assert chats[1][2]["role"] == "user" and "did not run" in chats[1][2]["content"]
    assert chats[2][1]["content"] == "bad("          # each retry starts from the original chat


def test_the_last_attempt_is_kept_when_none_runs():
    out = with_retry(scripted(["a", "b", "c"]), 2)([{"role": "user", "content": "q"}])
    assert out["output"] == "c" and out["generations"] == 3 and out["attempts"] == ["a", "b", "c"]


def test_no_extra_attempts_is_the_greedy_answer():
    out = with_retry(scripted(["oops"]), 0)([{"role": "user", "content": "q"}])
    assert out["output"] == "oops" and out["generations"] == 1


def test_the_vote_goes_to_the_largest_group_of_executed_answers():
    won = vote(["add(1, 2)", "subtract(5, 1)", "add(2, 1)", "oops", "add(0, 3)"])
    assert won == {"chosen": 0, "n_valid": 4, "votes": 3, "agreement": 0.6}


def test_a_tie_goes_to_the_group_sampled_first_and_the_first_program_of_the_group_is_chosen():
    assert vote(["add(1, 1)", "add(2, 2)", "add(2, 2)", "add(1, 1)"])["chosen"] == 0
    assert vote(["oops", "add(2, 2)", "add(1, 1)", "add(1, 1)", "add(2, 2)"])["chosen"] == 1


def test_all_invalid_samples_give_the_first_one_and_no_votes():
    assert vote(["a", "b"]) == {"chosen": 0, "n_valid": 0, "votes": 0, "agreement": 0.0}


def test_a_vote_over_samples_reports_its_cost():
    samples = [result(o) for o in ("add(1, 2)", "oops", "add(2, 1)", "add(9, 9)", "add(1, 2)")]
    out = with_vote(lambda chat: samples)([{"role": "user", "content": "q"}])
    assert out["output"] == "add(1, 2)" and out["generations"] == 5 and out["prompt_tokens"] == 50
    assert out["new_tokens"] == 15 and out["votes"] == 3 and out["n_valid"] == 4 and len(out["samples"]) == 5


def test_an_unknown_strategy_is_refused_before_anything_is_loaded():
    with pytest.raises(ValueError):
        extra_compute.generate_extra({}, None, [], "beam", {})


def test_first_conversations_stops_once_enough_turns_are_held():
    def conv(n):
        return {"id": str(n), "annotation": {
            "dialogue_break": ["q"] * n, "turn_program": ["add(1, 2)"] * n, "exe_ans_list": [3.0] * n,
            "qa_split": [0] * n,
        }}

    convs = [conv(2), conv(3), conv(4)]
    assert [c["id"] for c in first_conversations(convs, 3)] == ["2", "3"]
    assert [c["id"] for c in first_conversations(convs, 2)] == ["2"]
    assert [c["id"] for c in first_conversations(convs, 99)] == ["2", "3", "4"]
