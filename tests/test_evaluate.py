"""Running a prompt over a split (order, history) and scoring it, with a stub in place of the model."""

import pytest

from convfinqa.data.tables import turn_rows
from convfinqa.evaluate import NotFrozenError, RUNS, require_freeze, run_generation, score_rows, summarize
from convfinqa.probing import CORRECT


def conv(name, programs, answers, kind="Single"):
    return {
        "id": f"{kind}_X/2000/{name}.pdf-0", "pre_text": ["before."], "table": [["", "2000"], ["sales", "5"]],
        "post_text": ["after."],
        "annotation": {
            "dialogue_break": [f"question {k}" for k in range(len(programs))], "turn_program": programs,
            "exe_ans_list": answers, "qa_split": [0] * len(programs),
        },
    }


A = conv("a", ["5", "add(5, 1)", "multiply(#1, 2)"], [5, 6, 12])
B = conv("b", ["7", "subtract(7, 2)"], [7, 5], kind="Double")
C = conv("c", ["9"], [9])
CONVS = [A, B, C]
SYSTEM = "system message"


class Stub:
    """Answers `out<k>` for turn k, and records the chats it was asked."""

    def __init__(self, outputs=None):
        self.chats, self.outputs = [], outputs or {}

    def __call__(self, chat):
        self.chats.append(chat)
        k = sum(m["role"] == "user" for m in chat) - 1   # no examples: the turn index is the number of questions - 1
        return {"output": self.outputs.get(len(self.chats) - 1, f"out{k}"), "new_tokens": 2, "finish": "stop"}


def contents(chat):
    return [m["content"] for m in chat if m["role"] == "assistant"]


def test_gold_mode_asks_every_turn_with_gold_history_in_conversation_order():
    stub = Stub()
    rows = run_generation(CONVS, stub, system=SYSTEM, examples=[], mode="gold")
    assert [(r["id"][:10], r["turn"]) for r in rows] == [
        ("Single_X/2", 0), ("Single_X/2", 1), ("Single_X/2", 2), ("Double_X/2", 0), ("Double_X/2", 1), ("Single_X/2", 0)]
    assert contents(stub.chats[2]) == ["5", "add(5, 1)"]          # turn 3 of A: the gold programs
    assert all(c[0] == {"role": "system", "content": SYSTEM} for c in stub.chats)


def test_pred_mode_runs_one_round_per_turn_index_with_the_models_own_history():
    stub = Stub()
    rows = run_generation(CONVS, stub, system=SYSTEM, examples=[], mode="pred")
    assert [(r["id"][0], r["turn"]) for r in rows] == [
        ("S", 0), ("D", 0), ("S", 0), ("S", 1), ("D", 1), ("S", 2)]    # round 1: A, B, C; round 2: A, B; round 3: A
    assert len(stub.chats) == 6
    assert contents(stub.chats[5]) == ["out0", "out1"]                    # turn 3 of A reads the model's answers
    assert contents(stub.chats[3]) == ["out0"]


def test_an_invalid_answer_stays_in_the_history_as_written():
    stub = Stub({0: "```not a program```"})
    run_generation([A], stub, system=SYSTEM, examples=[], mode="pred")
    assert contents(stub.chats[1]) == ["```not a program```"]


def test_the_sink_sees_every_row_as_it_is_made():
    seen = []
    rows = run_generation(CONVS, Stub(), system=SYSTEM, examples=[], mode="gold", sink=seen.append)
    assert seen == rows and len(rows) == 6


def test_examples_come_first_and_a_bad_mode_is_refused():
    stub = Stub()
    run_generation([C], stub, system=SYSTEM, examples=[B], mode="gold")
    assert [m["role"] for m in stub.chats[0]] == ["system", "user", "assistant", "user", "assistant", "user"]
    with pytest.raises(ValueError):
        run_generation([C], stub, system=SYSTEM, examples=[], mode="both")


def predictions(outputs):
    return [{"id": c["id"], "turn": k, "output": outputs[(c["id"], k)], "finish": "stop", "new_tokens": 3}
            for c in CONVS for k in range(len(c["annotation"]["turn_program"]))]


def gold_outputs():
    return {(c["id"], k): p for c in CONVS for k, p in enumerate(c["annotation"]["turn_program"])}


def test_scoring_counts_gold_programs_as_correct_and_breaks_failures_down():
    outputs = gold_outputs()
    outputs[(A["id"], 1)] = "add(5, 2)"                    # runs; same operations, a different number
    outputs[(A["id"], 2)] = "```multiply(#1, 2)```"        # not a program
    outputs[(B["id"], 1)] = "subtract(7, 2"                # does not parse
    scores = score_rows(predictions(outputs), turn_rows(CONVS))
    by_key = {(s["id"], s["turn"]): s for s in scores}
    assert by_key[(A["id"], 0)]["kind"] == CORRECT and by_key[(A["id"], 1)]["kind"] == "wrong_numbers"
    assert by_key[(A["id"], 2)]["kind"] == "code_fence" and by_key[(B["id"], 1)]["kind"] == "syntax"
    summary = summarize(scores)
    assert summary["n"] == 6 and summary["accuracy"] == pytest.approx(3 / 6)
    assert summary["invalid_rate"] == pytest.approx(2 / 6)
    assert summary["by_kind"] == {"correct": 3, "wrong_numbers": 1, "code_fence": 1, "syntax": 1}
    assert summary["by_turn"]["1"] == {"n": 3, "accuracy": 1.0} and summary["by_turn"]["3"] == {"n": 1, "accuracy": 0.0}
    assert summary["by_type"]["II"]["n"] == 2 and summary["by_number_turn"]["number"]["n"] == 3


def test_scoring_needs_exactly_the_gold_turns():
    rows = predictions(gold_outputs())
    with pytest.raises(ValueError, match="1 missing"):
        score_rows(rows[:-1], turn_rows(CONVS))
    with pytest.raises(ValueError, match="1 repeated"):
        score_rows(rows + rows[:1], turn_rows(CONVS))


def test_truncated_outputs_are_counted():
    rows = predictions(gold_outputs())
    rows[0]["finish"] = "length"
    assert summarize(score_rows(rows, turn_rows(CONVS)))["truncated"] == 1


def test_runs_lists_the_outcomes_of_a_program_that_runs():
    assert CORRECT in RUNS and "code_fence" not in RUNS and "syntax" not in RUNS


def test_dev_is_refused_until_the_freeze_file_exists(tmp_path):
    freeze = tmp_path / "FROZEN"
    require_freeze("val", freeze)
    with pytest.raises(NotFrozenError):
        require_freeze("dev", freeze)
    freeze.write_text("frozen at commit abc\n")
    require_freeze("dev", freeze)
