import json

import pytest

from convfinqa.cli import example
from convfinqa.cli.page import validate_page

CONV = {
    "id": "Single_X/2000/page_1.pdf-0",
    "filename": "X/2000/page_1.pdf",
    "pre_text": ["before ."],
    "table_ori": [["Sales", "$5"]],
    "table": [["", "2000"], ["sales", "$ 5"]],
    "post_text": ["after ."],
    "qa": {"program": "secret"},
    "annotation": {
        "dialogue_break": ["q1", "q2"],
        "turn_program": ["5", "add(5, 1)"],
        "exe_ans_list": [5, 6],
        "qa_split": [0, 0],
    },
}


@pytest.mark.parametrize("spec, parsed", [("dev:3", ("dev", 3)), ("val:0", ("val", 0)), ("train:12", ("train", 12))])
def test_specs_parse(spec, parsed):
    assert example.parse_spec(spec) == parsed


@pytest.mark.parametrize("spec", ["dev", "dev:", "dev:-1", "test:1", "dev:x", ":3"])
def test_bad_specs_are_refused(spec):
    with pytest.raises(ValueError, match="SPLIT:N"):
        example.parse_spec(spec)


def test_the_page_has_no_gold_unless_asked():
    obj = example.example_object(CONV)
    assert list(obj) == ["id", "pre_text", "table", "post_text"]
    assert "secret" not in json.dumps(obj) and "5" not in json.dumps(obj["table"][0])
    assert obj["table"] == CONV["table"]  # the cleaned table, not table_ori


def test_gold_is_one_object_per_turn():
    gold = example.example_object(CONV, show_gold=True)["gold"]
    assert gold == [
        {"question": "q1", "program": "5", "answer": 5},
        {"question": "q2", "program": "add(5, 1)", "answer": 6},
    ]


def test_the_output_is_json_and_a_page_for_chat():
    text = example.to_json(example.example_object(CONV, show_gold=True))
    parsed = json.loads(text)
    assert parsed == example.example_object(CONV, show_gold=True)
    assert validate_page(parsed)["post_text"] == ["after ."]
    assert text.count("\n") > 8  # one table row, and one gold turn, per line


def test_main_prints_the_nth_conversation(monkeypatch, capsys):
    monkeypatch.setattr(example, "load_conversations", lambda split: [{**CONV, "id": f"{split}-{i}"} for i in range(3)])
    example.main(["dev:2"])
    assert json.loads(capsys.readouterr().out)["id"] == "dev-2"
    example.main(["--list", "val"])
    assert capsys.readouterr().out.splitlines()[:2] == ["val: 3 conversations", "0\tval-0"]


@pytest.mark.parametrize("argv", [["dev:3"], ["nonsense"], ["--list", "dev", "dev:0"], []])
def test_main_refuses_what_it_cannot_do(monkeypatch, capsys, argv):
    monkeypatch.setattr(example, "load_conversations", lambda split: [CONV] * 3)
    with pytest.raises(SystemExit) as stop:
        example.main(argv)
    assert stop.value.code == 2
