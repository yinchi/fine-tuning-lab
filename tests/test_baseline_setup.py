"""The baseline's example sets and prompts."""

import json
from pathlib import Path

import pytest

from convfinqa.data import page_group
from convfinqa.data.examples import conversation_type_of, draw_example_sets
from convfinqa.formatting import SYSTEM_PROMPT, conversation_messages, prompt_messages
from convfinqa.prompts import VARIANTS, parse_variant, prompt_chat

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "baseline.json").read_text())
SINGLES = [f"Single_X/{y}/page_{p}.pdf-{n}" for y in range(2000, 2010) for p in range(20) for n in (1, 2)]
DOUBLES = [f"Double_X/{y}/page_{p}.pdf" for y in range(2000, 2010) for p in range(20)]
IDS = SINGLES + DOUBLES
TYPES = ["I", "II", "I"]

CONV = {
    "id": "Single_X/2000/page_1.pdf-1", "pre_text": ["before."], "table": [["", "2000"], ["sales", "5"]],
    "post_text": ["after."],
    "annotation": {
        "dialogue_break": ["q1", "q2"], "turn_program": ["5", "add(5, 1)"], "exe_ans_list": [5, 6], "qa_split": [0, 0],
    },
}


def test_example_sets_are_deterministic_and_share_no_page():
    sets = draw_example_sets(IDS, [0, 1, 2], TYPES)
    assert sets == draw_example_sets(list(reversed(IDS)), [0, 1, 2], TYPES)      # independent of the pool's order
    assert list(sets) == ["0", "1", "2"] and all(len(v) == 3 for v in sets.values())
    flat = [i for v in sets.values() for i in v]
    assert len(set(flat)) == 9 and set(flat) <= set(IDS)
    assert len({page_group(i) for i in flat}) == 9                               # no two examples from one page
    assert sets != draw_example_sets(IDS, [3, 4, 5], TYPES)


def test_every_set_has_exactly_one_type_ii_conversation_even_in_its_first_two():
    for ids in draw_example_sets(IDS, [0, 1, 2, 3], TYPES).values():
        assert [conversation_type_of(i) for i in ids] == TYPES
        assert sum(conversation_type_of(i) == "II" for i in ids[:2]) == 1
        assert sum(conversation_type_of(i) == "II" for i in ids) == 1


def test_a_smaller_set_is_a_prefix_of_a_larger_one():
    three = draw_example_sets(IDS, [7], TYPES)["7"]
    two = draw_example_sets(IDS, [7], TYPES[:2])["7"]
    assert three[:2] == two


def test_the_seed_alone_decides_the_first_set():
    assert draw_example_sets(IDS, [0], TYPES)["0"] == draw_example_sets(IDS, [0, 1, 2], TYPES)["0"]


def test_a_pool_without_the_needed_type_is_refused():
    with pytest.raises(AssertionError):
        draw_example_sets(SINGLES, [0], TYPES)


def test_the_configured_pattern_has_one_type_ii_in_the_first_two_positions():
    assert CONFIG["example_types"] == TYPES


def test_variants():
    assert [parse_variant(v) for v in VARIANTS] == [
        (0, "plain"), (0, "description"), (2, "plain"), (2, "description"), (3, "plain"), (3, "description"), (3, "flat")]
    for bad in ("e1p", "e2x", "e1f", "2p", "e2pp", ""):
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_the_plain_message_is_the_training_message():
    assert CONFIG["system"]["plain"] == SYSTEM_PROMPT
    assert prompt_chat(CONV, 1, SYSTEM_PROMPT, [CONV]) == prompt_messages(CONV, 1, [CONV])


def test_the_description_replaces_only_the_system_message():
    plain = prompt_chat(CONV, 1, SYSTEM_PROMPT, [CONV], history=["9"])
    described = prompt_chat(CONV, 1, CONFIG["system"]["description"], [CONV], history=["9"])
    assert described[0]["content"] == CONFIG["system"]["description"] != plain[0]["content"]
    assert described[1:] == plain[1:]
    assert described[0]["role"] == "system"


def test_the_description_names_every_operation_and_constant_form():
    text = CONFIG["system"]["description"]
    for word in ("add", "subtract", "multiply", "divide", "exp", "greater", "#k", "const_m1", "const_1000000000",
                 "nothing else"):
        assert word in text


def test_with_no_examples_the_prompt_is_the_conversation_alone():
    chat = prompt_chat(CONV, 1, SYSTEM_PROMPT)
    assert [m["role"] for m in chat] == ["system", "user", "assistant", "user"]
    assert chat[1:] == conversation_messages(CONV)[1:4]


def test_the_flat_message_extends_the_description_with_the_no_nesting_rule():
    flat, description = CONFIG["system"]["flat"], CONFIG["system"]["description"]
    assert flat.startswith(description) and flat != description
    assert "never another operation" in flat and "divide(#0, 3)" in flat
