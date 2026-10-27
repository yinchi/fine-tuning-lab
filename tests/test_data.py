import pytest

from convfinqa import data
from convfinqa.dsl import steps
from convfinqa.formatting import conversation_messages

pytestmark = pytest.mark.skipif(
    not (data.RAW_DIR / "data" / "train.json").exists(), reason="run chapter 1.2 to download the data"
)


@pytest.fixture(scope="module")
def train():
    return data.load_split("train")


def test_split_sizes(train):
    assert len(train) == 3037
    assert len(data.load_split("dev")) == 421


def test_turn_files_match_conversations(train):
    assert sum(len(c["annotation"]["dialogue_break"]) for c in train) == len(data.load_split("train_turn"))


def test_turn_fields_aligned(train):
    for c in train:
        a = c["annotation"]
        assert len({len(a[k]) for k in ("dialogue_break", "turn_program", "exe_ans_list", "qa_split")}) == 1


def test_conversation_type(train):
    assert {data.conversation_type(c) for c in train} == {"I", "II"}


def test_is_number_turn():
    assert steps.is_number_turn("206588")
    assert not steps.is_number_turn("subtract(206588, 181001)")


def test_split_steps():
    assert steps.split_steps("subtract(206588, 181001), divide(#0, 181001)") == [
        "subtract(206588, 181001)",
        "divide(#0, 181001)",
    ]


def test_table_to_markdown():
    md = data.table_to_markdown([["a", "b"], ["x", "1"]])
    assert md.splitlines() == ["| a | b |", "|---|---|", "| x | 1 |"]


def test_messages_alternate(train):
    m = conversation_messages(train[0])
    assert [x["role"] for x in m][:3] == ["system", "user", "assistant"]
    assert m[2]["content"] == train[0]["annotation"]["turn_program"][0]
    assert len(m) == 1 + 2 * len(train[0]["annotation"]["dialogue_break"])


def test_copyable_numbers_follow_the_papers_rule():
    text = "net income $ 103102 , up 28% ( 28 % ) from 1,234.5 in june 30 , 2009 ."
    assert steps.copyable_numbers(text) == ["103102", "28%", "28", "%", "1,234.5", "30", "2009"]


def test_gold_constants_are_in_the_official_list(train):
    import re

    used = {c for conv in train for p in conv["annotation"]["turn_program"] for c in re.findall(r"const_\w+", p)}
    assert used <= set(steps.CONSTANTS)


def test_page_group():
    assert data.page_group("Single_JKHY/2009/page_28.pdf-3") == "JKHY/2009/page_28.pdf"
    assert data.page_group("Double_JKHY/2009/page_28.pdf-1") == "JKHY/2009/page_28.pdf"


def test_val_split_keeps_pages_apart(train):
    fit, val = data.make_val_split(train)
    assert len(fit) + len(val) == len(train)
    assert {data.page_group(c["id"]) for c in fit}.isdisjoint(data.page_group(c["id"]) for c in val)
    assert 0.07 < len(val) / len(train) < 0.13
    assert data.make_val_split(train) == (fit, val)
    assert data.make_val_split(train, seed=1)[1] != val


def test_turn_rows_match_the_official_turn_files(train):
    from convfinqa.data.tables import turn_rows

    rows = turn_rows(data.load_split("dev"))
    assert len(rows) == len(data.load_split("dev_turn")) == 1490
    assert {r["type"] for r in rows} == {"I", "II"}
    assert all(r["number_turn"] == ("(" not in r["program"]) for r in rows)
    assert max(r["turn"] for r in rows) + 1 == max(r["n_turns"] for r in rows)
