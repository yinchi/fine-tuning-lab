import pytest

from convfinqa import data
from convfinqa.formatting import SYSTEM_PROMPT, conversation_messages, prompt_messages, tokenize_conversation

CONV = {
    "id": "Single_X/2000/page_1.pdf-0",
    "pre_text": ["before."],
    "table": [["", "2000"], ["sales", "5"]],
    "post_text": ["after."],
    "annotation": {
        "dialogue_break": ["q1", "q2"],
        "turn_program": ["5", "add(5, 1)"],
        "exe_ans_list": [5, 6],
        "qa_split": [0, 0],
    },
}


def test_prompt_messages_zero_shot_uses_gold_history():
    m = prompt_messages(CONV, 1)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert m[2]["content"] == "5" and m[3]["content"] == "q2"
    assert m[0]["content"] == SYSTEM_PROMPT


def test_prompt_messages_history_and_examples():
    m = prompt_messages(CONV, 1, examples=[CONV], history=["9"])
    assert len(m) == 1 + 4 + 3
    assert m[-2]["content"] == "9"
    assert m[2]["content"] == "5"  # the example keeps its gold answers
    assert prompt_messages(CONV, 0)[-1]["role"] == "user"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")


def test_labels_are_exactly_the_assistant_programs(tok):
    ids, labels = tokenize_conversation(tok, conversation_messages(CONV))
    assert len(ids) == len(labels)
    trained = [i for i, l in zip(ids, labels) if l != -100]
    assert all(l in (-100, i) for i, l in zip(ids, labels))
    assert tok.decode(trained) == "5<|im_end|>add(5, 1)<|im_end|>"
