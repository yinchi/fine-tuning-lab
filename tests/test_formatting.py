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


def test_collate_pads_to_the_longest_and_masks_padding(tok):
    pytest.importorskip("torch")
    from convfinqa.step_cost import collate

    short = CONV
    long = {**CONV, "annotation": {**CONV["annotation"], "dialogue_break": ["q1", "q2", "q3"],
            "turn_program": ["5", "add(5, 1)", "add(5, 2)"], "exe_ans_list": [5, 6, 7], "qa_split": [0, 0, 0]}}
    b = collate(tok, [short, long], max_length=4096)
    assert b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape
    n_short = int(b["attention_mask"][0].sum())
    assert n_short < b["input_ids"].shape[1]
    assert (b["labels"][0, n_short:] == -100).all()
    assert (b["labels"][0] != -100).sum() == (b["labels"][0, :n_short] != -100).sum() > 0
    assert collate(tok, [long], max_length=10)["input_ids"].shape[1] == 10


def test_estimate_hours():
    from convfinqa.step_cost import estimate_hours

    assert estimate_hours(36.0, 2731, 2, 16) == 36.0 * 342 / 3600


def test_worst_case_batch_uses_real_lengths(tok):
    """`apply_chat_template(tokenize=True)` returns a mapping whose len() is 2: lengths must come from the ids."""
    ids = tok(tok.apply_chat_template(conversation_messages(CONV), tokenize=False), add_special_tokens=False)["input_ids"]
    assert len(ids) > 2
