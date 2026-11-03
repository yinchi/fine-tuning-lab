"""Training data for the Python adapter: Python answers under the Python system message."""

import pytest

from convfinqa.formatting import SYSTEM_PROMPT, tokenize_conversation
from convfinqa.python_eval import SYSTEM_PYTHON
from convfinqa.python_finetune import python_messages
from test_formatting import CONV, tok  # noqa: F401  (the fixture)


def test_the_chat_has_the_python_system_message_and_python_answers():
    chat = python_messages(CONV)
    assert chat[0] == {"role": "system", "content": SYSTEM_PYTHON}
    assert [m["content"] for m in chat if m["role"] == "assistant"] == ["ans = 5", "ans = 5 + 1"]
    assert SYSTEM_PROMPT not in [m["content"] for m in chat]
    assert CONV["annotation"]["turn_program"] == ["5", "add(5, 1)"]   # the conversation is untouched


def test_only_the_python_answers_are_trained_on(tok):  # noqa: F811
    ids, labels = tokenize_conversation(tok, python_messages(CONV))
    trained = [i for i, l in zip(ids, labels) if l != -100]
    assert tok.decode(trained) == "ans = 5<|im_end|>ans = 5 + 1<|im_end|>"


def test_the_dataset_is_cut_at_max_length(tok):  # noqa: F811
    pytest.importorskip("datasets")
    from convfinqa.python_finetune import python_dataset

    ds = python_dataset(tok, [CONV, CONV], max_length=12)
    assert len(ds) == 2 and all(len(r) <= 12 for r in ds["input_ids"])
    assert all(len(a) == len(b) for a, b in zip(ds["input_ids"], ds["labels"]))
