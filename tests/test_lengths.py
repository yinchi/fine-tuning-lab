"""Token lengths of conversations and the `max_length` rounding."""

import pytest

from convfinqa.lengths import conversation_lengths, max_length_for, n_tokens

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


class WordTokenizer:
    """Splits on spaces and joins the messages as a template would; records how it was called."""

    def __init__(self):
        self.special_flags = []

    def apply_chat_template(self, messages, tokenize):
        assert tokenize is False
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens):
        self.special_flags.append(add_special_tokens)
        return {"input_ids": text.split()}


def test_a_conversation_is_counted_after_the_template_without_adding_special_tokens():
    tok = WordTokenizer()
    n = n_tokens(tok, CONV)
    assert n > 10
    assert tok.special_flags == [False]


def test_lengths_have_one_row_per_conversation_in_the_order_of_the_splits():
    other = {**CONV, "id": "Single_Y/2001/page_2.pdf-0"}
    rows = conversation_lengths(WordTokenizer(), {"train": [CONV, other], "dev": [CONV]})
    assert [(r["split"], r["id"]) for r in rows] == [
        ("train", CONV["id"]), ("train", other["id"]), ("dev", CONV["id"])
    ]
    assert rows[0]["tokens"] == rows[2]["tokens"]


@pytest.mark.parametrize("longest, expected", [(3460, 3584), (3584, 3584), (3585, 3840), (1, 256)])
def test_max_length_is_the_longest_rounded_up_to_a_multiple_of_256(longest, expected):
    assert max_length_for(longest) == expected
