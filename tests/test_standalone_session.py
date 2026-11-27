import pytest

from convfinqa.cli.page import PageError
from convfinqa.cli.session import Session, format_answer
from convfinqa.formatting import prompt_messages

CONV = {
    "id": "Single_X/2000/page_1.pdf-0",
    "pre_text": ["before ."],
    "table": [["", "2000"], ["sales", "5"]],
    "post_text": ["after ."],
    "annotation": {
        "dialogue_break": ["q1", "q2", "q3"],
        "turn_program": ["5", "add(5, 1)", "add(5, 1), add(#0, 1)"],
        "exe_ans_list": [5, 6, 7],
        "qa_split": [0, 0, 0],
    },
}


def stub(outputs):
    """A `generate` that says the given outputs in turn, and records the chats it was asked."""
    asked = []
    queue = iter(outputs)

    def generate(chat):
        asked.append(chat)
        return {"output": next(queue), "finish": "stop"}

    generate.asked = asked
    return generate


def test_the_prompt_is_the_one_the_book_trains_and_evaluates_with():
    session = Session(CONV)
    generate = stub(["5", "add(5, 1)", "add(5, 1), add(#0, 1)"])
    for k, question in enumerate(["q1", "q2", "q3"]):
        session.ask(generate, question)
        history = ["5", "add(5, 1)"][:k]
        assert generate.asked[k] == prompt_messages(CONV, k, history=history)


def test_the_history_keeps_the_models_own_outputs_even_when_they_are_wrong():
    session = Session(CONV)
    generate = stub(["nonsense", "add(1, 2)"])
    first = session.ask(generate, "q1")
    session.ask(generate, "q2")
    assert first.error is not None and first.answer is None
    assert generate.asked[1][2] == {"role": "assistant", "content": "nonsense"}


def test_replies_carry_the_executed_answer():
    reply = Session(CONV).ask(stub(["subtract(7, 2)"]), "  q  ")
    assert (reply.question, reply.program, reply.answer, reply.error) == ("q", "subtract(7, 2)", 5.0, None)


def test_reset_starts_a_new_conversation_on_the_same_page():
    session = Session(CONV)
    generate = stub(["5", "5"])
    session.ask(generate, "q1")
    session.reset()
    session.ask(generate, "q1")
    assert generate.asked[0] == generate.asked[1]


def test_a_session_needs_a_valid_page():
    with pytest.raises(PageError):
        Session({"table": []})


@pytest.mark.parametrize("value, text", [(5.0, "5"), (0.14136, "0.14136"), ("yes", "yes"), (-2.0, "-2")])
def test_answers_are_shown_without_a_needless_decimal_point(value, text):
    assert format_answer(value) == text
