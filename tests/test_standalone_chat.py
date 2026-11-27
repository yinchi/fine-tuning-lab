from convfinqa.cli.chat import converse, show_reply
from convfinqa.cli.session import Reply, Session
from convfinqa.tuned_eval import MAX_NEW_TOKENS

PAGE = {"pre_text": [], "table": [["", "2000"], ["sales", "5"]], "post_text": []}


def stub(outputs):
    queue = iter(outputs)
    chats = []

    def generate(chat):
        chats.append(chat)
        return {"output": next(queue), "finish": "stop"}

    generate.chats = chats
    return generate


def run(lines, outputs, echo=False):
    out, session, generate = [], Session(PAGE), stub(outputs)
    converse(session, generate, lines, write=out.append, echo=echo)
    return out, session, generate


def test_each_question_gets_a_program_and_an_answer():
    out, session, _ = run(["what?", "again?"], ["5", "add(5, 1)"])
    assert out == ["  program: 5", "  answer:  5", "  program: add(5, 1)", "  answer:  6"]
    assert [q for q, _ in session.history] == ["what?", "again?"]


def test_an_invalid_program_is_reported_not_raised():
    out, session, _ = run(["what?", "next?"], ["nonsense", "7"])
    assert out[0] == "  program: nonsense" and out[1].startswith("  invalid: ")
    assert out[2:] == ["  program: 7", "  answer:  7"]
    assert session.history[0] == ("what?", "nonsense")


def test_reset_empties_the_history_and_quit_ends():
    out, session, generate = run(["a", "/reset", "b", "/quit", "never asked"], ["1", "2"])
    assert "(new conversation)" in out
    assert len(generate.chats) == 2 and len(generate.chats[1]) == 2  # system message and one user message
    assert session.history == [("b", "2")]


def test_empty_lines_are_skipped_and_questions_are_echoed_when_asked():
    out, _, generate = run(["", "  ", "what?"], ["5"], echo=True)
    assert out[0] == "> what?" and len(generate.chats) == 1


def test_a_cut_off_output_is_said_so():
    out = []
    show_reply(Reply("q", "add(1,", None, "syntax", finish="length"), out.append)
    assert out[-1] == f"  (the output was cut at {MAX_NEW_TOKENS} tokens)"
