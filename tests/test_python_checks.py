"""The literal and scale checks for Python output."""

from convfinqa.python_checks import check_python_rows, grounded


def conv(cid="a", text="sales were 1,200 in 2009 and 800 in 2008."):
    return {
        "id": cid, "pre_text": [text], "table": [["", "2009"], ["sales", "1200"]], "post_text": [],
        "annotation": {"dialogue_break": ["q1", "q2"], "turn_program": ["subtract(1200, 800)", "divide(#0, 800)"]},
    }


def turns(cid="a"):
    return [
        {"id": cid, "turn": 0, "program": "subtract(1200, 800)", "answer": 400.0, "question": "change?"},
        {"id": cid, "turn": 1, "program": "subtract(1200, 800), divide(#0, 800), multiply(#1, const_100)",
         "answer": 50.0, "question": "percent change?"},
    ]


def run(outputs):
    rows = [{"id": "a", "turn": i, "output": o} for i, o in enumerate(outputs)]
    return check_python_rows(rows, turns(), [conv()])


def test_the_page_numbers_and_the_dsl_constants_are_grounded_and_others_are_not():
    page = {"1200", "800"}
    assert grounded("ans = 1200 - 800", page)
    assert grounded("ans = (1200 - 800) / 800 * 100", page)       # 100 is the value of const_100
    assert grounded("ans = 1200 / 1000", page)                    # so is 1000
    assert not grounded("ans = 1200 - 799", page)
    assert grounded("ans = 1200", page) and grounded("ans = 12 / 12", {"12"})


def test_a_correct_program_matches_the_gold_literals_as_text_and_value():
    c = run(["ans = 1200 - 800", "ans = (1200 - 800) / 800 * 100"])
    assert [x["lit_text"] for x in c] == [True, True]
    assert all(x["runs"] and x["lit_grounded"] for x in c)


def test_the_same_numbers_written_differently_match_as_values_only():
    c = run(["ans = 1200.0 - 800", "ans = (1200 - 800) / 800 * 100"])
    assert not c[0]["lit_text"] and c[0]["lit_value"]


def test_an_invented_number_is_not_grounded_and_an_output_that_does_not_run_is_flagged():
    c = run(["ans = 1234 - 800", "I think it is 400"])
    assert not c[0]["lit_grounded"] and c[0]["runs"]
    assert not c[1]["runs"]


def test_the_scale_check_uses_the_executed_answer():
    c = run(["ans = 4", "ans = (1200 - 800) / 800"])        # 400 vs 4 (÷100), and 0.5 vs 50 (÷100)
    assert [x["scale"] for x in c] == ["factor_100", "factor_100"]
    assert c[1]["percent_style"] and c[1]["gold_percent"] and not c[0]["percent_style"]
