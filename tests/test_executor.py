import pytest

from convfinqa import data
from convfinqa.dsl import executor
from convfinqa.dsl.executor import ProgramError, answers_match, execute, parse_program, trace, try_execute


def test_bare_number():
    assert execute("206588") == 206588.0
    assert execute("1,234.5") == 1234.5


def test_example_from_the_paper():
    assert execute("subtract(206588, 181001)") == 25587.0
    assert execute("subtract(206588, 181001), divide(#0, 181001)") == 0.14136


@pytest.mark.parametrize(
    "program, expected",
    [
        ("add(2, 3)", 5.0),
        ("subtract(2, 3)", -1.0),
        ("multiply(4, 2.5)", 10.0),
        ("divide(1, 3)", 0.33333),
        ("exp(2, 3)", 8.0),
        ("greater(3, 2)", "yes"),
        ("greater(2, 3)", "no"),
        ("greater(2, 2)", "no"),
    ],
)
def test_operations(program, expected):
    assert execute(program) == expected


def test_percent_literal_is_divided_by_100():
    assert execute("multiply(900, 0.25%)") == 2.25
    assert execute("add(11%, 24%)") == 0.35


def test_constants():
    assert execute("divide(300, const_3)") == 100.0
    assert execute("multiply(5, const_m1)") == -5.0
    assert execute("multiply(2, const_1000000)") == 2000000.0
    assert execute("divide(1, const_1000000)") == 0.0  # 1e-06 rounds away at 5 places, as in the official code


def test_step_references():
    assert execute("add(1, 2), multiply(#0, 4), subtract(#1, #0)") == 9.0


def test_trace_gives_every_step_unrounded():
    assert trace("add(1, 2), multiply(#0, 4), greater(#1, 5)") == [3.0, 12.0, "yes"]
    assert trace("42") == [42.0]
    assert trace("divide(2, 3)") == [2 / 3]
    assert execute("divide(2, 3)") == 0.66667


def test_result_is_rounded_to_five_places():
    assert execute("divide(2, 3)") == 0.66667


def test_greater_result_cannot_be_reused_as_a_number():
    with pytest.raises(ProgramError) as e:
        execute("greater(2, 1), add(#0, 1)")
    assert e.value.kind == executor.BAD_REFERENCE


@pytest.mark.parametrize(
    "program, kind",
    [
        ("", executor.EMPTY),
        ("   ", executor.EMPTY),
        ("add(1, 2", executor.SYNTAX),
        ("add(1)", executor.SYNTAX),
        ("add(1, 2) divide(#0, 2)", executor.SYNTAX),
        ("add(1, add(2, 3))", executor.SYNTAX),
        ("sqrt(4, 2)", executor.UNKNOWN_OPERATOR),
        ("table_sum(revenue, 2)", executor.UNKNOWN_OPERATOR),
        ("add(one, 2)", executor.BAD_NUMBER),
        ("add(const_7x, 2)", executor.BAD_NUMBER),
        ("add(#1, 2)", executor.BAD_REFERENCE),
        ("add(1, 2), add(#1, 2)", executor.BAD_REFERENCE),
        ("add(1, 2), add(#x, 2)", executor.BAD_REFERENCE),
        ("divide(1, 0)", executor.DIVISION_BY_ZERO),
        ("exp(-8, 0.5)", executor.MATH_ERROR),
        ("exp(10, 1000)", executor.MATH_ERROR),
    ],
)
def test_errors_by_kind(program, kind):
    value, error = try_execute(program)
    assert value is None and error == kind


def test_parse_program_returns_steps():
    steps = parse_program("subtract(206588, 181001), divide(#0, 181001)")
    assert [s.op for s in steps] == ["subtract", "divide"]
    assert steps[1].args == ("#0", "181001")
    assert parse_program("42") == []


def test_answers_match_is_exact_after_rounding():
    assert answers_match(0.14136, 0.14136)
    assert not answers_match(0.14137, 0.14136)
    assert answers_match("yes", "yes") and not answers_match("yes", "no")
    assert not answers_match(None, 1.0)


@pytest.mark.skipif(
    not (data.RAW_DIR / "data" / "train.json").exists(), reason="run chapter 1.2 to download the data"
)
@pytest.mark.parametrize("split", ["train", "dev"])
def test_every_gold_program_reproduces_its_answer(split):
    bad = []
    for conv in data.load_split(split):
        a = conv["annotation"]
        for prog, gold in zip(a["turn_program"], a["exe_ans_list"]):
            if not answers_match(try_execute(prog)[0], gold):
                bad.append((conv["id"], prog, gold))
    assert not bad, bad[:5]
