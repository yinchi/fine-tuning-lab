"""Python as an output format: the translation and the safe evaluator."""

import pytest

from convfinqa import data
from convfinqa.dsl import executor
from convfinqa.dsl import python_format as pf
from convfinqa.dsl.executor import try_execute


@pytest.mark.parametrize("program, python", [
    ("subtract(5, 3), divide(#0, 3)", "ans = (5 - 3) / 3"),
    ("206588", "ans = 206588"),
    ("add(1, 2), add(#0, 3)", "ans = 1 + 2 + 3"),
    ("add(1, 2", None),
])
def test_translation_of_simple_programs(program, python):
    if python is None:
        with pytest.raises(executor.ProgramError):
            pf.to_python(program)
    else:
        assert pf.to_python(program) == python


def test_a_step_used_twice_is_named_and_one_used_once_is_inlined():
    code = pf.to_python("add(2, 3), multiply(#0, #0), subtract(#1, 4)")
    assert code == "v0 = 2 + 3\nans = v0 * v0 - 4"


def test_the_structure_of_the_program_is_kept_by_the_parentheses():
    assert pf.to_python("add(1, 2), multiply(#0, 3)") == "ans = (1 + 2) * 3"
    assert pf.to_python("add(1, 2), subtract(5, #0)") == "ans = 5 - (1 + 2)"
    assert pf.to_python("multiply(2, 3), add(#0, 4)") == "ans = 2 * 3 + 4"
    assert pf.to_python("exp(2, 3), exp(#0, 2)") == "ans = (2 ** 3) ** 2"


def test_constants_percentages_and_negative_numbers():
    assert pf.to_python("multiply(const_100, 5)") == "ans = 100 * 5"
    assert pf.to_python("add(const_m1, 5)") == "ans = (-1) + 5"
    assert pf.to_python("multiply(17%, 200)") == "ans = 17 / 100 * 200"
    assert pf.to_python("multiply(200, 17%)") == "ans = 200 * (17 / 100)"
    assert pf.to_python("17%") == "ans = 17 / 100"
    assert pf.to_python("subtract(-5, 3)") == "ans = (-5) - 3"
    assert pf.to_python("add(1234, 5)") == "ans = 1234 + 5"


def test_greater_is_a_comparison_and_gives_yes_or_no():
    code = pf.to_python("greater(5, 3)")
    assert code == "ans = 5 > 3"
    assert pf.execute_python(code) == "yes" and pf.execute_python("ans = 3 > 5") == "no"


def test_steps_the_result_does_not_depend_on_are_dropped():
    assert pf.to_python("add(9, 9), add(1, 2)") == "ans = 1 + 2"


@pytest.mark.parametrize("code, value", [
    ("ans = 1 / 3", 0.33333),
    ("x = 4\nans = x * 2.5", 10.0),
    ("a = 5\na += 2\nans = a", 7.0),
    ("ans = round(2.567, 1)", 2.6),
    ("ans = max(1, 5, 3) - min([4, 2])", 3.0),
    ("ans = sum([1, 2, 3]) / 2  # a comment", 3.0),
    ("ans = -(2 ** 3)", -8.0),
    ("ans = 3 < 4 < 5", "yes"),
])
def test_the_evaluator_runs_the_subset(code, value):
    assert pf.execute_python(code) == value


@pytest.mark.parametrize("code, kind", [
    ("", pf.EMPTY),
    ("```python\nans = 1\n```", pf.FENCED),
    ("ans = (1 +", pf.SYNTAX),
    ("ans = 17%", pf.SYNTAX),
    ("import math\nans = math.sqrt(4)", pf.OUTSIDE_SUBSET),
    ("def f():\n    return 1\nans = f()", pf.OUTSIDE_SUBSET),
    ("for i in range(3):\n    ans = i", pf.OUTSIDE_SUBSET),
    ("ans = math.sqrt(4)", pf.OUTSIDE_SUBSET),
    ("ans = __import__('os')", pf.OUTSIDE_SUBSET),
    ("ans = [x for x in (1, 2)]", pf.OUTSIDE_SUBSET),
    ("print(1)", pf.OUTSIDE_SUBSET),
    ("ans = open('f')", pf.OUTSIDE_SUBSET),
    ("ans = x + 1", pf.RUNTIME),
    ("ans = 1 / 0", pf.RUNTIME),
    ("ans = 10.0 ** 1000", pf.RUNTIME),
    ("ans = (-8) ** 0.5", pf.RUNTIME),
    ("ans = 10 ** 100000000", pf.RUNTIME),
    ("result = 5", pf.NO_ANS),
    ("ans = [1, 2]", pf.NOT_A_NUMBER),
    ("ans = (3 > 2) + 1", pf.RUNTIME),
])
def test_errors_are_reported_by_kind(code, kind):
    assert pf.try_execute_python(code) == (None, kind)


def test_a_huge_power_does_not_hang():
    assert pf.try_execute_python("ans = 9 ** 9 ** 9") == (None, pf.RUNTIME)


@pytest.mark.skipif(
    not (data.RAW_DIR / "data" / "train.json").exists(), reason="run chapter 1.2 to download the data"
)
@pytest.mark.parametrize("split", ["train", "dev"])
def test_every_gold_program_translates_to_python_with_the_same_answer(split):
    bad = []
    for conv in data.load_split(split):
        for prog in conv["annotation"]["turn_program"]:
            expected = try_execute(prog)[0]
            got = pf.try_execute_python(pf.to_python(prog))[0]
            if got != expected:
                bad.append((conv["id"], prog, expected, got))
    assert not bad, bad[:5]
