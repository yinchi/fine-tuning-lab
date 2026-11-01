"""The wrong-scale check and the literal check."""

from convfinqa.literals import literal_flags, literals, page_numbers, value
from convfinqa.scale_check import OTHER_POWER, PERCENT, percent_style, scale_kind


def test_a_factor_of_100_either_way():
    assert scale_kind(22.8588, 0.228588) == PERCENT
    assert scale_kind(0.14136, 14.136) == PERCENT


def test_other_powers_of_ten_are_reported_together():
    assert scale_kind(0.0021 * 10**4, 0.0021) == OTHER_POWER
    assert scale_kind(210.0, 0.021) == OTHER_POWER
    assert scale_kind(8.70854, 0.87085) == OTHER_POWER
    assert scale_kind(6000000.0, 6.0) == OTHER_POWER
    assert scale_kind(6.0, 6000.0) == OTHER_POWER
    assert scale_kind(13.0, 13000000.0) == OTHER_POWER


def test_rounding_of_the_gold_answer_is_allowed_for():
    assert scale_kind(14.136, 0.14136) == PERCENT
    assert scale_kind(14.14, 0.14136) is None


def test_no_scale_error():
    assert scale_kind(5.0, 5.0) is None
    assert scale_kind(7.0, 5.0) is None
    assert scale_kind(0.0, 5.0) is None
    assert scale_kind(5.0, 0.0) is None
    assert scale_kind(None, 5.0) is None
    assert scale_kind("yes", "no") is None
    assert scale_kind(0.0, 123.0) is None      # rounds to 0 at 5 decimals: no information
    assert scale_kind(12345.0, 5.0) is None


def test_percent_style_questions():
    assert percent_style("what percentage change does this represent?")
    assert percent_style("what is that in percent?")
    assert not percent_style("what is the difference?")


def test_literals():
    assert literals("subtract(1,234, 100.00), divide(#0, const_m1), add(17%, x1)") == ["1234", "100.00", "const_m1", "17%"]
    assert literals("add(206588, 181001)") == ["206588", "181001"]
    assert literals("subtract(-4027, 2.70), divide(#0, const_100)") == ["-4027", "2.70", "const_100"]
    assert literals("x = 5\ny = x1 + 17%") == ["5", "17%"]
    assert literals("add(1,234, 5)") == ["1234", "5"]


def test_values():
    assert value("2.70") == value("2.7") == 2.7
    assert value("17%") == 0.17
    assert value("const_100") == 100.0 and value("const_m1") == -1.0


def test_flags():
    page = page_numbers("sales of 206,588 and 181,001 in 2009, up 14 % ( 14 % ). losses ( 4,027 ).")
    same = literal_flags("subtract(206588, 181001)", "subtract(206588, 181001)", page)
    assert same == {"n_literals": 2, "lit_text": True, "lit_value": True, "lit_grounded": True}
    zeros = literal_flags("subtract(206588.0, 181001)", "subtract(206588, 181001)", page)
    assert (zeros["lit_text"], zeros["lit_value"]) == (False, True)
    constant = literal_flags("divide(5, 100)", "divide(5, const_100)", page)
    assert (constant["lit_text"], constant["lit_value"]) == (False, True)
    percent = literal_flags("add(14%, 5)", "add(0.14, 5)", page)
    assert (percent["lit_text"], percent["lit_value"]) == (False, True)
    invented = literal_flags("subtract(206588, 999)", "subtract(206588, 181001)", page)
    assert invented["lit_grounded"] is False
    sign = literal_flags("add(4027, 14)", "add(-4027, 14)", page)
    assert sign["lit_grounded"] is True and sign["lit_text"] is False
    assert literal_flags("divide(5, const_100)", "5", page)["lit_grounded"] is False
    assert literal_flags("const_100", "const_100", page)["lit_grounded"] is True
