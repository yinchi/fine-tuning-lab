import pytest

from convfinqa import merged_eval


def test_the_precisions():
    assert merged_eval.PRECISIONS == ("bf16", "fp32")


def test_an_unknown_precision_is_refused_before_anything_is_loaded():
    with pytest.raises(ValueError):
        merged_eval.generate_merged({}, None, [], "pred", "fp16")
