import json

import pytest

from convfinqa.cli.page import PageError, load_page, validate_page

PAGE = {"pre_text": ["before ."], "table": [["", "2000"], ["sales", "5"]], "post_text": []}


def test_a_page_keeps_only_its_three_fields():
    entry = {**PAGE, "id": "x", "table_ori": [["a"]], "annotation": {"turn_program": ["5"]}, "gold": []}
    assert validate_page(entry) == PAGE


def test_every_missing_key_is_named():
    with pytest.raises(PageError, match="missing `pre_text`.*missing `post_text`"):
        validate_page({"table": [["a"]]})


@pytest.mark.parametrize("bad, text", [
    ([], "JSON object"),
    ({**PAGE, "pre_text": "one string"}, "`pre_text` must be a list of strings"),
    ({**PAGE, "post_text": [1]}, "`post_text` must be a list of strings"),
    ({**PAGE, "table": []}, "non-empty list of rows"),
    ({**PAGE, "table": "x"}, "non-empty list of rows"),
    ({**PAGE, "table": [["a"], [1, 2]]}, "row 1"),
    ({**PAGE, "table": [["a"], "b"]}, "row 1"),
])
def test_bad_shapes_say_what_is_wrong(bad, text):
    with pytest.raises(PageError, match=text):
        validate_page(bad)


def test_empty_text_lists_are_fine():
    assert validate_page({**PAGE, "pre_text": []})["pre_text"] == []


def test_the_typo_of_a_key_is_reported_as_missing():
    page = {"pre_text": [], "table": [["a"]], "post_txt": []}
    with pytest.raises(PageError, match="missing `post_text`"):
        validate_page(page)


def test_load_page_reads_a_file_and_names_it_in_errors(tmp_path):
    good = tmp_path / "p.json"
    good.write_text(json.dumps({**PAGE, "id": "x"}))
    assert load_page(good) == PAGE
    for name, text, message in (("bad.json", "{oops", "not valid JSON"), ("odd.json", "[]", "JSON object")):
        (tmp_path / name).write_text(text)
        with pytest.raises(PageError, match=f"{name}.*{message}"):
            load_page(tmp_path / name)
    with pytest.raises(PageError, match="cannot read"):
        load_page(tmp_path / "missing.json")
