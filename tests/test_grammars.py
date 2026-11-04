"""The DSL and Python grammars of the constrained-decoding evaluation."""

import re

import pytest

from convfinqa import data, grammars
from convfinqa.constrained import compile_grammar, grammar_accepts
from convfinqa.dsl import python_format

TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
HAVE_DATA = (data.RAW_DIR / "data" / "train.json").exists()


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("xgrammar")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")


def compiled(tok, grammar):
    return compile_grammar(tok, grammar, vocab_size=151936)


@pytest.fixture(scope="module")
def gold_programs():
    if not HAVE_DATA:
        pytest.skip("data not downloaded")
    return {t.program for split in ("train", "dev") for c in data.load_split(split) for t in data.turns(c)}


@pytest.mark.parametrize("text", [
    "ans = 5", "ans = .5", "ans = -6", "ans = (5 - 3) / 3", "v0 = 5 - 3\nans = v0 / v0 * 2", "ans = 2 ** 3",
    "ans = 23 * (-1) > -25", "ans = 17 / 100 * 200", "v0 = 1\nv1 = v0 + 1\nans = v1",
])
def test_the_python_grammar_accepts_the_forms_the_translator_writes(tok, text):
    assert grammar_accepts(compiled(tok, grammars.PYTHON_GRAMMAR), text)


@pytest.mark.parametrize("text", [
    "", "5", "ans = ", "ans = 5 +", "ans = 1  + 2", "ans = 5\n", "x = 1\nans = x", "import math\nans = 1",
    "ans = round(2.5, 1)", "```python\nans = 1\n```", "ans = 1 // 2", "ans = 1 > 2 > 3", "The answer is ans = 5",
])
def test_the_python_grammar_rejects_everything_else(tok, text):
    assert not grammar_accepts(compiled(tok, grammars.PYTHON_GRAMMAR), text)


@pytest.mark.parametrize("bounded", [False, True])
def test_both_grammars_accept_every_gold_program(tok, gold_programs, bounded):
    dsl = compiled(tok, grammars.dsl_grammar(bounded))
    py = compiled(tok, grammars.python_grammar(bounded))
    assert [p for p in gold_programs if not grammar_accepts(dsl, p)] == []
    assert [p for p in gold_programs if not grammar_accepts(py, python_format.to_python(p))] == []


def test_the_bounds_are_the_digits_of_the_gold_programs(gold_programs):
    numbers = [m for p in gold_programs for m in re.findall(r"\d*\.?\d+", p)]
    assert max(len(n.split(".")[0]) for n in numbers) == grammars.MAX_INTEGER_DIGITS
    assert max(len(n.split(".")[1]) if "." in n else 0 for n in numbers) == grammars.MAX_DECIMAL_DIGITS


def test_the_bounded_grammars_stop_a_number_that_runs_on(tok):
    for grammar, prefix in [(grammars.dsl_grammar(True), ""), (grammars.python_grammar(True), "ans = ")]:
        g = compiled(tok, grammar)
        assert grammar_accepts(g, prefix + "1234567890") and not grammar_accepts(g, prefix + "12345678901")
        assert grammar_accepts(g, prefix + "2.7000") and not grammar_accepts(g, prefix + "2.70000")
    unbounded = compiled(tok, grammars.dsl_grammar(False))
    assert grammar_accepts(unbounded, "2.70000000000")


def test_the_bounded_dsl_grammar_differs_from_the_unbounded_one_only_in_the_number_rule():
    a, b = grammars.dsl_grammar(False).splitlines(), grammars.dsl_grammar(True).splitlines()
    assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] and len(a) == len(b)
    assert all(x.startswith("number") for x, y in zip(a, b) if x != y)


def test_a_constrained_generator_on_a_tiny_random_model_always_writes_the_grammar(tok):
    """A random model cannot write a program by itself: whatever it writes, the grammar makes it one."""
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, Qwen3Config

    from convfinqa.constrained_eval import make_constrained_generator

    torch.manual_seed(0)
    cfg = Qwen3Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=151936, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    for grammar, prefix in [(grammars.dsl_grammar(True), ""), (grammars.python_grammar(True), "ans = ")]:
        g = compiled(tok, grammar)
        row = make_constrained_generator(model, tok, g, max_new_tokens=60)(chat)
        assert row["finish"] in ("stop", "length") and row["prompt_tokens"] > 0
        if row["finish"] == "stop":
            assert grammar_accepts(g, row["output"]), row["output"]
        else:   # cut by the limit: the text so far is a prefix of the language, and so far it has followed the grammar
            import xgrammar as xgr
            m = xgr.GrammarMatcher(g, terminate_without_stop_token=True)
            assert m.accept_string(row["output"])
