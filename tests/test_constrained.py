import torch

from convfinqa.constrained import AllowedTokens


def test_only_allowed_tokens_survive():
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    out = AllowedTokens([1, 3])(torch.zeros(2, 1, dtype=torch.long), scores)
    assert torch.isinf(out[:, [0, 2, 4]]).all() and (out[:, [0, 2, 4]] < 0).all()
    assert torch.equal(out[:, [1, 3]], scores[:, [1, 3]])


def test_probabilities_renormalise_over_the_allowed_tokens():
    p = torch.softmax(AllowedTokens([0, 2])(torch.zeros(1, 1, dtype=torch.long), torch.tensor([[1.0, 9.0, 1.0]])), -1)
    assert p[0, 1] == 0 and torch.allclose(p[0, [0, 2]], torch.tensor([0.5, 0.5]))


# --- the DSL grammar (xgrammar) -------------------------------------------------------------------------

import pytest

from convfinqa import data
from convfinqa.constrained import DSL_GRAMMAR, compile_grammar, generate_constrained, grammar_accepts

TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def dsl():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("xgrammar")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")
    return compile_grammar(tok, DSL_GRAMMAR, vocab_size=151936)


@pytest.mark.parametrize("program", [
    "206588", "17%", "-3.5", ".5", "subtract(206588, 181001)", "subtract(1, 2), divide(#0, const_100)",
    "add(1, const_m1)", "greater(5.2%, #0)", "exp(2, 3), multiply(#0, 1000)",
])
def test_the_grammar_accepts_programs(dsl, program):
    assert grammar_accepts(dsl, program)


@pytest.mark.parametrize("text", [
    "", "average(1, 2)", "add(1)", "add(1,2)", "add(1, 2) ", "add(1, 2),divide(#0, 2)", "```add(1, 2)```",
    "divide(#0, 181001), ", "5 million", "add(1, const_11)", "add(1, 2", "add(1, 2))", "1,234",
])
def test_the_grammar_rejects_everything_else(dsl, text):
    assert not grammar_accepts(dsl, text)


@pytest.mark.skipif(not (data.RAW_DIR / "data" / "train.json").exists(), reason="data not downloaded")
def test_the_grammar_accepts_every_gold_program(dsl):
    programs = {t.program for split in ("train", "dev") for c in data.load_split(split) for t in data.turns(c)}
    assert len(programs) == 8981
    assert [p for p in programs if not grammar_accepts(dsl, p)] == []


def test_constrained_generation_always_follows_the_grammar():
    """A short run on the GPU: even an untuned model's answers are programs, because it cannot write anything else."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
        ).eval()
    except OSError:
        pytest.skip("model not cached")
    from convfinqa.formatting import prompt_messages

    compiled = compile_grammar(tok, DSL_GRAMMAR, vocab_size=model.config.vocab_size)
    convs = data.load_split("dev")[:3]
    chats = [prompt_messages(c, 0) for c in convs]
    outs = generate_constrained(model, tok, chats, compiled, batch_size=2, max_new_tokens=64)
    assert all(grammar_accepts(compiled, o) for o in outs), outs
