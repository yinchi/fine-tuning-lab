"""Probe a model on ConvFinQA: greedy answers, a sorted list of how they fail, and gold-program surprisal."""

from __future__ import annotations

from .dsl import executor, steps
from .formatting import tokenize_conversation

# Failure kinds beyond the executor's own error kinds (`executor.SYNTAX`, ...).
CORRECT = "correct"
FENCED = "code_fence"
WRONG_OPERATIONS = "wrong_operations"
WRONG_NUMBERS = "wrong_numbers"
WRONG_ANSWER_SAME_SHAPE = "wrong_answer"


def classify_output(output: str, gold_program: str, gold_answer) -> str:
    """Sort one model output into correct, or a kind of failure.

    The output is judged exactly as generated, with no cleaning, so it follows the headline rule of
    chapter 3.4 (one attempt, invalid counts as wrong). Outputs that do not run get the executor's
    error kind, or `code_fence` when the program is wrapped in a markdown fence. Outputs that run
    but give the wrong answer are split by comparing the program with the gold one: a different list
    of operations, a different set of numbers, or neither.

    Args:
        output: The model's text for one turn.
        gold_program: The gold DSL program.
        gold_answer: The gold execution result.

    Returns:
        `correct`, an executor error kind, `code_fence`, `wrong_operations`, `wrong_numbers` or
        `wrong_answer`.
    """
    value, error = executor.try_execute(output)
    if error is not None:
        return FENCED if "```" in output else error
    if executor.answers_match(value, gold_answer):
        return CORRECT
    ops = lambda p: [s.split("(")[0] for s in steps.split_steps(p)]
    if ops(output) != ops(gold_program):
        return WRONG_OPERATIONS
    args = lambda p: sorted(a for s in executor.parse_program(p) for a in s.args if not a.startswith("#"))
    return WRONG_NUMBERS if args(output) != args(gold_program) else WRONG_ANSWER_SAME_SHAPE


def generate_greedy(model, tok, chats: list[list[dict]], batch_size: int = 4, max_new_tokens: int = 96) -> list[str]:
    """Greedy answers to a list of chats, in the order given.

    The prompts are sorted by length once and run in fixed batches with left padding, because the
    batch composition changes bf16 logits slightly (see the 2026-10-22 decision).

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        chats: Each chat ends with a user message.
        batch_size: Prompts per batch; keep it fixed for reproducible output.
        max_new_tokens: Generation limit.

    Returns:
        The generated text for each chat, without special tokens.
    """
    import torch

    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [""] * len(texts)
    tok.padding_side = "left"
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
            )
        for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            out[i] = tok.decode(g, skip_special_tokens=True)
    return out


def gold_logprobs(model, tok, messages: list[dict]) -> tuple[list[int], list[float]]:
    """Log-probability the model gave each trained token of a chat.

    One forward pass over the whole conversation; token `t` is scored by the logits at `t - 1`, as
    in training. Only the assistant's tokens (`tokenize_conversation`'s labels) are returned.

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        messages: A full chat with gold assistant messages.

    Returns:
        The trained token ids and their log-probabilities (natural log, at most 0).
    """
    import torch

    ids, labels = tokenize_conversation(tok, messages)
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(x).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, x[0, 1:, None])[:, 0].cpu().tolist()
    keep = [t for t in range(1, len(ids)) if labels[t] != -100]
    return [ids[t] for t in keep], [logp[t - 1] for t in keep]


def mean_gold_nll(model, tok, chats: list[list[dict]]) -> dict:
    """Negative log-likelihood of the gold programs over a set of conversations.

    This is the quantity fine-tuning minimizes: the mean over trained tokens, so a conversation
    counts in proportion to its length.

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        chats: Full chats with gold assistant messages (`conversation_messages`).

    Returns:
        `nll` (mean per token), `tokens`, `conversations`, and `per_conversation`, a list of
        `[tokens, summed nll]`.
    """
    per = []
    for messages in chats:
        _, logp = gold_logprobs(model, tok, messages)
        per.append([len(logp), -sum(logp)])
    tokens = sum(n for n, _ in per)
    return {
        "nll": sum(s for _, s in per) / tokens,
        "tokens": tokens,
        "conversations": len(per),
        "per_conversation": per,
    }


def probe_sample(model, tok, sample: list[dict], shots: list[dict]) -> dict:
    """Everything chapter 2.4 shows for a sample of conversations.

    Every turn is asked with the gold history, zero-shot and with `shots` in front, and judged by
    `classify_output`. The gold programs' token log-probabilities are recorded per conversation.

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        sample: The conversations to probe.
        shots: Example conversations for the few-shot prompt.

    Returns:
        `answers`: one row per setting and turn (`setting`, `id`, `turn`, `output`, `gold`, `kind`);
        `surprisal`: per conversation, the `id`, trained token strings and their log-probabilities.
    """
    from .data import turns
    from .formatting import conversation_messages, prompt_messages

    items = [(c, k) for c in sample for k in range(len(c["annotation"]["dialogue_break"]))]
    answers = []
    for setting, examples in {"zero-shot": [], "few-shot": shots}.items():
        outs = generate_greedy(model, tok, [prompt_messages(c, k, examples) for c, k in items])
        for (c, k), out in zip(items, outs):
            t = turns(c)[k]
            answers.append({
                "setting": setting, "id": c["id"], "turn": k, "output": out, "gold": t.program,
                "kind": classify_output(out, t.program, t.answer),
            })
    surprisal = []
    for c in sample:
        ids, logp = gold_logprobs(model, tok, conversation_messages(c))
        surprisal.append({"id": c["id"], "tokens": [tok.decode([i]) for i in ids], "logprobs": logp})
    return {"answers": answers, "surprisal": surprisal}
