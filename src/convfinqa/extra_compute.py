"""Extra compute at inference time: retry with feedback and self-consistency, for the fine-tuned model.

Both are wrappers around one generation function, so `evaluate.run_generation` runs them unchanged and the
history they leave is defined by the `output` they return: the last attempt for a retry, and the first program
with the majority answer for a vote. Every wrapper also reports what it cost (`generations`, and the prompt and
generated tokens summed over them), because the point of the comparison is what the extra answers buy for it.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from .dsl import executor

STRATEGIES = ("retry", "vote")
RETRY_MESSAGE = "That program did not run: {error}. Reply with only the corrected program."


def run_error(program: str) -> str | None:
    """Why a program does not run, in the executor's words, or `None` if it runs."""
    try:
        executor.execute(program)
    except executor.ProgramError as e:
        return f"{e.kind}: {e}"
    return None


def with_retry(generate: Callable[[list[dict]], dict], max_extra: int) -> Callable[[list[dict]], dict]:
    """A generation function that answers again, with the error message, while the output does not run.

    The retry chat is the original chat, then the failed output as the assistant's turn, then the error as a user
    message; each retry starts from the original chat, not from the earlier retries.

    Args:
        generate: From a chat to a dict with `output`, `finish`, `prompt_tokens`, `new_tokens` and `seconds`.
        max_extra: The most extra attempts after the first.

    Returns:
        A function with the same signature. Its `output` is the last attempt; `generations`, `prompt_tokens`,
        `new_tokens` and `seconds` are totals over the attempts, and `attempts` lists every output.
    """

    def retrying(chat: list[dict]) -> dict:
        attempts, current = [], chat
        for _ in range(max_extra + 1):
            result = generate(current)
            attempts.append(result)
            error = run_error(result["output"])
            if error is None:
                break
            current = chat + [
                {"role": "assistant", "content": result["output"]},
                {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
            ]
        return {
            **attempts[-1],
            "generations": len(attempts),
            "attempts": [a["output"] for a in attempts],
            **{k: sum(a[k] for a in attempts) for k in ("prompt_tokens", "new_tokens", "seconds")},
        }

    return retrying


def vote(outputs: list[str]) -> dict:
    """Pick the answer most of the samples agree on.

    Programs that do not run are dropped. The programs are grouped by their executed answer; the largest group
    wins, a tie going to the group whose first program was sampled earliest.

    Args:
        outputs: The sampled programs, in sampling order.

    Returns:
        `chosen` (index of the first program of the winning group; 0 if none runs, so the answer is wrong),
        `n_valid`, `votes` (the size of the winning group) and `agreement` (`votes` over all samples).
    """
    groups: dict = {}
    for i, text in enumerate(outputs):
        value, error = executor.try_execute(text)
        if error is None:
            groups.setdefault(value, []).append(i)
    if not groups:
        return {"chosen": 0, "n_valid": 0, "votes": 0, "agreement": 0.0}
    best = max(groups.values(), key=lambda idx: (len(idx), -idx[0]))
    return {
        "chosen": best[0], "n_valid": sum(len(g) for g in groups.values()), "votes": len(best),
        "agreement": len(best) / len(outputs),
    }


def with_vote(sample: Callable[[list[dict]], list[dict]]) -> Callable[[list[dict]], dict]:
    """A generation function that samples several answers and returns the vote.

    Args:
        sample: From a chat to a list of generation dicts (`output`, `finish`, `prompt_tokens`, `new_tokens`).

    Returns:
        A function with the signature of `make_generator`'s. Its `output` is the winner of `vote`; the costs are
        totals over the samples, and `samples` lists every output.
    """

    def voting(chat: list[dict]) -> dict:
        results = sample(chat)
        result = vote([r["output"] for r in results])
        return {
            "output": results[result["chosen"]]["output"], "finish": results[result["chosen"]]["finish"],
            "generations": len(results), "samples": [r["output"] for r in results],
            **{k: sum(r[k] for r in results) for k in ("prompt_tokens", "new_tokens")},
            **{k: result[k] for k in ("n_valid", "votes", "agreement")},
        }

    return voting


def make_sampler(model, tok, max_new_tokens: int, n: int, temperature: float) -> Callable[[list[dict]], list[dict]]:
    """A function that samples `n` answers to one chat, at a plain temperature (no top-k or top-p).

    The prompt is processed once per sample, as `generate` expands the input; `prompt_tokens` is the length of
    the prompt, and the totals are summed by `with_vote`.

    Args:
        model: A causal language model.
        tok: Its tokenizer.
        max_new_tokens: Generation limit per sample.
        n: Samples per chat.
        temperature: Sampling temperature.

    Returns:
        A function from a chat to `n` dicts with `output`, `new_tokens`, `finish` and `prompt_tokens`.
    """
    import torch

    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def sample(chat: list[dict]) -> list[dict]:
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_tokens = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                num_return_sequences=n, pad_token_id=tok.pad_token_id,
            )
        results = []
        for row in out[:, prompt_tokens:].tolist():
            stop = next((i for i, t in enumerate(row) if t in eos), None)
            new = row if stop is None else row[: stop + 1]
            results.append({
                "output": tok.decode(new, skip_special_tokens=True), "new_tokens": len(new),
                "finish": "length" if stop is None else "stop", "prompt_tokens": prompt_tokens,
            })
        return results

    return sample


def first_conversations(convs: list[dict], min_turns: int) -> list[dict]:
    """The first conversations, in order, until they hold at least `min_turns` turns."""
    from .data import turns

    chosen, total = [], 0
    for conv in convs:
        if total >= min_turns:
            break
        chosen.append(conv)
        total += len(turns(conv))
    return chosen


def generate_extra(
    cfg: dict, adapter: Path, convs: list[dict], strategy: str, settings: dict, sink: Callable[[dict], None] | None = None
) -> list[dict]:
    """The fine-tuned model's answers to every turn with an extra-compute strategy, in the predicted history.

    Args:
        cfg: The settings of the training run (`model`, `memory_cap_gb`).
        adapter: The directory `finetune.train_lora` wrote the adapter to.
        convs: The conversations to answer.
        strategy: `retry` or `vote`.
        settings: `configs/extra_compute.json`.
        sink: Called with each row as soon as it exists.

    Returns:
        One row per turn, as `evaluate.run_generation` returns them, plus the cost fields of the wrapper.

    Raises:
        ValueError: If the strategy is unknown (before anything is loaded).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
    import torch
    from peft import PeftModel

    from .evaluate import load_model, make_generator, run_generation
    from .formatting import SYSTEM_PROMPT
    from .tuned_eval import MAX_NEW_TOKENS

    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    if strategy == "retry":
        generate = with_retry(make_generator(model, tok, MAX_NEW_TOKENS), settings["retry"]["max_extra"])
    else:
        generate = with_vote(
            make_sampler(model, tok, MAX_NEW_TOKENS, settings["vote"]["samples"], settings["vote"]["temperature"])
        )
    rows = run_generation(convs, generate, system=SYSTEM_PROMPT, examples=[], mode="pred", sink=sink)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
