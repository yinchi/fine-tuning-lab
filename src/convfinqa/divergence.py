"""Where a wrong program parts from the gold one: how close was the right token?

For a turn the fine-tuned model got wrong, both programs are written as the tokens the model would produce
(followed by the end-of-turn token). Up to the first token where they differ the two are the same text, so the
model's distribution at that position is the one it chose from, and the question is where the gold token stood in
it: its rank, and its probability against the probability of the token the model wrote.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable



def first_divergence(a: list[int], b: list[int]) -> int | None:
    """The index of the first position where two token lists differ, or `None` if they are the same.

    A list that is a proper prefix of the other differs from it at its own length, so `a` followed by the
    end-of-turn token against a longer `b` is a divergence at the end-of-turn token.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def choice_at(logits, gold_id: int, pred_id: int) -> dict:
    """What the model's distribution over the next token says about two candidates.

    Args:
        logits: The logits over the vocabulary at one position (a 1-D tensor).
        gold_id: The gold program's token.
        pred_id: The token the model's program has there.

    Returns:
        `logp_gold` and `logp_pred` (natural log), `rank_gold` and `rank_pred` (1 is the most probable token; ties
        share the better rank), `top` (the id of the most probable token) and `gold_over_pred` (the ratio of the
        probabilities).
    """
    import math

    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    rank = lambda t: 1 + int((logp > logp[t]).sum())
    lg, lp = float(logp[gold_id]), float(logp[pred_id])
    return {
        "logp_gold": lg, "logp_pred": lp, "rank_gold": rank(gold_id), "rank_pred": rank(pred_id),
        "top": int(logp.argmax()), "gold_over_pred": math.exp(lg - lp),
    }


def score_divergences(
    cfg: dict, adapter: Path, convs: list[dict], rows: list[dict], turn_rows: list[dict], wrong: set[tuple],
    sink: Callable[[dict], None] | None = None,
) -> list[dict]:
    """For every wrong turn, the model's choice at the first token where its program differs from the gold one.

    The prompt is the one the predicted-history run used: the training prompt and the model's own earlier answers.
    One forward pass over the prompt and the gold program gives the distribution at every position of the gold
    program; the distribution at the divergence is the same one the model wrote its own token from.

    Args:
        cfg: The settings of the training run (`model`, `memory_cap_gb`).
        adapter: The directory `finetune.train_lora` wrote the adapter to.
        convs: The conversations of the split.
        rows: The predictions of the predicted-history run (`id`, `turn`, `output`).
        turn_rows: The gold turns (`id`, `turn`, `program`).
        wrong: The `(id, turn)` pairs to explain.
        sink: Called with each result as soon as it exists.

    Returns:
        One dict per wrong turn: `id`, `turn`, `position` (index of the divergence in the program's tokens, the
        end-of-turn token included), `n_gold`, `n_pred`, `gold_token`, `pred_token`, `chosen_is_top` (whether the
        model's token is the most probable one when scored this way, which should hold for greedy decoding),
        `gold_total_logp`, and the fields of `choice_at`. A turn whose program equals the gold one in text is
        reported with `position` `None`.
    """
    import torch
    from peft import PeftModel

    from .evaluate import load_model
    from .formatting import SYSTEM_PROMPT
    from .prompts import prompt_chat

    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    gold = {(r["id"], r["turn"]): r["program"] for r in turn_rows}
    output = {(r["id"], r["turn"]): r["output"] for r in rows}
    results = []
    for conv in convs:
        n_turns = max(t for i, t in output if i == conv["id"]) + 1
        history = [output[(conv["id"], k)] for k in range(n_turns)]
        for k in range(n_turns):
            if (conv["id"], k) not in wrong:
                continue
            chat = prompt_chat(conv, k, SYSTEM_PROMPT, [], history)
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompt = tok(text, add_special_tokens=False)["input_ids"]
            gold_ids = tok(gold[(conv["id"], k)], add_special_tokens=False)["input_ids"] + [end]
            pred_ids = tok(history[k], add_special_tokens=False)["input_ids"] + [end]
            d = first_divergence(gold_ids, pred_ids)
            result = {"id": conv["id"], "turn": k, "position": d, "n_gold": len(gold_ids), "n_pred": len(pred_ids)}
            x = torch.tensor([prompt + gold_ids], device=model.device)
            with torch.no_grad():
                logits = model(x).logits[0].float()
            at = len(prompt) - 1  # the logits at position `at + i` give program token `i`
            logp = torch.log_softmax(logits[at: at + len(gold_ids)], dim=-1)
            result["gold_total_logp"] = float(logp.gather(1, torch.tensor(gold_ids, device=logp.device)[:, None]).sum())
            if d is not None:
                result.update(choice_at(logits[at + d], gold_ids[d], pred_ids[d]))
                result["chosen_is_top"] = result["top"] == pred_ids[d]
                result["gold_token"], result["pred_token"] = tok.decode([gold_ids[d]]), tok.decode([pred_ids[d]])
            results.append(result)
            if sink:
                sink(result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results
