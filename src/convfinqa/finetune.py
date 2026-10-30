"""LoRA fine-tuning of the chat model on training conversations, and a quick accuracy check of the adapter."""

from __future__ import annotations

import gc
import json
import random
import tempfile
import time
from pathlib import Path

from .train import make_dataset

GB = 1024**3


def _eval_at_end_callback():
    """A Trainer callback that evaluates after the last step even when it is not a multiple of `eval_steps`.

    The Trainer saves at the last step but only evaluates on multiples of `eval_steps`, so a final checkpoint
    would have no validation loss and could never be chosen as the best.
    """
    from transformers import TrainerCallback

    class EvalAtEnd(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= state.max_steps and state.global_step % args.eval_steps != 0:
                control.should_evaluate = True
            return control

    return EvalAtEnd()


def _token_weighted_trainer():
    """`SFTTrainer` whose evaluation loss is the mean over answer tokens, not over conversations.

    With an evaluation batch of one conversation, the Trainer averages the conversations' mean losses, so a short
    conversation counts as much as a long one. Training weighs every answer token equally (the batch loss is
    divided by the number of answer tokens in it), and so does `probing.mean_gold_nll`. This trainer scales each
    conversation's loss by its share of the average number of answer tokens, so that the average over conversations
    is the average over tokens. `mean_eval_tokens` must be set to that average before evaluating.
    """
    from trl import SFTTrainer

    class TokenWeightedSFTTrainer(SFTTrainer):
        mean_eval_tokens = 1.0

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            tokens = int((inputs["labels"][..., 1:] != -100).sum())
            return loss * (tokens / self.mean_eval_tokens), logits, labels

    return TokenWeightedSFTTrainer


def train_lora(cfg: dict, train_convs: list[dict], val_convs: list[dict], out: Path, seed: int) -> dict:
    """Fine-tune a model with a LoRA adapter, keep the adapter with the lowest validation loss, and save it.

    The base model's weights are bf16 and stay frozen; the adapter, which trains, is fp32 (both are checked).
    The loss counts the answer tokens of every conversation only (`formatting.tokenize_conversation`), in training
    and on the validation conversations alike: every answer token weighs the same (`_token_weighted_trainer`). The
    validation loss is measured before the first step, every `eval_every` steps and after the last one; the
    checkpoint with the lowest is loaded at the end and written to `out / "adapter"`.

    Args:
        cfg: The settings of the run (`train_config.run_config`).
        train_convs: Training conversations.
        val_convs: Conversations the validation loss is measured on, and the best checkpoint chosen by.
        out: Directory to write into (`adapter/`).
        seed: Seed of the run (initialisation of the adapter, data order, dropout).

    Returns:
        `log_history` (the Trainer's), `max_steps`, `step0_eval_loss`, `best_step`, `best_eval_loss`,
        `final_eval_loss`, `train_seconds`, `peak_gb`, `trainable_params`, `total_params`, and the `dtypes` of
        the base weights and of the adapter.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig

    torch.cuda.set_per_process_memory_fraction(
        min(1.0, cfg["memory_cap_gb"] / (torch.cuda.get_device_properties(0).total_memory / GB))
    )
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM",
    ))
    trainable = {p.dtype for p in model.parameters() if p.requires_grad}
    frozen = {p.dtype for p in model.parameters() if not p.requires_grad}
    if trainable != {torch.float32} or frozen != {torch.bfloat16}:
        raise RuntimeError(f"expected a bf16 base and an fp32 adapter, got {frozen} and {trainable}")
    train = make_dataset(tok, train_convs, cfg["max_length"])
    evaluation = make_dataset(tok, val_convs, cfg["max_length"])

    with tempfile.TemporaryDirectory() as scratch:
        args = SFTConfig(
            output_dir=scratch, per_device_train_batch_size=cfg["micro_batch"],
            gradient_accumulation_steps=cfg["accumulation"], per_device_eval_batch_size=1,
            num_train_epochs=cfg["epochs"], max_steps=cfg["max_steps"] or -1, learning_rate=cfg["lr"],
            lr_scheduler_type=cfg["scheduler"], warmup_steps=cfg["warmup_ratio"], weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg["max_grad_norm"], bf16=True, max_length=cfg["max_length"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps", eval_steps=cfg["eval_every"], eval_on_start=True,
            save_strategy="steps", save_steps=cfg["eval_every"], save_total_limit=1, save_only_model=True,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=1, report_to="none", disable_tqdm=True, seed=seed, data_seed=seed,
        )
        trainer = _token_weighted_trainer()(
            model=model, args=args, train_dataset=train, eval_dataset=evaluation, processing_class=tok,
            callbacks=[_eval_at_end_callback()],
        )
        trainer.mean_eval_tokens = sum(
            sum(l != -100 for l in row["labels"][1:]) for row in trainer.eval_dataset
        ) / len(trainer.eval_dataset)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trainer.train()
        seconds = time.perf_counter() - start
        best_checkpoint = trainer.state.best_model_checkpoint
        history, max_steps = trainer.state.log_history, trainer.state.max_steps
        trainer.model.save_pretrained(Path(out) / "adapter")
    evals = [h for h in history if "eval_loss" in h]
    result = {
        "log_history": history, "max_steps": max_steps,
        "step0_eval_loss": next(h["eval_loss"] for h in evals if h["step"] == 0),
        "best_step": int(Path(best_checkpoint).name.split("-")[-1]),
        "best_eval_loss": trainer.state.best_metric,
        "final_eval_loss": evals[-1]["eval_loss"],
        "train_seconds": seconds, "peak_gb": torch.cuda.max_memory_allocated() / GB,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "dtypes": {"base": str(next(iter(frozen))), "adapter": str(next(iter(trainable)))},
    }
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def check_adapter(cfg: dict, adapter: Path, convs: list[dict], turn_rows: list[dict], seed: int = 0) -> dict:
    """Execution accuracy of a trained adapter on a small sample of validation turns, with the gold history.

    The prompt is the training prompt (the one-line system message, no examples), the generation greedy and one
    prompt at a time, as in `evaluate`. It is a sanity check of the whole chain, not the evaluation of the model:
    that is done on every validation turn, with the predicted history, later.

    Args:
        cfg: The settings of the run (`check_turns`, `model`, `memory_cap_gb`).
        adapter: The directory `train_lora` wrote the adapter to.
        convs: The validation conversations.
        turn_rows: `eval_turns` rows of the validation split (the gold program and answer of every turn).
        seed: Seed of the draw of conversations.

    Returns:
        `summary` (`evaluate.summarize` of the scores), `scores` and `predictions`.
    """
    import torch
    from peft import PeftModel

    from .evaluate import load_model, make_generator, run_generation, score_rows, summarize
    from .formatting import SYSTEM_PROMPT

    order = random.Random(seed).sample(convs, len(convs))
    chosen, n = [], 0
    for conv in order:
        if n >= cfg["check_turns"]:
            break
        chosen.append(conv)
        n += len(conv["annotation"]["dialogue_break"])
    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    generate = make_generator(model, tok, 96)
    rows = run_generation(chosen, generate, system=SYSTEM_PROMPT, examples=[], mode="gold")
    keep = {(c["id"]) for c in chosen}
    scores = score_rows(rows, [r for r in turn_rows if r["id"] in keep])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "summary": summarize(scores), "scores": scores,
        "predictions": [{k: v for k, v in r.items() if k != "seconds"} for r in rows],
    }
