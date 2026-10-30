"""The settings of a training run: the common ones, with the changes a named run makes to them."""

from __future__ import annotations

REQUIRED = (
    "model", "lora", "lr", "scheduler", "warmup_ratio", "weight_decay", "max_grad_norm", "epochs", "max_steps",
    "micro_batch", "accumulation", "max_length", "eval_every", "gradient_checkpointing", "memory_cap_gb", "check_turns",
)


def run_config(cfg: dict, run: str) -> dict:
    """The settings of one named run: `cfg["common"]` with the run's own entries on top.

    A run is only a few overrides (its model, its length), so a change to one run's section leaves the others'
    settings, and anything built from them, as they were.

    Args:
        cfg: The parsed `configs/train.json`, with `common` and `runs`.
        run: The name of a run in `cfg["runs"]`.

    Returns:
        The merged settings, with every key of `REQUIRED`. The `lora` entry is merged key by key.

    Raises:
        KeyError: If the run is not defined.
        ValueError: If a setting is missing or unknown after the merge.
    """
    if run not in cfg["runs"]:
        raise KeyError(f"unknown training run {run!r}; defined: {sorted(cfg['runs'])}")
    merged = {**cfg["common"], **cfg["runs"][run]}
    if "lora" in cfg["runs"][run]:
        merged["lora"] = {**cfg["common"]["lora"], **cfg["runs"][run]["lora"]}
    missing, unknown = set(REQUIRED) - set(merged), set(merged) - set(REQUIRED)
    if missing or unknown:
        raise ValueError(f"run {run!r}: missing settings {sorted(missing)}, unknown settings {sorted(unknown)}")
    return merged
