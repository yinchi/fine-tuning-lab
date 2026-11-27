"""`train-adapter`: build the adapter from scratch, with the settings of the book's main run.

Downloads the data if needed, makes the book's page-grouped validation split of the training set, trains a LoRA adapter
(`finetune.train_lora`, the settings of the run `main` in `configs/train.json`), keeps the checkpoint with the lowest
validation loss, and writes the adapter directory with `train_info.json`. The development split is never read.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import tempfile
from pathlib import Path

from ..data import ROOT
from .info import format_fingerprint, write_info

DEFAULT_OUT = Path("outputs") / "adapter"
DEFAULT_CONFIG = ROOT / "configs" / "train.json"
EXTRA_FILES = ("train_log.json",)


def check_output(out: Path, force: bool) -> None:
    """Refuse a non-empty output directory unless `force`.

    Raises:
        FileExistsError: If `out` has files in it and `force` is false.
    """
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not force:
        raise FileExistsError(f"{out} already exists and is not empty; use --force to replace it")


def base_revision(model: str) -> str | None:
    """The commit of `model` in the local Hugging Face cache, or `None` if it cannot be found."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, local_files_only=True)).name
    except Exception:  # noqa: BLE001 - the revision is a note in a file, never a reason to lose a run
        return None


def make_info(cfg: dict, run: str, seed: int, result: dict, n_train: int, n_val: int) -> dict:
    """The content of `train_info.json` (see `info.check_adapter`): what the adapter was trained from, and how it went."""
    from ..artifacts.provenance import input_revisions, library_versions

    return {
        "format": format_fingerprint(),
        "base_model": cfg["model"],
        "base_revision": base_revision(cfg["model"]),
        "dataset_commit": input_revisions(ROOT)["dataset_commit"],
        "run": run, "seed": seed, "settings": cfg,
        "train_conversations": n_train, "val_conversations": n_val,
        "max_steps": result["max_steps"], "best_step": result["best_step"],
        "step0_eval_loss": result["step0_eval_loss"], "best_eval_loss": result["best_eval_loss"],
        "train_seconds": result["train_seconds"], "peak_gb": result["peak_gb"],
        "library_versions": library_versions(),
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train-adapter", description="Train the book's LoRA adapter from scratch (train and validation splits "
        "only; the development split is not read).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the training (default 0, the book's)")
    parser.add_argument("--run", default="main", help="the run of the settings file to train (default main)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="settings file (default configs/train.json)")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps, to try the pipeline")
    parser.add_argument("--force", action="store_true", help="replace a non-empty output directory")
    args = parser.parse_args(argv)

    from ..train_config import run_config

    try:
        check_output(args.out, args.force)
        cfg = run_config(json.loads(args.config.read_text()), args.run)
    except (OSError, ValueError, KeyError) as e:
        print(f"train-adapter: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.max_steps is not None:
        cfg = {**cfg, "max_steps": args.max_steps}

    import torch

    if not torch.cuda.is_available():
        print("train-adapter: no CUDA GPU is available; training runs on one", file=sys.stderr)
        raise SystemExit(2)

    from ..artifacts.seeding import seed_everything
    from ..data import download_data, load_split, make_val_split
    from ..finetune import train_lora

    download_data()
    fit, val = make_val_split(load_split("train"), fraction=0.1, seed=0)
    steps = cfg["max_steps"] or "the full number of"
    print(
        f"Training {cfg['model']} with LoRA r={cfg['lora']['r']} on {len(fit)} conversations ({steps} steps), "
        f"validating on {len(val)}; seed {args.seed}. The book's run took about 4 hours on a GB10; this is not a "
        f"prediction for another GPU. Writing to {args.out}.", file=sys.stderr,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as work:
        work = Path(work)
        seed_everything(args.seed)
        result = train_lora(cfg, fit, val, work, args.seed)
        info = make_info(cfg, args.run, args.seed, result, len(fit), len(val))
        (work / "adapter" / EXTRA_FILES[0]).write_text(
            json.dumps({k: result[k] for k in ("log_history", "max_steps")})
        )
        write_info(work / "adapter", info)
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.move(work / "adapter", args.out)
    print(
        f"Done: best validation loss {info['best_eval_loss']:.4f} at step {info['best_step']} of {info['max_steps']} "
        f"(step 0: {info['step0_eval_loss']:.4f}). Adapter in {args.out}.", file=sys.stderr,
    )
