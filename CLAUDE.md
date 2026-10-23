# fine-tuning-lab

A Quarto-book lab for learning LLM fine-tuning (Hugging Face, LoRA SFT) on ConvFinQA.

## Start of every session
1. Read `PLAN.md`: it's the source of truth for scope, decisions, and progress.
2. Find the first unchecked item in **Progress tracker** and continue from there, unless the user says otherwise.

## Keeping the plan current
- When an item is done **and verified**, tick it in **Progress tracker** with a short note (run id, headline number).
- When a decision is made or changed, add a dated entry to **Decision log**. Never rewrite old entries, and update the affected plan sections to match.
- Commit `PLAN.md` updates together with the change they describe.

## Conventions
- **Logic lives in `src/convfinqa/`** and is tested in `tests/`. Book chapters stay thin: narration, config, calls into the package, plots. Layers: `dsl` → `data` → model-facing modules (`formatting`, `constrained`, later `llm/`); `book` for presentation; `artifacts`/`builders` for infrastructure. Import downward only; keep torch out of `dsl` and `data` (`tests/test_layers.py`).
- **Stages pass data through files** in `artifacts/`, never through kernel state.
- **Artifacts** (`src/convfinqa/artifacts.py`): named results in `artifacts/`, with recursive freshness. A builder's signature is `(out, deps, files, *, <name fields>)` and nothing else: hyperparameters live in a declared file or a light config artifact. Freshness also compares a hash of the normalized source of the modules the builder imports (whitespace, comments and docstrings don't count); bump `version` for changes that can't see (library upgrades). Seeds are name fields or the decorator's `seed`; the framework seeds `random`/numpy/torch at the start of every build, and chapters seed their own stochastic cells locally. Heavy artifacts (training, evaluation, baselines) build in a subprocess only via `python -m convfinqa.artifacts build <name>|--all` or `RUN_HEAVY=1`; otherwise a stale or missing one raises, and `ALLOW_STALE=1` loads a stale-but-present one with a visible warning. List the book's artifacts in `src/convfinqa/targets.py`. Cells that call them carry `#HEAVY`. All chapters use `freeze: auto`, never `freeze: true`. `refresh` (run automatically by the `pre-render` hook) re-executes chapters that used a changed artifact. Build with `./build.sh` (builds the DAG, renders, previews) in `screen`.
- **Splits:** tune on `val` (page-grouped 10% of train). Use `dev` only for final reported numbers.
- **Chapter files** are named `docs/PCC_first_words.qmd` (P = part, CC = chapter). Appendices are `9CC`. Titles carry no number: Quarto numbers chapters itself (`number-sections: true`), so rendered numbers differ from the `PCC` codes.
- **Teaching style:** demonstrate claims with code on the real model and data. Use "Under the hood" and "Further reading" callouts. Cite via `docs/references.bib` and link terms to the glossary.

## Commands
- `uv sync`: install dependencies. torch comes from the cu132 index.
- `uv run pytest`: run the tests.
- `quarto render docs/` renders the book. `quarto render docs/<file>.qmd` re-executes one chapter.
