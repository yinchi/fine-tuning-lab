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
- **Artifacts** (`src/convfinqa/artifacts/`): named results in `artifacts/`, with recursive freshness. Declare a builder in the chapter cell that uses it (`@artifact(...)` on a function, then `get("name")`); the registry finds it by scanning the `.qmd` files, so the function must be a plain top-level `def` in a `{python}` cell (cells with `#| eval: false` are ignored; use a plain `python` fence for examples) and self-contained (parameters, builtins, imports inside the body). Every builder must be self-contained, in a module too: the code hash covers only the builder's own function and the `convfinqa` modules it imports, so editing the file around it never invalidates it. Names are letters, digits and underscores (`sft_dsl_0`). A builder's signature is `(out, deps, files, *, <name fields>)` and nothing else: hyperparameters live in a declared file or a light config artifact, and the heavy logic lives in `src/convfinqa/`. Freshness also compares a hash of the builder's normalized source and of the `convfinqa` modules it imports (whitespace, comments and docstrings don't count); bump `version` for changes that can't see (library upgrades). Seeds are name fields or the decorator's `seed`; the framework seeds `random`/numpy/torch at the start of every build, and chapters seed their own stochastic cells locally. Heavy artifacts (training, evaluation, baselines) build in a subprocess only via `python -m convfinqa.artifacts build <name>|--all` or `RUN_HEAVY=1`; otherwise a stale or missing one raises, and `ALLOW_STALE=1` loads a stale-but-present one with a visible warning. `build --all` builds every name that a chapter requests with `get("literal")`. Cells that call them carry `#HEAVY`. All chapters use `freeze: auto`, never `freeze: true`. `refresh` (run automatically by the `pre-render` hook) re-executes chapters that used a changed artifact. Build with `./build.sh` (builds the DAG, renders, previews) in `screen`.
- **Splits:** tune on `val` (page-grouped 10% of train). Use `dev` only for final reported numbers.
- **Chapter files** are named `docs/PCC_first_words.qmd` (P = part, CC = chapter). Appendices are `9CC`. Titles carry no number: Quarto numbers chapters itself (`number-sections: true`), so rendered numbers differ from the `PCC` codes. A chapter's title is its first heading, `# Title {#sec-name}` (a YAML `title:` can't be labelled). Refer to written chapters with `@sec-name`, never "chapter 2.4" (`tests/test_book_refs.py`); for a chapter not written yet, use words.
- **Teaching style:** demonstrate claims with code on the real model and data. Use "Under the hood" and "Further reading" callouts. Cite via `docs/references.bib` and link terms to the glossary.

## Commands
- `uv sync`: install dependencies. torch comes from the cu132 index.
- `uv run pytest`: run the tests.
- `quarto render docs/` renders the book. `quarto render docs/<file>.qmd` re-executes one chapter. Pass `--no-execute-daemon` (also to `quarto preview`): by default Quarto keeps a chapter's Jupyter kernel alive for 5 minutes after a render, and a reused kernel keeps its old environment variables (`RUN_HEAVY`, `ALLOW_STALE`) and loaded code; `execute: daemon: false` in `_quarto.yml` has no effect in Quarto 1.10.19. `./build.sh` already passes it.
