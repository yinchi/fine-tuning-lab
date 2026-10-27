"""Builders shared by several chapters. Most builders live in the chapter cell that uses them.

Importing this module registers its `@artifact` functions.

Every builder must be self-contained: it may use only its parameters, builtins and imports made inside
the function (absolute imports). That is checked when it is registered, and it is what lets the code
hash cover just the function and the `convfinqa` modules it imports, so editing another builder in
this file never makes an artifact stale. It also keeps heavy imports (torch, transformers) out of
module import time, which `status` and `refresh` pay for.
"""


from convfinqa.artifacts import artifact


@artifact("train_val_split", seed=0)
def build_train_val_split(out, deps):
    """`train.json` split into `train_fit.jsonl` and `val.jsonl`, grouped by report page."""
    import json

    from convfinqa import data

    fit, val = data.make_val_split(data.load_split("train"), fraction=0.1, seed=0)
    for name, convs in (("train_fit", fit), ("val", val)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in convs))


@artifact("eval_turns", deps=["train_val_split"])
def build_eval_turns(out, deps):
    """The turns of val and dev as tables (`val_turns.jsonl`, `dev_turns.jsonl`), and dev's conversations."""
    import json

    from convfinqa import data
    from convfinqa.data.tables import turn_rows

    val = list(deps["train_val_split"].jsonl("val.jsonl"))
    dev = data.load_split("dev")
    for name, rows in (("val_turns", turn_rows(val)), ("dev_turns", turn_rows(dev))):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "dev.jsonl").write_text("".join(json.dumps(c) + "\n" for c in dev))
