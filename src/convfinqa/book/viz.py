"""Small helpers for showing model internals in the book."""

from __future__ import annotations

from html import escape

# Hues for alternating token backgrounds. Semi-transparent, so text stays readable on light and dark pages.
_HUES = (210, 30, 130, 330, 50, 270, 170, 0)


def tokens_html(pieces: list[str], ids: list[int] | None = None) -> str:
    """Render tokens as coloured spans, like the OpenAI tokenizer page.

    Adjacent tokens get different colours. A newline is drawn as `↵` (and still breaks the line),
    a tab as `→`, and the token id is the hover text. A token that is only a space keeps its
    coloured background, so spaces are visible.

    Args:
        pieces: The list of token strings to render.
        ids: The list of token ids corresponding to `pieces`. If `None`, no hover text is shown.
    """
    spans = []
    for i, piece in enumerate(pieces):
        shown = escape(piece).replace("\n", "↵\n").replace("\t", "→")
        title = f' title="id {ids[i]}"' if ids is not None else ""
        colour = f"hsla({_HUES[i % len(_HUES)]}, 75%, 55%, 0.38)"
        spans.append(f'<span style="background:{colour}"{title}>{shown}</span>')
    return (
        '<div style="font-family:monospace; white-space:pre-wrap; line-height:1.9; margin:0.4em 0">'
        + "".join(spans)
        + "</div>"
    )


def tokenize(tok, text: str) -> tuple[list[int], list[str]]:
    """Token ids and the text of each token.

    A token that is a partial UTF-8 character shows as `�`.
    
    Args:
        tok: The tokenizer instance (e.g. via `transformers.AutoTokenizer.from_pretrained(...)`).
        text: The text to tokenize.

    Returns:
        A tuple of token ids and the corresponding token strings.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids, [tok.decode([i]) for i in ids]


def show_tokens(tok, text: str):
    """A displayable (IPython) colour-coded tokenization of `text`.
    
    Args:
        tok: The tokenizer instance (e.g. via `transformers.AutoTokenizer.from_pretrained(...)`).
        text: The text to tokenize.
    
    Returns:
        An IPython HTML object displaying the colour-coded tokenization.
    """
    from IPython.display import HTML

    ids, pieces = tokenize(tok, text)
    caption = f'<div style="font-size:0.8em; opacity:0.7">{len(ids)} tokens, {len(text)} characters</div>'
    return HTML(tokens_html(pieces, ids) + caption)


def next_token_stats(logits, temperature: float = 1.0, k: int = 10) -> dict:
    """Spread of a next-token distribution.

    Args:
        logits: The logits of the next-token distribution.
        temperature: The temperature to apply to the logits.
        k: The number of top tokens to consider.
    
    Returns:
        A dictionary with the top token's probability, the top-k mass, and the entropy (nats).
    """
    import torch

    p = torch.softmax(logits.float() / temperature, dim=-1)
    top = p.topk(k).values
    return {
        "top1": top[0].item(),
        f"top{k}_mass": top.sum().item(),
        "entropy": -(p * p.clamp_min(1e-30).log()).sum().item(),
    }


def plot_next_token(tok, logits, temperatures=(0.3, 1.0, 2.0), k: int = 10):
    """Bar charts of one next-token distribution at several temperatures.

    The bars are the `k` most likely tokens at temperature 1.0, so the same tokens appear
    in every panel; however, their probabilities depend on the selected temperature.

    Args:
        tok: The tokenizer instance (e.g. via `transformers.AutoTokenizer.from_pretrained(...)`).
        logits: The logits of the next-token distribution.
        temperatures: A sequence of temperatures to visualize.
        k: The number of top tokens to consider.

    Returns:
        A matplotlib figure object containing the bar charts.
    """
    import matplotlib.pyplot as plt
    import torch

    logits = logits.float()
    ids = torch.softmax(logits, dim=-1).topk(k).indices
    labels = [repr(tok.decode([int(i)])) for i in ids]
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(4.2 * len(temperatures), 3.4), sharey=True
    )
    for ax, t in zip(axes, temperatures):
        p = torch.softmax(logits / t, dim=-1)
        ax.bar(range(k), p[ids].cpu())
        ax.set_xticks(range(k))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title(f"T = {t}  (entropy {next_token_stats(logits, t, k)['entropy']:.2f})")
    axes[0].set_ylabel("probability")
    fig.tight_layout()
    return fig


def surprisal_html(pieces: list[str], logprobs: list[float], full: float = 4.0) -> str:
    """Render tokens shaded by how surprising they were to the model.

    White is a token the model expected; the strongest red is a surprisal of `full` nats or more
    (probability below `exp(-full)`, about 2%). The probability is the hover text.

    Args:
        pieces: The token strings.
        logprobs: Natural-log probability the model gave each token.
        full: Surprisal in nats that gets the full colour.
    """
    spans = []
    for piece, lp in zip(pieces, logprobs):
        shade = min(-lp / full, 1.0)
        shown = escape(piece).replace("\n", "↵\n")
        spans.append(
            f'<span style="background:hsla(0, 85%, 50%, {0.55 * shade:.2f})" '
            f'title="p = {2.718281828 ** lp:.3f}">{shown}</span>'
        )
    return (
        '<div style="font-family:monospace; white-space:pre-wrap; line-height:1.9; margin:0.4em 0">'
        + "".join(spans) + "</div>"
    )


def trained_html(pieces: list[str], trained: list[bool]) -> str:
    """Render tokens with the ones the loss counts in green and the rest in grey.

    Args:
        pieces: The token strings.
        trained: For each token, whether its label is kept (not `-100`).
    """
    spans = []
    for piece, kept in zip(pieces, trained):
        shown = escape(piece).replace("\n", "↵\n")
        colour = "hsla(130, 55%, 45%, 0.45)" if kept else "hsla(0, 0%, 55%, 0.18)"
        spans.append(f'<span style="background:{colour}">{shown}</span>')
    return (
        '<div style="font-family:monospace; white-space:pre-wrap; line-height:1.9; margin:0.4em 0">'
        + "".join(spans) + "</div>"
    )
