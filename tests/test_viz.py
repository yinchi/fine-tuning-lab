import re

from convfinqa.viz import tokens_html


def test_one_span_per_token_with_ids_as_hover_text():
    html = tokens_html(["what", " was", " 2001"], [10, 20, 30])
    assert html.count("<span") == 3
    assert 'title="id 20"' in html


def test_adjacent_tokens_get_different_colours():
    html = tokens_html(["a", "b", "c", "d"])
    colours = re.findall(r"background:(hsla\([^)]*\))", html)
    assert all(x != y for x, y in zip(colours, colours[1:]))


def test_html_is_escaped_and_whitespace_marked():
    html = tokens_html(["<b>", "\n", "\t"])
    assert "&lt;b&gt;" in html and "<b>" not in html
    assert "↵" in html and "→" in html


def test_no_ids_means_no_hover_text():
    assert "title=" not in tokens_html(["a"])


def test_next_token_stats_and_plot():
    import matplotlib

    matplotlib.use("Agg")
    import torch

    from convfinqa.viz import next_token_stats, plot_next_token

    class Tok:
        def decode(self, ids):
            return f"t{ids[0]}"

    logits = torch.tensor([3.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    cold, hot = next_token_stats(logits, 0.1, k=3), next_token_stats(logits, 10.0, k=3)
    assert cold["top1"] > 0.99 and hot["top1"] < 0.3
    assert cold["entropy"] < hot["entropy"] < torch.log(torch.tensor(6.0)).item()
    assert len(plot_next_token(Tok(), logits, (0.5, 1.0), k=3).axes) == 2
