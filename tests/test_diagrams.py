import math
import re
import xml.dom.minidom

import pytest

from convfinqa.book.diagrams import (
    attention, baseline_prompt, decoder, feedforward, history_rounds, label_shift, lora_layer, lora_matrix, pipeline, posttraining, sequence_layout, training_step,
)
from convfinqa.book.diagrams.svg import Box, Svg, arrowhead, rounded_path, waypoints


def test_waypoints_follow_the_moves():
    assert waypoints((10, 100), "r20 u100 l20") == [(10, 100), (30, 100), (30, 0), (10, 0)]
    assert waypoints((0, 0), "d5.5") == [(0, 0), (0, 5.5)]


@pytest.mark.parametrize("moves", ["", "x10", "r", "r10 q5", "10r"])
def test_bad_moves_are_rejected(moves):
    with pytest.raises(ValueError):
        waypoints((0, 0), moves)


def test_rounded_path_cuts_each_corner():
    d = rounded_path([(0, 0), (100, 0), (100, 100)], radius=10)
    assert d == "M 0 0 L 90.00 0.00 Q 100 0 100.00 10.00 L 100 100"


def test_radius_shrinks_to_fit_short_segments():
    d = rounded_path([(0, 0), (100, 0), (100, 8)], radius=10)  # the second segment only allows 4
    assert "L 96.00 0.00 Q 100 0 100.00 4.00" in d


def test_arrowhead_points_along_the_direction():
    tip, corner_a, corner_b = arrowhead((50, 50), (0, -1), length=10, half_width=5)  # an upward arrow
    assert tip == (50, 50)
    assert math.isclose(corner_a[1], 60) and math.isclose(corner_b[1], 60)  # the base is 10 below the tip
    assert sorted([round(corner_a[0], 6), round(corner_b[0], 6)]) == [45, 55]


def test_box_anchors():
    b = Box(10, 20, 100, 40)
    assert b.top == (60, 20) and b.bottom == (60, 60) and b.left == (10, 40) and b.right == (110, 40)


def test_route_returns_its_end_point_and_draws_a_head():
    svg = Svg(100, 100)
    assert svg.route((0, 0), "r20 d30") == (20, 30)
    out = svg.to_string()
    assert "<path" in out and "<polygon" in out

    plain = Svg(10, 10)
    plain.route((0, 0), "r20", arrow=False)
    assert "<polygon" not in plain.to_string()


def test_text_is_escaped():
    svg = Svg(100, 100)
    svg.text((5, 5), "a < b & c")
    assert "a &lt; b &amp; c" in svg.to_string()


def test_decoder_diagram_is_well_formed_and_mentions_every_part():
    out = decoder.build()
    xml.dom.minidom.parseString(out)
    for word in ["Embeddings", "Causal self-attention", "Feed-forward (SwiGLU)", "RMSNorm", "Linear (output head)", "36 ×", "token positions"]:
        assert word in out
    assert len(re.findall(r"<polygon", out)) >= 10  # arrowheads


def test_attention_diagram_shows_every_step_and_shape():
    out = attention.build()
    xml.dom.minidom.parseString(out)
    for word in ["W_Q", "W_K", "W_V", "W_O", "RMSNorm", "RoPE", "repeat each key/value head", "causal mask",
                 "(b, n_Q, s, s)", "(b, s, n_Q·d)", "(b, s, n_KV·d)", "split into heads", "matrix multiplication", "token positions"]:
        assert word in out


def test_feedforward_diagram_shows_the_gate():
    out = feedforward.build()
    xml.dom.minidom.parseString(out)
    for word in ["W_gate", "W_up", "W_down", "SiLU", "×", "intermediate_size"]:
        assert word in out


def test_pipeline_diagram_names_the_three_stages():
    out = pipeline.build()
    xml.dom.minidom.parseString(out)
    for word in ["Pretraining", "Post-training", "Our fine-tuning", "base model", "instruct model", "our model"]:
        assert word in out


def test_posttraining_diagram_shows_both_routes():
    out = posttraining.build()
    xml.dom.minidom.parseString(out)
    for word in ["Base models", "Long-CoT cold start", "Reasoning RL", "Thinking mode fusion", "General RL",
                 "Strong-to-weak distillation", "Flagship models", "Lightweight models", "teachers"]:
        assert word in out


def test_training_step_diagram_shows_the_loop_and_its_state():
    out = training_step.build()
    xml.dom.minidom.parseString(out)
    for word in ["Batch", "Forward pass", "Loss", "Backward pass", "Optimizer step", "Weights", "Gradients",
                 "Optimizer state", "repeat with the next batch"]:
        assert word in out


def test_label_shift_diagram_marks_the_answer_tokens():
    out = label_shift.build()
    xml.dom.minidom.parseString(out)
    for word in ["input_ids", "assistant", "&lt;|im_end|&gt;", "(nothing)", "yes", "no"]:
        assert word in out
    assert out.count(">yes<") == 4 and out.count(">no<") == 4


def test_lora_matrix_diagram_shows_both_views():
    out = lora_matrix.build()
    xml.dom.minidom.parseString(out)
    for word in ["As matrices", "As computation", "W′", "frozen", "(α/r)", "9,728 × 2,560", "196,608",
                 "is never built while training", "r = 16 numbers"]:
        assert word in out


def test_lora_layer_diagram_marks_the_seven_adapted_layers():
    out = lora_layer.build()
    xml.dom.minidom.parseString(out)
    for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        assert out.count(f">{name}<") == 1
    for word in ["36 ×", "Embeddings", "Output head", "RMSNorm", "252", "residual"]:
        assert word in out
    assert out.count("#d5e8d4") == 8  # seven adapted boxes and the legend swatch


def test_sequence_layout_diagram_shows_both_layouts():
    out = sequence_layout.build()
    xml.dom.minidom.parseString(out)
    for word in ["one sequence", "turn 1", "turn 2", "turn 3", "page + Q1", "system", "3 sequences"]:
        assert word in out
    assert out.count("#d5e8d4") >= 7  # trained answers, and the legend swatch


def test_baseline_prompt_diagram_shows_the_prompt_and_the_runs():
    out = baseline_prompt.build()
    xml.dom.minidom.parseString(out)
    for word in ["system", "example", "conversation A", "conversation B", "target conversation", "writes Ak",
                 "set 1", "set 2", "set 3", "mean ± spread"]:
        assert word in out


def test_history_rounds_diagram_shows_both_modes():
    out = history_rounds.build()
    xml.dom.minidom.parseString(out)
    for word in ["Gold history", "Predicted history", "round 1", "round 4", "conv 4"]:
        assert word in out
    assert out.count(">Q1<") == 8  # four conversations in each of the two grids
