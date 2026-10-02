"""Unit tests for the P4 reasoning renderer and the zero-contradiction gate.

Purpose:
    Freeze: byte-identical determinism for the same sample token (hash-based
    variant selection, verified against a manual sha256 computation), the
    empty-state sentence family (renders non-empty with zero objects and
    zero risks), and every P4.5 gate direction — extra categories, missing
    / extra risk phrases, traffic-light mentions.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_render.py -q``
"""

from __future__ import annotations

import hashlib

from drivealign.gt.config import load_phrase_map, load_templates
from drivealign.gt.render import check_zero_contradiction, render_reasoning


def _objects():
    return [
        {"category": "car", "bbox_2d": [0.0, 0.0, 10.0, 10.0], "motion_state": "stationary"},
        {"category": "pedestrian", "bbox_2d": [5.0, 5.0, 15.0, 25.0], "motion_state": "crossing"},
    ]


def _render(token="tok-a", objects=None, risks=None, action="KEEP_SPEED", yield_req=False):
    return render_reasoning(
        token,
        _objects() if objects is None else objects,
        risks or [],
        action,
        yield_req,
        load_templates(),
        load_phrase_map(),
    )


def test_deterministic_byte_identical_rerenders():
    first = _render()
    second = _render()
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_variant_selection_matches_manual_hash():
    segments = load_templates()["segments"]
    # Every segment's chosen variant is the one at hash(sample_token:salt) % n;
    # verify for perception by reconstructing the full expected sentence.
    segment = segments["perception"]
    idx = int(hashlib.sha256("tok-a:perception".encode()).hexdigest(), 16) % len(
        segment["variants"]
    )
    expected_perception = segment["variants"][idx].format(
        objects="a stationary car and a crossing pedestrian"
    )
    text = _render()
    assert text.startswith(expected_perception + " ")
    # Different token -> still deterministic, and picks from the same pool.
    other = _render(token="tok-b")
    assert other == _render(token="tok-b")
    assert any(other.startswith(v.split("{")[0]) for v in segment["variants"])


def test_mentions_all_objects_with_motion_phrases():
    text = _render()
    assert "a stationary car" in text
    assert "a crossing pedestrian" in text


def test_risk_phrases_appear_verbatim():
    text = _render(risks=["corridor_conflict", "congestion"])
    assert "corridor conflict" in text
    assert "congestion" in text


def test_decision_phrases_from_enum_map():
    text = _render(action="DECELERATE", yield_req=True)
    assert "decelerate" in text
    assert "yielding is required" in text
    keep = _render(action="KEEP_SPEED", yield_req=False)
    assert "maintain my current speed" in keep
    assert "no yielding is required" in keep


def test_empty_state_renders_non_empty_and_passes_gate():
    text = _render(objects=[], risks=[])
    assert text and len(text) >= 1
    assert check_zero_contradiction(text, [], [], load_phrase_map()) == []
    # Empty state contains no category and no risk phrase.
    for category in ("car", "pedestrian", "traffic cone"):
        assert category not in text


def test_gate_extra_category_violation():
    text = _render(objects=[], risks=[])
    problems = check_zero_contradiction(text, [], [], load_phrase_map())
    assert problems == []
    injected = text + " Also a bus is visible."
    problems = check_zero_contradiction(injected, [], [], load_phrase_map())
    assert any("bus" in p for p in problems)


def test_gate_missing_risk_phrase():
    text = _render(risks=["congestion"])
    problems = check_zero_contradiction(
        text, _objects(), ["congestion", "oncoming_traffic"], load_phrase_map()
    )
    assert any("missing" in p and "oncoming_traffic" in p for p in problems)


def test_gate_extra_risk_phrase():
    text = _render(risks=["congestion"])
    problems = check_zero_contradiction(text, _objects(), [], load_phrase_map())
    assert any("without a risk factor" in p for p in problems)


def test_gate_traffic_light_zero_tolerance():
    text = _render() + " The traffic light is green."
    problems = check_zero_contradiction(text, _objects(), [], load_phrase_map())
    assert any("traffic light" in p for p in problems)


def test_gate_mult_word_categories_use_space_display():
    # "traffic_cone" renders as "traffic cone" and must be accepted as an
    # in-set mention, but rejected when not in critical_objects.
    cone = [{"category": "traffic_cone", "bbox_2d": [0.0, 0.0, 5.0, 5.0], "motion_state": "stationary"}]
    text = _render(objects=cone, risks=[])
    assert "traffic cone" in text
    assert check_zero_contradiction(text, cone, [], load_phrase_map()) == []
    problems = check_zero_contradiction(text, [], [], load_phrase_map())
    assert any("traffic_cone" in p for p in problems)
