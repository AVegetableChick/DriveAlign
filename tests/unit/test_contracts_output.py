"""Unit tests for the Stage 03 output contract and prompt template.

Purpose:
    Freeze the Stage 03 Gate cases: valid samples pass; missing keys, wrong
    types, illegal bboxes, and illegal enumerations fail with categorized
    errors; the prompt never mentions the future.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit -q``
"""

from __future__ import annotations

import json

import pytest

from drivealign.contracts.output import (
    DEFAULT_CONTRACT_VERSION,
    OutputErrorCategory,
    SPEED_ACTIONS,
    parse_structured_output,
)
from drivealign.contracts.prompt import build_structured_prompt


def parse_v1(raw, **kwargs):
    """Parse against contract v1 (the default is now v3)."""
    return parse_structured_output(raw, contract_version="v1", **kwargs)


VALID_SAMPLE = {
    "critical_objects": [
        {
            "category": "car",
            "bbox_2d": [120, 300, 480, 560],
            "coarse_position": "front center",
            "behavior": "going ahead",
        }
    ],
    "risk_factors": ["pedestrian near curb"],
    "reasoning": "Car ahead is moving; keep speed and monitor.",
    "yield_required": False,
    "speed_action": "KEEP_SPEED",
}


def _variant(**changes):
    sample = json.loads(json.dumps(VALID_SAMPLE))
    sample.update(changes)
    return json.dumps(sample)


def _object_variant(**changes):
    sample = json.loads(json.dumps(VALID_SAMPLE))
    sample["critical_objects"][0].update(changes)
    return json.dumps(sample)


def test_valid_sample_parses_ok():
    result = parse_v1(json.dumps(VALID_SAMPLE))
    assert result.ok
    assert result.errors == ()
    assert result.output.speed_action == "KEEP_SPEED"
    assert result.output.critical_objects[0].bbox_2d == (120.0, 300.0, 480.0, 560.0)
    assert result.output.contract_version == "v1"


def test_to_dict_round_trip_is_stable():
    parsed = parse_v1(json.dumps(VALID_SAMPLE))
    reparsed = parse_v1(json.dumps(parsed.output.to_dict()))
    assert reparsed.ok and reparsed.output == parsed.output


def test_speed_actions_are_frozen():
    assert SPEED_ACTIONS == ("ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP")


@pytest.mark.parametrize(
    ("raw", "field_hint"),
    [
        (json.dumps({k: v for k, v in VALID_SAMPLE.items() if k != "reasoning"}), None),
        (_variant(yield_required="false"), "$.yield_required"),
        (_variant(speed_action="YIELD"), "$.speed_action"),
        (_variant(speed_action="keep_speed"), "$.speed_action"),
        (_variant(extra_field="future route"), None),
        (_object_variant(bbox_2d=[-10, 300, 480, 560]), "$.critical_objects[0].bbox_2d[0]"),
        (_object_variant(bbox_2d=[120, 300, 480]), "$.critical_objects[0].bbox_2d"),
        (_object_variant(category=""), "$.critical_objects[0].category"),
        (json.dumps([VALID_SAMPLE]), None),
    ],
)
def test_schema_violations_fail_with_category_and_field(raw, field_hint):
    result = parse_v1(raw)
    assert not result.ok and result.output is None
    categories = {error.category for error in result.errors}
    assert categories == {OutputErrorCategory.SCHEMA_FAILURE}
    if field_hint is not None:
        assert any(error.field == field_hint for error in result.errors)


@pytest.mark.parametrize(
    ("bbox", "image_size"),
    [
        ([480, 300, 120, 560], None),
        ([120, 560, 480, 300], None),
        ([0, 0, 1000, 720], (1000, 500)),
        ([0, 0, 1280, 900], (1280, 720)),
    ],
)
def test_semantic_range_failures(bbox, image_size):
    raw = _object_variant(bbox_2d=bbox)
    result = parse_v1(raw, image_size=image_size)
    assert not result.ok
    assert {error.category for error in result.errors} == {
        OutputErrorCategory.SEMANTIC_RANGE_FAILURE
    }


def test_bbox_inside_image_bounds_passes():
    result = parse_v1(json.dumps(VALID_SAMPLE), image_size=(1280, 720))
    assert result.ok


@pytest.mark.parametrize("raw", ["", "not json", "{truncated", None])
def test_json_parse_failures(raw):
    result = parse_v1(raw)
    assert not result.ok and result.output is None
    assert {error.category for error in result.errors} == {
        OutputErrorCategory.JSON_PARSE_FAILURE
    }


def test_fenced_valid_json_is_extracted_and_recorded():
    fenced = "```json\n" + json.dumps(VALID_SAMPLE) + "\n```"
    result = parse_v1(fenced)
    assert result.ok
    assert result.extracted_from_fences is True
    assert result.output.speed_action == "KEEP_SPEED"


def test_fenced_empty_object_becomes_schema_failure():
    result = parse_v1("```json\n{}\n```")
    assert not result.ok
    assert {error.category for error in result.errors} == {
        OutputErrorCategory.SCHEMA_FAILURE
    }


def test_fenced_invalid_inner_is_parse_failure():
    result = parse_v1("```json\n{broken\n```")
    assert not result.ok
    assert {error.category for error in result.errors} == {
        OutputErrorCategory.JSON_PARSE_FAILURE
    }
    assert "after fence extraction" in result.errors[0].message


def test_unterminated_fence_is_parse_failure():
    raw = "```json\n" + json.dumps(VALID_SAMPLE)
    result = parse_v1(raw)
    assert not result.ok
    assert {error.category for error in result.errors} == {
        OutputErrorCategory.JSON_PARSE_FAILURE
    }


def test_plain_json_has_no_extraction_flag():
    result = parse_v1(json.dumps(VALID_SAMPLE))
    assert result.ok and result.extracted_from_fences is False


def test_raw_text_is_echoed_for_traceability():
    raw = json.dumps(VALID_SAMPLE)
    result = parse_v1(raw)
    assert result.raw_text == raw
    failed = parse_v1("{broken")
    assert failed.raw_text == "{broken"


def test_prompt_without_speed_has_fixed_task_only():
    prompt = build_structured_prompt()
    assert "Currently available ego speed" not in prompt
    assert '"speed_action"' in prompt
    for action in SPEED_ACTIONS:
        assert action in prompt
    assert '"critical_objects"' in prompt and '"bbox_2d"' in prompt
    assert "future" not in prompt.lower()


def test_prompt_with_speed_only_prepends_one_line():
    speed = "current 8.3 m/s, past 1 s: 8.1 m/s"
    prompt = build_structured_prompt(speed)
    assert prompt.startswith(f"Currently available ego speed: {speed}\n\n")
    assert build_structured_prompt("   ") == build_structured_prompt(None)


def test_default_contract_version_is_frozen():
    assert DEFAULT_CONTRACT_VERSION == "v4"


def test_prompt_v2_states_envelope_rules_explicitly():
    prompt = build_structured_prompt(version="v2")
    assert "must start with { and end with }" in prompt
    assert "no code fences" in prompt
    assert "```" in prompt  # the rule forbidding fences is stated literally


def test_prompt_versions_v1_v2_v3_all_load_and_differ():
    prompts = {v: build_structured_prompt(version=v) for v in ("v1", "v2", "v3")}
    assert len({p for p in prompts.values()}) == 3
    # Shared invariants: no future wording, speed_action enum frozen in all.
    for prompt in prompts.values():
        assert "future" not in prompt.lower()
        for action in ("ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"):
            assert action in prompt
    # v1/v2 track schema v1 fields; v3 tracks schema v2 fields.
    for v in ("v1", "v2"):
        assert "coarse_position" in prompts[v] and "behavior" in prompts[v]
        assert "motion_state" not in prompts[v]
    assert "motion_state" in prompts["v3"]
    assert "coarse_position" not in prompts["v3"] and "behavior" not in prompts["v3"]
    assert "stationary" in prompts["v3"] and "pedestrian_crossing" in prompts["v3"]


def test_prompt_v3_lists_closed_vocabulary():
    prompt = build_structured_prompt(version="v3")
    for term in (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ):
        assert term in prompt
    for term in (
        "stationary",
        "same_direction",
        "oncoming",
        "crossing",
        "vehicle_merging",
        "lead_vehicle_braking",
        "corridor_conflict",
        "stationary_obstacle",
        "congestion",
        "oncoming_traffic",
    ):
        assert term in prompt
    assert "at most 8" in prompt


def test_prompt_unknown_version_rejected():
    import pytest

    with pytest.raises(ValueError, match="Unknown contract version"):
        build_structured_prompt(version="v9")


V2_VALID_SAMPLE = {
    "critical_objects": [
        {
            "category": "car",
            "bbox_2d": [120, 300, 480, 560],
            "motion_state": "same_direction",
        }
    ],
    "risk_factors": ["congestion"],
    "reasoning": "Car ahead is moving; keep speed and monitor.",
    "yield_required": False,
    "speed_action": "KEEP_SPEED",
}


def test_default_contract_is_v4_and_parses_ok():
    result = parse_structured_output(json.dumps(V2_VALID_SAMPLE))
    assert result.ok and result.errors == ()
    assert result.output.contract_version == DEFAULT_CONTRACT_VERSION == "v4"
    assert result.derived_positions == (None,)  # no image_size provided


def test_v2_normalizes_open_terms_via_vocab_tables():
    raw = json.dumps(
        {
            "critical_objects": [
                {
                    "category": "SUV",
                    "bbox_2d": [120, 300, 480, 560],
                    "motion_state": "Parked",
                }
            ],
            "risk_factors": ["Pedestrians crossing"],
            "reasoning": "r",
            "yield_required": False,
            "speed_action": "STOP",
        }
    )
    result = parse_structured_output(raw)
    assert result.ok
    assert ("SUV", "car") in result.normalized_terms
    assert ("Parked", "stationary") in result.normalized_terms
    assert ("Pedestrians crossing", "pedestrian_crossing") in result.normalized_terms
    assert result.unmapped_terms == ()
    assert result.output.critical_objects[0].category == "car"
    assert result.output.critical_objects[0].motion_state == "stationary"
    assert result.output.risk_factors == ("pedestrian_crossing",)


def test_v2_unmapped_terms_are_recorded_and_rejected():
    raw = json.dumps(
        {
            **V2_VALID_SAMPLE,
            "critical_objects": [
                {
                    "category": "vehicle",
                    "bbox_2d": [120, 300, 480, 560],
                    "motion_state": "moving",
                }
            ],
        }
    )
    result = parse_structured_output(raw)
    assert not result.ok and result.output is None
    assert {e.category for e in result.errors} == {OutputErrorCategory.SCHEMA_FAILURE}
    assert set(result.unmapped_terms) == {"vehicle", "moving"}


def test_v2_derives_coarse_position_from_bbox():
    result = parse_structured_output(
        json.dumps(V2_VALID_SAMPLE), image_size=(1280, 720)
    )
    assert result.ok
    # bbox center (300, 430) in 1280x720: column 0 -> left, row 1 -> middle.
    assert result.derived_positions == ("middle left",)
    # The derived position never enters the generation contract dict.
    assert "coarse_position" not in result.output.to_dict()["critical_objects"][0]


def test_v2_canonical_terms_pass_without_rewrites():
    result = parse_structured_output(json.dumps(V2_VALID_SAMPLE))
    assert result.ok and result.normalized_terms == () and result.unmapped_terms == ()


def test_v1_payload_fails_under_default_v3_schema():
    result = parse_structured_output(json.dumps(VALID_SAMPLE))
    assert not result.ok
    assert {e.category for e in result.errors} == {OutputErrorCategory.SCHEMA_FAILURE}


def test_truncated_json_reports_budget_hint():
    truncated = json.dumps(VALID_SAMPLE)[:-40]
    result = parse_v1(truncated)
    assert not result.ok
    (error,) = result.errors
    assert error.category is OutputErrorCategory.JSON_PARSE_FAILURE
    assert "truncated" in error.message


def test_v1_round_trip_still_works_with_explicit_version():
    parsed = parse_v1(json.dumps(VALID_SAMPLE))
    reparsed = parse_v1(json.dumps(parsed.output.to_dict()))
    assert reparsed.ok and reparsed.output == parsed.output


def test_unknown_contract_version_rejected():
    import pytest

    with pytest.raises(ValueError, match="Unknown contract version"):
        parse_structured_output("{}", contract_version="v9")
