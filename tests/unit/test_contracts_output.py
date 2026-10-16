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
    assert DEFAULT_CONTRACT_VERSION == "v7"


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


V6_VALID_SAMPLE = {k: v for k, v in V2_VALID_SAMPLE.items() if k != "reasoning"}

V7_VALID_SAMPLE = {
    "critical_objects": [
        {
            "id": 0,
            "category": "car",
            "bbox_2d": [120, 300, 480, 560],
            "motion_state": "same_direction",
            "bev_x": 17.8,
            "bev_y": 0.0,
        }
    ],
    # V7-A1：risk_factors 条目 = 归因对象 {factor, objects}。
    "risk_factors": [{"factor": "object_near", "objects": [0]}],
    "yield_required": False,
    "speed_action": "KEEP_SPEED",
}


def test_v6_face_still_parses_v6_sample():
    """The frozen v6 face keeps validating its own 4-field shape (no BEV)."""
    result = parse_structured_output(
        json.dumps(V6_VALID_SAMPLE), contract_version="v6"
    )
    assert result.ok and result.errors == ()
    assert result.output.contract_version == "v6"
    assert result.output.reasoning is None


def test_default_contract_is_v7_and_parses_ok():
    result = parse_structured_output(json.dumps(V7_VALID_SAMPLE))
    assert result.ok and result.errors == ()
    assert result.output.contract_version == DEFAULT_CONTRACT_VERSION == "v7"
    assert result.output.reasoning is None
    assert "reasoning" not in result.output.to_dict()
    assert result.derived_positions == (None,)  # no image_size provided
    obj = result.output.critical_objects[0]
    assert obj.bev_x == 17.8 and obj.bev_y == 0.0


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
    result = parse_structured_output(raw, contract_version="v3")
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
    result = parse_structured_output(raw, contract_version="v3")
    assert not result.ok and result.output is None
    assert {e.category for e in result.errors} == {OutputErrorCategory.SCHEMA_FAILURE}
    assert set(result.unmapped_terms) == {"vehicle", "moving"}


def test_v2_derives_coarse_position_from_bbox():
    result = parse_structured_output(
        json.dumps(V2_VALID_SAMPLE), image_size=(1280, 720), contract_version="v3"
    )
    assert result.ok
    # bbox center (300, 430) in 1280x720: column 0 -> left, row 1 -> middle.
    assert result.derived_positions == ("middle left",)
    # The derived position never enters the generation contract dict.
    assert "coarse_position" not in result.output.to_dict()["critical_objects"][0]


def test_v2_canonical_terms_pass_without_rewrites():
    result = parse_structured_output(json.dumps(V2_VALID_SAMPLE), contract_version="v3")
    assert result.ok and result.normalized_terms == () and result.unmapped_terms == ()


def test_v1_payload_fails_under_default_schema():
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


# ---------------------------------------------------------------------------
# S13：bbox_source_size（模型 bbox 发射空间 → 原图空间的 parse 端内联反变换）。
# 背景：base（未训）模型在 max_pixels=360000 下输出 784x448 smart-resize 空间
# 的 bbox_2d；S13 起 runner 在 parse 端确定性反变换回原图坐标后才落盘，
# 下游 eval/BEV 只见原图坐标（predictions 文件约定 = 原图坐标系）。
# ---------------------------------------------------------------------------

# 784x448 源空间 bbox，中心 (100, 100)；
# scale = (1600/784, 900/448) = (2.040816…, 2.008929…)。
SOURCE_SPACE_BBOX = [10, 10, 190, 190]


def _source_space_sample():
    sample = json.loads(json.dumps(VALID_SAMPLE))
    sample["critical_objects"][0]["bbox_2d"] = list(SOURCE_SPACE_BBOX)
    return json.dumps(sample)


def test_bbox_source_size_rescales_to_original_space():
    """bbox 按声明空间线性反变换回原图坐标，其余字段逐字节不变。"""
    result = parse_v1(
        _source_space_sample(),
        image_size=(1600, 900),
        bbox_source_size=(784, 448),
    )
    assert result.ok
    scale_x, scale_y = 1600.0 / 784.0, 900.0 / 448.0
    bbox = result.output.critical_objects[0].bbox_2d
    expected = tuple(v * s for v, s in zip(SOURCE_SPACE_BBOX, (scale_x, scale_y, scale_x, scale_y)))
    assert bbox == pytest.approx(expected)
    # 变换只作用于几何字段，语义字段不受影响。
    assert result.output.critical_objects[0].category == "car"
    assert result.output.speed_action == "KEEP_SPEED"


def test_bbox_source_size_default_none_keeps_coords():
    """缺省（SFT 面 / 历史口径）零行为变化：bbox 原样保留。"""
    result = parse_v1(
        _source_space_sample(),
        image_size=(1600, 900),
    )
    assert result.ok
    assert result.output.critical_objects[0].bbox_2d == (
        10.0, 10.0, 190.0, 190.0,
    )


def test_bbox_source_size_requires_image_size():
    """只给 bbox_source_size 不给 image_size 是配置错误，显式报错拒绝。"""
    with pytest.raises(ValueError, match="bbox_source_size requires image_size"):
        parse_v1(_source_space_sample(), bbox_source_size=(784, 448))


def test_bbox_rescale_precedes_bounds_check_and_coarse_position():
    """bounds 检查与 coarse_position 派生都发生在反变换之后的原图坐标系。

    源空间右侧 bbox 变换后落在原图右列（"middle right"）：
    若变换未先行执行，(740, 224) 中心会被误判进原图左列。
    """
    sample = json.loads(json.dumps(V7_VALID_SAMPLE))
    sample["critical_objects"][0]["bbox_2d"] = [700, 160, 780, 288]
    result = parse_structured_output(
        json.dumps(sample),
        image_size=(1600, 900),
        bbox_source_size=(784, 448),
    )
    assert result.ok and result.errors == ()
    # 中心 (740, 224) → 原图 (1510.2, 450.0)：第 3 列、中间行。
    assert result.derived_positions[0] == "middle right"
    # 反变换后的 bbox 仍在原图界内（x_max = 780 * 1600/784 ≈ 1591.8）。
    assert result.output.critical_objects[0].bbox_2d[2] == pytest.approx(
        780.0 * 1600.0 / 784.0
    )


def test_bbox_out_of_original_bounds_fails_after_rescale():
    """bounds 检查消费的是反变换后的坐标：原始发射值 [10,10,800,460] 按
    原图口径检查本可放行（800<1600、460<900），反变换后越界 → 失败，
    证明检查发生在变换之后且从不静默修复。"""
    sample = json.loads(json.dumps(VALID_SAMPLE))
    sample["critical_objects"][0]["bbox_2d"] = [10, 10, 800, 460]
    result = parse_v1(
        json.dumps(sample),
        image_size=(1600, 900),
        bbox_source_size=(784, 448),
    )
    assert not result.ok
    assert {e.category for e in result.errors} == {
        OutputErrorCategory.SEMANTIC_RANGE_FAILURE
    }


def test_bbox_rescale_source_edge_maps_exactly_to_original_edge():
    """源空间贴边 bbox（截断物体常态）必须精确映射到原图贴边。

    回归 pin：乘法在前的实现吃浮点噪声（448*(900/448)=900.0000000000001
    > 900），把贴底边物体误判成越界；除法在前 (448/448)*900=900.0 精确。
    """
    sample = json.loads(json.dumps(VALID_SAMPLE))
    sample["critical_objects"][0]["bbox_2d"] = [10, 10, 380, 448]
    result = parse_v1(
        json.dumps(sample),
        image_size=(1600, 900),
        bbox_source_size=(784, 448),
    )
    assert result.ok and result.errors == ()
    bbox = result.output.critical_objects[0].bbox_2d
    assert bbox[3] == 900.0  # 精确贴边，非近似
    assert bbox[2] == pytest.approx(380.0 * 1600.0 / 784.0)
