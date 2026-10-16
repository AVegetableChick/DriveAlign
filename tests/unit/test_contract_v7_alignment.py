"""Contract v7 alignment pins (S13 contract v7 ruling).

Purpose:
    Pin the S13 v7 transition (revised risk taxonomy + BEV fields) so the
    deltas that motivated it can never silently reappear and the frozen
    history stays byte-valid:

    1. v7 model face: the risk enum narrows 7 -> 6 (corridor_conflict and
       congestion removed; vehicle_merging renamed vehicle_crossing;
       pedestrian_crossing widened to vru_crossing) and every
       ``critical_object`` additionally requires ``bev_x`` / ``bev_y``
       (ego-frame nearest-footprint-point, 0.1 m granularity).
    1b. v7 risk attribution (ruling V7-A1): each ``critical_object`` carries
       an integer ``id`` (0-7) and ``risk_factors`` entries are attribution
       objects ``{factor, objects}`` whose ``objects`` reference those ids
       (schema-bounded, parser-checked: unknown/duplicate ids and duplicate
       factors are semantic failures).
    2. v7 vocabulary inheritance: category_vocab / motion_vocab are verbatim
       v6 copies (the hash face must not drift); the v6 face itself stays
       byte-nailed so the frozen S11 archives remain reproducible.
    3. Two-way consistency: schema enum order == risk_taxonomy canonical
       order. Per ruling V7-C1 the reasoning assets (reasoning_templates /
       enum_phrase_map) are retired at v7: the model face has had no
       reasoning since v6, the SFT supervision targets are 4-field, and the
       GT render chain only ever consumed the frozen v4 assets.
    4. v7 owns its model face: ``MODEL_FACE_LINEAGE["v7"] == "v7"``; a v7
       request hash differs from the v6 face for identical causal inputs.
    5. gt_rule_config v7 is a PROVISIONAL draft whose five calibration slots
       are registered in ``calibration_targets`` (Step 4 freezes them).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest tests/unit/test_contract_v7_alignment.py -q``
"""

import hashlib
import json
from pathlib import Path

from drivealign.contracts.output import parse_structured_output
from drivealign.contracts.prompt import build_structured_prompt
from drivealign.contracts.versions import (
    AVAILABLE_CONTRACT_VERSIONS,
    CONTRACTS_DIR,
    DEFAULT_CONTRACT_VERSION,
    MODEL_FACE_LINEAGE,
    contract_file,
    model_face_version,
)
from drivealign.records.record import FrameInput, ModelInputs
from drivealign.records.serializer import InputPolicy, serialize

# --- v7 risk taxonomy (schema enum order = canonical output order). ---
V7_RISK_TERMS = (
    "vru_crossing",
    "vehicle_crossing",
    "lead_vehicle_braking",
    "stationary_obstacle",
    "object_near",
    "oncoming_traffic",
)

# --- v6-era terms with no v7 meaning (unmapped -> semantic failure). ---
REMOVED_TERMS = (
    "vehicle_merging",
    "corridor_conflict",
    "congestion",
)

# --- The v7 widening keeps the old pedestrian phrase a legal synonym. ---
RENAMED_SYNONYMS = {"pedestrian_crossing": "vru_crossing"}

#: Frozen v6 hash-face byte-nails (sha256 of prompt/category/motion), so the
#: S11 v6 face stays reproducible after the DEFAULT switch.
V6_HASH_FACE_PINS = {
    "prompt.txt": "242b632b279e87d9cf355fe42803f7381fa983adebe82d07b5597cbbe692ec10",
    "category_vocab.json": "649470dccc3634d29c8e595a3c75a4b6999ea6078b55b5d87b90f15cfe6b34aa",
    "motion_vocab.json": "6e8e8338dec888724eec9d6ddaf170b03ae0aadcadf404bd83e94c46a90a1b61",
}


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v7_schema() -> dict:
    return json.loads(contract_file("v7", "output_schema.json").read_text())


def _v7_valid_payload() -> dict:
    return {
        "critical_objects": [
            {
                "id": 0,
                "category": "car",
                "bbox_2d": [1.0, 2.0, 3.0, 4.0],
                "motion_state": "same_direction",
                "bev_x": 17.8,
                "bev_y": 0.0,
            }
        ],
        "risk_factors": [],
        "yield_required": False,
        "speed_action": "KEEP_SPEED",
    }


def _model_inputs() -> ModelInputs:
    return ModelInputs(
        frames=[
            FrameInput("samples/CAM_FRONT/n0.jpg", "tok0", 1510000000000000, 0.0)
        ],
        ego_speed_mps=5.2,
    )


# --------------------------------------------------------------------------
# ① v7 model face: BEV fields required; revised 6-item taxonomy enforced.
# --------------------------------------------------------------------------


def test_v7_valid_payload_parses_ok_default_and_explicit():
    payload = json.dumps(_v7_valid_payload())
    r_default = parse_structured_output(payload)
    assert r_default.ok and r_default.errors == ()
    assert r_default.output.contract_version == DEFAULT_CONTRACT_VERSION == "v7"
    r_v7 = parse_structured_output(payload, contract_version="v7")
    assert r_v7.ok and r_v7.output.contract_version == "v7"


def test_v7_requires_bev_fields():
    for missing in ("bev_x", "bev_y"):
        items = [
            {k: v for k, v in obj.items() if k != missing}
            for obj in _v7_valid_payload()["critical_objects"]
        ]
        payload = dict(_v7_valid_payload(), critical_objects=items)
        r = parse_structured_output(json.dumps(payload), contract_version="v7")
        assert not r.ok, f"v7 should require {missing!r}"


def test_v7_rejects_removed_and_old_taxonomy_terms():
    for term in REMOVED_TERMS:
        payload = dict(
            _v7_valid_payload(),
            risk_factors=[{"factor": term, "objects": [0]}],
        )
        r = parse_structured_output(json.dumps(payload), contract_version="v7")
        assert not r.ok, f"v7 should reject removed term {term!r}"


def test_v7_renamed_pedestrian_phrase_normalizes_to_vru_crossing():
    """The v7 vru_crossing widening keeps 'pedestrian_crossing' (underscore or
    space) a legal synonym so legacy phrasing maps onto the new canonical."""
    payload = dict(
        _v7_valid_payload(),
        risk_factors=[{"factor": "pedestrian_crossing", "objects": [0]}],
    )
    r = parse_structured_output(json.dumps(payload), contract_version="v7")
    assert r.ok
    assert r.output.risk_factors == ("vru_crossing",)
    assert r.normalized_terms == (("pedestrian_crossing", "vru_crossing"),)


def test_v7_enum_type_and_extra_property_violations_fail():
    bad_enum = dict(_v7_valid_payload(), speed_action="YIELD")
    assert not parse_structured_output(json.dumps(bad_enum), contract_version="v7").ok
    bad_type = dict(_v7_valid_payload(), yield_required="false")
    assert not parse_structured_output(json.dumps(bad_type), contract_version="v7").ok
    extra = dict(_v7_valid_payload(), reasoning="clear")
    assert not parse_structured_output(json.dumps(extra), contract_version="v7").ok


def test_v7_schema_caps_and_bev_types():
    schema = _v7_schema()
    assert schema["properties"]["critical_objects"]["maxItems"] == 8
    rf = schema["properties"]["risk_factors"]
    assert rf["maxItems"] == 6
    obj = schema["$defs"]["critical_object"]
    assert obj["additionalProperties"] is False
    assert set(obj["required"]) == {
        "id",
        "category",
        "bbox_2d",
        "motion_state",
        "bev_x",
        "bev_y",
    }
    assert obj["properties"]["bev_x"]["type"] == "number"
    assert obj["properties"]["bev_y"]["type"] == "number"


def test_v7_schema_attribution_item_shape():
    """V7-A1：risk_factors 条目 = {factor, objects}；factor 走六项枚举，
    objects 是 1..8 个不重复的整数 id（0-7）。"""
    item = _v7_schema()["properties"]["risk_factors"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"factor", "objects"}
    assert item["properties"]["factor"]["type"] == "string"
    objects = item["properties"]["objects"]
    assert objects["minItems"] == 1 and objects["maxItems"] == 8
    assert objects["uniqueItems"] is True
    assert objects["items"] == {"type": "integer", "minimum": 0, "maximum": 7}
    oid = _v7_schema()["$defs"]["critical_object"]["properties"]["id"]
    assert oid["type"] == "integer"
    assert oid["minimum"] == 0 and oid["maximum"] == 7


def test_v7_attribution_payload_parses_and_emits():
    """V7-A1 端到端：{factor, objects} 条目解析进归因视图、名字元组视图
    保持不变（evaluator/gt_distribution 版本无感），to_dict() 发射归因
    对象形状而非旧版裸名列表。"""
    payload = dict(
        _v7_valid_payload(),
        risk_factors=[
            {"factor": "vru_crossing", "objects": [0]},
            {"factor": "object_near", "objects": [0]},
        ],
    )
    r = parse_structured_output(json.dumps(payload), contract_version="v7")
    assert r.ok and r.errors == ()
    out = r.output
    # 名字视图：与 v6 形状同构（因子名元组）。
    assert out.risk_factors == ("vru_crossing", "object_near")
    # 归因视图：(factor, objects) 二元组序列。
    assert [(a.factor, a.objects) for a in out.risk_attribution] == [
        ("vru_crossing", (0,)),
        ("object_near", (0,)),
    ]
    # to_dict：risk_factors 发射归因对象形状；critical_object 发射 id 且
    # 键序钉死（id 在 bbox_2d 之后、motion_state 之前）。
    emitted = out.to_dict()
    assert emitted["risk_factors"] == [
        {"factor": "vru_crossing", "objects": [0]},
        {"factor": "object_near", "objects": [0]},
    ]
    assert list(emitted["critical_objects"][0]) == [
        "category", "bbox_2d", "id", "motion_state", "bev_x", "bev_y"
    ]


def test_v7_attribution_schema_violations_fail():
    """V7-A1 硬界（schema 层）：objects 非空、objects/id 均 0-7、id 必填。"""
    base = _v7_valid_payload()

    empty_objects = dict(
        base, risk_factors=[{"factor": "object_near", "objects": []}]
    )
    assert not parse_structured_output(
        json.dumps(empty_objects), contract_version="v7"
    ).ok
    out_of_range = dict(
        base, risk_factors=[{"factor": "object_near", "objects": [8]}]
    )
    assert not parse_structured_output(
        json.dumps(out_of_range), contract_version="v7"
    ).ok
    no_id = dict(
        base,
        critical_objects=[
            {k: v for k, v in base["critical_objects"][0].items() if k != "id"}
        ],
    )
    assert not parse_structured_output(json.dumps(no_id), contract_version="v7").ok


def test_v7_attribution_semantic_failures():
    """V7-A1 语义校验（parser 层）：重复 critical_objects id、重复因子、
    条目内重复对象 id、未知对象引用——全部判 SEMANTIC_RANGE_FAILURE。"""
    base = _v7_valid_payload()

    def _obj(idx):
        return dict(base["critical_objects"][0], id=idx)

    # ① 重复 critical_objects id（critical_objects 无 schema uniqueItems，
    #    由 parser 语义层拦截）。
    dup_ids = dict(base, critical_objects=[_obj(0), _obj(0)])
    r = parse_structured_output(json.dumps(dup_ids), contract_version="v7")
    assert not r.ok and r.errors
    # ② 重复因子（同 factor 出现两条归因）。
    dup_factor = dict(
        base,
        risk_factors=[
            {"factor": "object_near", "objects": [0]},
            {"factor": "object_near", "objects": [0]},
        ],
    )
    r = parse_structured_output(json.dumps(dup_factor), contract_version="v7")
    assert not r.ok and r.errors
    # ③ 条目内重复对象 id（schema uniqueItems 兜底，parser 语义双保险）。
    dup_ref = dict(
        base, risk_factors=[{"factor": "object_near", "objects": [0, 0]}]
    )
    r = parse_structured_output(json.dumps(dup_ref), contract_version="v7")
    assert not r.ok and r.errors
    # ④ 未知对象引用：id 在 0-7 合法界内但不在本响应的 critical_objects 里。
    unknown = dict(
        base, risk_factors=[{"factor": "object_near", "objects": [3]}]
    )
    r = parse_structured_output(json.dumps(unknown), contract_version="v7")
    assert not r.ok and r.errors


# --------------------------------------------------------------------------
# ② vocabulary inheritance + frozen v6 hash face.
# --------------------------------------------------------------------------


def test_v7_vocabs_inherit_v6_byte_for_byte():
    for name in ("category_vocab.json", "motion_vocab.json"):
        assert contract_file("v7", name).read_bytes() == contract_file(
            "v6", name
        ).read_bytes(), f"v7 {name} drifted from v6"


def test_v6_hash_face_is_byte_nailed():
    for filename, expected in V6_HASH_FACE_PINS.items():
        path = CONTRACTS_DIR / "v6" / filename
        assert path.is_file(), f"missing v6/{filename}"
        assert _sha256_bytes(path) == expected, f"v6/{filename} drifted"


def test_v7_prompt_teaches_bev_and_new_taxonomy_only():
    prompt = contract_file("v7", "prompt.txt").read_text(encoding="utf-8")
    assert "reasoning" not in prompt
    for term in V7_RISK_TERMS:
        assert term in prompt
    for term in REMOVED_TERMS:
        assert term not in prompt
    assert "bev_x" in prompt and "bev_y" in prompt
    assert "nearest point" in prompt and "0.1 m" in prompt
    assert "at most 8" in prompt
    # V7-A1：prompt 教学 id（0-7 锚点）与归因条目形状。
    assert '"id"' in prompt and "0 to 7" in prompt
    assert "objects" in prompt and "factor" in prompt


# --------------------------------------------------------------------------
# ③ two-way consistency: schema enum == taxonomy canonical order.
#    (v7 ruling V7-C1: reasoning assets retired — no enum_phrase_map.)
# --------------------------------------------------------------------------


def test_v7_schema_taxonomy_consistent():
    risk_items = _v7_schema()["properties"]["risk_factors"]["items"]
    schema_enum = risk_items["properties"]["factor"]["enum"]
    taxonomy = json.loads(
        contract_file("v7", "risk_taxonomy.json").read_text(encoding="utf-8")
    )
    canonical = tuple(t["canonical"] for t in taxonomy["terms"])
    assert schema_enum == list(V7_RISK_TERMS)
    assert canonical == V7_RISK_TERMS
    assert taxonomy["schema_ref"].startswith("v7/")
    assert taxonomy["version"] == "v7"


# --------------------------------------------------------------------------
# ④ registry + runtime default resolve to v7; v7 owns its model face.
# --------------------------------------------------------------------------


def test_v7_registry_and_face():
    assert AVAILABLE_CONTRACT_VERSIONS[-2:] == ("v6", "v7")
    assert AVAILABLE_CONTRACT_VERSIONS[-1] == "v7"
    assert DEFAULT_CONTRACT_VERSION == "v7"
    assert MODEL_FACE_LINEAGE["v7"] == "v7"
    assert model_face_version("v7") == "v7"
    # The older faces are untouched.
    assert model_face_version("v6") == "v6"
    assert model_face_version("v5") == "v5"
    assert model_face_version("v4") == "v3"


def _request_for(version: str, mi: ModelInputs):
    """Rebuild the 1F request exactly as the serializer does, pinned to a
    specific face version (validated against the real ``serialize``)."""
    from drivealign.cli.base_benchmark import ego_speed_line
    from drivealign.records.serializer import ModelRequest

    anchor = mi.frames[-1]
    prompt = build_structured_prompt(
        ego_speed_line(mi.ego_speed_mps), version=version
    )
    return ModelRequest(
        policy=InputPolicy.ONE_FRAME,
        contract_version=model_face_version(version),
        prompt=prompt,
        image_relpaths=(anchor.image_relpath,),
        frame_tokens=(anchor.frame_token,),
        time_offsets_s=(anchor.time_offset_s,),
        ego_speed_mps=mi.ego_speed_mps,
    )


def test_v7_owns_its_face_vs_v6():
    """Identical causal inputs, different face -> different request hash."""
    mi = _model_inputs()
    v7_request = _request_for("v7", mi)
    v6_request = _request_for("v6", mi)
    assert v6_request.contract_version == "v6"
    assert v7_request.contract_version == "v7"
    assert "bev_x" not in v6_request.prompt
    assert "bev_x" in v7_request.prompt
    assert v7_request.canonical_hash() != v6_request.canonical_hash()
    # The reconstruction is faithful: default serialize == v7 face hash.
    assert serialize(mi, InputPolicy.ONE_FRAME).canonical_hash() == (
        v7_request.canonical_hash()
    )


# --------------------------------------------------------------------------
# ⑤ gt_rule_config v7: FROZEN (Stage C 2026-10-12) with the five calibrated
# slots pinned to their final sweep values.
# --------------------------------------------------------------------------


def test_v7_rule_config_frozen_structure():
    cfg = json.loads(
        contract_file("v7", "gt_rule_config.json").read_text(encoding="utf-8")
    )
    assert cfg["status"] == "frozen"
    assert cfg["contract_version"] == "v7"
    # Five calibration slots, explicitly registered AND pinned to the frozen
    # sweep conclusions (base=2.0 / thr=0.8 / r=5.5 / Y_brake=1.185 / Y_st=3.0).
    assert set(cfg["calibration_targets"]) == {
        "firing.base_m",
        "crossing_gate.lateral_thr_mps",
        "object_near.radius_m",
        "braking.y_brake_m",
        "stationary_obstacle.y_st_m",
    }
    assert cfg["firing"]["base_m"] == 2.0
    assert cfg["firing"]["assumption"] == "single"  # 分裂假设不被数据支持
    assert cfg["crossing_gate"]["lateral_thr_mps"] == 0.8
    assert cfg["object_near"]["radius_m"] == 5.5
    assert cfg["braking"]["y_brake_m"] == 1.185
    assert cfg["stationary_obstacle"]["y_st_m"] == 3.0  # 3.5/4.0 补扫被否：破 25% 警戒线
    # Crossing arrival test (rules v2.4): T fixed at 5.0, NOT a calibration slot.
    assert cfg["crossing_gate"]["t_arrive_max_s"] == 5.0
    # Pool regroup (P-1) with per-class cone override for two-wheelers.
    groups = cfg["pool"]["class_groups"]
    assert set(groups) == {"vru", "vehicle", "static"}
    assert set(groups["vru"]) == {"pedestrian", "bicycle", "motorcycle"}
    assert "bicycle" not in groups["vehicle"] and "motorcycle" not in groups["vehicle"]
    assert cfg["pool"]["cone_range_by_class_m"] == {
        "bicycle": 45.0,
        "motorcycle": 45.0,
    }
    # Corridor constants kept; d_risk_* replaced by the firing engine.
    assert "d_risk_vehicle_m" not in cfg["corridor"]
    assert "d_risk_pedestrian_m" not in cfg["corridor"]
    assert cfg["corridor"]["t_risk_s"] == 3.0
    assert cfg["corridor"]["time_step_s"] == 0.25
    assert "congestion" not in cfg
    # In-path half-width constraint: Y_st >= Y_brake (frozen values).
    assert cfg["stationary_obstacle"]["y_st_m"] >= cfg["braking"]["y_brake_m"]
    # Invariants: motion/action semantics frozen across generations.
    assert cfg["motion"] == {
        "stationary_speed_mps": 0.5,
        "band_half_width_deg": 45.0,
    }
    assert cfg["action"]["delta"] == 0.15


def test_v7_assets_present_and_reasoning_retired():
    """V7-C1: the reasoning assets (reasoning_templates / enum_phrase_map)
    are removed at v7 — the model face has not consumed reasoning since v6,
    the SFT supervision targets are 4-field, and the GT render chain consumes
    the frozen v4 assets only. The remaining six v7 assets must exist."""
    for name in (
        "prompt.txt",
        "output_schema.json",
        "category_vocab.json",
        "motion_vocab.json",
        "risk_taxonomy.json",
        "gt_rule_config.json",
    ):
        assert contract_file("v7", name).is_file(), f"missing v7/{name}"
    for name in ("reasoning_templates.json", "enum_phrase_map.json"):
        assert not (CONTRACTS_DIR / "v7" / name).exists(), (
            f"v7/{name} must stay retired (ruling V7-C1)"
        )
