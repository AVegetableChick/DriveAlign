"""Contract v6 alignment pins (S11 Step 0 model-face ruling).

Purpose:
    Pin the S11 v6 model-face transition so the deltas that motivated it can
    never silently reappear and the frozen history stays byte-valid:

    1. v6 drops ``reasoning`` from the model output: the v6 prompt no longer
       enumerates it and the v6 output schema neither requires nor defines it
       (with ``additionalProperties: false`` a reasoning key is rejected).
    2. v6 is otherwise a verbatim copy of v5: the hash-face vocab tables and
       every GT-side asset are byte-identical; the ONLY deltas are the
       prompt/schema reasoning removal.
    3. The pre-v6 hash face is frozen: prompt.txt / category_vocab /
       motion_vocab for v1-v5 are byte-nailed so a drifting S09 archive can
       be caught immediately.
    4. v6 owns its model face: ``MODEL_FACE_LINEAGE["v6"] == "v6"``, the
       runtime default resolves to the 4-field v6 prompt, and a v6 request
       hash differs from the v5 face for identical inputs.
    5. The dataset_v4 records are zero-impact: each record carries its own
       contract_version (v4) and validates/serializes against that face,
       independent of the DEFAULT switch.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest tests/unit/test_contract_v6_alignment.py -q``
"""

import hashlib
import json
from pathlib import Path

import pytest

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
from drivealign.records.record import DriveAlignRecord, FrameInput, ModelInputs
from drivealign.records.serializer import InputPolicy, serialize

# --- v6 field set (the 4-generation fields that define the model face). ---
V6_FIELDS = ("critical_objects", "risk_factors", "yield_required", "speed_action")

# --- Exact clause removed from the v5 prompt to form the v6 prompt. ---
_REASONING_CLAUSE = (
    '- "reasoning": a concise explanation grounded in the current image '
    "and the available ego speed, if provided.\n"
)

#: Frozen pre-v6 hash-face byte-nails (sha256 of prompt/category/motion).
#: v1/v2 predate the closed vocabularies, so only prompt.txt pins apply.
HASH_FACE_PINS = {
    ("v1", "prompt.txt"): "2ef3487ab366de5551d10f1adf3645f0c4132a017f2f122c586125ff93f81124",
    ("v2", "prompt.txt"): "47b59ec1c6550b436048b72a8491ead97f1ca2be048d9c6b808a3801609e71fe",
    ("v3", "prompt.txt"): "a5f8b44caa087cab173b12de04fd12d6842258d1aabb86259607192453a63f9c",
    ("v4", "prompt.txt"): "a5f8b44caa087cab173b12de04fd12d6842258d1aabb86259607192453a63f9c",
    ("v5", "prompt.txt"): "16533b5708d7319e549f512d45e6f5bf8d9c9ed96d999047560915047b8b426e",
    ("v3", "category_vocab.json"): "649470dccc3634d29c8e595a3c75a4b6999ea6078b55b5d87b90f15cfe6b34aa",
    ("v4", "category_vocab.json"): "649470dccc3634d29c8e595a3c75a4b6999ea6078b55b5d87b90f15cfe6b34aa",
    ("v5", "category_vocab.json"): "649470dccc3634d29c8e595a3c75a4b6999ea6078b55b5d87b90f15cfe6b34aa",
    ("v3", "motion_vocab.json"): "6e8e8338dec888724eec9d6ddaf170b03ae0aadcadf404bd83e94c46a90a1b61",
    ("v4", "motion_vocab.json"): "6e8e8338dec888724eec9d6ddaf170b03ae0aadcadf404bd83e94c46a90a1b61",
    ("v5", "motion_vocab.json"): "6e8e8338dec888724eec9d6ddaf170b03ae0aadcadf404bd83e94c46a90a1b61",
}

# Files that v6 must inherit byte-for-byte from v5 (everything but the two
# reasoning-removal deltas: prompt.txt and output_schema.json).
V6_INHERITED = (
    "category_vocab.json",
    "motion_vocab.json",
    "risk_taxonomy.json",
    "gt_rule_config.json",
    "reasoning_templates.json",
    "enum_phrase_map.json",
)


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v6_schema() -> dict:
    return json.loads(contract_file("v6", "output_schema.json").read_text())


def _v6_valid_payload() -> dict:
    return {
        "critical_objects": [
            {"category": "car", "bbox_2d": [1.0, 2.0, 3.0, 4.0],
             "motion_state": "same_direction"}
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
# ① 4-field schema: valid 4-field passes; reasoning rejected; required/missing
#    and enum violations fail against the v6 contract.
# --------------------------------------------------------------------------


def test_v6_four_field_valid_payload_parses_ok():
    r = parse_structured_output(json.dumps(_v6_valid_payload()))
    assert r.ok and r.errors == ()
    assert r.output.contract_version == "v6"
    assert r.output.reasoning is None
    assert set(r.output.to_dict()) == set(V6_FIELDS)


def test_v6_rejects_reasoning_as_extra_property():
    payload = dict(_v6_valid_payload(), reasoning="clear")
    r = parse_structured_output(json.dumps(payload))
    assert not r.ok and r.output is None


def test_v6_requires_all_four_fields():
    for missing in V6_FIELDS:
        payload = {k: v for k, v in _v6_valid_payload().items() if k != missing}
        r = parse_structured_output(json.dumps(payload))
        assert not r.ok, f"v6 should require {missing!r}"


def test_v6_enum_and_type_violations_fail():
    bad_enum = dict(_v6_valid_payload(), speed_action="YIELD")
    assert not parse_structured_output(json.dumps(bad_enum)).ok
    bad_type = dict(_v6_valid_payload(), yield_required="false")
    assert not parse_structured_output(json.dumps(bad_type)).ok


# --------------------------------------------------------------------------
# ①② v6 = v5 with only the reasoning deltas; pre-v6 hash face byte-nailed.
# --------------------------------------------------------------------------


def test_v6_inherits_v5_byte_for_byte_except_prompt_and_schema():
    for name in V6_INHERITED:
        assert contract_file("v6", name).read_bytes() == contract_file(
            "v5", name
        ).read_bytes(), f"v6 {name} drifted from v5"


def test_v6_prompt_is_v5_reasoning_line_removed():
    v5 = contract_file("v5", "prompt.txt").read_text(encoding="utf-8")
    v6 = contract_file("v6", "prompt.txt").read_text(encoding="utf-8")
    assert _REASONING_CLAUSE in v5
    assert "reasoning" not in v6
    assert v6 == v5.replace(_REASONING_CLAUSE, "")
    # The rest of the v5 wording (risk taxonomy, envelope rules) is preserved.
    for word in ("risk_factors", "critical_objects", "yield_required", "speed_action"):
        assert word in v6


def test_v6_schema_removes_reasoning_semantics():
    v5 = json.loads(contract_file("v5", "output_schema.json").read_text())
    v6 = _v6_schema()
    assert v6["additionalProperties"] is False
    # v5 required it and defined it; v6 neither requires nor defines it.
    assert "reasoning" in v5["required"] and "reasoning" in v5["properties"]
    assert "reasoning" not in v6["required"]
    assert "reasoning" not in v6["properties"]
    # All four remaining field constraints are carried over unchanged.
    assert set(v6["required"]) == set(V6_FIELDS)
    for field in V6_FIELDS:
        assert v5["properties"][field] == v6["properties"][field]


def test_pre_v6_hash_face_is_byte_nailed():
    """The S09 v5 face (and its ancestors) stay frozen so archives stay valid."""
    for (version, filename), expected in HASH_FACE_PINS.items():
        path = CONTRACTS_DIR / version / filename
        assert path.is_file(), f"missing {version}/{filename}"
        assert _sha256_bytes(path) == expected, f"{version}/{filename} drifted"


def test_v6_hash_face_vocabs_inherit_v5():
    assert contract_file("v6", "category_vocab.json").read_bytes() == contract_file(
        "v5", "category_vocab.json"
    ).read_bytes()
    assert contract_file("v6", "motion_vocab.json").read_bytes() == contract_file(
        "v5", "motion_vocab.json"
    ).read_bytes()


# --------------------------------------------------------------------------
# ③④ registry + runtime default resolve to v6; v6 owns its model face.
# --------------------------------------------------------------------------


def test_v6_registry_and_face():
    assert AVAILABLE_CONTRACT_VERSIONS[-2:] == ("v5", "v6")
    assert DEFAULT_CONTRACT_VERSION == "v6"
    assert MODEL_FACE_LINEAGE["v6"] == "v6"
    assert model_face_version("v6") == "v6"
    # The v5 face and v3/v4 face history are untouched.
    assert model_face_version("v5") == "v5"
    assert model_face_version("v4") == "v3"


def test_runtime_default_resolves_to_v6_face():
    frozen = build_structured_prompt()
    assert "reasoning" not in frozen
    request = serialize(_model_inputs(), InputPolicy.ONE_FRAME)
    assert request.contract_version == "v6"
    assert request.prompt.endswith(frozen)


def _request_for(version: str, mi: ModelInputs) -> "ModelRequest":
    """Rebuild the 1F request exactly as the serializer does, but pinned to a
    specific face version. Validated against the real ``serialize`` below."""
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


def test_v6_owns_its_face_vs_v5():
    """Identical causal inputs, different face -> different request hash."""
    mi = _model_inputs()
    v6_request = _request_for("v6", mi)
    v5_request = _request_for("v5", mi)
    # v6 vs v5: distinct face -> distinct stamp and distinct hash.
    assert v5_request.contract_version == "v5"
    assert v6_request.contract_version == "v6"
    assert "reasoning" in v5_request.prompt
    assert "reasoning" not in v6_request.prompt
    assert v6_request.canonical_hash() != v5_request.canonical_hash()
    # The reconstruction is faithful: default serialize == v6 face hash.
    assert serialize(mi, InputPolicy.ONE_FRAME).canonical_hash() == (
        v6_request.canonical_hash()
    )


# --------------------------------------------------------------------------
# ⑤ dataset_v4 zero-impact: records bind their OWN contract_version (v4),
#    independent of the DEFAULT switch.
# --------------------------------------------------------------------------


DATASET_V4 = Path(__file__).resolve().parents[3] / "data" / "dataset_v4"


def _v4_records(limit=5):
    shard = DATASET_V4 / "train" / "shard-0000.jsonl"
    if not shard.is_file():
        pytest.skip("dataset_v4 not present")
    records = []
    with shard.open("r", encoding="utf-8") as fh:
        for raw in fh:
            records.append(DriveAlignRecord.from_dict(json.loads(raw)))
            if len(records) >= limit:
                break
    return records


def test_dataset_v4_records_bind_own_face_unaffected_by_default():
    """A frozen v4 record's GT validates against its OWN recorded version,
    independent of the DEFAULT switch (which now resolves to v6). Records stay
    byte-identical on disk; this proves the v6 transition did not touch them."""
    records = _v4_records()
    assert records, "expected at least one v4 record"
    for rec in records:
        assert rec.provenance.contract_version == "v4"
        # GT expected_output (5-field, carries reasoning) validates against v4
        # **explicitly**, so DEFAULT=v6 cannot change its validity.
        expected = rec.training_targets.expected_output
        assert expected is not None and "reasoning" in expected
        pr = parse_structured_output(json.dumps(expected), contract_version="v4")
        assert pr.ok, f"v4 record GT failed self-validation for {rec.sample_token}"
        # Requests on frozen records now carry the CURRENT DEFAULT face (v6);
        # that is a face-stamp change only, not a record mutation.
        request = serialize(rec.model_inputs, InputPolicy.ONE_FRAME)
        assert request.contract_version == "v6"


def test_archived_v5_request_hash_is_reproducible_and_differs_from_v6():
    """S09's frozen v5-face manifest (data/dataset_v4/anchor_policy_manifest.json,
    sha fd0d4dc4...) records each anchor's v5 request hash. A real record
    rebuilt under the v5 face must reproduce that hash; the v6 face must differ
    (the whole point of owning the model face)."""
    manifest_path = DATASET_V4 / "anchor_policy_manifest.json"
    shard = DATASET_V4 / "train" / "shard-0000.jsonl"
    if not (manifest_path.is_file() and shard.is_file()):
        pytest.skip("dataset_v4 not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "v5"  # archived v5-face authority
    # Pick the first anchor that lives in train/shard-0000.jsonl.
    target = None
    for token in sorted(manifest["anchors"]):
        entry = manifest["anchors"][token]
        if entry["split"] == "train" and entry["shard"].startswith("train/shard-0000"):
            target = (token, entry)
            break
    assert target is not None, "no train/shard-0000 anchor found"
    token, entry = target
    with shard.open("r", encoding="utf-8") as fh:
        record = DriveAlignRecord.from_dict(json.loads(list(fh)[int(entry["line"])]))
    v5_request = _request_for("v5", record.model_inputs)
    assert v5_request.contract_version == "v5"
    # Reproduces the archived v5 request hash -> the frozen v5 face is intact.
    assert v5_request.canonical_hash() == entry["request_hash_1f"]
    # The v6 face for the same causal inputs is a different request.
    assert _request_for("v6", record.model_inputs).canonical_hash() != entry[
        "request_hash_1f"
    ]