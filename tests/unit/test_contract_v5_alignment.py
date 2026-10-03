"""Contract v5 alignment pins (S09 Step 1).

Purpose:
    Pin the v5 model-face correction and the frozen-asset invariants so the
    prompt/schema gap that motivated v5 can never silently reappear:

    1. The v5 prompt enumerates exactly the 7-term risk taxonomy and no
       longer teaches ``small_following_gap`` (a term the v4+ output schema
       rejects) — the prompt↔schema alignment is checked across the prompt
       text, the schema enum and the taxonomy table.
    2. v5 is otherwise a verbatim copy of v4: the hash-face vocab tables,
       the parse layer (output_schema/risk_taxonomy) and the three GT-side
       assets are byte-identical; the ONLY delta is the prompt's removal of
       the ``small_following_gap`` enumeration clause.
    3. v4 assets stay frozen: its prompt still carries the legacy 8-item
       wording (v3 verbatim heritage — an in-place "fix" of v4 is
       forbidden), and its hash face remains byte-identical to v3.
    4. The registry (AVAILABLE_CONTRACT_VERSIONS / DEFAULT_CONTRACT_VERSION
       / MODEL_FACE_LINEAGE) and the runtime default path resolve to the
       v5 face: serialize() stamps ``v5`` and the default prompt is the
       corrected v5 text.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest \
    tests/unit/test_contract_v5_alignment.py -q``
"""

import json
import re

from drivealign.contracts.prompt import build_structured_prompt
from drivealign.contracts.versions import (
    AVAILABLE_CONTRACT_VERSIONS,
    DEFAULT_CONTRACT_VERSION,
    MODEL_FACE_FILES,
    contract_file,
    model_face_version,
)
from drivealign.records.record import FrameInput, ModelInputs
from drivealign.records.serializer import InputPolicy, serialize

#: The 7 canonical risk terms of the frozen v4+ taxonomy (S08 gap removal).
V5_RISK_TERMS = {
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic",
}

#: The exact clause deleted from the v4 prompt to form the v5 prompt.
_REMOVED_CLAUSE = '"small_following_gap" (very short distance to the vehicle ahead), '

_RISK_ITEM_RE = re.compile(r'"([a-z_]+)" \(')


def _risk_items(prompt_text: str) -> set[str]:
    """Extract the enumerated risk terms from a prompt's risk_factors line."""
    line = next(l for l in prompt_text.splitlines() if '"risk_factors"' in l)
    return set(_RISK_ITEM_RE.findall(line))


def _schema_risk_enum(version: str) -> set[str]:
    schema = json.loads(contract_file(version, "output_schema.json").read_text())
    return set(schema["properties"]["risk_factors"]["items"]["enum"])


def _taxonomy_terms(version: str) -> set[str]:
    taxonomy = json.loads(contract_file(version, "risk_taxonomy.json").read_text())
    return {t["canonical"] for t in taxonomy["terms"]}


def test_v5_prompt_schema_taxonomy_alignment():
    """The pin that motivated v5: prompt wording == schema enum == taxonomy."""
    prompt = contract_file("v5", "prompt.txt").read_text(encoding="utf-8")
    items = _risk_items(prompt)
    assert items == _schema_risk_enum("v5") == _taxonomy_terms("v5") == V5_RISK_TERMS
    assert "small_following_gap" not in prompt


def test_v5_is_verbatim_v4_except_prompt():
    """v5 inherits v4 byte-for-byte on everything except prompt.txt."""
    inherited = (
        "category_vocab.json",
        "motion_vocab.json",
        "output_schema.json",
        "risk_taxonomy.json",
        "gt_rule_config.json",
        "reasoning_templates.json",
        "enum_phrase_map.json",
    )
    for name in inherited:
        assert contract_file("v5", name).read_bytes() == contract_file(
            "v4", name
        ).read_bytes(), f"v5 {name} drifted from v4"


def test_v5_prompt_is_v4_with_single_clause_removed():
    v4 = contract_file("v4", "prompt.txt").read_text(encoding="utf-8")
    v5 = contract_file("v5", "prompt.txt").read_text(encoding="utf-8")
    assert _REMOVED_CLAUSE in v4
    assert v5 == v4.replace(_REMOVED_CLAUSE, "")


def test_v4_assets_stay_frozen():
    """v4 must keep the legacy 8-item prompt and the v3 hash face."""
    v4_prompt = contract_file("v4", "prompt.txt").read_text(encoding="utf-8")
    items = _risk_items(v4_prompt)
    assert "small_following_gap" in items and len(items) == 8
    for name in ("prompt.txt", "category_vocab.json", "motion_vocab.json"):
        assert contract_file("v4", name).read_bytes() == contract_file(
            "v3", name
        ).read_bytes(), f"v4 {name} drifted from v3"


def test_v5_registry_and_face():
    assert AVAILABLE_CONTRACT_VERSIONS[-2:] == ("v4", "v5")
    assert DEFAULT_CONTRACT_VERSION == "v5"
    assert model_face_version("v5") == "v5"
    assert model_face_version("v4") == "v3"
    for name in MODEL_FACE_FILES:
        assert contract_file("v5", name).is_file()


def test_runtime_default_resolves_to_v5_face():
    frozen = build_structured_prompt()
    assert "small_following_gap" not in frozen
    assert _risk_items(frozen) == V5_RISK_TERMS
    model_inputs = ModelInputs(
        frames=[
            FrameInput(
                "samples/CAM_FRONT/n0.jpg", "tok0", 1510000000000000, 0.0
            )
        ],
        ego_speed_mps=5.2,
    )
    request = serialize(model_inputs, InputPolicy.ONE_FRAME)
    assert request.contract_version == "v5"
    assert request.prompt.endswith(frozen)
