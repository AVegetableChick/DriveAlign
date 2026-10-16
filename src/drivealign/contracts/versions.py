"""Single source of truth for DriveAlign contract versions.

Purpose:
    Freeze the version registry for the Stage 03 output contract. One
    contract version binds together every frozen artifact of that pairing:
    ``configs/contracts/v<N>/prompt.txt``, ``output_schema.json`` and, from
    v3 on, the closed-vocabulary tables. The contract version is the only
    version number used by the parser, the prompt builder, the runner, and
    all records, so a sample can always be traced to one folder on disk.

    Pairing history (frozen):
    - v1: brief format rules + free-text schema (first sanity round).
    - v2: strong envelope rules + free-text schema (recorded S03 round).
    - v3: v2 envelope rules + closed vocabularies and nine-grid derived
      coarse_position.
    - v4: v3 five model-face artifacts copied verbatim + three new GT-side
      assets (gt_rule_config.json, reasoning_templates.json,
      enum_phrase_map.json) carrying the S08 backfill rules. The model face
      is unchanged, so v4 request hashes stay identical to v3 (S08 gate 1).
    - v5: v4 eight artifacts copied verbatim except ``prompt.txt``, whose
      risk_factors wording is corrected from 8 to 7 items (the frozen
      ``small_following_gap`` enumeration line is removed) so the prompt no
      longer teaches the model to emit a term the v4+ schema rejects
      (S09 model-face ruling). v5 owns its model face
      (``MODEL_FACE_LINEAGE["v5"] == "v5"``), so v5 request hashes differ
      from the frozen v3/v4-face history; dataset assets are NOT rebuilt
      (records' GT/model_inputs are prompt-independent) and S09 derives a
      v5-face anchor manifest into runs/ instead.
    - v6: v5 eight artifacts copied verbatim except ``prompt.txt`` (the
      ``reasoning`` field entry is removed) and ``output_schema.json`` (the
      ``reasoning`` property is dropped from both required and properties,
      so a reasoning key is rejected as an extra property). v6 owns its
      model face (``MODEL_FACE_LINEAGE["v6"] == "v6"``), so v6 request
      hashes differ from the v5 face; the prompt stops teaching reasoning
      (S11 model-face ruling). reasoning_templates.json is retained as a
      record-side GT provenance asset only; the model face no longer
      consumes it. dataset_v4 records are NOT rebuilt and v5 archives stay
      valid; S11 derives a v6-face anchor manifest into runs/ instead.
    - v7: the S13 contract v7 ruling (S13_contract_v7_decision_log.md) is
      applied to BOTH faces of the v6 generation. Per ruling V7-C1 the two
      reasoning assets are RETIRED — v7 carries SIX artifacts (prompt.txt,
      output_schema.json, risk_taxonomy.json, gt_rule_config.json,
      category_vocab.json, motion_vocab.json):
      ``prompt.txt`` re-teaches the revised 6-item risk taxonomy
      (vru_crossing / vehicle_crossing / lead_vehicle_braking /
      stationary_obstacle / object_near / oncoming_traffic) and the new
      per-object ``bev_x`` / ``bev_y`` fields; ``output_schema.json`` narrows
      the risk enum 7 -> 6 (corridor_conflict and congestion removed) and
      adds the two BEV numbers to ``critical_object`` (required);
      ``risk_taxonomy.json`` / ``gt_rule_config.json`` are rewritten
      for the v7 rule engine and taxonomy. Per ruling V7-C1 (S13) the
      reasoning assets ``reasoning_templates.json`` / ``enum_phrase_map.json``
      are RETIRED at v7 — the model face has had no reasoning since v6, the
      SFT supervision targets are 4-field, and the GT render chain only ever
      consumed the frozen v4 assets — so v7 carries six artifacts and the GT
      render chain retires with the v7 generation (expected_output drops to
      the 4 contract fields; the v7 output_schema becomes the unified
      GT/model validator). v7 owns its model face
      (``MODEL_FACE_LINEAGE["v7"] == "v7"``), so v7 request hashes differ
      from the v5/v6 faces; the GT-side structure change also moves the
      record-hash generation, so dataset assets are rebuilt as dataset_v7
      (new physical directory + ``data/face_manifests/v7/``), while
      dataset_v4 and the v5/v6 face manifests stay byte-frozen. The two
      hash-face vocab tables (category/motion) are verbatim v6 copies.
"""

from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = REPO_ROOT / "configs" / "contracts"

AVAILABLE_CONTRACT_VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7")

#: Current frozen pairing (v7 model face: revised 6-item risk taxonomy +
#: per-object BEV fields; GT-side assets rewritten for the v7 rule engine,
#: calibration slots provisional until the S13 sweep freezes).
DEFAULT_CONTRACT_VERSION = "v7"

#: Model-face lineage: which contract version's model face each version
#: inherits for REQUEST purposes. The request hash covers the prompt text
#: (``prompt.txt``) plus the causal inputs. v4 keeps the v3 prompt verbatim
#: (as well as category_vocab/motion_vocab); the only v4 model-face deltas
#: are the frozen gap removal in ``output_schema.json`` /
#: ``risk_taxonomy.json`` (risk enum 8 -> 7, S08_parameter_freeze 9.1),
#: which never enter the request hash, so v4 request hashes stay identical
#: to the frozen v3 anchor manifest (S08 gate 1: v4-vs-v3 differences live
#: in GT fields only).
#: v5 corrects the prompt itself (risk_factors wording 8 -> 7 items), which
#: DOES enter the request hash, so v5 owns its model face and the S09
#: benchmark/training requests carry the ``v5`` stamp. The frozen v3/v4-face
#: history remains valid for dataset audits; never compare request hashes
#: across face eras.
#: v6 drops reasoning from the prompt/schema, which also enters the request
#: hash, so v6 owns its model face and S11 requests carry the ``v6`` stamp.
#: v7 rewrites the prompt (6-item taxonomy + BEV fields) and the schema
#: (risk enum 7 -> 6 + required bev_x/bev_y), so v7 owns its model face and
#: S13+ requests carry the ``v7`` stamp.
#: Every new contract version MUST add an entry here; a test pins the
#: byte-identity of the hash-relevant artifacts.
MODEL_FACE_LINEAGE = {
    "v1": "v1",
    "v2": "v2",
    "v3": "v3",
    "v4": "v3",
    "v5": "v5",
    "v6": "v6",
    "v7": "v7",
}

#: The five model-face artifact files (prompt + parse/validate layer).
MODEL_FACE_FILES = (
    "prompt.txt",
    "output_schema.json",
    "category_vocab.json",
    "motion_vocab.json",
    "risk_taxonomy.json",
)

#: Model-face artifacts that must remain byte-identical across a lineage
#: step (everything the request hash can see). output_schema/risk_taxonomy
#: may differ when a new version deliberately narrows the closed taxonomy.
HASH_FACE_FILES = ("prompt.txt", "category_vocab.json", "motion_vocab.json")


def model_face_version(version: str) -> str:
    """Return the contract version whose model face ``version`` inherits.

    Records carry ``version`` (it binds the GT-side assets too), but the
    serialized model request is stamped with the model-face version so the
    request hash tracks what the model actually sees (S08 gate 1).
    """
    if version not in MODEL_FACE_LINEAGE:
        raise ValueError(
            f"Unknown contract version {version!r}; "
            f"available: {', '.join(AVAILABLE_CONTRACT_VERSIONS)}"
        )
    return MODEL_FACE_LINEAGE[version]


def contract_dir(version: str) -> Path:
    """Return the frozen artifact folder for ``version``."""
    if version not in AVAILABLE_CONTRACT_VERSIONS:
        raise ValueError(
            f"Unknown contract version {version!r}; "
            f"available: {', '.join(AVAILABLE_CONTRACT_VERSIONS)}"
        )
    return CONTRACTS_DIR / version


@lru_cache(maxsize=None)
def contract_file(version: str, filename: str) -> Path:
    """Return (and cache) one frozen artifact path, validating existence."""
    path = contract_dir(version) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Contract {version} is missing required artifact: {path}"
        )
    return path
