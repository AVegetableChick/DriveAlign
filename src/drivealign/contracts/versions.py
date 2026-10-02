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
"""

from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = REPO_ROOT / "configs" / "contracts"

AVAILABLE_CONTRACT_VERSIONS = ("v1", "v2", "v3", "v4")

#: Current frozen pairing (v3 model face + S08 GT-side assets).
DEFAULT_CONTRACT_VERSION = "v4"

#: Model-face lineage: which contract version's model face each version
#: inherits for REQUEST purposes. The request hash covers the prompt text
#: (``prompt.txt``) plus the causal inputs, and v4 keeps the v3 prompt
#: verbatim (as well as category_vocab/motion_vocab); the only v4 model-face
#: deltas are the frozen gap removal in ``output_schema.json`` /
#: ``risk_taxonomy.json`` (risk enum 8 -> 7, S08_parameter_freeze 9.1),
#: which never enter the request hash. Model requests therefore keep the
#: ``v3`` stamp and request hashes stay identical to the frozen v3 anchor
#: manifest (S08 gate 1: v4-vs-v3 differences live in GT fields only).
#: Every new contract version MUST add an entry here; a test pins the
#: byte-identity of the hash-relevant artifacts.
MODEL_FACE_LINEAGE = {"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v3"}

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
