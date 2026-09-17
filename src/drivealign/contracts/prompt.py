"""Versioned fixed structured-output prompt templates for Stage 03.

Purpose:
    Load the frozen structured-output prompt templates stored as versioned
    text files under ``configs/contracts/prompts/`` (structured_output_v1.txt,
    structured_output_v2.txt, structured_output_v3.txt). Every prompt asks the
    base model for exactly one JSON object matching the DriveAlign output
    contract (``configs/contracts/output_schema_v1.json`` for prompt v1/v2;
    ``output_schema_v2.json`` for prompt v3). The prompt only carries the
    fixed task description plus an optional causally available ego speed
    string; the image itself is attached by the Stage 02 runner, never by
    this module. Nothing about the future is allowed in the prompt.

    Version history (frozen, never edited in place):
    - v1: first contract prompt; brief format rule. Lost the format battle
      with markdown fences in the S03 sanity run (3/3 fenced outputs).
    - v2: strong start/end + restated output rules; fixed fences on
      original/shuffled but blank images still degenerated (2/3 ok).
      Produced the recorded S03 sanity evidence paired with schema v1.
    - v3: v2 format rules plus closed-vocabulary enumeration matching
      output_schema_v2 (canonical category, motion_state, risk_terms).
      Default version, shipped together with the v2 parser.

Startup command:
    This module is a library imported by the Stage 03 structured runner and
    tests. Print the default frozen prompt from the workspace root:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.contracts.prompt import build_structured_prompt; print(build_structured_prompt())"``
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = REPO_ROOT / "configs" / "contracts" / "prompts"

AVAILABLE_PROMPT_VERSIONS = ("v1", "v2", "v3")

#: Default frozen prompt version. v3 pairs with output_schema_v2 and the v2
#: parser (closed vocabularies); they shipped together as one unit, replacing
#: the v2+v1 pair that produced the recorded S03 sanity evidence.
PROMPT_VERSION = "v3"

_TEMPLATE_FILENAME = "structured_output_{}.txt"


@lru_cache(maxsize=None)
def _load_template(version: str) -> str:
    path = PROMPTS_DIR / _TEMPLATE_FILENAME.format(version)
    return path.read_text(encoding="utf-8")


def build_structured_prompt(
    available_speed: str | None = None,
    version: str | None = None,
) -> str:
    """Return the frozen structured prompt of ``version`` for one image.

    ``version`` defaults to ``PROMPT_VERSION``. ``available_speed`` is an
    optional caller-provided string describing only causally available speed
    information (current or past speed); it is inserted as one leading line.
    Future-derived speed information must never be passed here. Empty or
    whitespace-only values are treated as absent.
    """
    resolved = version if version is not None else PROMPT_VERSION
    if resolved not in AVAILABLE_PROMPT_VERSIONS:
        raise ValueError(
            f"Unknown prompt version {resolved!r}; "
            f"available: {', '.join(AVAILABLE_PROMPT_VERSIONS)}"
        )
    template = _load_template(resolved)
    if available_speed is None or not available_speed.strip():
        return template
    return f"Currently available ego speed: {available_speed.strip()}\n\n{template}"
