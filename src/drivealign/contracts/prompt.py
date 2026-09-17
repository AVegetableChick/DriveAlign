"""Versioned fixed structured-output prompt templates for Stage 03.

Purpose:
    Load the frozen structured-output prompt for one contract version from
    ``configs/contracts/v<N>/prompt.txt`` (see ``contracts/versions.py`` for
    the pairing registry). Every prompt asks the base model for exactly one
    JSON object matching that version's ``output_schema.json``; v3 adds the
    closed-vocabulary enumeration. The prompt only carries the fixed task
    description plus an optional causally available ego speed string; the
    image itself is attached by the Stage 02 runner, never by this module.
    Nothing about the future is allowed in the prompt. Templates are frozen
    artifacts: fixing a typo means opening a new contract version, never
    editing an existing folder in place.

Startup command:
    This module is a library imported by the Stage 03 structured runner and
    tests. Print the default frozen prompt from the workspace root:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.contracts.prompt import build_structured_prompt; print(build_structured_prompt())"``
"""

from __future__ import annotations

from functools import lru_cache

from drivealign.contracts.versions import (
    DEFAULT_CONTRACT_VERSION,
    AVAILABLE_CONTRACT_VERSIONS,
    contract_file,
)

__all__ = [
    "AVAILABLE_CONTRACT_VERSIONS",
    "DEFAULT_CONTRACT_VERSION",
    "build_structured_prompt",
]


@lru_cache(maxsize=None)
def _load_template(version: str) -> str:
    return contract_file(version, "prompt.txt").read_text(encoding="utf-8")


def build_structured_prompt(
    available_speed: str | None = None,
    version: str | None = None,
) -> str:
    """Return the frozen structured prompt of contract ``version``.

    ``version`` defaults to ``DEFAULT_CONTRACT_VERSION``. ``available_speed``
    is an optional caller-provided string describing only causally available
    speed information (current or past speed); it is inserted as one leading
    line. Future-derived speed information must never be passed here. Empty
    or whitespace-only values are treated as absent.
    """
    resolved = version if version is not None else DEFAULT_CONTRACT_VERSION
    template = _load_template(resolved)
    if available_speed is None or not available_speed.strip():
        return template
    return f"Currently available ego speed: {available_speed.strip()}\n\n{template}"
