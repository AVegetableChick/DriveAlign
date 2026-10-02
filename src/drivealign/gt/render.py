"""Deterministic reasoning renderer + zero-contradiction gate (S08 P4).

Purpose:
    Render the ``reasoning`` string from frozen English templates (P4.1):
    three segments (perception / risk / decision), each 3-5 wording
    variants; the variant index comes from
    ``sha256(sample_token + ':' + segment salt)`` so the same input renders
    byte-identical text forever. Variants only change wording, never
    information: object items are ``{article} {motion_phrase} {category}``,
    risk items are the FIXED phrases of ``enum_phrase_map.json`` verbatim,
    decision wording comes from the speed_action / yield phrases. English
    only, no distance or speed numbers, zero traffic-light mentions (P4.4).

    ``check_zero_contradiction`` is the P4.5 build gate (fail-fast in the
    backfill, re-checked over the full dataset afterwards):
      1. categories mentioned in the text (word-boundary match over the
         closed 10-class vocabulary) are a subset of critical_objects
         categories;
      2. risk_factors <-> risk wording is bidirectionally exact: every
         listed factor's fixed phrase appears, and NO other risk phrase
         appears (with empty risk_factors, zero phrases may appear);
      3. the traffic-light word family has zero regex hits;
      4. the rendered text is non-empty.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.config import load_templates, load_phrase_map; \
    from drivealign.gt.render import render_reasoning; \
    print(render_reasoning('tok', [], [], 'KEEP_SPEED', False, \
    load_templates(), load_phrase_map()))"``
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Sequence

# Closed 10-class vocabulary (v3 category_vocab.json canonical terms).
CATEGORY_VOCAB = (
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
)

TRAFFIC_LIGHT_PATTERN = re.compile(
    r"\btraffic[_\s-]?lights?\b|\b(red|green|yellow)\s+light\b",
    re.IGNORECASE,
)

_SALTS = {"perception", "risk", "decision"}


def _variant_index(sample_token: str, salt: str, n: int) -> int:
    digest = hashlib.sha256(f"{sample_token}:{salt}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def _pick(sample_token: str, segment: Dict, key: str) -> str:
    variants = segment[key]
    return variants[_variant_index(sample_token, segment["salt"], len(variants))]


def _join_and(items: Sequence[str]) -> str:
    """English list join: 'A, B and C' (deterministic, no Oxford comma)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def render_reasoning(
    sample_token: str,
    critical_objects: Sequence[Dict],
    risk_factors: Sequence[str],
    speed_action: str,
    yield_required: bool,
    templates: Dict,
    phrase_map: Dict,
) -> str:
    """Render the three-segment reasoning text; deterministic per token."""
    segments = templates["segments"]

    # Perception segment: every critical object is mentioned exactly once.
    if critical_objects:
        motion_phrases = phrase_map["motion_state"]
        items = []
        for obj in critical_objects:
            adj = motion_phrases[obj["motion_state"]]
            display = obj["category"].replace("_", " ")
            items.append(f"{_article(adj)} {adj} {display}")
        perception = _pick(sample_token, segments["perception"], "variants").format(
            objects=_join_and(items)
        )
    else:
        perception = _pick(sample_token, segments["perception"], "empty_variants")

    # Risk segment: the fixed phrases verbatim, in taxonomy-listed order.
    if risk_factors:
        risk_phrases = phrase_map["risk_factors"]
        risks = _join_and([risk_phrases[r] for r in risk_factors])
        risk_text = _pick(sample_token, segments["risk"], "variants").format(
            risks=risks
        )
    else:
        risk_text = _pick(sample_token, segments["risk"], "empty_variants")

    # Decision segment: speed_action + yield phrases from the shared map.
    decision = _pick(sample_token, segments["decision"], "variants").format(
        speed_action=phrase_map["speed_action"][speed_action],
        yield_phrase=phrase_map["yield_required"][
            "true" if yield_required else "false"
        ],
    )

    return f"{perception} {risk_text} {decision}"


def check_zero_contradiction(
    reasoning: str,
    critical_objects: Sequence[Dict],
    risk_factors: Sequence[str],
    phrase_map: Dict,
) -> List[str]:
    """Return every P4.5 violation; an empty list means the gate passes."""
    problems: List[str] = []
    if not reasoning:
        problems.append("reasoning is empty")
        return problems

    # 1) Object mentions (word-boundary) must stay inside critical_objects.
    # Categories render with spaces ("traffic cone"), so match that form too.
    mentioned = set()
    for c in CATEGORY_VOCAB:
        display = re.escape(c.replace("_", " ")).replace(r"\ ", r"\s+")
        if re.search(rf"\b{display}\b", reasoning) is not None:
            mentioned.add(c)
    critical_categories = {obj["category"] for obj in critical_objects}
    extra_categories = sorted(mentioned - critical_categories)
    if extra_categories:
        problems.append(
            f"reasoning mentions categories outside critical_objects: "
            f"{extra_categories}"
        )

    # 2) Bidirectional risk phrase correspondence.
    lowered = reasoning.lower()
    risk_phrases = phrase_map["risk_factors"]
    found = {
        enum for enum, phrase in risk_phrases.items() if phrase.lower() in lowered
    }
    expected = set(risk_factors)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing:
        problems.append(f"risk factors missing their fixed phrase: {missing}")
    if extra:
        problems.append(f"risk phrases present without a risk factor: {extra}")

    # 3) Traffic lights: zero mentions, ever.
    if TRAFFIC_LIGHT_PATTERN.search(reasoning):
        problems.append("reasoning mentions traffic lights (forbidden)")

    return problems
