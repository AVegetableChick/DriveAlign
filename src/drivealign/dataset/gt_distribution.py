"""GT distribution accounting for the v4 dataset build (S08).

Purpose:
    Accumulate the label-side distributions of the five ``expected_output``
    fields WHILE the build writes shards (zero extra I/O pass), then emit a
    single ``gt_distribution_report.{json,md}`` pair next to the build
    report. Quarantined records carry no GT and are never counted; the
    ``frames`` denominator of every split therefore equals its
    ``backfilled_count`` (gate: ``gt_backfill_full`` makes that == valid).

    All enum vocabularies are read from the frozen v4 contract assets —
    categories / motion states / speed actions from ``output_schema.json``
    and the seven risk terms from ``risk_taxonomy.json`` — so zero-count
    values still appear as explicit rows and any contract drift is caught
    instead of silently re-columned. Counting is deterministic: identical
    builds produce identical payloads (sorted keys, closed vocabularies).

    The per-split payload feeds the markdown renderer; an ``all`` payload
    (sum over splits) is appended by the caller.

Example:
    ``from drivealign.dataset.gt_distribution import GtDistribution``
    ``stats = GtDistribution(); stats.update(expected_output_dict)``
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping

from drivealign.contracts.versions import contract_file


def _schema_enums() -> tuple:
    """(categories, motion_states, speed_actions) from the frozen v4 schema."""
    schema = _load_json(contract_file("v4", "output_schema.json"))
    obj = schema["$defs"]["critical_object"]["properties"]
    actions = schema["properties"]["speed_action"]["enum"]
    return tuple(obj["category"]["enum"]), tuple(obj["motion_state"]["enum"]), tuple(
        sorted(actions)
    )


def _risk_terms() -> tuple:
    taxonomy = _load_json(contract_file("v4", "risk_taxonomy.json"))
    return tuple(sorted(t["canonical"] for t in taxonomy["terms"]))


def _load_json(path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


CATEGORIES, MOTION_STATES, SPEED_ACTIONS = _schema_enums()
RISK_TERMS = _risk_terms()
YIELD_KEYS = ("true", "false")


def _pctl(sorted_values: List[int], q: float) -> int:
    """Nearest-rank percentile of an ascending list (q in [0, 1])."""
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1)))
    return sorted_values[index]


class GtDistribution:
    """Mutable accumulator over backfilled ``expected_output`` records."""

    def __init__(self) -> None:
        self.frames = 0
        self.speed_action: Counter = Counter()
        self.yield_required: Counter = Counter()
        self.risk_terms: Counter = Counter()
        self.risk_frames: Counter = Counter()  # n_risks per frame, 0..7
        self.categories: Counter = Counter()
        self.motion_states: Counter = Counter()
        self.category_motion: Dict[str, Counter] = {}
        self.objects_per_frame: Counter = Counter()  # 0..8
        self.speed_action_x_yield: Counter = Counter()  # (action, "true"/"false")
        self.total_objects = 0
        self.reasoning_lengths: List[int] = []

    def update(self, expected_output: Mapping) -> None:
        """Count one GT-filled record (quarantined None never reaches here)."""
        self.frames += 1
        action = expected_output["speed_action"]
        self.speed_action[action] += 1
        yield_key = "true" if expected_output["yield_required"] else "false"
        self.yield_required[yield_key] += 1
        self.speed_action_x_yield[(action, yield_key)] += 1
        risks = list(expected_output["risk_factors"])
        self.risk_frames[len(risks)] += 1
        self.risk_terms.update(risks)
        objects = list(expected_output["critical_objects"])
        self.objects_per_frame[len(objects)] += 1
        self.total_objects += len(objects)
        for obj in objects:
            self.categories[obj["category"]] += 1
            self.motion_states[obj["motion_state"]] += 1
            self.category_motion.setdefault(obj["category"], Counter())[
                obj["motion_state"]
            ] += 1
        self.reasoning_lengths.append(len(expected_output["reasoning"]))

    def merge(self, other: "GtDistribution") -> None:
        """Fold ``other`` into self (used for the cross-split TOTAL)."""
        self.frames += other.frames
        self.speed_action.update(other.speed_action)
        self.yield_required.update(other.yield_required)
        self.risk_terms.update(other.risk_terms)
        self.risk_frames.update(other.risk_frames)
        self.categories.update(other.categories)
        self.motion_states.update(other.motion_states)
        for category, counts in other.category_motion.items():
            self.category_motion.setdefault(category, Counter()).update(counts)
        self.objects_per_frame.update(other.objects_per_frame)
        self.speed_action_x_yield.update(other.speed_action_x_yield)
        self.total_objects += other.total_objects
        self.reasoning_lengths.extend(other.reasoning_lengths)

    def to_payload(self) -> dict:
        """Deterministic JSON-ready dict (sorted keys, explicit zero rows)."""
        lengths = sorted(self.reasoning_lengths)
        mean = round(sum(lengths) / len(lengths), 1) if lengths else 0.0

        def sorted_counts(counter: Counter, vocabulary) -> Dict[str, int]:
            return {str(name): int(counter.get(name, 0)) for name in sorted(vocabulary)}

        return {
            "frames": self.frames,
            "reasoning_length": {"mean": mean, "p50": _pctl(lengths, 0.50), "p90": _pctl(lengths, 0.90)},
            "risk_frames": sorted_counts(self.risk_frames, range(len(RISK_TERMS) + 1)),
            "risk_terms": sorted_counts(self.risk_terms, RISK_TERMS),
            "speed_action": sorted_counts(self.speed_action, SPEED_ACTIONS),
            "total_objects": self.total_objects,
            "yield_required": sorted_counts(self.yield_required, YIELD_KEYS),
            "category_motion": {
                category: sorted_counts(self.category_motion.get(category, Counter()), MOTION_STATES)
                if category in self.category_motion
                else {name: 0 for name in sorted(MOTION_STATES)}
                for category in sorted(CATEGORIES)
            },
            "categories": sorted_counts(self.categories, CATEGORIES),
            "motion_states": sorted_counts(self.motion_states, MOTION_STATES),
            "speed_action_x_yield": {
                action: {
                    key: int(self.speed_action_x_yield.get((action, key), 0))
                    for key in YIELD_KEYS
                }
                for action in SPEED_ACTIONS
            },
            "objects_per_frame": sorted_counts(self.objects_per_frame, range(9)),
        }


def _pct(part: int, total: int) -> str:
    return f"{part} ({part / total * 100:.1f}%)" if total else "0"


def render_gt_distribution_markdown(per_split: Dict[str, dict], meta: Mapping) -> str:
    """Render the frozen-table markdown report (``all`` row required)."""
    splits = [s for s in per_split if s != "all"]
    columns = splits + ["all"]
    lines = [
        "# S08 v4 GT distribution report",
        "",
        "## Meta",
        "",
    ]
    for key in sorted(meta):
        lines.append(f"- {key}: `{meta[key]}`")
    lines += [
        "",
        "Frames = GT-backfilled (valid) records; quarantined records carry no "
        "GT and are excluded. Enum vocabularies are the frozen v4 contract "
        "assets, so zero-count rows stay visible.",
        "",
    ]

    def table(title: str, header: List[str], rows: List[List[str]]) -> None:
        lines.extend([f"## {title}", "", "| " + " | ".join(header) + " |",
                      "|" + "---|" * len(header)])
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        lines.append("")

    # 1) speed_action
    rows = [
        [action] + [_pct(per_split[c]["speed_action"][action], per_split[c]["frames"]) for c in columns]
        for action in SPEED_ACTIONS
    ]
    table("speed_action", ["split", *columns], rows)

    # 2) yield_required
    rows = [
        [key] + [_pct(per_split[c]["yield_required"][key], per_split[c]["frames"]) for c in columns]
        for key in YIELD_KEYS
    ]
    table("yield_required", ["split", *columns], rows)

    # 3) risk_factors (risk_factors is unique per frame -> count == frames-with-term)
    frames = {c: per_split[c]["frames"] for c in columns}
    rows = [
        [term] + [_pct(per_split[c]["risk_terms"][term], frames[c]) for c in columns]
        for term in RISK_TERMS
    ]
    rows.append(
        ["frames_with_risk"]
        + [_pct(frames[c] - per_split[c]["risk_frames"]["0"], frames[c]) for c in columns]
    )
    table("risk_factors (unique per frame: count == frames)", ["split", *columns], rows)

    # 4) critical_objects: category (counts are objects, not frames)
    rows = [
        [category] + [str(per_split[c]["categories"][category]) for c in columns]
        for category in CATEGORIES
    ]
    rows.append(["total_objects"] + [str(per_split[c]["total_objects"]) for c in columns])
    table("critical_objects: category", ["split", *columns], rows)

    # 5) critical_objects: motion_state
    rows = [
        [state] + [str(per_split[c]["motion_states"][state]) for c in columns]
        for state in MOTION_STATES
    ]
    table("critical_objects: motion_state", ["split", *columns], rows)

    # 6) category x motion_state (all)
    rows = [
        [category]
        + [str(per_split["all"]["category_motion"][category][state]) for state in MOTION_STATES]
        + [str(sum(per_split["all"]["category_motion"][category].values()))]
        for category in CATEGORIES
    ]
    table("critical_objects: category x motion_state (all)", ["category", *MOTION_STATES, "total"], rows)

    # 7) objects-per-frame histogram
    rows = [
        [f"{n}"] + [str(per_split[c]["objects_per_frame"][str(n)]) for c in columns]
        for n in range(9)
    ]
    table("objects per frame (critical_objects maxItems=8)", ["n_objects", *columns], rows)

    # 8) speed_action x yield_required (all)
    total_frames = per_split["all"]["frames"]
    cross = per_split["all"]["speed_action_x_yield"]
    rows = [
        [action]
        + [_pct(cross[action][key], total_frames) for key in YIELD_KEYS]
        + [_pct(per_split["all"]["speed_action"][action], total_frames)]
        for action in SPEED_ACTIONS
    ]
    table("speed_action x yield_required (all)", ["speed_action", *YIELD_KEYS, "total"], rows)

    # 9) reasoning length
    rows = [
        [stat]
        + [str(per_split[c]["reasoning_length"][stat]) for c in columns]
        for stat in ("mean", "p50", "p90")
    ]
    table("reasoning length (chars)", ["stat", *columns], rows)
    return "\n".join(lines) + "\n"
