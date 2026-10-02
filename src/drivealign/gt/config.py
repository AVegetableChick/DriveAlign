"""Loader for the three v4 GT-side contract assets.

Purpose:
    Single access point for the S08 rule engine inputs, all frozen under
    ``configs/contracts/v4/``:

    - ``gt_rule_config.json``: every threshold used by the GT derivation
      (P1). Parsed into a frozen :class:`RuleConfig` tree so no module can
      invent its own numbers; ``status`` flipped to ``frozen`` after the
      P1.2 train-only calibration plus the truncation-risk verification
      (runs/S08_gt_backfill/, 2026-10-02).
    - ``enum_phrase_map.json``: enum -> fixed phrase, consumed by both the
      rules (canonical enums) and the renderer (P4.2 one-table-two-consumers).
    - ``reasoning_templates.json``: the three reasoning segments with their
      wording variants and salts (P4.1).

    Also hosts the nuScenes full-category -> canonical 10-class mapping
    (mirrors the devkit detection-name folding; anything outside the closed
    v3 category vocabulary is dropped from GT derivation).

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.config import load_rule_config, canonical_category; \
    cfg = load_rule_config(); \
    print(cfg.rule_version, canonical_category('human.pedestrian.adult'))"``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

from drivealign.contracts.versions import contract_file

RULE_CONFIG_FILE = "gt_rule_config.json"
TEMPLATES_FILE = "reasoning_templates.json"
PHRASE_MAP_FILE = "enum_phrase_map.json"

# nuScenes full category -> canonical term of the v3 category vocabulary.
# Mirrors nuscenes_eval category_to_detection_name; entries folding to none
# (animal, debris, stroller, ...) are excluded from GT derivation entirely.
NUSCENES_CATEGORY_MAP: Dict[str, Optional[str]] = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.wheelchair": None,
    "human.pedestrian.stroller": None,
    "human.pedestrian.personal_mobility": None,
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": None,
    "movable_object.debris": None,
    "static_object.bicycle_rack": None,
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": None,
    "vehicle.emergency.police": None,
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def canonical_category(full_name: str) -> Optional[str]:
    """Map a nuScenes full category name to the closed v3 vocabulary."""
    return NUSCENES_CATEGORY_MAP.get(full_name)


@dataclass(frozen=True)
class PoolConfig:
    cone_half_angle_deg: float
    proximity_disc_radius_m: float
    cone_range_by_group_m: Dict[str, float]
    class_groups: Dict[str, Tuple[str, ...]]

    def group_of(self, category: str) -> str:
        for group, members in self.class_groups.items():
            if category in members:
                return group
        raise ValueError(f"category {category!r} not in any class group")


@dataclass(frozen=True)
class ObservabilityConfig:
    min_depth_m: float
    min_visible_area_frac: float


@dataclass(frozen=True)
class MotionConfig:
    stationary_speed_mps: float
    band_half_width_deg: float


@dataclass(frozen=True)
class CorridorConfig:
    t_risk_s: float
    time_step_s: float
    ego_length_m: float
    ego_width_m: float
    width_margin_m: float
    d_risk_vehicle_m: float
    d_risk_pedestrian_m: float


@dataclass(frozen=True)
class BrakingConfig:
    evidence_window_s: float
    speed_drop_mps: float


@dataclass(frozen=True)
class StationaryObstacleConfig:
    evidence_window_s: float
    max_speed_mps: float


@dataclass(frozen=True)
class CongestionConfig:
    lead_speed_mps: float
    min_queue_vehicles: int


@dataclass(frozen=True)
class OncomingConfig:
    range_m: float


@dataclass(frozen=True)
class ActionConfig:
    window_s: float
    delta: float
    stop_speed_mps: float


@dataclass(frozen=True)
class RuleConfig:
    rule_version: str
    status: str
    contract_version: str
    pool: PoolConfig
    observability: ObservabilityConfig
    motion: MotionConfig
    corridor: CorridorConfig
    braking: BrakingConfig
    stationary_obstacle: StationaryObstacleConfig
    congestion: CongestionConfig
    oncoming: OncomingConfig
    action: ActionConfig
    calibration_targets: Tuple[str, ...] = field(default_factory=tuple)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=None)
def _load_json(version: str, filename: str) -> Tuple[dict, ...]:
    payload = json.loads(contract_file(version, filename).read_text(encoding="utf-8"))
    return payload,  # single-element tuple so the cache stays hashable-safe


def load_rule_config(version: str = "v4") -> RuleConfig:
    """Parse and validate ``gt_rule_config.json`` into a frozen tree."""
    payload = _load_json(version, RULE_CONFIG_FILE)[0]
    _require(payload["status"] in ("provisional", "frozen"), "status invalid")

    pool = payload["pool"]
    groups = {g: tuple(members) for g, members in pool["class_groups"].items()}
    pool_cfg = PoolConfig(
        cone_half_angle_deg=float(pool["cone_half_angle_deg"]),
        proximity_disc_radius_m=float(pool["proximity_disc_radius_m"]),
        cone_range_by_group_m={
            g: float(v) for g, v in pool["cone_range_by_group_m"].items()
        },
        class_groups=groups,
    )
    _require(
        set(pool_cfg.cone_range_by_group_m) == set(groups),
        "cone_range_by_group_m keys must match class_groups keys",
    )
    obs = payload["observability"]
    motion = payload["motion"]
    corridor = payload["corridor"]
    braking = payload["braking"]
    stationary = payload["stationary_obstacle"]
    congestion = payload["congestion"]
    oncoming = payload["oncoming"]
    action = payload["action"]
    return RuleConfig(
        rule_version=str(payload["version"]),
        status=str(payload["status"]),
        contract_version=str(payload["contract_version"]),
        pool=pool_cfg,
        observability=ObservabilityConfig(
            min_depth_m=float(obs["min_depth_m"]),
            min_visible_area_frac=float(obs["min_visible_area_frac"]),
        ),
        motion=MotionConfig(
            stationary_speed_mps=float(motion["stationary_speed_mps"]),
            band_half_width_deg=float(motion["band_half_width_deg"]),
        ),
        corridor=CorridorConfig(
            t_risk_s=float(corridor["t_risk_s"]),
            time_step_s=float(corridor["time_step_s"]),
            ego_length_m=float(corridor["ego_length_m"]),
            ego_width_m=float(corridor["ego_width_m"]),
            width_margin_m=float(corridor["width_margin_m"]),
            d_risk_vehicle_m=float(corridor["d_risk_vehicle_m"]),
            d_risk_pedestrian_m=float(corridor["d_risk_pedestrian_m"]),
        ),
        braking=BrakingConfig(
            evidence_window_s=float(braking["evidence_window_s"]),
            speed_drop_mps=float(braking["speed_drop_mps"]),
        ),
        stationary_obstacle=StationaryObstacleConfig(
            evidence_window_s=float(stationary["evidence_window_s"]),
            max_speed_mps=float(stationary["max_speed_mps"]),
        ),
        congestion=CongestionConfig(
            lead_speed_mps=float(congestion["lead_speed_mps"]),
            min_queue_vehicles=int(congestion["min_queue_vehicles"]),
        ),
        oncoming=OncomingConfig(range_m=float(oncoming["range_m"])),
        action=ActionConfig(
            window_s=float(action["window_s"]),
            delta=float(action["delta"]),
            stop_speed_mps=float(action["stop_speed_mps"]),
        ),
        calibration_targets=tuple(payload.get("calibration_targets", ())),
    )


def load_phrase_map(version: str = "v4") -> dict:
    """Load the shared enum -> phrase table (P4.2)."""
    return _load_json(version, PHRASE_MAP_FILE)[0]


def load_templates(version: str = "v4") -> dict:
    """Load the P4.1 reasoning template library; validates variant counts."""
    templates = _load_json(version, TEMPLATES_FILE)[0]
    for name, segment in templates["segments"].items():
        for key in ("variants", "empty_variants"):
            if key in segment:
                _require(
                    3 <= len(segment[key]) <= 5,
                    f"segment {name!r} {key} must hold 3-5 variants, "
                    f"got {len(segment[key])}",
                )
    return templates
