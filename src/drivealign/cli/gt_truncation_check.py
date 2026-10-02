"""Truncation risk verification for the frozen pool geometry (S08 P1.2).

Purpose:
    The P1.2 calibration chose the pool geometry (cone 30 deg, disc 15 m,
    range scale 0.75) accepting an ~11.3% truncation-frame share against
    the 5% target, on the safety argument that the objects cut by the
    distance-ascending top-8 rule are the FARTHEST ones, whose corridor-
    conflict probability is low at urban speeds. This CLI quantifies that
    argument on the train split ONLY: it replays the frozen pool, collects
    every object the top-8 cut drops (rank 9+ among geometric survivors),
    and runs the frozen corridor conflict test on them (t0 bilateral CV
    counterfactual, T_risk / d_risk values read from ``gt_rule_config.json``
    so the check always validates the actual frozen file).

    Reported metrics:
    - truncation frame share and truncated-object count (sanity vs. the
      calibration report);
    - share of truncated objects that would fire ANY corridor-family risk;
    - frame-level missed evidence: the pool fires NO corridor-family risk
      while a truncated object would — under the frozen rules these are
      exactly the frames whose ``yield_required`` GT would flip to True;
    - corridor factors fired by truncated objects, per taxonomy term;
    - distance distribution (P50/P90) of truncated objects.

    HARD CONSTRAINT: train-only — the CLI reads the
    ``train_scene_manifest.json`` ONLY; val/test never enter and their
    manifests are never opened here.

    Outputs: ``runs/S08_gt_backfill/truncation_risk_report.{json,md}``.

Example launch command (full train sweep, loads ~10-15 GB RAM, tmux):
    ``tmux new-session -d -s s08_trunc \
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
    cd /root/autodl-tmp/drivealign_workspace && \
    set -o pipefail && PYTHONPATH=DriveAlign/src python -m drivealign.cli.gt_truncation_check \
    2>&1 | tee runs/S08_gt_backfill/truncation_check.log'``

Smoke (2 scenes):
    ``PYTHONPATH=DriveAlign/src python -m drivealign.cli.gt_truncation_check \
    --max-scenes 2 --out runs/S08_gt_backfill/truncation_smoke``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from nuscenes.nuscenes import NuScenes

from drivealign.data.nuscenes_io import load_nuscenes
from drivealign.dataset.build_dataset import load_scene_manifest, scene_keyframes
from drivealign.evaluation.projection import project_box
from drivealign.gt.backfill import NUM_FUTURE_FRAMES, collect_anchor_evidence
from drivealign.gt.config import PoolConfig, RuleConfig, canonical_category, load_rule_config
from drivealign.gt.observability import UNSCORABLE, classify_observability
from drivealign.gt.pool import (
    MAX_POOL_SIZE,
    PoolCandidate,
    azimuth_deg_from_center,
    select_objects,
)
from drivealign.gt.risk import (
    EgoState,
    ObjectState,
    corridor_conflict,
    corridor_distance_curve,
)

DEFAULT_MANIFESTS = "data/dataset_v1"
DEFAULT_OUT = "runs/S08_gt_backfill"
DEFAULT_DATAROOT = "data/nuscenes/trainval"
DEFAULT_VERSION = "v1.0-trainval"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path(DEFAULT_DATAROOT))
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--scene-manifests",
        type=Path,
        default=Path(DEFAULT_MANIFESTS),
        help="Directory holding the frozen scene manifests (train read ONLY).",
    )
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument(
        "--max-scenes", type=int, default=None, help="Smoke: first N train scenes."
    )
    parser.add_argument(
        "--max-anchors", type=int, default=None, help="Smoke: first N anchors."
    )
    return parser.parse_args(argv)


def _backward_speed(nusc: NuScenes, token: str) -> float:
    """t0 backward-difference planar speed (m/s)."""
    sample = nusc.get("sample", token)
    prev_token = sample["prev"]
    if not prev_token:
        return 0.0
    prev = nusc.get("sample", prev_token)

    def _ego_xy(s: dict) -> Tuple[float, float]:
        sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        return float(pose["translation"][0]), float(pose["translation"][1])

    x0, y0 = _ego_xy(prev)
    x1, y1 = _ego_xy(sample)
    dt = (sample["timestamp"] - prev["timestamp"]) / 1e6
    if dt <= 0:
        return 0.0
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt


def _geometric_survivors(
    candidates: List[PoolCandidate], pool_cfg: PoolConfig
) -> List[PoolCandidate]:
    """The full geometric survivor list BEFORE the top-8 cut.

    Mirrors ``pool.select_objects`` minus the ``[:MAX_POOL_SIZE]`` slice;
    keep the two in sync if the frozen rule changes.
    """
    cone_half = pool_cfg.cone_half_angle_deg
    disc = pool_cfg.proximity_disc_radius_m
    kept = []
    for cand in candidates:
        group = pool_cfg.group_of(cand.category)
        in_cone = (
            abs(cand.azimuth_deg) <= cone_half
            and cand.distance_m < pool_cfg.cone_range_by_group_m[group]
        )
        if in_cone or cand.distance_m < disc:
            kept.append(cand)
    kept.sort(key=lambda c: (round(c.distance_m, 3), c.ann_token))
    return kept


def _pNN(hist: Counter, total: int, q: float) -> Optional[float]:
    """Nearest-rank percentile of an integer histogram {value: count}."""
    if total <= 0:
        return None
    target = max(1, int(round(q * total)))
    cumulative = 0
    for value in sorted(hist):
        cumulative += hist[value]
        if cumulative >= target:
            return float(value)
    return None


class _Aggregates:
    """All accumulators for one truncation-check run."""

    def __init__(self) -> None:
        self.anchors = 0
        self.truncated_frames = 0
        self.truncated_objects = 0
        self.truncated_conflict_objects = 0
        self.factor_hits: Counter = Counter()
        self.category_hist: Counter = Counter()
        self.distance_hist: Counter = Counter()  # int-meter bins
        self.missed_conflict_frames = 0  # pool silent, a truncated object fires


def _process_anchor(
    nusc: NuScenes,
    token: str,
    config: RuleConfig,
    agg: _Aggregates,
) -> None:
    evidence = collect_anchor_evidence(nusc, token, NUM_FUTURE_FRAMES)
    agg.anchors += 1

    ego_state = EgoState(
        x=evidence.ego_x,
        y=evidence.ego_y,
        yaw=evidence.ego_yaw,
        speed_mps=_backward_speed(nusc, token),
        length_m=config.corridor.ego_length_m,
        width_m=config.corridor.ego_width_m,
    )

    candidates: List[PoolCandidate] = []
    for box in evidence.camera_boxes:
        category = canonical_category(box.name)
        projection = project_box(box, evidence.intrinsics)
        gated = classify_observability(
            projection.corners_2d,
            projection.depth,
            evidence.image_size,
            config.observability.min_depth_m,
            config.observability.min_visible_area_frac,
        )
        if category is None or gated.state == UNSCORABLE:
            continue
        ann = evidence.ann_records.get(box.token)
        if ann is None:
            continue
        candidates.append(
            PoolCandidate(
                ann_token=box.token,
                instance_token=ann.instance_token,
                category=category,
                distance_m=projection.distance_m,
                azimuth_deg=azimuth_deg_from_center(
                    float(box.center[0]), float(box.center[2])
                ),
                bbox_2d=gated.bbox_2d,
                visibility_state=gated.state,
                global_x=ann.x,
                global_y=ann.y,
                velocity_x_mps=ann.velocity_x_mps,
                velocity_y_mps=ann.velocity_y_mps,
            )
        )

    survivors = _geometric_survivors(candidates, config.pool)
    if len(survivors) <= MAX_POOL_SIZE:
        return  # no truncation on this frame
    agg.truncated_frames += 1
    pool = select_objects(candidates, config.pool)
    pool_tokens = {c.ann_token for c in pool}

    def _object_state(c: PoolCandidate) -> ObjectState:
        return ObjectState(
            ann_token=c.ann_token,
            category=c.category,
            x=c.global_x,
            y=c.global_y,
            velocity_x_mps=c.velocity_x_mps,
            velocity_y_mps=c.velocity_y_mps,
            distance_m=c.distance_m,
            azimuth_deg=c.azimuth_deg,
        )

    def _fires(o: ObjectState) -> Optional[str]:
        """Taxonomy term this object's corridor conflict would produce (None = no conflict)."""
        curve = corridor_distance_curve(
            ego_state,
            o,
            config.corridor.t_risk_s,
            config.corridor.time_step_s,
            config.corridor.width_margin_m,
        )
        d_risk = (
            config.corridor.d_risk_pedestrian_m
            if o.category == "pedestrian"
            else config.corridor.d_risk_vehicle_m
        )
        if not corridor_conflict(curve, d_risk):
            return None
        if o.category == "pedestrian":
            return "pedestrian_crossing"
        if config.pool.group_of(o.category) == "vehicle":
            return "vehicle_merging"
        return "corridor_conflict"

    # Pool side: does ANY in-pool object already fire the corridor family?
    pool_conflict = any(_fires(_object_state(c)) for c in pool)

    # Truncated side: rank-9+ geometric survivors only.
    truncated = [c for c in survivors[MAX_POOL_SIZE:] if c.ann_token not in pool_tokens]
    frame_missed = False
    for c in truncated:
        agg.truncated_objects += 1
        agg.category_hist[c.category] += 1
        agg.distance_hist[int(c.distance_m)] += 1
        o = _object_state(c)
        factor = _fires(o)
        if factor is None:
            continue
        agg.truncated_conflict_objects += 1
        agg.factor_hits[factor] += 1
        agg.factor_hits["corridor_conflict"] += 1
        if not pool_conflict:
            frame_missed = True
    if frame_missed:
        agg.missed_conflict_frames += 1


def _write_report(
    agg: _Aggregates, config: RuleConfig, out_dir: Path, config_note: dict
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(agg.anchors, 1)
    n_trunc_frames = max(agg.truncated_frames, 1)

    report = {
        "anchors_total": agg.anchors,
        "truncated_frames": agg.truncated_frames,
        "truncated_frame_share": round(agg.truncated_frames / n, 5),
        "truncated_objects_total": agg.truncated_objects,
        "truncated_objects_per_truncated_frame_mean": (
            round(agg.truncated_objects / agg.truncated_frames, 3)
            if agg.truncated_frames
            else None
        ),
        "truncated_conflict_objects": agg.truncated_conflict_objects,
        "truncated_conflict_object_share": round(
            agg.truncated_conflict_objects / max(agg.truncated_objects, 1), 5
        ),
        "truncated_factor_hits": dict(sorted(agg.factor_hits.items())),
        "truncated_category_hist": dict(sorted(agg.category_hist.items())),
        "truncated_distance_m_p50": _pNN(agg.distance_hist, agg.truncated_objects, 0.50),
        "truncated_distance_m_p90": _pNN(agg.distance_hist, agg.truncated_objects, 0.90),
        "missed_conflict_frames": agg.missed_conflict_frames,
        "missed_conflict_frame_share": round(agg.missed_conflict_frames / n, 5),
        "missed_conflict_share_of_truncated_frames": round(
            agg.missed_conflict_frames / n_trunc_frames, 5
        ),
        "frozen_parameters": {
            "cone_half_angle_deg": config.pool.cone_half_angle_deg,
            "proximity_disc_radius_m": config.pool.proximity_disc_radius_m,
            "cone_range_by_group_m": config.pool.cone_range_by_group_m,
            "t_risk_s": config.corridor.t_risk_s,
            "d_risk_vehicle_m": config.corridor.d_risk_vehicle_m,
            "d_risk_pedestrian_m": config.corridor.d_risk_pedestrian_m,
        },
        "notes": [
            "under the frozen rules a 'missed conflict frame' is exactly a frame "
            "whose yield_required GT would flip to True if the pool cap were lifted",
            "factor naming mirrors the frozen risk rule: a conflicting pedestrian "
            "adds pedestrian_crossing, a conflicting vehicle-group object adds "
            "vehicle_merging, other categories only raise corridor_conflict",
        ],
    }
    (out_dir / "truncation_risk_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    md = [
        "# Pool truncation risk verification (train-only, S08 P1.2)",
        "",
        "Quantifies what the distance-ascending top-8 cut drops under the",
        "frozen pool geometry: corridor-family evidence carried ONLY by",
        "rank-9+ geometric survivors.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
        f"| anchors | {agg.anchors} |",
        f"| truncated frames | {agg.truncated_frames} ({agg.truncated_frames / n:.2%}) |",
        f"| truncated objects | {agg.truncated_objects} |",
        f"| truncated objects firing corridor conflict | {agg.truncated_conflict_objects} "
        f"({agg.truncated_conflict_objects / max(agg.truncated_objects, 1):.2%} of truncated) |",
        f"| missed-evidence frames (pool silent, truncated fires) | {agg.missed_conflict_frames} "
        f"({agg.missed_conflict_frames / n:.2%} of all, "
        f"{agg.missed_conflict_frames / n_trunc_frames:.2%} of truncated) |",
        f"| truncated distance P50 / P90 (m) | "
        f"{report['truncated_distance_m_p50']} / {report['truncated_distance_m_p90']} |",
        "",
        "## Factor hits by truncated objects",
        "",
        "| factor | hits |",
        "|---|---|",
    ]
    for factor, hits in sorted(agg.factor_hits.items()):
        md.append(f"| {factor} | {hits} |")
    if not agg.factor_hits:
        md.append("| (none) | 0 |")
    md += [
        "",
        "## Truncated objects by category",
        "",
        "| category | count |",
        "|---|---|",
    ]
    for cat, cnt in sorted(agg.category_hist.items()):
        md.append(f"| {cat} | {cnt} |")
    md += [
        "",
        "## Frozen parameters used",
        "",
        "```json",
        json.dumps(report["frozen_parameters"], indent=2),
        "```",
        "",
        "## Notes",
        "",
        f"- {report['notes'][0]}.",
        f"- {report['notes'][1]}.",
        "",
        "## Run provenance",
        "",
        f"- train scene manifest: `{config_note['train_scene_manifest']}` "
        f"(sha256 {config_note['train_scene_manifest_sha256'][:12]}...)",
        f"- max_scenes: {config_note['max_scenes']}, max_anchors: {config_note['max_anchors']}",
    ]
    (out_dir / "truncation_risk_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_rule_config()
    manifest_path = args.scene_manifests.expanduser() / "train_scene_manifest.json"
    scene_tokens, fingerprint = load_scene_manifest(manifest_path)
    if args.max_scenes is not None:
        scene_tokens = scene_tokens[: args.max_scenes]

    print(f"TRAIN-ONLY truncation check: manifest {manifest_path} ({fingerprint[:12]}...)")
    print(f"loading {args.version} from {args.dataroot} ...", flush=True)
    nusc = load_nuscenes(args.dataroot.expanduser(), args.version)
    print("nuscenes loaded", flush=True)

    agg = _Aggregates()
    n_scenes = len(scene_tokens)
    processed = 0
    for s_pos, scene_token in enumerate(scene_tokens, start=1):
        tokens = scene_keyframes(nusc, scene_token)
        for walk_index, token in enumerate(tokens):
            if walk_index < 3:
                continue  # 4F window anchors only (S07 candidate rule)
            if args.max_anchors is not None and processed >= args.max_anchors:
                break
            _process_anchor(nusc, token, config, agg)
            processed += 1
            if processed % 500 == 0:
                print(f"anchors processed: {processed}", flush=True)
        if args.max_anchors is not None and processed >= args.max_anchors:
            break
        if s_pos % 20 == 0 or s_pos == n_scenes:
            print(f"[train] scene {s_pos}/{n_scenes} anchors={processed}", flush=True)

    config_note = {
        "train_scene_manifest": str(manifest_path),
        "train_scene_manifest_sha256": fingerprint,
        "max_scenes": args.max_scenes,
        "max_anchors": args.max_anchors,
    }
    _write_report(agg, config, args.out.expanduser(), config_note)
    print(f"anchors={agg.anchors} -> {args.out / 'truncation_risk_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
