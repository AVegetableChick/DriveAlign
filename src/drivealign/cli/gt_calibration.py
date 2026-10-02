"""P1.2 train-only GT threshold calibration (Stage 08).

Purpose:
    Sweep the frozen P1 parameter grids over ALL train anchors and emit the
    recommendation report the user needs to freeze ``gt_rule_config.json``
    (provisional -> frozen). HARD CONSTRAINT: train-only — the CLI reads the
    ``train_scene_manifest.json`` ONLY; val/test never enter any fitted
    state and their manifests are never opened here.

    One memory-light pass over the anchors: per anchor the geometric atoms
    (observability state, distance, azimuth, t0 global states, corridor CV
    curves, future tracks, ego future speeds) are computed ONCE with the
    widest grid reach, then every grid cell is evaluated cheaply from those
    atoms. Only aggregates accumulate — no per-anchor data is retained.

    Grids (S08 implementation plan section 4):
    1. pool geometry: cone {20,25,30,35,40} deg x disc {8,10,12,15} m x
       class-range scale {0.5,0.75,1.0,1.5} -> truncation-frame share
       (target < 5%), pool size P50/P90, in-pool observability mix;
    2. corridor family: T_risk {2,3,4,6} s x d_risk vehicle {1.0,1.5,2.0,3.0} m
       x pedestrian {0.75,1.0,1.5,2.0} m -> conflict detection rate
       (sparsity check: > 50% = no discrimination, < 1% = GT starvation);
    3. motion: theta {30,45,60} deg x stationary {0.3,0.5,0.8} m/s ->
       four-value distribution over base-pool objects;
    4. action: window {2,3,4} s x delta {0.1,0.15,0.2} -> four-action share
       (by construction this IS the expert behavior distribution).

    small_following_gap was removed from the taxonomy by the P1.2
    calibration (<= 0.64% detection, starved) — no gap grid anymore.

    Outputs: ``runs/S08_gt_backfill/calibration_report.{json,md}``.

Example launch command (full sweep, loads ~10-15 GB RAM, run inside tmux):
    ``tmux new-session -d -s s08_calib \
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
    cd /root/autodl-tmp/drivealign_workspace && \
    set -o pipefail && PYTHONPATH=DriveAlign/src python -m drivealign.cli.gt_calibration \
    2>&1 | tee runs/S08_gt_backfill/calibration.log'``

Smoke (2 scenes):
    ``PYTHONPATH=DriveAlign/src python -m drivealign.cli.gt_calibration \
    --max-scenes 2 --out runs/S08_gt_backfill/smoke``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nuscenes.nuscenes import NuScenes

from drivealign.data.nuscenes_io import load_nuscenes
from drivealign.dataset.build_dataset import load_scene_manifest, scene_keyframes
from drivealign.evaluation.projection import project_box
from drivealign.gt.action_yield import (
    derive_speed_action,
    future_ego_speeds,
)
from drivealign.gt.backfill import NUM_FUTURE_FRAMES, collect_anchor_evidence
from drivealign.gt.config import RuleConfig, canonical_category, load_rule_config
from drivealign.gt.motion_state import classify_motion_state, relative_angle_deg
from drivealign.gt.observability import (
    INFERABLE,
    OBSERVABLE,
    UNSCORABLE,
    classify_observability,
)
from drivealign.gt.pool import (
    MAX_POOL_SIZE,
    PoolCandidate,
    azimuth_deg_from_center,
    select_objects,
    truncated_count,
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

CONES = (20.0, 25.0, 30.0, 35.0, 40.0)
DISCS = (8.0, 10.0, 12.0, 15.0)
RANGE_SCALES = (0.5, 0.75, 1.0, 1.5)
T_RISKS = (2.0, 3.0, 4.0, 6.0)
D_RISK_VEHICLE = (1.0, 1.5, 2.0, 3.0)
D_RISK_PEDESTRIAN = (0.75, 1.0, 1.5, 2.0)
MOTION_THETAS = (30.0, 45.0, 60.0)
MOTION_STATIONARY = (0.3, 0.5, 0.8)
ACTION_WINDOWS = (2.0, 3.0, 4.0)
ACTION_DELTAS = (0.1, 0.15, 0.2)

MAX_T_REACH = 6.0  # widest grid reach for curves/tracks/time steps
TIME_STEP = 0.25


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


def _backward_speed(nusc: NuScenes, token: str) -> Tuple[float, float]:
    """t0 backward-difference planar speed and the keyframe dt (seconds)."""
    sample = nusc.get("sample", token)
    prev_token = sample["prev"]
    if not prev_token:
        return 0.0, 0.5
    prev = nusc.get("sample", prev_token)

    def _ego_xy(s: dict) -> Tuple[float, float]:
        sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        return float(pose["translation"][0]), float(pose["translation"][1])

    x0, y0 = _ego_xy(prev)
    x1, y1 = _ego_xy(sample)
    dt = (sample["timestamp"] - prev["timestamp"]) / 1e6
    if dt <= 0:
        return 0.0, 0.5
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt, dt


def _future_ego_poses(
    nusc: NuScenes, token: str, max_frames: int = NUM_FUTURE_FRAMES
) -> List[Tuple[float, float, int]]:
    poses = []
    next_token = nusc.get("sample", token)["next"]
    while next_token and len(poses) < max_frames:
        sample = nusc.get("sample", next_token)
        sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        poses.append((float(pose["translation"][0]), float(pose["translation"][1]), int(sample["timestamp"])))
        next_token = sample["next"]
    return poses


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
    """All grid accumulators for one calibration run."""

    def __init__(self) -> None:
        self.anchors = 0
        self.obs_box_total = 0
        self.obs_states: Counter = Counter()  # observable/inferable/unscorable
        self.obs_reasons: Counter = Counter()
        self.pool_grid: Dict[Tuple, Dict[str, object]] = {}
        self.corridor_grid: Dict[Tuple, Counter] = {}
        self.motion_grid: Dict[Tuple, Counter] = {}
        self.action_grid: Dict[Tuple, Counter] = {}
        self.action_empty_future = 0

    @staticmethod
    def _pool_cell(store: Dict, key: Tuple) -> Dict[str, object]:
        cell = store.get(key)
        if cell is None:
            cell = {
                "frames": 0,
                "truncated": 0,
                "size_hist": Counter(),
                "observable_hist": Counter(),
                "pool_states": Counter(),
            }
            store[key] = cell
        return cell


def _process_anchor(
    nusc: NuScenes,
    token: str,
    config: RuleConfig,
    agg: _Aggregates,
) -> None:
    evidence = collect_anchor_evidence(nusc, token, NUM_FUTURE_FRAMES)
    v0, _ = _backward_speed(nusc, token)
    future_poses = _future_ego_poses(nusc, token)
    agg.anchors += 1

    ego_state = EgoState(
        x=evidence.ego_x,
        y=evidence.ego_y,
        yaw=evidence.ego_yaw,
        speed_mps=v0,
        length_m=config.corridor.ego_length_m,
        width_m=config.corridor.ego_width_m,
    )

    # --- per-box atoms: observability + geometry + t0 global state -------
    candidates: List[PoolCandidate] = []
    for box in evidence.camera_boxes:
        category = canonical_category(box.name)
        agg.obs_box_total += 1
        projection = project_box(box, evidence.intrinsics)
        gated = classify_observability(
            projection.corners_2d,
            projection.depth,
            evidence.image_size,
            config.observability.min_depth_m,
            config.observability.min_visible_area_frac,
        )
        agg.obs_states[gated.state] += 1
        if gated.state == UNSCORABLE:
            agg.obs_reasons[gated.reason] += 1
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

    object_states = [
        ObjectState(
            ann_token=c.ann_token,
            category=c.category,
            x=c.global_x,
            y=c.global_y,
            velocity_x_mps=c.velocity_x_mps,
            velocity_y_mps=c.velocity_y_mps,
            distance_m=c.distance_m,
            azimuth_deg=c.azimuth_deg,
        )
        for c in candidates
    ]

    # Corridor CV curves once at the widest reach.
    curves = {
        o.ann_token: corridor_distance_curve(
            ego_state, o, MAX_T_REACH, TIME_STEP, config.corridor.width_margin_m
        )
        for o in object_states
    }

    # --- grid 1: pool geometry -------------------------------------------
    base_pool_cfg = config.pool
    for cone in CONES:
        for disc in DISCS:
            for scale in RANGE_SCALES:
                pool_cfg = base_pool_cfg.__class__(
                    cone_half_angle_deg=cone,
                    proximity_disc_radius_m=disc,
                    cone_range_by_group_m={
                        g: r * scale
                        for g, r in base_pool_cfg.cone_range_by_group_m.items()
                    },
                    class_groups=base_pool_cfg.class_groups,
                )
                cell = _Aggregates._pool_cell(agg.pool_grid, (cone, disc, scale))
                cell["frames"] += 1  # type: ignore[operator]
                n_geo = truncated_count(candidates, pool_cfg)
                if n_geo > MAX_POOL_SIZE:
                    cell["truncated"] += 1  # type: ignore[operator]
                pool = select_objects(candidates, pool_cfg)
                cell["size_hist"][len(pool)] += 1  # type: ignore[operator]
                cell["observable_hist"][
                    sum(1 for c in pool if c.visibility_state == OBSERVABLE)
                ] += 1  # type: ignore[operator]
                for c in pool:
                    cell["pool_states"][c.visibility_state] += 1  # type: ignore[operator]

    # Base pool drives the motion / corridor default-reporting slices.
    base_pool = select_objects(candidates, base_pool_cfg)

    # --- grid 2: corridor family (pool-independent: over ALL gate survivors,
    # so the corridor thresholds are calibrated before the pool freezes) ----
    for t_risk in T_RISKS:
        for d_veh in D_RISK_VEHICLE:
            for d_ped in D_RISK_PEDESTRIAN:
                key = (t_risk, d_veh, d_ped)
                counts = agg.corridor_grid.setdefault(key, Counter())
                any_conflict = False
                any_ped = False
                any_veh = False
                for o in object_states:
                    curve = [
                        (t, d) for t, d in curves[o.ann_token] if t <= t_risk + 1e-9
                    ]
                    if not curve:
                        continue
                    d_risk = (
                        d_ped
                        if o.category == "pedestrian"
                        else d_veh
                    )
                    if corridor_conflict(curve, d_risk):
                        any_conflict = True
                        if o.category == "pedestrian":
                            any_ped = True
                        else:
                            any_veh = True
                if any_conflict:
                    counts["corridor_conflict"] += 1
                if any_ped:
                    counts["pedestrian_crossing"] += 1
                if any_veh:
                    counts["vehicle_merging"] += 1

    # --- grid 3: motion distribution over base-pool objects ----------------
    for theta in MOTION_THETAS:
        for stat in MOTION_STATIONARY:
            counts = agg.motion_grid.setdefault((theta, stat), Counter())
            for c in base_pool:
                rel = relative_angle_deg(
                    c.velocity_x_mps, c.velocity_y_mps, evidence.ego_yaw
                )
                counts[
                    classify_motion_state(
                        c.speed_mps, rel, stat, theta
                    )
                ] += 1

    # --- grid 4: speed_action windows --------------------------------------
    for window in ACTION_WINDOWS:
        speeds = future_ego_speeds(
            evidence.ego_x,
            evidence.ego_y,
            evidence.ego_timestamp_us,
            future_poses,
            window,
        )
        if not speeds:
            agg.action_empty_future += 1
        for delta in ACTION_DELTAS:
            counts = agg.action_grid.setdefault((window, delta), Counter())
            counts[
                derive_speed_action(v0, speeds, delta, config.action.stop_speed_mps)
            ] += 1


def _percentiles(hist: Counter, frames: int) -> Tuple[Optional[float], Optional[float]]:
    return _pNN(hist, frames, 0.50), _pNN(hist, frames, 0.90)


def _write_report(agg: _Aggregates, config: RuleConfig, out_dir: Path, config_note: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(agg.anchors, 1)

    # --- pool table + recommendation ---------------------------------------
    pool_rows = []
    passing = []
    for (cone, disc, scale), cell in sorted(agg.pool_grid.items()):
        frames = cell["frames"]  # type: ignore[operator]
        trunc_share = cell["truncated"] / frames  # type: ignore[operator]
        p50, p90 = _percentiles(cell["size_hist"], frames)  # type: ignore[operator]
        states = cell["pool_states"]  # type: ignore[operator]
        row = {
            "cone_half_angle_deg": cone,
            "proximity_disc_radius_m": disc,
            "range_scale": scale,
            "truncation_share": round(trunc_share, 5),
            "pool_size_p50": p50,
            "pool_size_p90": p90,
            "observable_share_in_pool": round(
                states[OBSERVABLE] / max(sum(states.values()), 1), 5
            ),
        }
        pool_rows.append(row)
        if trunc_share < 0.05:
            passing.append((cone, disc, scale))
    recommendation = None
    if passing:
        base = (
            config.pool.cone_half_angle_deg,
            config.pool.proximity_disc_radius_m,
            1.0,
        )
        recommendation = min(passing, key=lambda c: sum(abs(a - b) for a, b in zip(c, base)))

    report = {
        "manifest_type": "gt_calibration_report",
        "rule_version": config.rule_version,
        "rule_status_at_calibration": config.status,
        "train_only": True,
        "config_note": config_note,
        "anchors": agg.anchors,
        "observability": {
            "boxes_total": agg.obs_box_total,
            "states": dict(agg.obs_states),
            "unscorable_reasons": dict(agg.obs_reasons),
            "state_shares": {
                k: round(v / max(agg.obs_box_total, 1), 5)
                for k, v in agg.obs_states.items()
            },
        },
        "pool_grid": pool_rows,
        "pool_recommendation": {
            "cone_half_angle_deg": recommendation[0],
            "proximity_disc_radius_m": recommendation[1],
            "range_scale": recommendation[2],
            "note": "passing cell (truncation < 5%) closest to the provisional defaults; full table above for the user decision",
        }
        if recommendation
        else {
            "note": "NO passing cell under the 5% truncation target — widen ranges or raise the cap after user review"
        },
        "corridor_grid": {
            f"{t}|{dv}|{dp}": {
                "corridor_conflict": round(c["corridor_conflict"] / n, 5),
                "pedestrian_crossing": round(c["pedestrian_crossing"] / n, 5),
                "vehicle_merging": round(c["vehicle_merging"] / n, 5),
            }
            for (t, dv, dp), c in sorted(agg.corridor_grid.items())
        },
        "motion_grid": {
            f"theta={theta}|stationary={stat}": {
                k: round(v / max(sum(c.values()), 1), 5) for k, v in c.items()
            }
            for (theta, stat), c in sorted(agg.motion_grid.items())
        },
        "action_grid": {
            f"window={w}|delta={d}": {
                k: round(v / n, 5) for k, v in sorted(c.items())
            }
            for (w, d), c in sorted(agg.action_grid.items())
        },
        "action_empty_future_share": round(agg.action_empty_future / n, 5),
        "notes": [
            "corridor/motion grids are evaluated over ALL observability gate survivors; the pool geometry grid is calibrated separately (the frozen pool then restricts which objects can fire risks)",
            "action distribution == expert behavior distribution by construction (derived from the actual ego trajectory); the check is for degenerate labels, not for expert agreement",
            "corridor sparsity check: detection share > 0.5 means no discrimination, < 0.01 means GT starvation",
        ],
    }
    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    md = [
        "# S08 P1.2 train-only GT calibration report",
        "",
        f"- anchors: {agg.anchors}",
        f"- rule version: `{config.rule_version}` ({config.status})",
        "",
        "## Observability (base params)",
        "",
        "| state | share |",
        "|---|---|",
    ]
    for state in (OBSERVABLE, INFERABLE, UNSCORABLE):
        share = agg.obs_states[state] / max(agg.obs_box_total, 1)
        md.append(f"| {state} | {share:.4f} |")
    md += ["", "## Pool grid (target: truncation < 5%)", ""]
    md += [
        "| cone | disc | scale | trunc | size P50 | size P90 | obs in pool |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in pool_rows:
        md.append(
            f"| {row['cone_half_angle_deg']} | {row['proximity_disc_radius_m']} | "
            f"{row['range_scale']} | {row['truncation_share']:.4f} | "
            f"{row['pool_size_p50']} | {row['pool_size_p90']} | "
            f"{row['observable_share_in_pool']:.4f} |"
        )
    md += ["", f"Recommendation: `{report['pool_recommendation']}`", ""]
    md += [
        "## Corridor grid (detection shares)",
        "",
        "| T_risk | d_veh | d_ped | corridor | ped_crossing | veh_merging |",
        "|---|---|---|---|---|---|",
    ]
    for (t, dv, dp), c in sorted(agg.corridor_grid.items()):
        md.append(
            f"| {t} | {dv} | {dp} | {c['corridor_conflict'] / n:.4f} | "
            f"{c['pedestrian_crossing'] / n:.4f} | {c['vehicle_merging'] / n:.4f} |"
        )
    md += ["", "## Motion grid (shares)", "", "| theta | stationary | stationary | same | crossing | oncoming |", "|---|---|---|---|---|---|"]
    for (theta, stat), c in sorted(agg.motion_grid.items()):
        total = max(sum(c.values()), 1)
        md.append(
            f"| {theta} | {stat} | {c.get('stationary', 0) / total:.4f} | "
            f"{c.get('same_direction', 0) / total:.4f} | "
            f"{c.get('crossing', 0) / total:.4f} | "
            f"{c.get('oncoming', 0) / total:.4f} |"
        )
    md += ["", "## Action grid (shares)", "", "| window | delta | ACCELERATE | KEEP_SPEED | DECELERATE | STOP |", "|---|---|---|---|---|---|"]
    for (w, d), c in sorted(agg.action_grid.items()):
        md.append(
            f"| {w} | {d} | {c.get('ACCELERATE', 0) / n:.4f} | "
            f"{c.get('KEEP_SPEED', 0) / n:.4f} | {c.get('DECELERATE', 0) / n:.4f} | "
            f"{c.get('STOP', 0) / n:.4f} |"
        )
    md += [""]
    (out_dir / "calibration_report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_rule_config()
    manifest_path = args.scene_manifests.expanduser() / "train_scene_manifest.json"
    scene_tokens, fingerprint = load_scene_manifest(manifest_path)
    if args.max_scenes is not None:
        scene_tokens = scene_tokens[: args.max_scenes]

    print(f"TRAIN-ONLY calibration: manifest {manifest_path} ({fingerprint[:12]}...)")
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
        "grids": {
            "cones": CONES,
            "discs": DISCS,
            "range_scales": RANGE_SCALES,
            "t_risk": T_RISKS,
            "d_risk_vehicle": D_RISK_VEHICLE,
            "d_risk_pedestrian": D_RISK_PEDESTRIAN,
            "motion_thetas": MOTION_THETAS,
            "motion_stationary": MOTION_STATIONARY,
            "action_windows": ACTION_WINDOWS,
            "action_deltas": ACTION_DELTAS,
        },
    }
    _write_report(agg, config, args.out.expanduser(), config_note)
    print(f"anchors={agg.anchors} -> {args.out / 'calibration_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
