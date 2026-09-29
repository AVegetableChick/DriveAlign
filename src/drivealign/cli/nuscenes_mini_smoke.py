"""Run the Stage 04 nuScenes-mini four-frame smoke with GT projection output.

Purpose:
    Read a CAM_FRONT keyframe history chain (anchor + 3 predecessors along
    sample.prev), validate same-scene / strictly-increasing timestamps /
    ~0.5 s gaps, project each frame's own keyframe GT boxes into its image,
    and write the four-frame visualization grid plus machine-readable
    timeline / projection / gate reports. This verifies data plumbing only
    and never proves driving ability.

Startup command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.nuscenes_mini_smoke``
    Re-run with ``--out runs/nuscenes_mini_smoke_rerun`` and diff the two
    timeline/projection JSONs for the determinism gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from drivealign.data.nuscenes_io import (
    load_nuscenes,
    read_keyframe_chain,
    resolve_anchor,
)
from drivealign.evaluation.box_render import compose_grid, render_frame
from drivealign.evaluation.projection import (
    camera_boxes_for_frame,
    classify_visibility,
    project_box,
)

DEFAULT_DATAROOT = "data/nuscenes/mini"
DEFAULT_OUT = "runs/nuscenes_mini_smoke"
JPEG_QUALITY = 95
VISIBILITY_MARGIN = 0.5

DRAWABLE_DECISIONS = frozenset({"visible", "clipped"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path(DEFAULT_DATAROOT))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=50.0,
        help="Only boxes whose sensor-frame center distance is within this "
        "radius are drawn (all boxes stay in the report).",
    )
    return parser.parse_args()


def _offset_tag(offset_s: float) -> str:
    """-1.5 -> 't-1500ms', -0.5 -> 't-0500ms', 0.0 -> 't0'."""
    ms = int(round(offset_s * 1000))
    if ms == 0:
        return "t0"
    sign = "-" if ms < 0 else "+"
    return f"t{sign}{abs(ms):04d}ms"


def _rounded_corners(corners_2d) -> list[list[float]] | None:
    """8x2 rounded coordinates, or None when any value is non-finite."""
    flat = [float(v) for row in corners_2d for v in row]
    if any(v != v or v in (float("inf"), float("-inf")) for v in flat):
        return None
    return [[round(float(x), 3), round(float(y), 3)] for x, y in corners_2d]


def _write_json(path: Path, payload: dict) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n")
    return (text + "\n").encode("utf-8")


def main() -> int:
    args = _parse_args()
    args.out.expanduser().mkdir(parents=True, exist_ok=True)

    config_echo = {
        "dataroot": str(args.dataroot.expanduser().resolve()),
        "version": args.version,
        "scene_name": args.scene_name,
        "sample_token": args.sample_token,
        "max_distance_m": args.max_distance_m,
        "visibility_margin": VISIBILITY_MARGIN,
    }

    nusc = load_nuscenes(args.dataroot.expanduser(), args.version)
    anchor_token = resolve_anchor(
        nusc, scene_name=args.scene_name, sample_token=args.sample_token
    )
    chain = read_keyframe_chain(nusc, anchor_token)
    if chain.frames is None:
        summary = {
            **config_echo,
            "anchor_sample_token": anchor_token,
            "chain_ok": False,
            "reason_code": chain.reason_code,
            "all_pass": False,
        }
        _write_json(args.out / "summary.json", summary)
        print(f"FAIL: chain read failed with reason_code={chain.reason_code}")
        return 1

    frames = chain.frames
    timeline = _write_json(
        args.out / "timeline.json",
        {
            **config_echo,
            "anchor_sample_token": anchor_token,
            "scene_token": chain.scene_token,
            "scene_name": chain.scene_name,
            "devkit": {
                "nuscenes_module": __import__("nuscenes").__file__,
                "sha_reference": "artifacts/environment/nuscenes-devkit-sha.txt",
            },
            "frames": [
                {
                    "sample_token": f.sample_token,
                    "sample_data_token": f.sample_data_token,
                    "timestamp_us": f.timestamp_us,
                    "offset_s": f.offset_s,
                    "offset_tag": _offset_tag(f.offset_s),
                    "image_relpath": f.image_relpath,
                    "image_size": list(f.image_size),
                    "is_anchor": f.is_anchor,
                }
                for f in frames
            ],
        },
    )

    report_frames = []
    panels = []
    for record in frames:
        frame_data = camera_boxes_for_frame(nusc, record.sample_data_token)
        box_entries = []
        drawn_boxes = []
        for box in frame_data.boxes:
            projection = project_box(box, frame_data.intrinsics)
            decision = classify_visibility(
                projection.corners_2d,
                projection.depth,
                record.image_size,
                margin=VISIBILITY_MARGIN,
            )
            drawn = decision in DRAWABLE_DECISIONS and (
                projection.distance_m <= args.max_distance_m
            )
            box_entries.append(
                {
                    "annotation_token": box.token,
                    "category": box.name,
                    "distance_m": round(projection.distance_m, 3),
                    "depth_min_m": round(float(projection.depth.min()), 3),
                    "corners_2d": _rounded_corners(projection.corners_2d),
                    "visibility_decision": decision,
                    "drawn": drawn,
                }
            )
            if drawn:
                drawn_boxes.append(
                    (
                        projection.corners_2d,
                        projection.depth,
                        box.name,
                        projection.distance_m,
                    )
                )

        caption = (
            f"{_offset_tag(record.offset_s)}  offset={record.offset_s:+.3f}s  "
            f"ts={record.timestamp_us}  scene={chain.scene_name}  "
            f"CAM_FRONT keyframe"
        )
        panel = render_frame(frame_data.image_path, drawn_boxes, caption)
        stem = _offset_tag(record.offset_s)
        panel.save(args.out / f"frame_{stem}.jpg", quality=JPEG_QUALITY)
        panels.append(panel)

        report_frames.append(
            {
                "sample_data_token": record.sample_data_token,
                "offset_s": record.offset_s,
                "offset_tag": stem,
                "drawn_count": len(drawn_boxes),
                "boxes": box_entries,
            }
        )

    projection_report = _write_json(
        args.out / "projection_report.json",
        {
            **config_echo,
            "image_size": list(frames[-1].image_size),
            "frames": report_frames,
        },
    )

    grid = compose_grid(panels)
    grid.save(args.out / "four_frame_grid.jpg", quality=JPEG_QUALITY)

    offsets = [record.offset_s for record in frames]
    tokens = [record.sample_data_token for record in frames]
    anchor_report = next(f for f in report_frames if f["offset_s"] == 0.0)
    gates = {
        "chain_ok": True,
        "same_scene": True,  # enforced inside read_keyframe_chain
        "strictly_increasing": all(a < b for a, b in zip(offsets, offsets[1:])),
        "tokens_unique": len(set(tokens)) == len(tokens) == 4,
        "finite_values": all(
            entry["corners_2d"] is not None
            for frame in report_frames
            for entry in frame["boxes"]
            if entry["drawn"]
        ),
        "at_least_one_visible_gt": anchor_report["drawn_count"] >= 1,
    }
    gates["all_pass"] = all(gates.values())
    _write_json(
        args.out / "summary.json",
        {
            **config_echo,
            "anchor_sample_token": anchor_token,
            "scene_token": chain.scene_token,
            "scene_name": chain.scene_name,
            "offsets_s": offsets,
            "reason_code": None,
            "gates": gates,
            "report_sha256": hashlib.sha256(timeline + projection_report).hexdigest(),
        },
    )

    for frame in report_frames:
        print(
            f"{frame['offset_tag']:>10s}: {frame['drawn_count']} GT boxes drawn "
            f"({len(frame['boxes'])} annotations)"
        )
    if gates["all_pass"]:
        print(f"PASS: all gates ok -> {args.out / 'four_frame_grid.jpg'}")
    else:
        print(f"FAIL: gates={gates}")
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
