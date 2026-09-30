"""Run the Stage 06 record-build smoke over disjoint 4F windows.

Purpose:
    Enumerate the 32 mutually disjoint CAM_FRONT windows of the v1.0-mini
    train split (8 scenes x 4 windows; anchor stride 10 keyframes = 4F
    history + 3.0 s future), build one DriveAlignRecord per window via the
    Stage 06 adapter, derive 1F/4F requests via the serializer, scan every
    payload layer for future-field fingerprints, and evaluate the five
    Stage 06 gates. Writes summary.json + windows.json; re-running into a
    second --out directory and diffing windows.json doubles as a
    cross-process determinism check.

Startup command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.record_build_smoke``
    Determinism re-run: ``PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.record_build_smoke --out runs/record_build_smoke_rerun``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from nuscenes.utils.splits import mini_train

from drivealign.data.nuscenes_io import load_nuscenes
from drivealign.records.adapter import NUM_FUTURE_POSES, build_record
from drivealign.records.record import validate_record
from drivealign.records.serializer import (
    InputPolicy,
    collate,
    future_fingerprints,
    scan_future_fingerprints,
    serialize,
    to_processor_inputs,
)

DEFAULT_DATAROOT = "data/nuscenes/mini"
DEFAULT_OUT = "runs/record_build_smoke"
ANCHOR_INDEX_OFFSET = 3  # first window anchor needs 3 keyframe prevs
WINDOW_STRIDE = 10  # 4F history + 3.0 s future = 10 disjoint keyframes
EXPECTED_WINDOWS = 31  # strict-disjoint total; scene-0061 has 39 keyframes
# (3+10k <= 39-7 => k in {0,1,2}), so 7x4 + 1x3 = 31, not 32.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path(DEFAULT_DATAROOT))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument(
        "--builder-commit",
        default=None,
        help="Git commit recorded in provenance; defaults to `git rev-parse HEAD`.",
    )
    return parser.parse_args()


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[3]  # .../DriveAlign
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: dict) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n")
    return (text + "\n").encode("utf-8")


def _scene_samples(nusc, scene: dict) -> list[str]:
    """Ordered keyframe sample tokens of one scene (oldest -> newest)."""
    tokens = []
    current = scene["first_sample_token"]
    while current:
        tokens.append(current)
        current = nusc.get("sample", current)["next"]
    return tokens


def _disjoint_anchor_indices(n_samples: int) -> list[int]:
    """Anchor indices of disjoint windows: 3 prevs + 6 futures, stride 10."""
    last_usable = n_samples - 1 - NUM_FUTURE_POSES
    return list(range(ANCHOR_INDEX_OFFSET, last_usable + 1, WINDOW_STRIDE))


def _parity_ok(record, req1, req4) -> bool:
    """1F/4F must share anchor frame, current image, prompt and speed."""
    return (
        req1.prompt == req4.prompt
        and req1.frame_tokens[-1] == req4.frame_tokens[-1] == record.sample_token
        and req4.image_relpaths[-1] == req1.image_relpaths[0]
        and req1.ego_speed_mps == req4.ego_speed_mps == record.model_inputs.ego_speed_mps
        and record.training_targets.expected_output is None
        and record.training_targets.language_reference is None
    )


def main() -> int:
    args = _parse_args()
    args.out.expanduser().mkdir(parents=True, exist_ok=True)
    builder_commit = args.builder_commit or _git_commit()

    config_echo = {
        "dataroot": str(args.dataroot.expanduser().resolve()),
        "version": args.version,
        "builder_commit": builder_commit,
        "anchor_index_offset": ANCHOR_INDEX_OFFSET,
        "window_stride": WINDOW_STRIDE,
        "expected_windows": EXPECTED_WINDOWS,
        "future_poses": NUM_FUTURE_POSES,
    }

    nusc = load_nuscenes(args.dataroot.expanduser(), args.version)
    available = {s["name"]: s for s in nusc.scene}

    entries = []
    for scene_name in mini_train:  # frozen devkit order, 8 scenes
        scene = available.get(scene_name)
        if scene is None:
            continue
        samples = _scene_samples(nusc, scene)
        for index in _disjoint_anchor_indices(len(samples)):
            token = samples[index]
            built = build_record(nusc, token, builder_commit)
            entry: dict = {
                "scene_name": scene_name,
                "anchor_index": index,
                "sample_token": token,
                "outcome": "valid" if built.record is not None else "quarantine",
                "reason_code": built.reason_code,
                "detail": built.detail,
            }
            if built.record is not None:
                record = built.record
                req1 = serialize(record.model_inputs, InputPolicy.ONE_FRAME)
                req4 = serialize(record.model_inputs, InputPolicy.FOUR_FRAME)
                procs = [to_processor_inputs(r) for r in (req1, req4)]
                batch = collate(procs)
                scan = scan_future_fingerprints(
                    future_fingerprints(record),
                    {
                        "prompt_1f": req1.prompt,
                        "prompt_4f": req4.prompt,
                        "processor": json.dumps(
                            [p.to_dict() for p in procs], sort_keys=True
                        ),
                        "batch": json.dumps(batch.to_dict(), sort_keys=True),
                    },
                )
                rebuilt = build_record(nusc, token, builder_commit)
                entry.update(
                    {
                        "record": record.to_dict(),
                        "record_hash": record.canonical_hash(),
                        "record_hash_stable": rebuilt.record is not None
                        and rebuilt.record.canonical_hash() == record.canonical_hash(),
                        "request_hash_1f": req1.canonical_hash(),
                        "request_hash_4f": req4.canonical_hash(),
                        "policy_parity": _parity_ok(record, req1, req4),
                        "future_pose_count": len(record.oracle_only.future_ego_poses),
                        "validate_problems": validate_record(record),
                        "scan_hits": scan,
                    }
                )
            entries.append(entry)

    valid = [e for e in entries if e["outcome"] == "valid"]
    quarantine = [e for e in entries if e["outcome"] == "quarantine"]
    reason_counts: dict[str, int] = {}
    for entry in quarantine:
        reason_counts[entry["reason_code"]] = (
            reason_counts.get(entry["reason_code"], 0) + 1
        )

    gates = {
        "frames_contract": all(
            not e["validate_problems"] and e["future_pose_count"] == NUM_FUTURE_POSES
            for e in valid
        ),
        "policy_parity": all(e["policy_parity"] and e["record_hash_stable"] for e in valid),
        "leak_free": all(not e["scan_hits"] for e in valid),
        "full_coverage": len(entries) == EXPECTED_WINDOWS
        and len(valid) + len(quarantine) == len(entries),
        "determinism": all(e["record_hash_stable"] for e in valid),
    }
    gates["all_pass"] = all(gates.values())

    windows_bytes = _write_json(args.out / "windows.json", {**config_echo, "windows": entries})
    summary = {
        **config_echo,
        "candidates": len(entries),
        "valid": len(valid),
        "quarantine": len(quarantine),
        "quarantine_reasons": reason_counts,
        "gates": gates,
        "report_sha256": hashlib.sha256(windows_bytes).hexdigest(),
    }
    _write_json(args.out / "summary.json", summary)

    print(
        f"windows={len(entries)} valid={len(valid)} quarantine={len(quarantine)} "
        f"reasons={reason_counts or '{}'}"
    )
    if gates["all_pass"]:
        print(f"PASS: all Stage 06 gates ok -> {args.out / 'summary.json'}")
    else:
        print(f"FAIL: gates={gates}")
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
