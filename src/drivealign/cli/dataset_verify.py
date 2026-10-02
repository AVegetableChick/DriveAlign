"""Stage 07 determinism verification: rebuild into scratch + byte diff.

Purpose:
    Close the Stage 07 gate "same raw version, code and configuration ->
    identical manifests and record hashes". The frozen assets under
    ``--assets`` are rebuilt from the frozen scene manifests into a scratch
    directory (never touching the real assets) by re-running the Step 2
    builder and the Step 3 manifest derivation in-process, then every
    produced file is compared byte-by-byte against the frozen copy:

    - shards (all splits), quarantine.jsonl, dataset_manifest.json
    - anchor_policy_manifest.json
    - build_report.{json,md} plus whichever distribution reports the frozen
      build produced (v1: record_distribution_report; v4: gt_distribution_report)

    The builder commit is pinned to the one recorded inside the frozen
    ``dataset_manifest.json`` so a moved git HEAD cannot masquerade as a
    determinism failure; a genuine code/config drift shows up as a byte
    difference. The scene manifests are read from the frozen config's
    ``scene_manifests_dir`` (falling back to the asset dir for v1, which
    stored them inline). Exit code 0 iff every compared file is identical.

Usage (v4 assets; loads ~10-15 GB RAM, run inside tmux):
    ``tmux new-session -d -s s08_verify \
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
    cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.dataset_verify \
    --assets data/dataset_v4 --reports runs/S08_gt_backfill/reports \
    --scratch runs/S08_gt_backfill/verify_scratch \
    --scratch-reports runs/S08_gt_backfill/verify_scratch_reports \
    2>&1 | tee runs/S08_gt_backfill/verify.log'``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from drivealign.dataset import build_dataset, manifest

DEFAULT_ASSETS = "data/dataset_v1"
DEFAULT_SCRATCH = "runs/S07_dataset/verify_out"
DEFAULT_SCRATCH_REPORTS = "runs/S07_dataset/verify_reports"
DEFAULT_REPORTS = "runs/S07_dataset/reports"


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, default=Path(DEFAULT_ASSETS),
        help="Frozen dataset asset directory (reference, never modified).",
    )
    parser.add_argument(
        "--scratch", type=Path, default=Path(DEFAULT_SCRATCH),
        help="Rebuild target directory; wiped at startup.",
    )
    parser.add_argument(
        "--scratch-reports", type=Path, default=Path(DEFAULT_SCRATCH_REPORTS),
        help="Reports of the rebuild run; wiped at startup.",
    )
    parser.add_argument(
        "--reports", type=Path, default=Path(DEFAULT_REPORTS),
        help="Where the frozen reports live (for byte comparison).",
    )
    return parser.parse_args(argv)


def _diff_files(pairs: List[Tuple[Path, Path]]) -> List[str]:
    """Return the reference-relative names of byte-differing file pairs."""
    differing = []
    for reference, candidate in pairs:
        if not candidate.is_file():
            differing.append(f"{reference.name}: MISSING in rebuild")
            continue
        if reference.read_bytes() != candidate.read_bytes():
            differing.append(reference.name)
    return differing


def main(argv=None) -> int:
    args = _parse_args(argv)
    assets = args.assets.expanduser().resolve()
    scratch = args.scratch.expanduser().resolve()
    scratch_reports = args.scratch_reports.expanduser().resolve()
    frozen_reports = args.reports.expanduser().resolve()

    frozen_manifest_path = assets / "dataset_manifest.json"
    if not frozen_manifest_path.is_file():
        print(f"FAIL: no frozen dataset manifest at {frozen_manifest_path}")
        return 1
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen_commit = frozen["config"]["builder_commit"]
    dataroot = frozen["config"]["dataroot"]
    version = frozen["config"]["nuscenes_version"]
    # The frozen manifests live inside the v1 asset dir; later builds record
    # their (external) scene-manifest dir in the frozen config instead.
    scene_manifests_dir = Path(frozen["config"].get("scene_manifests_dir", str(assets)))

    for path in (scratch, scratch_reports):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    # 1) Rebuild records + shards + quarantine + root manifest (Step 2).
    rc_build = build_dataset.main(
        [
            "--dataroot", dataroot,
            "--version", version,
            "--scene-manifests", str(scene_manifests_dir),
            "--out", str(scratch),
            "--reports", str(scratch_reports),
            "--builder-commit", frozen_commit,
        ]
    )
    # 2) Re-derive the 1F/4F anchor policy manifest (Step 3).
    for split in ("train", "val", "test"):
        shutil.copy2(
            scene_manifests_dir / f"{split}_scene_manifest.json",
            scratch / f"{split}_scene_manifest.json",
        )
    rc_manifest = manifest.main(
        [
            "--dataroot", dataroot,
            "--version", version,
            "--assets", str(scratch),
            "--reports", str(scratch_reports),
            "--builder-commit", frozen_commit,
        ]
    )

    # 3) Byte-compare every produced file against the frozen copy.
    #    Report files are compared only when the frozen build produced them
    #    (v1 also wrote record_distribution_report; v4 writes gt_distribution_report).
    pairs = [
        (assets / "dataset_manifest.json", scratch / "dataset_manifest.json"),
        (assets / "quarantine.jsonl", scratch / "quarantine.jsonl"),
        (assets / "anchor_policy_manifest.json", scratch / "anchor_policy_manifest.json"),
    ]
    pairs += [
        (frozen_reports / name, scratch_reports / name)
        for name in (
            "build_report.json",
            "build_report.md",
            "record_distribution_report.json",
            "record_distribution_report.md",
            "gt_distribution_report.json",
            "gt_distribution_report.md",
        )
        if (frozen_reports / name).is_file()
    ]
    for split in ("train", "val", "test"):
        for shard in frozen["splits"][split]["shards"]:
            pairs.append((assets / shard, scratch / shard))
    differing = _diff_files(pairs)

    gates = {
        "rebuild_gates_pass": rc_build == 0 and rc_manifest == 0,
        "byte_identical": not differing,
    }
    gates["all_pass"] = all(gates.values())

    report = {
        "manifest_type": "determinism_report",
        "frozen_builder_commit": frozen_commit,
        "compared_files": len(pairs),
        "differing": differing,
        "gates": gates,
    }
    reports_dir = frozen_reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "determinism_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"compared {len(pairs)} files; differing={differing or 'none'}")
    if gates["all_pass"]:
        print(f"PASS: deterministic rebuild -> {reports_dir / 'determinism_report.json'}")
    else:
        print(f"FAIL: gates={gates}")
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
