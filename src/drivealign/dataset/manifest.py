"""Stage 07 manifests: 1F/4F anchor-policy table + record-level distributions.

Purpose:
    Derive the two Step 3 deliverables on top of the frozen dataset assets
    (decisions D3/D4) without touching any record:

    1. ``anchor_policy_manifest.json`` (data asset next to
       ``dataset_manifest.json``): for every valid record, the 1F and 4F
       model requests derived through the Stage 06 serializer, fingerprinted
       by their canonical hashes. The Stage 07 gate "1F/4F manifest anchor
       tokens correspond 1:1 with equal counts" is checked here: both
       policies must cover exactly the same anchor set, share the prompt,
       and the 1F image must be the 4F newest frame.
    2. ``record_distribution_report.{json,md}`` (run artifacts): valid-record
       level location / time_of_day / scene-description keyword
       distributions per split, with deltas against the scene-level Step 1
       baseline (same ``distribution_stats`` methodology). This audits that
       the BAD_GAP quarantines did not shift the split representativeness.
       Report only -- never a filtering or reweighting input (D4).

    Inputs are integrity-checked before use: ``dataset_manifest.json`` and
    the three scene manifests are re-hashed, and the scene fingerprints must
    match the dataset manifest binding. Every shard line is re-verified
    against the index (placement + ``sha256(line) == record_hash``) while
    the 1F/4F requests are derived.

Example launch command (full; loads ~10-15 GB RAM, run inside tmux):
    ``tmux new-session -d -s s07_manifest \
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
    cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.dataset.manifest \
    2>&1 | tee runs/S07_dataset/manifest.log'``

Smoke (200 records per split, asset written to a scratch directory):
    ``PYTHONPATH=DriveAlign/src python -m drivealign.dataset.manifest \
    --limit-records 200 --out runs/S07_dataset/smoke_out \
    --reports runs/S07_dataset/reports``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nuscenes.nuscenes import NuScenes

from drivealign.contracts.versions import DEFAULT_CONTRACT_VERSION
from drivealign.data.nuscenes_io import load_nuscenes
from drivealign.dataset.split import distribution_stats
from drivealign.records.record import DriveAlignRecord
from drivealign.records.serializer import InputPolicy, serialize

DEFAULT_DATAROOT = "data/nuscenes/trainval"
DEFAULT_VERSION = "v1.0-trainval"
DEFAULT_ASSETS = "data/dataset_v1"
DEFAULT_REPORTS = "runs/S07_dataset/reports"

SPLITS = ("train", "val", "test")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path(DEFAULT_DATAROOT))
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path(DEFAULT_ASSETS),
        help="Directory holding dataset_manifest.json (read).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write anchor_policy_manifest.json (default: --assets).",
    )
    parser.add_argument("--reports", type=Path, default=Path(DEFAULT_REPORTS))
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Smoke mode: derive at most N anchors per split.",
    )
    parser.add_argument(
        "--builder-commit",
        default=None,
        help="Git commit echoed in config; defaults to `git rev-parse HEAD`. "
        "Pass the frozen dataset builder commit when reproducing assets.",
    )
    return parser.parse_args(argv)


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[3]  # .../DriveAlign
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verified_payload(path: Path) -> Tuple[dict, str]:
    """Load a canonical JSON asset; verify sha256; return (payload, fingerprint)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_sha256")
    actual = _sha256(_canonical_json(payload))
    if expected != actual:
        raise ValueError(f"{path}: manifest sha256 mismatch ({expected} != {actual})")
    return payload, expected


def _scene_meta(nusc: NuScenes) -> Dict[str, dict]:
    """scene_token -> {location, time_of_day, description} for every scene."""
    meta: Dict[str, dict] = {}
    for scene in nusc.scene:
        first = nusc.get("sample", scene["first_sample_token"])
        hour = datetime.fromtimestamp(first["timestamp"] / 1e6, tz=timezone.utc).hour
        meta[scene["token"]] = {
            "description": scene.get("description", ""),
            "location": nusc.get("log", scene["log_token"])["location"],
            "time_of_day": "day" if 6 <= hour < 18 else "night",
        }
    return meta


def _derive_anchor(
    record: DriveAlignRecord, token: str
) -> Tuple[Optional[dict], Optional[str]]:
    """Derive the 1F/4F fingerprints of one record; None + problem on parity failure."""
    req1 = serialize(record.model_inputs, InputPolicy.ONE_FRAME)
    req4 = serialize(record.model_inputs, InputPolicy.FOUR_FRAME)
    if req1.prompt != req4.prompt:
        return None, "1F/4F prompts differ"
    if not (len(req1.image_relpaths) == 1 and len(req4.image_relpaths) == 4):
        return None, "unexpected image counts"
    if req4.image_relpaths[-1] != req1.image_relpaths[0]:
        return None, "1F image is not the 4F newest frame"
    if req1.frame_tokens[-1] != req4.frame_tokens[-1] or req1.frame_tokens[-1] != token:
        return None, "anchor frame token mismatch"
    if req1.ego_speed_mps != req4.ego_speed_mps:
        return None, "ego speed mismatch between policies"
    return (
        {
            "anchor_image_relpath": req1.image_relpaths[0],
            "request_hash_1f": req1.canonical_hash(),
            "request_hash_4f": req4.canonical_hash(),
        },
        None,
    )


def build_anchor_policy_manifest(
    dataset_fingerprint: str,
    scene_fingerprints: Dict[str, str],
    anchors: Dict[str, dict],
    config: dict,
) -> dict:
    payload = {
        "manifest_type": "anchor_policy_manifest",
        "contract_version": DEFAULT_CONTRACT_VERSION,
        "config": config,
        "dataset_manifest_sha256": dataset_fingerprint,
        "scene_manifest_sha256": scene_fingerprints,
        "counts": {
            split: sum(1 for a in anchors.values() if a["split"] == split)
            for split in SPLITS
        },
        "anchors": anchors,
    }
    payload["counts"]["total"] = sum(payload["counts"].values())
    digest = _sha256(_canonical_json(payload))
    return {**payload, "manifest_sha256": digest}


def _keyword_counts(texts: List[str]) -> Counter:
    """Description keyword counts (mirrors split.py methodology)."""
    import re

    stopwords = frozenset(
        """a an and are as at be by for from has have in into is it its of on
        or that the their there this to with with-area without""".split()
    )
    counts: Counter = Counter()
    for text in texts:
        counts.update(
            w for w in re.findall(r"[a-z][a-z'-]*", text.lower()) if w not in stopwords
        )
    return counts


def record_distribution(
    anchors: Dict[str, dict], scene_meta: Dict[str, dict]
) -> Dict[str, dict]:
    """Valid-record level distribution per split (location/tod/keywords)."""
    per_split: Dict[str, dict] = {}
    for split in SPLITS:
        tokens = [t for t, a in anchors.items() if a["split"] == split]
        locations: Counter = Counter()
        tods: Counter = Counter()
        descriptions: List[str] = []
        for token in tokens:
            meta = scene_meta[anchors[token]["scene_token"]]
            locations[meta["location"]] += 1
            tods[meta["time_of_day"]] += 1
            descriptions.append(meta["description"])
        n = len(tokens)
        keywords = _keyword_counts(descriptions)
        per_split[split] = {
            "record_count": n,
            "locations": dict(locations),
            "time_of_day": dict(tods),
            "keywords": [
                {"keyword": w, "count": c, "per_1k_records": round(c / n * 1000, 1)}
                for w, c in keywords.most_common(20)
            ],
        }
    return per_split


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 1) if total else 0.0


def _markdown_report(config: dict, scene_stats: dict, record_stats: dict) -> str:
    lines = ["# Stage 07 Step 3: record-level distribution report", ""]
    lines += ["## Config", ""]
    for key in sorted(config):
        lines.append(f"- {key}: `{config[key]}`")
    lines += [
        "",
        "Baseline = scene-level Step 1 stats (same methodology); deltas in "
        "percentage points measure the BAD_GAP quarantine effect on valid "
        "records. Report only, never a filtering input (D4).",
    ]
    for field, title in (("time_of_day", "Time of day"), ("locations", "Location")):
        lines += [
            "",
            f"## {title}: scene% vs record%",
            "",
            "| category | " + " | ".join(
                f"{s} scene% / rec% / Δpp" for s in SPLITS
            ) + " |",
            "|---|" + "---|" * len(SPLITS),
        ]
        categories = sorted(scene_stats["per_split"][SPLITS[0]][field].keys())
        for category in categories:
            cells = []
            for split in SPLITS:
                s = scene_stats["per_split"][split]
                r = record_stats[split]
                s_pct = _pct(s[field].get(category, 0), s["scene_count"])
                r_pct = _pct(r[field].get(category, 0), r["record_count"])
                cells.append(f"{s_pct} / {r_pct} / {round(r_pct - s_pct, 1)}")
            lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "## Record-level keywords (top 20, per 1k records)", ""]
    lines += ["| keyword | " + " | ".join(f"{s} /1k" for s in SPLITS) + " |"]
    lines += ["|---|" + "---|" * len(SPLITS)]
    union = sorted(
        {k["keyword"] for split in SPLITS for k in record_stats[split]["keywords"]}
    )
    for keyword in union:
        cells = []
        for split in SPLITS:
            match = next(
                (
                    k["per_1k_records"]
                    for k in record_stats[split]["keywords"]
                    if k["keyword"] == keyword
                ),
                0.0,
            )
            cells.append(str(match))
        lines.append(f"| {keyword} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    assets = args.assets.expanduser()
    out_dir = (args.out.expanduser() if args.out is not None else assets)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.reports.expanduser()
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Frozen assets: dataset manifest + scene manifests, all re-hashed.
    dataset_payload, dataset_fingerprint = _verified_payload(
        assets / "dataset_manifest.json"
    )
    scene_fingerprints: Dict[str, str] = {}
    scene_tokens: Dict[str, List[str]] = {}
    for split in SPLITS:
        path = assets / f"{split}_scene_manifest.json"
        payload, fingerprint = _verified_payload(path)
        if fingerprint != dataset_payload["scene_manifest_sha256"][split]:
            raise ValueError(f"{path}: fingerprint differs from dataset_manifest binding")
        scene_tokens[split] = payload["scene_tokens"]
        scene_fingerprints[split] = fingerprint

    config = {
        "builder_commit": args.builder_commit or _git_commit(),
        "contract_version": DEFAULT_CONTRACT_VERSION,
        "dataroot": str(args.dataroot.expanduser().resolve()),
        "dataset_builder_commit": dataset_payload["config"]["builder_commit"],
        "limit_records": args.limit_records,
        "mode": "smoke" if args.limit_records is not None else "full",
        "nuscenes_version": args.version,
        "policy_hashes": "request.canonical_hash()",
    }

    # 2) Load nuScenes once for scene metadata and the scene-level baseline.
    print(f"loading {args.version} from {args.dataroot} ...", flush=True)
    nusc = load_nuscenes(args.dataroot.expanduser(), args.version)
    print("nuscenes loaded", flush=True)
    scene_meta = _scene_meta(nusc)
    scene_stats = distribution_stats(nusc, scene_tokens)

    # 3) Re-read every shard line, verify placement, derive 1F/4F anchors.
    index = dataset_payload["records"]
    anchors: Dict[str, dict] = {}
    parity_problems: List[str] = []
    placement_problems: List[str] = []
    for split in SPLITS:
        n_done = 0
        for shard_rel in dataset_payload["splits"][split]["shards"]:
            with (assets / shard_rel).open("r", encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh):
                    line = raw.rstrip("\n")
                    if not line:
                        placement_problems.append(f"{shard_rel}:{line_no}: blank line")
                        continue
                    record_dict = json.loads(line)
                    token = record_dict["sample_token"]
                    entry = index.get(token)
                    if (
                        entry is None
                        or entry["shard"] != shard_rel
                        or entry["line"] != line_no
                        or _sha256(line) != entry["record_hash"]
                    ):
                        placement_problems.append(f"{shard_rel}:{line_no}: {token}")
                        continue
                    anchor_fields, problem = _derive_anchor(
                        DriveAlignRecord.from_dict(record_dict), token
                    )
                    if problem is not None:
                        parity_problems.append(f"{token}: {problem}")
                        continue
                    anchors[token] = {
                        "line": line_no,
                        "record_hash": entry["record_hash"],
                        "scene_token": record_dict["scene_token"],
                        "shard": shard_rel,
                        "split": split,
                        **anchor_fields,
                    }
                    n_done += 1
                    if args.limit_records is not None and n_done >= args.limit_records:
                        break
                if args.limit_records is not None and n_done >= args.limit_records:
                    break
        print(f"[{split}] anchors={n_done}", flush=True)

    # 4) Anchor-policy manifest (data asset) and distribution report.
    manifest = build_anchor_policy_manifest(
        dataset_fingerprint, scene_fingerprints, anchors, config
    )
    manifest_path = out_dir / "anchor_policy_manifest.json"
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")

    record_stats = record_distribution(anchors, scene_meta)
    report = {
        "manifest_type": "record_distribution_report",
        "config": config,
        "scene_level": scene_stats,
        "record_level": record_stats,
    }
    (reports_dir / "record_distribution_report.json").write_text(
        _canonical_json(report) + "\n", encoding="utf-8"
    )
    (reports_dir / "record_distribution_report.md").write_text(
        _markdown_report(config, scene_stats, record_stats), encoding="utf-8"
    )

    # 5) Gates.
    expected = {
        split: (
            dataset_payload["splits"][split]["record_count"]
            if args.limit_records is None
            else min(args.limit_records, dataset_payload["splits"][split]["record_count"])
        )
        for split in SPLITS
    }
    counts_match = all(
        manifest["counts"][split] == expected[split] for split in SPLITS
    )
    splits_ok = all(
        a["shard"].startswith(a["split"] + "/") for a in anchors.values()
    )
    gates = {
        "assets_intact": True,  # _verified_payload raised otherwise
        "scene_binding_match": scene_fingerprints
        == dataset_payload["scene_manifest_sha256"],
        "anchor_sets_identical": counts_match
        and manifest["counts"]["total"] == len(anchors),
        "policy_parity": not parity_problems,
        "placement_reverified": not placement_problems,
        "split_paths_consistent": splits_ok,
    }
    gates["all_pass"] = all(gates.values())

    for split in SPLITS:
        print(
            f"{split}: anchors={manifest['counts'][split]} (expected {expected[split]})"
        )
    print(f"anchor manifest: {manifest_path}")
    print(f"report: {reports_dir / 'record_distribution_report.md'}")
    if gates["all_pass"]:
        print("PASS: all Step 3 gates ok")
    else:
        print(
            f"FAIL: gates={gates} "
            f"parity={parity_problems[:5]} placement={placement_problems[:5]}"
        )
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
