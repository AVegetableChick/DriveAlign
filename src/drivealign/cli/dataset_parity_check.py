"""S08 v3<->v4 parity check: anchor sets + request hashes + GT presence.

Purpose:
    Pre-verify (Step 3 smoke) and finally verify (Step 5 full) that the v4
    dataset differs from the frozen v3 dataset ONLY in the GT fields
    (S08 hard gates 1-3):

    1. anchor subset/set equality: every v4 anchor token exists in the
       frozen v3 ``anchor_policy_manifest.json`` under the SAME split
       (smoke mode checks the subset; full mode additionally requires the
       token sets to be identical);
    2. request_hash parity (HARD): the 1F/4F request hashes re-derived from
       the v4 shard records are byte-equal to the frozen v3 manifest values
       for every compared anchor — the serializer stamps the model-face
       contract (v4 inherits the v3 model face verbatim), so any difference
       here means the model face drifted;
    3. GT presence (HARD): every compared v4 valid record carries a
       non-empty ``training_targets.expected_output`` with the five S08
       fields;
    4. record-hash generation change (HARD): every compared token's v4
       record_hash DIFFERS from its v3 value — expected exactly because the
       GT fields enter the canonical payload (this is why contract v4
       exists); an unchanged hash would mean the GT silently did not land.

    Placement integrity (shard / line / sha256(line) == record_hash) is
    re-verified for every v4 line read.

Example launch command (Step 3 smoke parity after the 2-scene smoke build):
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.dataset_parity_check \
    --v4-assets runs/S08_gt_backfill/smoke_out --reports runs/S08_gt_backfill``

Full gate (Step 5, after the full v4 rebuild; loads only JSON + lines):
    ``PYTHONPATH=DriveAlign/src python -m drivealign.cli.dataset_parity_check \
    --v4-assets data/dataset_v4 --mode full --reports runs/S08_gt_backfill``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from drivealign.records.record import DriveAlignRecord
from drivealign.records.serializer import InputPolicy, serialize

DEFAULT_V3_ASSETS = "data/dataset_v1"
DEFAULT_REPORTS = "runs/S08_gt_backfill"
SPLITS = ("train", "val", "test")

#: The five S08 backfill fields every valid v4 record must carry.
EXPECTED_OUTPUT_KEYS = frozenset(
    {"critical_objects", "risk_factors", "reasoning", "yield_required", "speed_action"}
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-assets",
        type=Path,
        default=Path(DEFAULT_V3_ASSETS),
        help="Frozen v3 asset dir holding anchor_policy_manifest.json (read).",
    )
    parser.add_argument(
        "--v4-assets",
        type=Path,
        required=True,
        help="v4 build output dir holding dataset_manifest.json + shards.",
    )
    parser.add_argument("--reports", type=Path, default=Path(DEFAULT_REPORTS))
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Smoke: compare at most N anchors per split.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="full additionally gates on v3/v4 token-set equality.",
    )
    return parser.parse_args(argv)


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verified_payload(path: Path) -> Tuple[dict, str]:
    """Load a canonical JSON manifest; verify sha256; return (payload, fp)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_sha256")
    actual = _sha256(_canonical_json(payload))
    if expected != actual:
        raise ValueError(f"{path}: manifest sha256 mismatch ({expected} != {actual})")
    return payload, expected


@dataclass(frozen=True)
class AnchorComparison:
    """Per-anchor outcome of the v3<->v4 comparison."""

    token: str
    split: str
    found_in_v3: bool
    split_match: bool
    request_hash_1f_match: bool
    request_hash_4f_match: bool
    record_hash_changed: bool
    gt_present: bool


def evaluate_parity(
    comparisons: List[AnchorComparison],
    placement_problems: List[str],
    *,
    mode: str,
    v3_total: int,
) -> Dict[str, bool]:
    """Pure gate evaluation over the per-anchor comparisons."""
    n = len(comparisons)
    gates = {
        "anchor_subset": all(c.found_in_v3 for c in comparisons),
        "split_assignment_match": all(c.split_match for c in comparisons),
        "request_hash_parity": all(
            c.request_hash_1f_match and c.request_hash_4f_match
            for c in comparisons
        ),
        "gt_backfill_present": all(c.gt_present for c in comparisons),
        "record_hash_new_generation": n > 0 and all(
            c.record_hash_changed for c in comparisons
        ),
        "placement_integrity": not placement_problems,
    }
    gates["anchor_set_equality"] = (
        mode == "full" and v3_total == n and gates["anchor_subset"]
    ) or mode == "smoke"
    gates["all_pass"] = all(gates.values())
    return gates


def _request_hashes(record: DriveAlignRecord) -> Tuple[str, str]:
    """Re-derive the 1F/4F canonical request hashes of a v4 record."""
    req1 = serialize(record.model_inputs, InputPolicy.ONE_FRAME)
    req4 = serialize(record.model_inputs, InputPolicy.FOUR_FRAME)
    return req1.canonical_hash(), req4.canonical_hash()


def _gt_present(record_dict: dict) -> bool:
    expected = record_dict["training_targets"]["expected_output"]
    return isinstance(expected, dict) and set(expected) == EXPECTED_OUTPUT_KEYS


def _markdown_report(
    config: dict,
    comparisons: List[AnchorComparison],
    placement_problems: List[str],
    gates: Dict[str, bool],
) -> str:
    n = len(comparisons)
    mismatches_1f = [c.token for c in comparisons if not c.request_hash_1f_match]
    mismatches_4f = [c.token for c in comparisons if not c.request_hash_4f_match]
    missing_gt = [c.token for c in comparisons if not c.gt_present]
    unchanged = [c.token for c in comparisons if not c.record_hash_changed]
    per_split = {
        split: sum(1 for c in comparisons if c.split == split) for split in SPLITS
    }
    lines = [
        "# S08 v3<->v4 parity report",
        "",
        "## Config",
        "",
    ]
    for key in sorted(config):
        lines.append(f"- {key}: `{config[key]}`")
    lines += [
        "",
        "## Comparison",
        "",
        f"- anchors compared: {n} ({per_split})",
        f"- placement problems: {len(placement_problems)}",
        f"- 1F request-hash mismatches: {len(mismatches_1f)}"
        + (f" e.g. {mismatches_1f[:5]}" if mismatches_1f else ""),
        f"- 4F request-hash mismatches: {len(mismatches_4f)}"
        + (f" e.g. {mismatches_4f[:5]}" if mismatches_4f else ""),
        f"- GT missing: {len(missing_gt)}" + (f" e.g. {missing_gt[:5]}" if missing_gt else ""),
        f"- record hash NOT in a new generation: {len(unchanged)}"
        + (f" e.g. {unchanged[:5]}" if unchanged else ""),
        "",
        "## Gates",
        "",
        "| gate | result |",
        "|---|---|",
    ]
    for name, ok in gates.items():
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    v3_dir = args.v3_assets.expanduser()
    v4_dir = args.v4_assets.expanduser()
    reports_dir = args.reports.expanduser()
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Frozen v3 side: anchor manifest (request hashes) + dataset manifest
    #    (record hashes of the v3 generation), both sha256-verified.
    v3_anchor_payload, v3_anchor_fp = _verified_payload(
        v3_dir / "anchor_policy_manifest.json"
    )
    v3_anchors: Dict[str, dict] = v3_anchor_payload["anchors"]
    v3_dataset_payload, _ = _verified_payload(v3_dir / "dataset_manifest.json")
    v3_record_hashes: Dict[str, str] = {
        token: entry["record_hash"]
        for token, entry in v3_dataset_payload["records"].items()
    }

    # 2) v4 side: the build output's dataset manifest, sha256-verified.
    v4_payload, _ = _verified_payload(v4_dir / "dataset_manifest.json")
    v4_index: Dict[str, dict] = v4_payload["records"]

    config = {
        "mode": args.mode,
        "limit_records": args.limit_records,
        "v3_assets": str(v3_dir.resolve()),
        "v3_anchor_manifest_sha256": v3_anchor_fp,
        "v4_assets": str(v4_dir.resolve()),
        "v4_contract_version": v4_payload["contract_version"],
    }

    # 3) Compare every v4 anchor against the frozen v3 manifest.
    comparisons: List[AnchorComparison] = []
    placement_problems: List[str] = []
    per_split_done: Dict[str, int] = {split: 0 for split in SPLITS}
    for token in sorted(v4_index):
        entry = v4_index[token]
        split = entry["split"]
        if args.limit_records is not None and per_split_done[split] >= args.limit_records:
            continue
        shard_path = v4_dir / entry["shard"]
        with shard_path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh):
                if line_no == entry["line"]:
                    break
            else:
                placement_problems.append(f"{token}: line {entry['line']} missing")
                continue
        line = raw.rstrip("\n")
        if _sha256(line) != entry["record_hash"]:
            placement_problems.append(f"{token}: sha256(line) != record_hash")
            continue
        record_dict = json.loads(line)
        if record_dict["sample_token"] != token:
            placement_problems.append(f"{token}: shard line holds {record_dict['sample_token']}")
            continue
        per_split_done[split] += 1

        record = DriveAlignRecord.from_dict(record_dict)
        hash1, hash4 = _request_hashes(record)
        v3 = v3_anchors.get(token)
        v3_record_hash = v3_record_hashes.get(token)
        comparisons.append(
            AnchorComparison(
                token=token,
                split=split,
                found_in_v3=v3 is not None,
                split_match=v3 is not None and v3["split"] == split,
                request_hash_1f_match=v3 is not None
                and v3["request_hash_1f"] == hash1,
                request_hash_4f_match=v3 is not None
                and v3["request_hash_4f"] == hash4,
                record_hash_changed=v3_record_hash is not None
                and v3_record_hash != entry["record_hash"],
                gt_present=_gt_present(record_dict),
            )
        )

    gates = evaluate_parity(
        comparisons,
        placement_problems,
        mode=args.mode,
        v3_total=len(v3_anchors),
    )

    report = {
        "manifest_type": "s08_parity_report",
        "config": config,
        "compared": len(comparisons),
        "per_split_compared": per_split_done,
        "placement_problems": placement_problems[:50],
        "mismatches_1f": [c.token for c in comparisons if not c.request_hash_1f_match][:50],
        "mismatches_4f": [c.token for c in comparisons if not c.request_hash_4f_match][:50],
        "gt_missing": [c.token for c in comparisons if not c.gt_present][:50],
        "hash_unchanged": [c.token for c in comparisons if not c.record_hash_changed][:50],
        "gates": gates,
    }
    (reports_dir / "parity_report.json").write_text(
        _canonical_json(report) + "\n", encoding="utf-8"
    )
    (reports_dir / "parity_report.md").write_text(
        _markdown_report(config, comparisons, placement_problems, gates),
        encoding="utf-8",
    )

    print(
        f"compared {len(comparisons)} anchors "
        f"({ {s: per_split_done[s] for s in SPLITS} }) of {len(v3_anchors)} v3 anchors"
    )
    print(f"report: {reports_dir / 'parity_report.md'}")
    if gates["all_pass"]:
        print("PASS: v3<->v4 parity gates ok")
    else:
        failing = [k for k, ok in gates.items() if not ok]
        print(
            f"FAIL: gates={failing} placement={placement_problems[:5]} "
            f"mismatch_1f={report['mismatches_1f'][:5]} "
            f"mismatch_4f={report['mismatches_4f'][:5]}"
        )
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
