"""S13 v7<->v4 parity check: non-GT fields identical, generation stamps correct.

Purpose:
    Verify that the v7 dataset differs from the frozen v4 dataset ONLY in the
    fields the S13 contract revision is allowed to touch (implementation plan
    §4.4). Unlike :mod:`drivealign.cli.dataset_parity_check` (v3<->v4), the
    request hashes and record hashes are EXPECTED to change generation-wide
    (v7 has its own model face + four-field training targets, decision log
    §2.3), so they are checked as positive generation gates instead of parity:

    Per-anchor checks (full sweep over every valid record, not a sample):
    1. manifest index alignment: v4 and v7 hold the same token set with the
       same split / shard / line placement;
    2. placement integrity: sha256(line) == record_hash on BOTH generations;
    3. model_inputs byte-equal between v4 and v7 (adapter untouched, S08 gate 1
       invariant carried forward);
    4. oracle_only byte-equal between v4 and v7;
    5. metadata equal (record_version / scene_token / sample_token);
    6. provenance drift limited to builder_commit + contract_version, with
       v7 stamped "v7" and the shared fields (nuscenes_version /
       temporal_policy_version) unchanged;
    7. record hash generation change: v7 record_hash differs from the v4
       value for every anchor (unchanged would mean training_targets landed
       byte-identically, i.e. the v7 GT silently did not enter the payload);
    8. request hash generation change: the v7 face manifest's 1F/4F request
       hashes differ from the frozen v6 face manifest values for every anchor
       (equal would mean the v7 model face did not reach the serializer);
    9. GT presence: every v7 expected_output carries exactly the four v7
       contract fields (V7-C1: reasoning retired).

Example launch command (full sweep; reads only JSON/JSONL, minutes):
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.v7_parity_check \
    --reports runs/S13_v7_calibration/full_reports``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from drivealign.cli.dataset_parity_check import EXPECTED_OUTPUT_KEYS

DEFAULT_V4_ASSETS = "data/dataset_v4"
DEFAULT_V7_ASSETS = "data/dataset_v7"
DEFAULT_V7_FACE = "runs/S13_v7_calibration/face_v7/anchor_policy_manifest.json"
DEFAULT_V6_FACE = "data/face_manifests/v6/anchor_policy_manifest.json"
DEFAULT_REPORTS = "runs/S13_v7_calibration/full_reports"

SPLITS = ("train", "val", "test")

#: provenance 的预期代际漂移字段（其余字段必须逐值相同）。
EXPECTED_PROVENANCE_DRIFT = frozenset({"builder_commit", "contract_version"})


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-assets", type=Path, default=Path(DEFAULT_V4_ASSETS))
    parser.add_argument("--v7-assets", type=Path, default=Path(DEFAULT_V7_ASSETS))
    parser.add_argument("--face-v7-manifest", type=Path, default=Path(DEFAULT_V7_FACE))
    parser.add_argument("--face-v6-manifest", type=Path, default=Path(DEFAULT_V6_FACE))
    parser.add_argument("--reports", type=Path, default=Path(DEFAULT_REPORTS))
    return parser.parse_args(argv)


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verified_payload(path: Path) -> Tuple[dict, str]:
    """加载 canonical JSON manifest 并校验内嵌 sha256，返回 (payload, fingerprint)。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_sha256")
    actual = _sha256(_canonical_json(payload))
    if expected != actual:
        raise ValueError(f"{path}: manifest sha256 mismatch ({expected} != {actual})")
    return payload, expected


def _read_shard_line(assets: Path, shard_rel: str, line_no: int) -> str:
    """读取指定 shard 的指定行（校验存在性由调用方负责）。"""
    with (assets / shard_rel).open("r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh):
            if i == line_no:
                return raw.rstrip("\n")
    raise FileNotFoundError(f"{assets / shard_rel}:{line_no}: line missing")


def diff_tree(a, b, path: str = "") -> List[str]:
    """递归对比两棵 JSON 树，返回差异路径列表（标量不等 / 结构不等 / 缺键）。"""
    if isinstance(a, dict) and isinstance(b, dict):
        problems: List[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a:
                problems.append(f"{path}.{key}: missing in v4")
            elif key not in b:
                problems.append(f"{path}.{key}: missing in v7")
            else:
                problems.extend(diff_tree(a[key], b[key], f"{path}.{key}"))
        return problems
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: list length {len(a)} != {len(b)}"]
        problems = []
        for i, (x, y) in enumerate(zip(a, b)):
            problems.extend(diff_tree(x, y, f"{path}[{i}]"))
        return problems
    # 标量（含类型不一致的情形）：直接比较
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@dataclass
class RecordComparison:
    """单条 anchor 的 v4<->v7 对比结果（True 表示该项通过）。"""

    token: str
    split: str
    model_inputs_identical: bool
    oracle_only_identical: bool
    metadata_identical: bool
    provenance_ok: bool
    record_hash_changed: bool
    request_hash_changed: Optional[bool]  # None = face manifest 未参与对比
    gt_present: bool
    problems: List[str] = field(default_factory=list)


def compare_record(
    token: str,
    split: str,
    v4_dict: dict,
    v7_dict: dict,
    v4_record_hash: str,
    v7_record_hash: str,
    v6_face: Optional[dict],
    v7_face: Optional[dict],
) -> RecordComparison:
    """对一条记录执行 §docstring 的 2-6/9 项检查（hash 门由调用方传入双代际值）。"""
    problems: List[str] = []

    model_ok = v4_dict["model_inputs"] == v7_dict["model_inputs"]
    if not model_ok:
        problems.append(
            "model_inputs: " + "; ".join(
                diff_tree(v4_dict["model_inputs"], v7_dict["model_inputs"])[:3]
            )
        )

    oracle_ok = v4_dict["oracle_only"] == v7_dict["oracle_only"]
    if not oracle_ok:
        problems.append(
            "oracle_only: " + "; ".join(
                diff_tree(v4_dict["oracle_only"], v7_dict["oracle_only"])[:3]
            )
        )

    meta_ok = (
        v4_dict["record_version"] == v7_dict["record_version"]
        and v4_dict["scene_token"] == v7_dict["scene_token"]
        and v4_dict["sample_token"] == v7_dict["sample_token"]
    )
    if not meta_ok:
        problems.append("metadata: record_version/scene_token/sample_token differ")

    # provenance：除白名单字段外逐值相同；v7 戳必须为 "v7"
    prov4, prov7 = v4_dict["provenance"], v7_dict["provenance"]
    prov_drift = {
        k for k in set(prov4) | set(prov7)
        if prov4.get(k) != prov7.get(k)
    }
    prov_ok = (
        prov_drift <= EXPECTED_PROVENANCE_DRIFT
        and prov7.get("contract_version") == "v7"
    )
    if not prov_ok:
        problems.append(
            f"provenance: drift={sorted(prov_drift)} v7_stamp={prov7.get('contract_version')!r}"
        )

    hash_changed = v4_record_hash != v7_record_hash

    if v6_face is not None and v7_face is not None:
        a6, a7 = v6_face["anchors"], v7_face["anchors"]
        req_changed = (
            token in a6
            and token in a7
            and a6[token]["request_hash_1f"] != a7[token]["request_hash_1f"]
            and a6[token]["request_hash_4f"] != a7[token]["request_hash_4f"]
        )
    else:
        req_changed = None

    expected = v7_dict["training_targets"]["expected_output"]
    gt_ok = isinstance(expected, dict) and set(expected) == EXPECTED_OUTPUT_KEYS
    if not gt_ok:
        keys = sorted(expected) if isinstance(expected, dict) else expected
        problems.append(f"gt: expected_output keys={keys!r}")

    return RecordComparison(
        token=token,
        split=split,
        model_inputs_identical=model_ok,
        oracle_only_identical=oracle_ok,
        metadata_identical=meta_ok,
        provenance_ok=prov_ok,
        record_hash_changed=hash_changed,
        request_hash_changed=req_changed,
        gt_present=gt_ok,
        problems=problems,
    )


def _markdown_report(config: dict, comparisons: List[RecordComparison], gates: dict,
                     per_split: Dict[str, int]) -> str:
    n = len(comparisons)
    fail = lambda pred: [c.token for c in comparisons if not pred(c)]  # noqa: E731
    lines = [
        "# S13 v7<->v4 parity report",
        "",
        "## Config",
        "",
    ]
    for key in sorted(config):
        lines.append(f"- {key}: `{config[key]}`")
    lines += [
        "",
        "## Sweep",
        "",
        f"- anchors compared: {n} ({per_split})",
        f"- model_inputs diffs: {len(fail(lambda c: c.model_inputs_identical))}",
        f"- oracle_only diffs: {len(fail(lambda c: c.oracle_only_identical))}",
        f"- metadata diffs: {len(fail(lambda c: c.metadata_identical))}",
        f"- provenance violations: {len(fail(lambda c: c.provenance_ok))}",
        f"- record hash NOT in new generation: {len(fail(lambda c: c.record_hash_changed))}",
        f"- request hash NOT in new generation (v6 face): "
        f"{len(fail(lambda c: c.request_hash_changed is not False))}",
        f"- v7 GT missing/malformed: {len(fail(lambda c: c.gt_present))}",
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
    v4_dir, v7_dir = args.v4_assets.expanduser(), args.v7_assets.expanduser()
    reports_dir = args.reports.expanduser()
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) 双方 dataset manifest（校验内嵌 sha256）+ index 对齐检查。
    v4_payload, v4_fp = _verified_payload(v4_dir / "dataset_manifest.json")
    v7_payload, v7_fp = _verified_payload(v7_dir / "dataset_manifest.json")
    v4_index: Dict[str, dict] = v4_payload["records"]
    v7_index: Dict[str, dict] = v7_payload["records"]

    index_problems: List[str] = []
    if set(v4_index) != set(v7_index):
        only4 = sorted(set(v4_index) - set(v7_index))[:5]
        only7 = sorted(set(v7_index) - set(v4_index))[:5]
        index_problems.append(f"token sets differ: only_v4={only4} only_v7={only7}")
    for token, e7 in v7_index.items():
        e4 = v4_index.get(token)
        if e4 is None:
            continue
        if (e4["split"], e4["shard"], e4["line"]) != (e7["split"], e7["shard"], e7["line"]):
            index_problems.append(f"{token}: placement differs")
            if len(index_problems) >= 20:
                break

    # 2) face manifests（v6 冻结 / v7 新派生）：request hash 代际检查的两侧。
    v7_face_payload, v7_face_fp = _verified_payload(args.face_v7_manifest)
    v6_face_payload, v6_face_fp = _verified_payload(args.face_v6_manifest)
    face_problems: List[str] = []
    a6, a7 = v6_face_payload["anchors"], v7_face_payload["anchors"]
    if set(a6) != set(a7):
        face_problems.append(
            f"anchor sets differ: v6={len(a6)} v7={len(a7)} "
            f"only_v6={sorted(set(a6) - set(a7))[:3]} only_v7={sorted(set(a7) - set(a6))[:3]}"
        )

    config = {
        "face_v6_manifest": str(args.face_v6_manifest),
        "face_v6_sha256": v6_face_fp,
        "face_v7_manifest": str(args.face_v7_manifest),
        "face_v7_sha256": v7_face_fp,
        "v4_assets": str(v4_dir.resolve()),
        "v4_contract_version": v4_payload["contract_version"],
        "v4_manifest_sha256": v4_fp,
        "v7_assets": str(v7_dir.resolve()),
        "v7_contract_version": v7_payload["contract_version"],
        "v7_manifest_sha256": v7_fp,
    }

    # 3) 全量逐行对比（index 对齐保证行序一致性，直接按 placement 取行）。
    comparisons: List[RecordComparison] = []
    placement_problems: List[str] = []
    per_split: Dict[str, int] = {s: 0 for s in SPLITS}
    tokens = sorted(v7_index)
    for done, token in enumerate(tokens, 1):
        e7 = v7_index[token]
        e4 = v4_index.get(token)
        split = e7["split"]
        if e4 is None or (e4["split"], e4["shard"], e4["line"]) != (split, e7["shard"], e7["line"]):
            continue  # index 问题已记录，此处跳过避免误报
        line7 = _read_shard_line(v7_dir, e7["shard"], e7["line"])
        line4 = _read_shard_line(v4_dir, e4["shard"], e4["line"])
        if _sha256(line7) != e7["record_hash"]:
            placement_problems.append(f"v7 {token}: sha256(line) != record_hash")
            continue
        if _sha256(line4) != e4["record_hash"]:
            placement_problems.append(f"v4 {token}: sha256(line) != record_hash")
            continue
        comparisons.append(
            compare_record(
                token, split, json.loads(line4), json.loads(line7),
                e4["record_hash"], e7["record_hash"], v6_face_payload, v7_face_payload,
            )
        )
        per_split[split] += 1
        if done % 5000 == 0:
            print(f"[progress] {done}/{len(tokens)} anchors compared", flush=True)

    # 4) gate 汇总。
    gates = {
        "manifest_index_alignment": not index_problems,
        "face_anchor_sets_aligned": not face_problems,
        "placement_integrity": not placement_problems,
        "model_inputs_identical": all(c.model_inputs_identical for c in comparisons),
        "oracle_only_identical": all(c.oracle_only_identical for c in comparisons),
        "metadata_identical": all(c.metadata_identical for c in comparisons),
        "provenance_generation_drift": all(c.provenance_ok for c in comparisons),
        "record_hash_generation_change": bool(comparisons) and all(
            c.record_hash_changed for c in comparisons
        ),
        "request_hash_generation_change": bool(comparisons) and all(
            c.request_hash_changed is True for c in comparisons
        ),
        "gt_present_v7": all(c.gt_present for c in comparisons),
    }
    gates["all_pass"] = all(gates.values())

    report = {
        "manifest_type": "s13_v7_parity_report",
        "config": config,
        "compared": len(comparisons),
        "per_split_compared": per_split,
        "index_problems": index_problems[:20],
        "face_problems": face_problems[:20],
        "placement_problems": placement_problems[:50],
        "problem_samples": {
            c.token: c.problems
            for c in comparisons
            if c.problems
        } | {},
        "gates": gates,
    }
    # 只保留前 50 条问题样例，报告体积可控
    report["problem_samples"] = dict(
        list(report["problem_samples"].items())[:50]
    )
    (reports_dir / "v7_parity_report.json").write_text(
        _canonical_json(report) + "\n", encoding="utf-8"
    )
    (reports_dir / "v7_parity_report.md").write_text(
        _markdown_report(config, comparisons, gates, per_split), encoding="utf-8"
    )

    print(
        f"compared {len(comparisons)} anchors ({per_split}) of {len(v7_index)} v7 records"
    )
    print(f"report: {reports_dir / 'v7_parity_report.md'}")
    if gates["all_pass"]:
        print("PASS: v7<->v4 parity gates ok (generation drift as designed)")
    else:
        failing = [k for k, ok in gates.items() if not ok]
        sample = next((c for c in comparisons if c.problems), None)
        print(
            f"FAIL: gates={failing} index={index_problems[:3]} "
            f"placement={placement_problems[:3]} "
            f"sample_problems={sample.problems if sample else 'n/a'}"
        )
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
