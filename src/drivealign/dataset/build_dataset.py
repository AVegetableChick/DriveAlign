"""Stage 07/08 dataset build: frozen scene manifests -> sharded records.

Purpose:
    Build the v4 dataset from the three frozen scene manifests produced by
    :mod:`drivealign.dataset.split` (decisions D3/D4). For every scene, in
    manifest token order, walk its keyframes oldest -> newest and take every
    sample whose walk index is >= 3 (exactly ``sample.prev`` depth >= 3 in
    the linear keyframe chain) as one 4F-window candidate anchor. Each
    candidate is built through the Stage 06 adapter (``build_record``, the
    only construction path): it either yields a valid record or a stable
    quarantine reason code. Scene end truncates future poses naturally and
    is NOT a quarantine condition (Stage 06 behavior, unchanged).

    S08 v4 backfill (this stage's addition): every VALID record gets
    ``training_targets.expected_output`` filled through the pure S08 GT
    pipeline (:mod:`drivealign.gt.backfill` — observability gating, pool
    top-8, motion_state, risk_factors, speed_action, yield_required,
    deterministic reasoning render, v4 output-schema + zero-contradiction
    gates, fail-fast). The GT inputs come from the SAME record's frozen
    derivations (``model_inputs.ego_speed_mps`` and
    ``oracle_only.future_ego_poses``) plus the anchor-frame evidence the gt
    package reads from nuScenes. Quarantined records keep ``None`` (no GT
    for quarantined anchors). ``adapter.py`` and the request-hash path are
    untouched: model_inputs are byte-identical to v3 (S08 gate 1); the
    record hash moves to a new generation exactly because the GT fields are
    added (why contract v4 exists).

    Outputs (data assets under ``--out``, decision D3):
    - ``<split>/shard-NNNN.jsonl``: one canonical record JSON per line
      (sort_keys compact, so sha256(line) == ``record.canonical_hash()``),
      1000 records per shard. Records are ordered by (scene manifest order,
      scene walk order) within each split, so the shard file of a record is
      ``global_index // 1000`` and its 0-based line number inside that file
      is ``global_index % 1000``.
    - ``quarantine.jsonl``: one compact JSON line per quarantined candidate
      with split / scene / anchor index / sample token / reason code /
      detail, in deterministic build order.
    - ``dataset_manifest.json``: sample_token -> {split, shard, line,
      record_hash} index over RELATIVE paths only, plus the scene-manifest
      sha256 fingerprints and the manifest's own sha256 (canonical JSON over
      the payload without the hash field).

    Run artifacts (``--reports``): ``build_report.{json,md}`` with per-split
    counts, the quarantine reason distribution and the Step 2 gates; plus
    ``gt_distribution_report.{json,md}`` with the S08 GT label distributions
    (speed_action / yield_required / risk_factors / critical_objects /
    reasoning length, per split + total), counted inline while shards are
    written (:mod:`drivealign.dataset.gt_distribution`).

    Gates enforced here:
    - scene manifests intact (embedded sha256 recomputed) and pairwise disjoint
    - full coverage: valid + quarantine == candidates for every split
    - quarantine reason codes drawn from the frozen 8-code taxonomy, and no
      quarantined token leaking into the record index
    - GT backfill: 100% of valid records written with a non-empty
      ``expected_output`` (fail-fast: any rule/gate failure aborts the build
      naming the anchor)
    - shard roundtrip: re-reading every shard line reproduces the manifest
      index exactly (token, line, record hash, roundtripped canonical hash)

Example launch command (full build; loads ~10-15 GB RAM, run inside tmux):
    ``tmux new-session -d -s s08_build \
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
    cd /root/autodl-tmp/drivealign_workspace && \
    set -o pipefail && PYTHONPATH=DriveAlign/src python -m drivealign.dataset.build_dataset \
    --out /root/autodl-tmp/datasets/drivealign_dataset/v4 \
    2>&1 | tee runs/S08_gt_backfill/build_dataset.log'``

Smoke (2 scenes per split into a scratch directory, never the real assets):
    ``PYTHONPATH=DriveAlign/src python -m drivealign.dataset.build_dataset \
    --max-scenes 2 --out runs/S08_gt_backfill/smoke_out \
    --reports runs/S08_gt_backfill/smoke_reports``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, IO, List, Optional, Tuple

from nuscenes.nuscenes import NuScenes

from drivealign.contracts.versions import DEFAULT_CONTRACT_VERSION, contract_file
from drivealign.data.nuscenes_io import (
    BAD_GAP,
    CROSS_SCENE,
    INSUFFICIENT_HISTORY,
    MISSING_IMAGE,
    NO_CAM_FRONT,
    NON_INCREASING_TIME,
    load_nuscenes,
)
from drivealign.dataset.gt_distribution import GtDistribution, render_gt_distribution_markdown
from drivealign.gt.backfill import (
    build_expected_output,
    collect_anchor_evidence,
)
from drivealign.gt.config import RuleConfig, load_phrase_map, load_rule_config, load_templates
from drivealign.records.adapter import (
    MISSING_CALIBRATION,
    NUMERIC_ANOMALY,
    build_record,
)
from drivealign.records.record import (
    DriveAlignRecord,
    TrainingTargets,
    validate_record,
)

DEFAULT_DATAROOT = "data/nuscenes/trainval"
DEFAULT_VERSION = "v1.0-trainval"
DEFAULT_ASSETS = "data/dataset_v1"
DEFAULT_REPORTS = "runs/S08_gt_backfill/reports"

SHARD_SIZE = 1000  # records per shard file (decision D3)
MIN_PREVS = 3  # 4F window: anchor + 3 keyframe prevs
SPLITS = ("train", "val", "test")

# Frozen 8-code quarantine taxonomy (6 chain codes + 2 adapter codes).
QUARANTINE_CODES = frozenset(
    {
        INSUFFICIENT_HISTORY,
        CROSS_SCENE,
        NON_INCREASING_TIME,
        MISSING_IMAGE,
        BAD_GAP,
        NO_CAM_FRONT,
        MISSING_CALIBRATION,
        NUMERIC_ANOMALY,
    }
)

#: The frozen GT-side contract asset carrying every S08 threshold.
GT_RULE_CONFIG_FILE = "gt_rule_config.json"


@dataclass(frozen=True)
class GtAssets:
    """Frozen S08 rule assets, loaded once per build (deterministic inputs)."""

    rule_config: RuleConfig
    templates: Dict
    phrase_map: Dict

    @classmethod
    def load(cls) -> "GtAssets":
        return cls(
            rule_config=load_rule_config(),
            templates=load_templates(),
            phrase_map=load_phrase_map(),
        )

    @property
    def rule_config_sha256(self) -> str:
        return hashlib.sha256(
            contract_file("v4", GT_RULE_CONFIG_FILE).read_bytes()
        ).hexdigest()


def _backfill_record(
    nusc: NuScenes, record: DriveAlignRecord, gt: GtAssets
) -> DriveAlignRecord:
    """Fill ``training_targets.expected_output`` for one valid record (fail-fast).

    The GT derivation consumes the record's OWN frozen values
    (``model_inputs.ego_speed_mps``, ``oracle_only.future_ego_poses``) so the
    labels always match what evaluation will replay; anchor-frame evidence
    comes from the gt package's nuScenes reader. Any rule or gate failure
    raises ValueError naming the anchor (S08 fail-fast gate plan) — the
    build aborts instead of writing a half-filled dataset.
    """
    evidence = collect_anchor_evidence(nusc, record.sample_token)
    expected_output = build_expected_output(
        evidence,
        ego_speed_mps=record.model_inputs.ego_speed_mps,
        future_ego_poses=[
            (p.x, p.y, p.timestamp_us) for p in record.oracle_only.future_ego_poses
        ],
        config=gt.rule_config,
        templates=gt.templates,
        phrase_map=gt.phrase_map,
    )
    filled = replace(
        record,
        training_targets=TrainingTargets(
            expected_output=expected_output, language_reference=None
        ),
    )
    problems = validate_record(filled)
    if problems:
        raise ValueError(
            f"backfilled record for {record.sample_token} fails validation "
            f"(derivation bug): {'; '.join(problems)}"
        )
    return filled


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path(DEFAULT_DATAROOT))
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--scene-manifests",
        type=Path,
        default=Path(DEFAULT_ASSETS),
        help="Directory holding the frozen {split}_scene_manifest.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_ASSETS),
        help="Output root for shards, quarantine.jsonl and dataset_manifest.json.",
    )
    parser.add_argument("--reports", type=Path, default=Path(DEFAULT_REPORTS))
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Smoke mode: process only the first N scenes of every split.",
    )
    parser.add_argument(
        "--builder-commit",
        default=None,
        help="Git commit recorded in provenance; defaults to `git rev-parse HEAD`.",
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


def load_scene_manifest(path: Path) -> Tuple[List[str], str]:
    """Read a frozen scene manifest; verify sha256; return (tokens, fingerprint)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("manifest_sha256")
    actual = _sha256(_canonical_json(payload))
    if expected != actual:
        raise ValueError(f"{path}: manifest sha256 mismatch ({expected} != {actual})")
    if payload["split"] != path.name.split("_")[0]:
        raise ValueError(f"{path}: split field {payload['split']!r} does not match filename")
    return payload["scene_tokens"], expected


def scene_keyframes(nusc: NuScenes, scene_token: str) -> List[str]:
    """Ordered keyframe sample tokens of one scene (oldest -> newest)."""
    tokens = []
    current = nusc.get("scene", scene_token)["first_sample_token"]
    while current:
        tokens.append(current)
        current = nusc.get("sample", current)["next"]
    return tokens


class _ShardWriter:
    """Append canonical record lines to ``<split>/shard-NNNN.jsonl`` files."""

    def __init__(self, out_root: Path, split: str, shard_size: int):
        self.split = split
        self.shard_size = shard_size
        self.split_dir = out_root / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.shards: List[str] = []  # relpaths, e.g. "train/shard-0000.jsonl"
        self._handle: Optional[IO[str]] = None
        self._line_in_shard = 0

    def write(self, record_dict: dict) -> Tuple[str, int]:
        """Write one record; return (relative shard path, 0-based line number)."""
        if self._handle is None or self._line_in_shard == self.shard_size:
            if self._handle is not None:
                self._handle.close()
            name = f"shard-{len(self.shards):04d}.jsonl"
            self._handle = (self.split_dir / name).open("w", encoding="utf-8")
            self.shards.append(f"{self.split}/{name}")
            self._line_in_shard = 0
        self._handle.write(
            json.dumps(record_dict, sort_keys=True, separators=(",", ":")) + "\n"
        )
        shard = self.shards[-1]
        line_no = self._line_in_shard
        self._line_in_shard += 1
        return shard, line_no

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def build_split(
    nusc: NuScenes,
    split: str,
    scene_tokens: List[str],
    out_root: Path,
    builder_commit: str,
    gt: Optional[GtAssets] = None,
    gt_stats: Optional[GtDistribution] = None,
) -> Tuple[dict, List[dict], Dict[str, dict]]:
    """Build every 4F candidate of one split.

    ``gt`` carries the frozen S08 rule assets; when given, every valid
    record is GT-backfilled (v4 semantics; quarantined records keep None).
    ``gt_stats`` (optional) accumulates the GT label distributions of the
    backfilled records inline — no extra pass over the shards. Returns
    (stats, quarantine_entries, record_index) where record_index maps
    sample_token -> {split, shard, line, record_hash} for the valid records.
    """
    writer = _ShardWriter(out_root, split, SHARD_SIZE)
    quarantine: List[dict] = []
    index: Dict[str, dict] = {}
    reasons: Counter = Counter()
    n_keyframes = 0
    n_valid = 0
    n_backfilled = 0
    n_candidates = 0

    for s_pos, scene_token in enumerate(scene_tokens, start=1):
        scene_name = nusc.get("scene", scene_token)["name"]
        tokens = scene_keyframes(nusc, scene_token)
        n_keyframes += len(tokens)
        for anchor_index, token in enumerate(tokens):
            if anchor_index < MIN_PREVS:
                continue  # cannot form a 4F window; not a candidate
            n_candidates += 1
            result = build_record(nusc, token, builder_commit)
            if result.record is None:
                reasons[result.reason_code] += 1
                quarantine.append(
                    {
                        "anchor_index": anchor_index,
                        "detail": result.detail,
                        "reason_code": result.reason_code,
                        "scene_name": scene_name,
                        "scene_token": scene_token,
                        "sample_token": token,
                        "split": split,
                    }
                )
            else:
                record = result.record
                if gt is not None:
                    record = _backfill_record(nusc, record, gt)
                    if record.training_targets.expected_output is not None:
                        n_backfilled += 1
                        if gt_stats is not None:
                            gt_stats.update(record.training_targets.expected_output)
                shard, line_no = writer.write(record.to_dict())
                index[token] = {
                    "line": line_no,
                    "record_hash": record.canonical_hash(),
                    "shard": shard,
                    "split": split,
                }
                n_valid += 1
        if s_pos % 50 == 0 or s_pos == len(scene_tokens):
            print(
                f"[{split}] scene {s_pos}/{len(scene_tokens)} "
                f"candidates={n_candidates} valid={n_valid} "
                f"backfilled={n_backfilled} quarantine={len(quarantine)}",
                flush=True,
            )
    writer.close()

    stats = {
        "scene_count": len(scene_tokens),
        "keyframe_count": n_keyframes,
        "candidate_count": n_candidates,
        "valid_count": n_valid,
        "backfilled_count": n_backfilled,
        "quarantine_count": len(quarantine),
        "shard_count": len(writer.shards),
        "shards": list(writer.shards),
        "reasons": dict(sorted(reasons.items())),
    }
    return stats, quarantine, index


def verify_shard_roundtrip(
    out_root: Path,
    per_split: Dict[str, dict],
    index: Dict[str, dict],
) -> List[str]:
    """Re-read every shard line and check it against the manifest index.

    Per line: sha256(line) must equal the indexed record_hash (the stored
    line is the canonical payload), and the record must roundtrip through
    DriveAlignRecord.from_dict with the same canonical hash. Token placement
    (split / shard / line) and per-split token sets must match the index.
    """
    problems: List[str] = []
    index_by_split: Dict[str, Dict[str, dict]] = {split: {} for split in SPLITS}
    for token, entry in index.items():
        index_by_split[entry["split"]][token] = entry

    for split, info in per_split.items():
        seen: set = set()
        n_lines = 0
        for shard_rel in info["shards"]:
            path = out_root / shard_rel
            with path.open("r", encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh):
                    line = raw.rstrip("\n")
                    if not line:
                        problems.append(f"{shard_rel}:{line_no}: blank line")
                        continue
                    n_lines += 1
                    try:
                        record_dict = json.loads(line)
                        token = record_dict["sample_token"]
                    except (json.JSONDecodeError, KeyError) as exc:
                        problems.append(f"{shard_rel}:{line_no}: unparseable ({exc})")
                        continue
                    entry = index_by_split[split].get(token)
                    if entry is None:
                        problems.append(
                            f"{shard_rel}:{line_no}: token {token} not in index"
                        )
                        continue
                    if entry["shard"] != shard_rel or entry["line"] != line_no:
                        problems.append(
                            f"{shard_rel}:{line_no}: placement mismatch for {token} "
                            f"(index says {entry['shard']}:{entry['line']})"
                        )
                    if _sha256(line) != entry["record_hash"]:
                        problems.append(
                            f"{shard_rel}:{line_no}: sha256(line) != record_hash "
                            f"for {token}"
                        )
                    try:
                        record = DriveAlignRecord.from_dict(record_dict)
                        if record.canonical_hash() != entry["record_hash"]:
                            problems.append(
                                f"{shard_rel}:{line_no}: roundtrip hash mismatch "
                                f"for {token}"
                            )
                    except ValueError as exc:
                        problems.append(
                            f"{shard_rel}:{line_no}: from_dict failed for {token} "
                            f"({exc})"
                        )
                    seen.add(token)
        missing = sorted(set(index_by_split[split]) - seen)
        if missing:
            problems.append(f"{split}: {len(missing)} indexed tokens missing in shards")
        if n_lines != info["record_count"]:
            problems.append(
                f"{split}: shard lines {n_lines} != record_count "
                f"{info['record_count']}"
            )
    return problems


def build_dataset_manifest(
    config: dict,
    scene_fingerprints: Dict[str, str],
    per_split: Dict[str, dict],
    index: Dict[str, dict],
    quarantine_count: int,
) -> dict:
    """Root index over relative paths only, with its own sha256."""
    payload = {
        "manifest_type": "dataset_manifest",
        "contract_version": DEFAULT_CONTRACT_VERSION,
        "config": config,
        "scene_manifest_sha256": scene_fingerprints,
        "splits": per_split,
        "records": index,
        "quarantine": {"path": "quarantine.jsonl", "count": quarantine_count},
    }
    digest = _sha256(_canonical_json(payload))
    return {**payload, "manifest_sha256": digest}


def _markdown_report(
    config: dict, stats: Dict[str, dict], gates: Dict[str, bool]
) -> str:
    lines = ["# Stage 07/08 dataset build report (contract v4 + GT backfill)", "", "## Config", ""]
    for key in sorted(config):
        lines.append(f"- {key}: `{config[key]}`")
    lines += [
        "",
        "## Per-split build",
        "",
        "| split | scenes | keyframes | candidates | valid | backfilled | quarantine | shards |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        s = stats[split]
        lines.append(
            f"| {split} | {s['scene_count']} | {s['keyframe_count']} | "
            f"{s['candidate_count']} | {s['valid_count']} | "
            f"{s['backfilled_count']} | "
            f"{s['quarantine_count']} | {s['shard_count']} |"
        )
    reasons_total: Counter = Counter()
    for s in stats.values():
        reasons_total.update(s["reasons"])
    lines += ["", "## Quarantine reasons", ""]
    if reasons_total:
        lines += ["| reason_code | count |", "|---|---|"]
        for code, count in sorted(reasons_total.items()):
            lines.append(f"| {code} | {count} |")
    else:
        lines.append("none (all candidates built valid records)")
    lines += ["", "## Gates", "", "| gate | result |", "|---|---|"]
    for name, ok in gates.items():
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_root = args.out.expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    reports_dir = args.reports.expanduser()
    reports_dir.mkdir(parents=True, exist_ok=True)
    builder_commit = args.builder_commit or _git_commit()

    # 1) Frozen scene manifests: integrity + pairwise disjointness.
    scene_tokens_by_split: Dict[str, List[str]] = {}
    fingerprints: Dict[str, str] = {}
    for split in SPLITS:
        path = args.scene_manifests.expanduser() / f"{split}_scene_manifest.json"
        tokens, fingerprint = load_scene_manifest(path)
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"{path}: duplicate scene tokens")
        scene_tokens_by_split[split] = tokens
        fingerprints[split] = fingerprint
    sets = {split: set(tokens) for split, tokens in scene_tokens_by_split.items()}
    scene_sets_disjoint = not (
        (sets["train"] & sets["val"])
        or (sets["train"] & sets["test"])
        or (sets["val"] & sets["test"])
    )

    gt = GtAssets.load()
    if gt.rule_config.status != "frozen":
        raise ValueError(
            f"gt_rule_config.json status is {gt.rule_config.status!r}; "
            "the v4 build requires the frozen S08 rules"
        )
    print(
        f"gt rules: version={gt.rule_config.rule_version} "
        f"status={gt.rule_config.status} "
        f"sha256={gt.rule_config_sha256[:12]}...",
        flush=True,
    )

    config = {
        "builder_commit": builder_commit,
        "contract_version": DEFAULT_CONTRACT_VERSION,
        "dataroot": str(args.dataroot.expanduser().resolve()),
        "gt_backfill": True,
        "gt_rule_config_sha256": gt.rule_config_sha256,
        "gt_rule_status": gt.rule_config.status,
        "gt_rule_version": gt.rule_config.rule_version,
        "max_scenes": args.max_scenes,
        "min_prevs": MIN_PREVS,
        "mode": "smoke" if args.max_scenes is not None else "full",
        "nuscenes_version": args.version,
        "scene_manifests_dir": str(args.scene_manifests.expanduser().resolve()),
        "shard_line_base": 0,
        "shard_size": SHARD_SIZE,
    }

    # 2) Load nuScenes once and build every split through the adapter.
    print(f"loading {args.version} from {args.dataroot} ...", flush=True)
    nusc = load_nuscenes(args.dataroot.expanduser(), args.version)
    print("nuscenes loaded", flush=True)

    index: Dict[str, dict] = {}
    quarantine: List[dict] = []
    stats: Dict[str, dict] = {}
    per_split: Dict[str, dict] = {}
    distributions: Dict[str, GtDistribution] = {
        split: GtDistribution() for split in SPLITS
    }
    for split in SPLITS:
        tokens = scene_tokens_by_split[split]
        if args.max_scenes is not None:
            tokens = tokens[: args.max_scenes]
        stats[split], q, idx = build_split(
            nusc, split, tokens, out_root, builder_commit, gt,
            gt_stats=distributions[split],
        )
        index.update(idx)
        quarantine.extend(q)
        per_split[split] = {
            "record_count": stats[split]["valid_count"],
            "shard_count": stats[split]["shard_count"],
            "shards": stats[split]["shards"],
        }

    # 3) Write quarantine.jsonl and the root dataset_manifest.json.
    quarantine_path = out_root / "quarantine.jsonl"
    with quarantine_path.open("w", encoding="utf-8") as fh:
        for entry in quarantine:
            fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")

    manifest = build_dataset_manifest(
        config, fingerprints, per_split, index, len(quarantine)
    )
    (out_root / "dataset_manifest.json").write_text(
        _canonical_json(manifest) + "\n", encoding="utf-8"
    )

    # 4) Gates.
    full_coverage = all(
        s["valid_count"] + s["quarantine_count"] == s["candidate_count"]
        for s in stats.values()
    ) and sum(s["candidate_count"] for s in stats.values()) == len(index) + len(
        quarantine
    )
    index_tokens = set(index)
    quarantine_ok = all(
        entry["reason_code"] in QUARANTINE_CODES
        and bool(entry["detail"])
        and entry["sample_token"] not in index_tokens
        for entry in quarantine
    )
    roundtrip_problems = verify_shard_roundtrip(out_root, per_split, index)
    gt_backfill_full = all(
        s["backfilled_count"] == s["valid_count"] for s in stats.values()
    )
    gates = {
        "scene_manifests_intact": True,  # load_scene_manifest raised otherwise
        "scene_sets_disjoint": scene_sets_disjoint,
        "full_coverage": full_coverage,
        "quarantine_codes_valid": quarantine_ok,
        "gt_backfill_full": gt_backfill_full,
        "shard_roundtrip": not roundtrip_problems,
    }
    gates["all_pass"] = all(gates.values())

    report = {
        "manifest_type": "build_report",
        "config": config,
        "per_split": {
            split: {k: v for k, v in stats[split].items() if k != "shards"}
            for split in SPLITS
        },
        "quarantine_reasons_total": dict(
            sorted(
                sum(
                    (Counter(s["reasons"]) for s in stats.values()),
                    Counter(),
                ).items()
            )
        ),
        "roundtrip_problems": roundtrip_problems,
        "gates": gates,
    }
    (reports_dir / "build_report.json").write_text(
        _canonical_json(report) + "\n", encoding="utf-8"
    )
    (reports_dir / "build_report.md").write_text(
        _markdown_report(config, stats, gates), encoding="utf-8"
    )

    # 5) GT label distribution report (S08): inline-counted during build.
    total = GtDistribution()
    for distribution in distributions.values():
        total.merge(distribution)
    per_split_gt = {split: d.to_payload() for split, d in distributions.items()}
    per_split_gt["all"] = total.to_payload()
    gt_meta = {
        key: config[key]
        for key in (
            "builder_commit",
            "contract_version",
            "gt_rule_config_sha256",
            "gt_rule_status",
            "gt_rule_version",
            "mode",
        )
    }
    (reports_dir / "gt_distribution_report.json").write_text(
        _canonical_json(per_split_gt) + "\n", encoding="utf-8"
    )
    (reports_dir / "gt_distribution_report.md").write_text(
        render_gt_distribution_markdown(per_split_gt, gt_meta), encoding="utf-8"
    )

    for split in SPLITS:
        s = stats[split]
        print(
            f"{split}: scenes={s['scene_count']} keyframes={s['keyframe_count']} "
            f"candidates={s['candidate_count']} valid={s['valid_count']} "
            f"backfilled={s['backfilled_count']} "
            f"quarantine={s['quarantine_count']} shards={s['shard_count']}"
        )
    print(f"quarantine reasons: {report['quarantine_reasons_total'] or '{}'}")
    print(f"manifest: {out_root / 'dataset_manifest.json'}")
    print(f"gt distribution: {reports_dir / 'gt_distribution_report.md'}")
    if gates["all_pass"]:
        print(f"PASS: all Step 2 gates ok -> {reports_dir / 'build_report.json'}")
    else:
        print(f"FAIL: gates={gates} roundtrip_problems={roundtrip_problems[:10]}")
    return 0 if gates["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
