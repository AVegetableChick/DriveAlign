"""Unit tests for the S08 v4 build backfill and the v3<->v4 parity check.

Purpose:
    Freeze the Step 3 mechanics: ``GtAssets`` loads the frozen rule set,
    ``_backfill_record`` fills the five S08 fields on a real v1.0-mini
    anchor deterministically, ``build_split`` writes shards whose every
    valid line carries a GT-filled record (quarantined ones keep None) and
    passes the roundtrip verifier, and the parity CLI gates v3<->v4 anchor
    identity on a synthetic asset pair (subset/split/request-hash/GT/
    generation-change/placement) before the real-data smoke run.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_dataset_build_v4.py -q``
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from drivealign.cli.dataset_parity_check import (
    AnchorComparison,
    evaluate_parity,
    main as parity_main,
)
from drivealign.dataset.build_dataset import (
    GtAssets,
    _backfill_record,
    build_split,
    verify_shard_roundtrip,
)
from drivealign.gt.config import load_rule_config
from drivealign.records.adapter import build_record
from drivealign.records.record import DriveAlignRecord

MINI_ROOT = Path(__file__).resolve().parents[3] / "data" / "nuscenes" / "mini"


# --- GtAssets ----------------------------------------------------------------


def test_gt_assets_load_frozen_rules():
    gt = GtAssets.load()
    assert gt.rule_config.status == "frozen"
    assert gt.rule_config.contract_version == "v4"
    assert len(gt.rule_config_sha256) == 64
    assert gt.templates["segments"] and gt.phrase_map


# --- backfill on real mini (skipped when absent) ------------------------------


def _first_mini_anchor(nusc):
    scene = nusc.scene[0]
    token = scene["first_sample_token"]
    index = 0
    while index < 3:
        token = nusc.get("sample", token)["next"]
        index += 1
    return token


@pytest.mark.skipif(not MINI_ROOT.is_dir(), reason="mini dataroot not present")
def test_backfill_record_fills_five_fields_and_is_deterministic(tmp_path):
    from drivealign.data.nuscenes_io import load_nuscenes

    nusc = load_nuscenes(MINI_ROOT)
    gt = GtAssets.load()
    token = _first_mini_anchor(nusc)
    result = build_record(nusc, token, builder_commit="deadbeef")
    assert result.record is not None, result.reason_code

    record = _backfill_record(nusc, result.record, gt)
    expected = record.training_targets.expected_output
    assert expected is not None
    assert set(expected) == {
        "critical_objects",
        "risk_factors",
        "reasoning",
        "yield_required",
        "speed_action",
    }
    assert len(expected["critical_objects"]) <= 8
    assert expected["speed_action"] in {"STOP", "DECELERATE", "ACCELERATE", "KEEP_SPEED"}
    assert isinstance(expected["yield_required"], bool)
    assert len(expected["reasoning"]) >= 1
    # language_reference stays closed (DriveLM enhancement package is off).
    assert record.training_targets.language_reference is None

    again = _backfill_record(nusc, result.record, gt)
    assert again.canonical_hash() == record.canonical_hash()


@pytest.mark.skipif(not MINI_ROOT.is_dir(), reason="mini dataroot not present")
def test_build_split_writes_gt_filled_shards(tmp_path):
    from drivealign.data.nuscenes_io import load_nuscenes
    from drivealign.dataset.gt_distribution import GtDistribution

    nusc = load_nuscenes(MINI_ROOT)
    gt = GtAssets.load()
    scene_token = nusc.scene[0]["token"]
    gt_stats = GtDistribution()
    stats, quarantine, index = build_split(
        nusc, "train", [scene_token], tmp_path, "deadbeef", gt, gt_stats=gt_stats
    )
    assert stats["valid_count"] >= 1
    assert stats["backfilled_count"] == stats["valid_count"]
    assert stats["valid_count"] + stats["quarantine_count"] == stats["candidate_count"]
    # Inline distribution accounting matches the shard contents exactly.
    assert gt_stats.frames == stats["backfilled_count"]
    assert sum(gt_stats.speed_action.values()) == stats["backfilled_count"]

    for token, entry in index.items():
        line = (tmp_path / entry["shard"]).read_text(encoding="utf-8").splitlines()[
            entry["line"]
        ]
        assert hashlib.sha256(line.encode("utf-8")).hexdigest() == entry["record_hash"]
        record_dict = json.loads(line)
        expected = record_dict["training_targets"]["expected_output"]
        assert isinstance(expected, dict) and len(expected) == 5
        # provenance carries the record-side contract version (v4)...
        assert record_dict["provenance"]["contract_version"] == "v4"

    per_split = {
        "train": {
            "record_count": stats["valid_count"],
            "shard_count": stats["shard_count"],
            "shards": stats["shards"],
        }
    }
    assert verify_shard_roundtrip(tmp_path, per_split, index) == []


def test_build_split_without_gt_keeps_targets_none(tmp_path):
    """No gt bundle -> the pure Stage 07 behavior (targets stay None)."""
    from drivealign.data.nuscenes_io import load_nuscenes

    if not MINI_ROOT.is_dir():
        pytest.skip("mini dataroot not present")
    nusc = load_nuscenes(MINI_ROOT)
    scene_token = nusc.scene[0]["token"]
    stats, _, index = build_split(nusc, "train", [scene_token], tmp_path, "deadbeef")
    assert stats["backfilled_count"] == 0
    token = next(iter(index))
    line = (tmp_path / index[token]["shard"]).read_text(encoding="utf-8").splitlines()[
        index[token]["line"]
    ]
    assert json.loads(line)["training_targets"]["expected_output"] is None


# --- parity gate evaluation (pure) --------------------------------------------


def _cmp(**overrides) -> AnchorComparison:
    fields = dict(
        token="tok",
        split="train",
        found_in_v3=True,
        split_match=True,
        request_hash_1f_match=True,
        request_hash_4f_match=True,
        record_hash_changed=True,
        gt_present=True,
    )
    fields.update(overrides)
    return AnchorComparison(**fields)


def test_evaluate_parity_all_pass_smoke_and_full():
    good = [_cmp(token=f"tok{i}") for i in range(3)]
    smoke = evaluate_parity(good, [], mode="smoke", v3_total=100)
    assert smoke["all_pass"] and smoke["anchor_set_equality"]
    full = evaluate_parity(good, [], mode="full", v3_total=3)
    assert full["all_pass"] and full["anchor_set_equality"]
    unequal = evaluate_parity(good, [], mode="full", v3_total=100)
    assert not unequal["anchor_set_equality"] and not unequal["all_pass"]


def test_evaluate_parity_each_gate_failure_flips():
    cases = {
        "anchor_subset": [_cmp(found_in_v3=False)],
        "split_assignment_match": [_cmp(split_match=False)],
        "request_hash_parity": [
            _cmp(request_hash_1f_match=False),
            _cmp(request_hash_4f_match=False),
        ],
        "gt_backfill_present": [_cmp(gt_present=False)],
        "record_hash_new_generation": [_cmp(record_hash_changed=False)],
    }
    for expected_gate, comparisons in cases.items():
        gates = evaluate_parity(comparisons, [], mode="smoke", v3_total=10)
        assert not gates[expected_gate] and not gates["all_pass"]
    gates = evaluate_parity([_cmp()], ["tok: sha256 mismatch"], mode="smoke", v3_total=10)
    assert not gates["placement_integrity"] and not gates["all_pass"]


# --- parity CLI on a synthetic asset pair -------------------------------------


GT_SAMPLE = {
    "critical_objects": [],
    "risk_factors": [],
    "reasoning": "The road ahead is clear.",
    "yield_required": False,
    "speed_action": "KEEP_SPEED",
}


def _record_dict(contract_version="v4", expected=GT_SAMPLE):
    frames = [
        {
            "image_relpath": f"samples/CAM_FRONT/n{i}.jpg",
            "frame_token": "tok" if i == 3 else f"tok-f{i}",
            "timestamp_us": 1_510_000_000_000_000 + i * 500_000,
            "time_offset_s": (i - 3) * 0.5,
        }
        for i in range(4)
    ]
    return {
        "record_version": "v1",
        "sample_token": "tok",
        "scene_token": "scene-0",
        "model_inputs": {"frames": frames, "ego_speed_mps": 5.2},
        "training_targets": {"expected_output": expected, "language_reference": None},
        "oracle_only": {
            "future_ego_poses": [
                {
                    "sample_token": f"tok-fu{k}",
                    "timestamp_us": 1_510_000_000_002_000_000 + k * 500_000,
                    "x": 1.0 + k,
                    "y": 2.0,
                    "yaw": 0.1,
                }
                for k in range(2)
            ]
        },
        "provenance": {
            "contract_version": contract_version,
            "temporal_policy_version": "v1",
            "builder_commit": "deadbeef",
            "nuscenes_version": "v1.0-mini",
        },
    }


def _hash_manifest(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(body, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_assets(tmp_path, record_dict):
    """Synthetic v3 (frozen) + v4 (built) asset pair sharing one anchor."""
    from drivealign.records.serializer import InputPolicy, serialize

    record = DriveAlignRecord.from_dict(record_dict)
    line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
    record_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
    (tmp_path / "v4" / "train").mkdir(parents=True)
    (tmp_path / "v4" / "train" / "shard-0000.jsonl").write_text(line + "\n", encoding="utf-8")

    req1 = serialize(record.model_inputs, InputPolicy.ONE_FRAME)
    req4 = serialize(record.model_inputs, InputPolicy.FOUR_FRAME)
    v4_manifest = {
        "manifest_type": "dataset_manifest",
        "contract_version": "v4",
        "config": {},
        "scene_manifest_sha256": {},
        "splits": {},
        "records": {
            "tok": {
                "line": 0,
                "record_hash": record_hash,
                "shard": "train/shard-0000.jsonl",
                "split": "train",
            }
        },
        "quarantine": {"path": "quarantine.jsonl", "count": 0},
    }
    v4_manifest["manifest_sha256"] = _hash_manifest(v4_manifest)
    (tmp_path / "v4" / "dataset_manifest.json").write_text(
        json.dumps(v4_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    v3_anchors = {
        "manifest_type": "anchor_policy_manifest",
        "contract_version": "v3",
        "config": {},
        "dataset_manifest_sha256": "0" * 64,
        "scene_manifest_sha256": {},
        "counts": {"train": 1, "total": 1},
        "anchors": {
            "tok": {
                "line": 0,
                "record_hash": "f" * 64,  # v3 generation: necessarily different
                "scene_token": "scene-0",
                "shard": "train/shard-0000.jsonl",
                "split": "train",
                "anchor_image_relpath": "samples/CAM_FRONT/n3.jpg",
                "request_hash_1f": req1.canonical_hash(),
                "request_hash_4f": req4.canonical_hash(),
            }
        },
    }
    v3_anchors["manifest_sha256"] = _hash_manifest(v3_anchors)
    v3_records = {
        "manifest_type": "dataset_manifest",
        "contract_version": "v3",
        "config": {},
        "scene_manifest_sha256": {},
        "splits": {},
        "records": {"tok": {"record_hash": "f" * 64}},
        "quarantine": {"path": "quarantine.jsonl", "count": 0},
    }
    v3_records["manifest_sha256"] = _hash_manifest(v3_records)
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3" / "anchor_policy_manifest.json").write_text(
        json.dumps(v3_anchors, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "v3" / "dataset_manifest.json").write_text(
        json.dumps(v3_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record_hash


def test_parity_cli_passes_on_matching_pair(tmp_path):
    record_dict = _record_dict()
    _write_assets(tmp_path, record_dict)
    rc = parity_main(
        [
            "--v3-assets", str(tmp_path / "v3"),
            "--v4-assets", str(tmp_path / "v4"),
            "--reports", str(tmp_path / "reports"),
            "--mode", "full",
        ]
    )
    assert rc == 0
    report = json.loads((tmp_path / "reports" / "parity_report.json").read_text())
    assert report["gates"]["all_pass"]


def test_parity_cli_fails_on_request_hash_drift(tmp_path):
    record_dict = _record_dict()
    _write_assets(tmp_path, record_dict)
    manifest_path = tmp_path / "v3" / "anchor_policy_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["anchors"]["tok"]["request_hash_1f"] = "0" * 64  # simulate drift
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = _hash_manifest(payload)  # honest re-sign
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rc = parity_main(
        [
            "--v3-assets", str(tmp_path / "v3"),
            "--v4-assets", str(tmp_path / "v4"),
            "--reports", str(tmp_path / "reports"),
        ]
    )
    assert rc == 1
    report = json.loads((tmp_path / "reports" / "parity_report.json").read_text())
    assert not report["gates"]["request_hash_parity"]


def test_parity_cli_fails_when_gt_missing(tmp_path):
    record_dict = _record_dict(expected=None)  # quarantined-style record
    _write_assets(tmp_path, record_dict)
    rc = parity_main(
        [
            "--v3-assets", str(tmp_path / "v3"),
            "--v4-assets", str(tmp_path / "v4"),
            "--reports", str(tmp_path / "reports"),
        ]
    )
    assert rc == 1
    report = json.loads((tmp_path / "reports" / "parity_report.json").read_text())
    assert not report["gates"]["gt_backfill_present"]


# --- GT distribution accounting (S08 Step 4 report) ----------------------------


def _eo(
    *,
    objects=(("car", "stationary"),),
    risks=(),
    yield_required=False,
    speed_action="KEEP_SPEED",
    reasoning="r" * 10,
):
    return {
        "critical_objects": [
            {"category": c, "bbox_2d": [1.0, 2.0, 3.0, 4.0], "motion_state": m}
            for c, m in objects
        ],
        "risk_factors": list(risks),
        "reasoning": reasoning,
        "yield_required": yield_required,
        "speed_action": speed_action,
    }


def test_gt_distribution_counts_synthetic_and_deterministic():
    from drivealign.dataset.gt_distribution import GtDistribution

    stats = GtDistribution()
    stats.update(_eo(objects=(("car", "stationary"), ("car", "crossing")),
                     risks=("pedestrian_crossing", "congestion"),
                     yield_required=True, speed_action="STOP", reasoning="abcd"))
    stats.update(_eo(objects=()))  # empty scene: no objects, no risks
    payload = stats.to_payload()

    assert payload["frames"] == 2
    assert payload["speed_action"] == {"ACCELERATE": 0, "DECELERATE": 0, "KEEP_SPEED": 1, "STOP": 1}
    assert payload["yield_required"] == {"false": 1, "true": 1}
    assert payload["risk_terms"]["pedestrian_crossing"] == 1
    assert payload["risk_terms"]["corridor_conflict"] == 0  # zero rows stay visible
    assert payload["risk_frames"] == {
        "0": 1, "1": 0, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0
    }
    assert payload["categories"]["car"] == 2 and payload["categories"]["bus"] == 0
    assert payload["motion_states"]["crossing"] == 1 and payload["motion_states"]["oncoming"] == 0
    assert payload["category_motion"]["car"] == {"crossing": 1, "oncoming": 0, "same_direction": 0, "stationary": 1}
    assert payload["objects_per_frame"] == {
        "0": 1, "1": 0, "2": 1, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0
    }
    assert payload["total_objects"] == 2
    assert payload["reasoning_length"]["mean"] == 7.0
    assert payload["speed_action_x_yield"]["STOP"] == {"false": 0, "true": 1}

    again = GtDistribution()
    again.update(_eo(objects=(("car", "stationary"), ("car", "crossing")),
                     risks=("pedestrian_crossing", "congestion"),
                     yield_required=True, speed_action="STOP", reasoning="abcd"))
    again.update(_eo(objects=()))
    assert again.to_payload() == payload  # deterministic


def test_gt_distribution_merge_sums_and_cross_tab():
    from drivealign.dataset.gt_distribution import GtDistribution

    a, b = GtDistribution(), GtDistribution()
    a.update(_eo(speed_action="STOP", yield_required=True, risks=("corridor_conflict",)))
    b.update(_eo(speed_action="STOP", yield_required=False))
    b.update(_eo(speed_action="KEEP_SPEED"))
    total = GtDistribution()
    total.merge(a)
    total.merge(b)
    payload = total.to_payload()
    assert payload["frames"] == 3
    assert payload["speed_action"]["STOP"] == 2
    assert payload["speed_action_x_yield"]["STOP"] == {"false": 1, "true": 1}
    assert payload["risk_frames"]["1"] == 1 and payload["risk_frames"]["0"] == 2


def test_render_gt_distribution_markdown_tables():
    from drivealign.dataset.gt_distribution import GtDistribution, render_gt_distribution_markdown

    per_split = {}
    for name, count in (("train", 2), ("val", 1)):
        stats = GtDistribution()
        for _ in range(count):
            stats.update(_eo(speed_action="STOP", yield_required=True))
        per_split[name] = stats.to_payload()
    merged = GtDistribution()
    for count in (2, 1):
        for _ in range(count):
            merged.update(_eo(speed_action="STOP", yield_required=True))
    per_split["all"] = merged.to_payload()

    md = render_gt_distribution_markdown(per_split, {"contract_version": "v4"})
    assert md.startswith("# S08 v4 GT distribution report")
    for section in (
        "## speed_action",
        "## yield_required",
        "## risk_factors",
        "## critical_objects: category",
        "## critical_objects: motion_state",
        "## critical_objects: category x motion_state (all)",
        "## objects per frame",
        "## speed_action x yield_required (all)",
        "## reasoning length (chars)",
    ):
        assert section in md, section
    assert "| STOP | 2 (100.0%) | 1 (100.0%) | 3 (100.0%) |" in md
    assert "| pedestrian_crossing | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |" in md
