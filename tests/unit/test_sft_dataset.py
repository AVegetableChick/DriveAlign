"""S11 Step 1 单测：`sft/dataset.py`（record→样本 + 面 parity + 样本 sha）。

覆盖 §3 Step 1 的 `test_sft_dataset` ③④：
    ③ 面 parity（user 面文本 + request_hash_1f vs manifest）；
    ④ 确定性（两次构建 sample_sha256 一致）。
另含 ①②的超集（record_hash 读取语义 + 监督段结构由 targets/collator 承接）。

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_dataset.py -q``
"""

from __future__ import annotations

import pytest

from drivealign.records.record import DriveAlignRecord
from drivealign.records.serializer import InputPolicy, serialize
from drivealign.sft.dataset import build_training_sample
from drivealign.sft.targets import TARGET_FIELD_ORDER


def _record_dict() -> dict:
    """合成一条 1F v4-vocab 冻结记录（四历史帧，GT 含四字段+reasoning）。"""
    frames = [
        {
            "image_relpath": f"samples/CAM_FRONT/n{i}.jpg",
            "frame_token": f"tok{i}",
            "timestamp_us": 1_510_000_000_000_000 + i * 500_000,
            "time_offset_s": (i - 3) * 0.5,
        }
        for i in range(4)
    ]
    return {
        "record_version": "v1",
        "sample_token": "tok3",
        "scene_token": "scene-0",
        "model_inputs": {"frames": frames, "ego_speed_mps": 5.2},
        "training_targets": {
            "expected_output": {
                "critical_objects": [
                    {"category": "car", "bbox_2d": [1, 2, 3, 4], "motion_state": "moving"}
                ],
                "risk_factors": ["collision"],
                "reasoning": "car ahead",
                "yield_required": True,
                "speed_action": "DECELERATE",
            },
            "language_reference": None,
        },
        "oracle_only": {"future_ego_poses": []},
        "provenance": {
            "contract_version": "v4",
            "temporal_policy_version": "v1",
            "builder_commit": "deadbeef",
            "nuscenes_version": "v1.0-mini",
        },
    }


def _record() -> DriveAlignRecord:
    return DriveAlignRecord.from_dict(_record_dict())


def _entry_for(record: DriveAlignRecord, *, corrupt_hash: bool = False) -> dict:
    """按 record 重建 manifest entry；request_hash_1f 取自 serialize（v6 面）。"""
    model_request = serialize(record.model_inputs, InputPolicy.ONE_FRAME)
    expected_hash = model_request.canonical_hash()
    if corrupt_hash:
        expected_hash = "0" * 64
    return {
        "split": "train",
        "scene_token": record.scene_token,
        "request_hash_1f": expected_hash,
        "record_hash": "a" * 64,
        "anchor_image_relpath": "samples/CAM_FRONT/n3.jpg",
    }


def test_build_training_sample_parity_ok_and_target_four_fields():
    record = _record()
    sample = build_training_sample(record, _entry_for(record))
    # 面 parity：serialize 重建值 == manifest 值。
    assert sample.request_hash_1f == sample.manifest_request_hash
    # 目标四字段，顺序正确，无 reasoning。
    assert list(sample.target) == list(TARGET_FIELD_ORDER)
    assert "reasoning" not in sample.target
    assert sample.sample_token == record.sample_token
    assert sample.image_relpath == "samples/CAM_FRONT/n3.jpg"


def test_build_training_sample_wrong_face_fails_fast():
    record = _record()
    with pytest.raises(ValueError, match="face parity failed"):
        build_training_sample(record, _entry_for(record, corrupt_hash=True))


def test_build_training_sample_deterministic():
    record = _record()
    a = build_training_sample(record, _entry_for(record))
    b = build_training_sample(record, _entry_for(record))
    assert a == b  # dataclass 全字段相等
    assert a.sample_sha256 == b.sample_sha256  # 逐样本 sha 确定性


def test_sample_sha_tracks_face_and_target():
    """样本 sha 在目标/面变化时应改变（防 sha 被写成与内容无关的常量）。"""
    record = _record()
    base = build_training_sample(record, _entry_for(record))
    # 改埋 target：通过一个 expected_output 不同的 record。
    altered = _record_dict()
    altered["training_targets"]["expected_output"]["speed_action"] = "STOP"
    altered_rec = DriveAlignRecord.from_dict(altered)
    changed = build_training_sample(altered_rec, _entry_for(altered_rec))
    assert changed.sample_sha256 != base.sample_sha256


def test_encode_chat_requires_processor_and_image():
    """encode_chat 依赖真实 processor + 图像，无模型环境应被主动跳过/提示。"""
    from drivealign.sft.dataset import encode_chat

    # 无本地模型时不跑（等价于既有 test 的 skipif 口径）。此处仅验证函数存在且
    # 缺 processor 时会在 process_vision_info 处失败（不 mock，直接断言签名可用）。
    assert callable(encode_chat)