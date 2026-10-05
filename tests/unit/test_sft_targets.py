"""S11 Step 1 单测：`sft/targets.py`（四字段目标 + 确定性序列化）。

覆盖 §3 Step 1 的 `test_sft_targets` 清单：
    ① 整串 dump == 内部断言；② 合成异常输入 fail-fast 语义；
    ③ 四字段显式列举防漏防混。

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_targets.py -q``
"""

from __future__ import annotations

import json

import pytest

from drivealign.sft.targets import (
    TARGET_FIELD_ORDER,
    ALL_EXPECTED_OUTPUT_FIELDS,
    DUMP,
    as_target,
    serialize_target,
)


def _full_expected() -> dict:
    """一个完整的记录侧五字段 expected_output（含 reasoning，应被丢弃）。"""
    return {
        "critical_objects": [
            {"category": "car", "bbox_2d": [1, 2, 3, 4], "motion_state": "moving"}
        ],
        "risk_factors": ["collision"],
        "reasoning": "car ahead, will decelerate",
        "yield_required": True,
        "speed_action": "DECELERATE",
    }


def test_field_order_is_four_and_matches_v6():
    # 四字段、且必属记录侧白名单；reasoning 不在训练发射顺序里。
    assert TARGET_FIELD_ORDER == (
        "critical_objects",
        "risk_factors",
        "yield_required",
        "speed_action",
    )
    assert set(TARGET_FIELD_ORDER) <= ALL_EXPECTED_OUTPUT_FIELDS


def test_as_target_drops_reasoning_keeps_order():
    target = as_target(_full_expected())
    assert list(target) == list(TARGET_FIELD_ORDER)  # 顺序即发射顺序
    assert "reasoning" not in target  # reasoning 是 GT provenance，不进训练目标


def test_serialize_target_is_whole_string_and_roundtrip():
    target = as_target(_full_expected())
    full = serialize_target(target)
    # ①整串 dump：一次性 json.dumps，非逐段拼接。
    assert full == json.dumps(target, **DUMP)
    # 语义 round-trip：读回同名 dict。
    assert json.loads(full) == target


def test_as_target_none_fails_fast():
    with pytest.raises(ValueError, match="expected_output is None"):
        as_target(None)


def test_as_target_missing_field_fails_fast():
    d = _full_expected()
    del d["yield_required"]
    with pytest.raises(ValueError, match="missing required target fields"):
        as_target(d)


def test_as_target_unexpected_field_fails_fast():
    d = _full_expected()
    d["novel_key"] = 1  # 未来 record 若加字段，这里必须显式暴露而非静默忽略
    with pytest.raises(ValueError, match="unexpected expected_output fields"):
        as_target(d)


def test_serialize_deterministic_two_calls_identical():
    target = as_target(_full_expected())
    assert serialize_target(target) == serialize_target(target)


def test_as_target_is_pure_copy_not_mutation():
    """as_target 不应改动原 dict，且返回结果是按顺序重建的拷贝。"""
    original = _full_expected()
    snapshot = dict(original)
    as_target(original)
    assert original == snapshot  # 原 dict 未被写坏