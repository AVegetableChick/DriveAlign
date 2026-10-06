"""S11 单测：`sft/gate.py`（G1 硬层状态保真 + 软层四字段语义 + G7 离散字段变更）。

覆盖 `S11_implementation_plan.md` §7-I 的判定逻辑：
    逐张量 shape/dtype/Δ 比对、NaN/Inf 检出、期望 global step 推导、
    四字段语义相等（忽略 bbox 数值与顺序、检出各字段漂移）。
覆盖 §7-J 的 G7 判定原语：
    blank-image 反事实只看离散字段（speed_action / yield_required /
    critical_objects 集合），risk_factors 抖动不算"视觉依赖"证据。
全部用合成 tensor / 纯 dict，不依赖 GPU、模型或图像文件。

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_gate.py -q``
"""

from __future__ import annotations

import torch

from drivealign.sft.gate import (
    DISCRETE_FIELD_NAMES,
    adapter_state_dict,
    changed_semantic_fields,
    compare_state_dicts,
    discrete_fields_changed,
    expected_global_step,
    four_fields_equal,
    nonfinite_parameter_names,
    semantic_fields,
)

#: 覆盖阶段合并口径的最小 config（§7-F/G）。
CONFIG = {
    "training": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
    },
    "train32": {"training": {"num_train_epochs": 2}},
    "overfit": {"training": {"num_train_epochs": 3}},
}


def _obj(category, motion_state=None, bbox=(0.0, 0.0, 1.0, 1.0)):
    return {"category": category, "motion_state": motion_state, "bbox_2d": list(bbox)}


class _FakePeftModel(torch.nn.Module):
    """最小复刻 peft 命名：基座权重 + `lora_*` adapter 权重。"""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.q_proj = torch.nn.Module()
        self.q_proj.lora_A = torch.nn.Module()
        self.q_proj.lora_A.default = torch.nn.Module()
        self.q_proj.lora_A.default.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.q_proj.lora_B = torch.nn.Module()
        self.q_proj.lora_B.default = torch.nn.Module()
        self.q_proj.lora_B.default.weight = torch.nn.Parameter(torch.zeros(2, 2))


def test_adapter_state_dict_selects_lora_params_by_name():
    model = _FakePeftModel()
    state = adapter_state_dict(model)
    assert set(state) == {
        "q_proj.lora_A.default.weight",
        "q_proj.lora_B.default.weight",
    }
    # 落盘参照必须是 CPU 上的独立副本（不共享计算图/设备）。
    assert all(not t.requires_grad for t in state.values())


def test_adapter_state_dict_survives_inference_mode_freeze():
    # 回归防护（S11 实测假 FAIL）：peft `from_pretrained` 默认 is_trainable=False，
    # 会把已加载的 adapter 权重置 requires_grad=False。按名字筛必须仍然选中它们，
    # 否则复验侧得到空 dict → 硬层退化为"全 missing"。
    model = _FakePeftModel()
    for param in model.parameters():
        param.requires_grad_(False)  # 模拟推理模式加载后的全冻结状态
    state = adapter_state_dict(model)
    assert len(state) == 2
    assert "q_proj.lora_A.default.weight" in state


def test_compare_state_dicts_identical_passes():
    ref = {"a": torch.zeros(2, dtype=torch.bfloat16), "b": torch.ones(3)}
    new = {k: v.clone() for k, v in ref.items()}
    out = compare_state_dicts(ref, new)
    assert out["passed"] is True
    assert out["worst_abs_diff"] == 0.0
    assert out["missing"] == [] and out["unexpected"] == []


def test_compare_state_dicts_detects_over_tolerance_delta():
    ref = {"a": torch.zeros(4)}
    new = {"a": torch.tensor([0.0, 0.0, 2e-6, 0.0])}
    out = compare_state_dicts(ref, new, atol=1e-6)
    assert out["passed"] is False
    assert out["worst_abs_diff"] > 1e-6
    # 恰好等于容差也应 FAIL（判定是严格小于）。用 float64 保证 1e-6 可精确表示。
    ref64 = {"a": torch.zeros(4, dtype=torch.float64)}
    boundary = compare_state_dicts(
        ref64, {"a": torch.tensor([1e-6, 0, 0, 0], dtype=torch.float64)}, atol=1e-6
    )
    assert boundary["worst_abs_diff"] == 1e-6
    assert boundary["passed"] is False


def test_compare_state_dicts_detects_missing_unexpected_shape_dtype():
    ref = {"a": torch.zeros(2), "b": torch.zeros(3), "c": torch.zeros(1, dtype=torch.float32)}
    new = {"a": torch.zeros(3), "b": torch.zeros(3, dtype=torch.bfloat16), "d": torch.zeros(1)}
    out = compare_state_dicts(ref, new)
    assert out["passed"] is False
    assert out["missing"] == ["c"]
    assert out["unexpected"] == ["d"]
    assert out["shape_mismatch"] == {"a": ([2], [3])}
    assert out["dtype_mismatch"] == {"b": ("torch.float32", "torch.bfloat16")}


def test_nonfinite_parameter_names_detects_nan_and_inf():
    state = {
        "ok": torch.tensor([1.0, -2.0]),
        "nan": torch.tensor([0.0, float("nan")]),
        "inf": torch.tensor([float("inf")]),
    }
    assert nonfinite_parameter_names(state) == ["nan", "inf"]


def test_expected_global_step_train32_and_overfit():
    # train32: ceil(32/4)=8 update/epoch × 2 epoch = 16。
    assert expected_global_step(CONFIG, 32) == 16
    # overfit128: ceil(128/4)=32 × 3 = 96。
    assert expected_global_step(CONFIG, 128, overfit=True) == 96
    # 不整除时向上取整（dataloader 不丢尾批）。
    assert expected_global_step(CONFIG, 30) == 16


def test_semantic_fields_ignores_bbox_and_order():
    a = {
        "speed_action": "KEEP_SPEED",
        "yield_required": False,
        "risk_factors": ["crossing", "occlusion"],
        "critical_objects": [_obj("car", "moving", (0, 0, 1, 1)), _obj("pedestrian", "stationary")],
    }
    # bbox 数值不同、对象与风险顺序颠倒 → 语义仍应相等。
    b = {
        "yield_required": False,
        "speed_action": "KEEP_SPEED",
        "critical_objects": [_obj("pedestrian", "stationary", (9, 9, 9, 9)), _obj("car", "moving", (5, 5, 5, 5))],
        "risk_factors": ["occlusion", "crossing"],
    }
    equal, diffs = four_fields_equal(a, b)
    assert equal is True and diffs == []


def test_four_fields_equal_detects_each_field_drift():
    base = {
        "speed_action": "KEEP_SPEED",
        "yield_required": False,
        "risk_factors": ["crossing"],
        "critical_objects": [_obj("car", "moving")],
    }
    # 逐个字段制造漂移，确认 diff 能定位到具体字段。
    for field, value in (
        ("speed_action", "STOP"),
        ("yield_required", True),
        ("risk_factors", []),
        ("critical_objects", [_obj("pedestrian", "stationary")]),
    ):
        mutated = dict(base)
        mutated[field] = value
        equal, diffs = four_fields_equal(base, mutated)
        assert equal is False
        assert diffs == [field]


def test_semantic_fields_tolerates_missing_optional_fields():
    # 缺字段不应抛异常，而应被规范化为 None/空集，交由 diff 判定。
    fields = semantic_fields({})
    assert fields == {
        "speed_action": None,
        "yield_required": None,
        "critical_objects": frozenset(),
        "risk_factors": frozenset(),
    }


def test_changed_semantic_fields_ignores_bbox_values_and_order():
    # 语义比较的口径来自 `semantic_fields`：bbox 数值与顺序都不算"变化"，
    # 否则 G7 的 blank-image 反事实会被 token 级抖动淹没。
    a = {
        "speed_action": "KEEP_SPEED",
        "yield_required": False,
        "critical_objects": [_obj("car", "moving", (0, 0, 1, 1))],
        "risk_factors": ["crossing", "occlusion"],
    }
    b = {
        "yield_required": False,
        "speed_action": "KEEP_SPEED",
        "risk_factors": ["occlusion", "crossing"],
        "critical_objects": [_obj("car", "moving", (7, 7, 8, 8))],
    }
    assert changed_semantic_fields(a, b) == []


def test_discrete_fields_changed_detects_each_discrete_field():
    # §7-J 的三个离散字段逐个制造漂移，确认都能被 G7 判定原语命中。
    base = {
        "speed_action": "KEEP_SPEED",
        "yield_required": False,
        "critical_objects": [_obj("car", "moving")],
        "risk_factors": ["crossing"],
    }
    for field, value in (
        ("speed_action", "STOP"),
        ("yield_required", True),
        ("critical_objects", [_obj("pedestrian", "stationary")]),
    ):
        mutated = dict(base)
        mutated[field] = value
        assert discrete_fields_changed(base, mutated) == [field]


def test_discrete_fields_changed_excludes_risk_factor_drift():
    # risk_factors 对光照/纹理类扰动更敏感，按 §7-J 不计入"视觉依赖"判定；
    # 但 `changed_semantic_fields` 仍应如实报告它变了（信息项与判定项分离）。
    base = {
        "speed_action": "KEEP_SPEED",
        "yield_required": False,
        "critical_objects": [_obj("car", "moving")],
        "risk_factors": ["crossing"],
    }
    mutated = dict(base, risk_factors=["occlusion"])
    assert changed_semantic_fields(base, mutated) == ["risk_factors"]
    assert discrete_fields_changed(base, mutated) == []


def test_discrete_field_names_matches_frozen_scope():
    # 固化口径：G7 只看这三项（§7-J 明文列举），防后续顺手把 risk_factors 加回去。
    assert DISCRETE_FIELD_NAMES == ("speed_action", "yield_required", "critical_objects")
