"""S11 Gate 判定纯逻辑（G1 硬/软两层 + G7 视觉依赖抽查原语）。

这部分内容包含：

1. **G1 判定**（`S11_implementation_plan.md` §7-I，2026-10-05 拍板）
------------------------------------------------------------
- **硬层（状态保真，确定性）**：reload 后权重**逐张量** shape+dtype 同构可比
  （容差 `max|Δ| < 1e-6`）、训练 global step 与配置推导的期望步数相等、无 NaN/Inf。
- **软层（功能冒烟，容忍 token 级非确定）**：固定输入 forward 的 logits 有限；
  贪心生成文本可被 `parse_structured_output` 严格解析，且**四字段语义相等**
  （speed_action / yield_required 相等，critical_objects 集合相等，risk_factors
  集合相等）。允许 bbox 数值、token 序列等逐位差异（flashattention/kernel 与残留
  dropout 可致逐 token 非确定）。

2. **G7 判定原语**（§7-J）：blank-image 反事实下，overfit 模型的**离散字段**
   （speed_action / yield_required / critical_objects 集合）至少 1 项改变。

为什么硬层只比对 **adapter** 权重
---------------------------------
基座权重由"训练进程"与"复验进程"分别从**同一冻结 checkpoint**（`config.model.path`）
加载；LoRA 在训练期冻结基座，基座不产生任何更新。真正被"训练 → 保存 adapter →
新进程重载"这条链路影响的状态，只有 LoRA（adapter）权重。全量 3B 基座权重
（bf16 ≈ 6GB）逐张量落盘做参照既无必要（来源同一冻结文件，逐位同构）又会拖慢
smoke，故硬层比对范围 = adapter 参数集合（即 `save_pretrained` 落盘的内容）。

选取口径由 `adapter_state_dict` 负责（按 peft 命名 `lora_` 筛，**不是**按
`requires_grad`——原因见该函数 docstring）。

本模块只依赖 torch 与普通 Python 对象，不含模型加载/生成/图像 IO，因此可被 CPU
单测完全覆盖（见 `tests/unit/test_sft_gate.py`）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

#: 硬层逐张量容差（§7-I）：`max|Δ|` 必须严格小于该值。
DEFAULT_ATOL = 1e-6


#: LoRA adapter 参数名里必然出现的子串（peft 命名约定，如
#: ``...self_attn.q_proj.lora_A.default.weight`` / ``...lora_B.default.weight``）。
_ADAPTER_NAME_MARKER = "lora_"


def adapter_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """取模型 **adapter（LoRA）参数**的 state_dict（G1 硬层比对对象）。

    为什么按**名字**筛，而不是按 `requires_grad`
    --------------------------------------------
    peft 的 `PeftModel.from_pretrained` 默认 `is_trainable=False`（推理模式），会把
    已正常加载的 LoRA 权重整体置为 `requires_grad=False`。若拿 `requires_grad` 当
    筛子，复验侧会得到一个**空 dict**，硬层比对就退化成"参照里每个张量都 missing"
    的**假 FAIL**（S11 实测：`n_trainable_tensors=0`、`missing` 列了全部 696 个张量，
    而 adapter 其实已正确加载）。名字含 `lora_` 是 peft 的稳定命名约定，训练端
    （`get_peft_model`）与推理端（`from_pretrained`）命名一致，才能两端口径对齐。

    基座权重不入选：两进程从同一冻结 checkpoint 加载、训练期被 LoRA 冻结，不属于
    "训练 → 保存 → 重载"这条链路影响的状态（详见模块 docstring）。统一 detach 并
    搬到 CPU 后 clone，避免比对受设备/计算图影响。
    """
    return {
        name: param.detach().to("cpu").clone()
        for name, param in model.named_parameters()
        if _ADAPTER_NAME_MARKER in name
    }


def nonfinite_parameter_names(state: Mapping[str, torch.Tensor]) -> list[str]:
    """列出含 NaN/Inf 的张量名（硬层：无 NaN/Inf）。

    注意检查是 `isnan | isinf`——只查 NaN 会漏掉 Inf（例如 bf16 溢出得到 ±inf）。
    """
    return [
        name
        for name, tensor in state.items()
        if not bool(torch.isfinite(tensor).all())
    ]


def compare_state_dicts(
    reference: Mapping[str, torch.Tensor],
    reloaded: Mapping[str, torch.Tensor],
    *,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    """逐张量比对两组 state_dict（G1 硬层核心）。

    判定维度（缺一即 FAIL）：
        - 名字集合完全一致：`missing`（参照有、reload 无）/ `unexpected`（反之）。
        - shape 同构：`shape_mismatch`。
        - dtype 同构：`dtype_mismatch`。
        - 逐张量 `max|Δ| < atol`：`max_abs_diff`（逐名）、`worst_abs_diff`（全局最大）。

    参数
    ----
    reference : 训练端落盘的参照 state_dict（`train_state_ref.pt` 的 `adapter_state`）。
    reloaded  : 新进程重载后的 adapter state_dict（`adapter_state_dict(model)`）。
    atol      : 逐张量容差上界（严格小于）。

    返回
    ----
    dict，字段见上；`passed` 为全部维度通过的布尔，供上层直接做 Gate 判定。
    """
    missing = sorted(set(reference) - set(reloaded))
    unexpected = sorted(set(reloaded) - set(reference))

    shape_mismatch: dict[str, tuple[list[int], list[int]]] = {}
    dtype_mismatch: dict[str, tuple[str, str]] = {}
    max_abs_diff: dict[str, float] = {}

    for name in sorted(set(reference) & set(reloaded)):
        ref_t = reference[name]
        new_t = reloaded[name]
        if tuple(ref_t.shape) != tuple(new_t.shape):
            shape_mismatch[name] = (list(ref_t.shape), list(new_t.shape))
            # 形状不一致无法逐元素相减，跳过 Δ 计算（该项已由 shape_mismatch 判 FAIL）。
            continue
        if ref_t.dtype != new_t.dtype:
            dtype_mismatch[name] = (str(ref_t.dtype), str(new_t.dtype))
            # dtype 不同时数值仍可比（提升后再减），继续算 Δ 以保留诊断信息。
        delta = (ref_t.to(torch.float64) - new_t.to(torch.float64)).abs().max()
        max_abs_diff[name] = float(delta)

    # 全局最大 Δ：无同名张量时记 0.0（缺失项已由 missing/unexpected 判 FAIL）。
    worst_abs_diff = max(max_abs_diff.values(), default=0.0)

    passed = (
        not missing
        and not unexpected
        and not shape_mismatch
        and not dtype_mismatch
        and worst_abs_diff < atol
    )
    return {
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "max_abs_diff": max_abs_diff,
        "worst_abs_diff": worst_abs_diff,
        "atol": atol,
        "passed": passed,
    }


def expected_global_step(
    config: Mapping[str, Any], n_samples: int, *, overfit: bool = False
) -> int:
    """由配置 + 样本数推导期望的 optimizer step（硬层：global step 相等）。

    HF `Trainer.state.global_step` 计的是**优化器更新次数**（已按梯度累积折算）。
    推导式：

        steps_per_epoch = ceil(n_samples / (per_device_batch * grad_accum))
        expected        = steps_per_epoch * num_train_epochs

    与 `sft/train.py: make_trainer` 合并阶段参数的口径一致（顶层 `training` 打底、
    阶段段覆盖），因此 train32（2 epoch）与 overfit128（3 epoch）分别得到各自步数。
    该式让复验进程能**独立**判"训练确实按注册调度跑满、参照是跑完那一刻落盘的"。
    """
    seg = "overfit" if overfit else "train32"
    params = dict(config["training"])
    params.update(config.get(seg, {}).get("training", {}))
    batch = int(params["per_device_train_batch_size"])
    accum = int(params["gradient_accumulation_steps"])
    epochs = int(params["num_train_epochs"])
    per_epoch = math.ceil(n_samples / (batch * accum)) if batch * accum else 0
    return per_epoch * epochs


def semantic_fields(four_fields: Mapping[str, Any]) -> dict[str, Any]:
    """把四字段输出压成**语义可比**形式（剥离 token 级噪声）。

    规则（§7-I）：
        - `speed_action`：值相等（离散枚举）。
        - `yield_required`：值相等（布尔）。
        - `critical_objects`：取 `(category, motion_state)` 的**集合**——bbox 数值
          会因 token 级采样/核函数差异抖动，故不入语义键；对象顺序也不计。
        - `risk_factors`：字符串**集合**（顺序不计）。

    入参既可是 `StructuredDrivingOutput.to_dict()`，也可是 GT 的四字段 dict
    （记录侧 `expected_output` 子集）。缺字段按 None/空集处理，交由 diff 暴露。
    """
    objects = four_fields.get("critical_objects") or []
    object_keys = frozenset(
        (obj.get("category"), obj.get("motion_state"))
        for obj in objects
        if isinstance(obj, Mapping)
    )
    return {
        "speed_action": four_fields.get("speed_action"),
        "yield_required": four_fields.get("yield_required"),
        "critical_objects": object_keys,
        "risk_factors": frozenset(four_fields.get("risk_factors") or []),
    }


def four_fields_equal(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    """软层判定：两组四字段输出的语义是否相等。

    返回 `(相等布尔, 不等的字段名列表)`——字段名列表用于把失败原因写进报告，
    避免只报"PASS/FAIL"而无法定位是哪个字段漂了。
    """
    sa, sb = semantic_fields(a), semantic_fields(b)
    diffs = [key for key in sa if sa[key] != sb[key]]
    return (not diffs), diffs


# ---------------------------------------------------------------------------
# G7：视觉依赖抽查（blank-image 反事实）的判定原语
# ---------------------------------------------------------------------------

#: G7 只看**离散字段**（规划 §7-J 明文列举）：speed_action / yield_required /
#: critical_objects 集合。不含 risk_factors，因为它对光照/纹理类扰动更敏感，
#: 容易把"视觉确实在用"与"输出抖动"混为一谈。
DISCRETE_FIELD_NAMES: tuple[str, ...] = (
    "speed_action",
    "yield_required",
    "critical_objects",
)


def changed_semantic_fields(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> list[str]:
    """列出两组四字段输出里**语义上变了**的字段名（顺序同 `semantic_fields`）。

    语义比较规则复用 `semantic_fields`（bbox 数值与对象/风险顺序不计），
    因此该函数天然忽略 token 级噪声，可直接用于 G7 的反事实判定。
    """
    sa, sb = semantic_fields(a), semantic_fields(b)
    return [key for key in sa if sa[key] != sb[key]]


def discrete_fields_changed(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> list[str]:
    """G7 判定原语：两侧输出中**离散字段**发生变化的字段名列表。

    判定条件（§7-J）= 该列表非空，即"blank-image 反事实下至少 1 项离散字段改变"。
    """
    changed = changed_semantic_fields(a, b)
    return [name for name in changed if name in DISCRETE_FIELD_NAMES]
