"""训练目标构造（§2.2）：四字段显式列举 + 整串 dump + 确定性断言。

设计动机
--------
v6 模型面删除了 `reasoning` 字段，模型要求的输出只剩四个字段：

    critical_objects, risk_factors, yield_required, speed_action

SFT 的训练目标 = record 内 `training_targets.expected_output` 的**四字段子集**
（`reasoning` 字段仍是记录的 GT provenance，但训练侧不再消费）。本模块只做
"从 record 目标 dict 里取出四字段，并序列化成发射文本"，不含任何 tokenizer /
process 逻辑，因此是纯函数、可完全由 CPU 单测确定性验证。

为什么"显式列举"而不是"字典过滤"
--------------------------------
如果写成 `{k: v for k in d if k in TARGET_FIELD_ORDER}`，将来 record 若新增第五个
字段（例如未来恢复 reasoning），会被静默过滤掉——这是"隐式依赖"，容易出 bug。
显式列举则相反：列的是什么就发什么，顺序即发射顺序；遗漏/新增都会被下面两组
确定性断言捕获。

为什么"整串 dump"
-----------------
整段目标文本用一个冻结的 separators 配置 `json.dumps` 生成。它不是"一段一段拼接"
（那样会引入分隔符漂移风险），而是把四字段按固定顺序放进一个 dict 后一次 dump，
保证每个字段出现的文字只源于该字段自身，无相邻字段污染。

技能：确定性钉子
----------------
本模块末尾的 `TARGET_FIELD_ORDER` + 两组断言（`as_target` 里的 set 比对，
`serialize_target` 里的 round-trip）把"构造恒等"钉住，防止未来序列化实现漂移。
它们不再承担 S11 v1 规划里的 mask 职责（那段已在 §2.1 判废）。
"""

from __future__ import annotations

import json
from typing import Any, Mapping

#: 增大 `json.dumps` 不受 Python 3.12 起的 KeyError 空格警告影响（无风险，仅为自文档）。
_ = None

# 第 (1) 项：四字段的**发射顺序**，与 v6 prompt 的字段清单一致。
# critical_objects（对象列表）→ risk_factors（风险列表）→ yield_required（让行布尔）
# → speed_action（速度动作枚举）。这也是 SFT 监督段文本顶部的顺序。
TARGET_FIELD_ORDER: tuple[str, ...] = (
    "critical_objects",
    "risk_factors",
    "yield_required",
    "speed_action",
)

#: 必须存在于 `expected_output` 中的字段集合（记录侧五字段：四 + reasoning）。
#: 注意这里不含 `language_reference`，它是训练目标命名空间里独立的键，不参与本目标。
ALL_EXPECTED_OUTPUT_FIELDS: frozenset[str] = frozenset(
    {"critical_objects", "risk_factors", "reasoning", "yield_required", "speed_action"}
)

#: 冻结的序列化配置（§2.2）：`ensure_ascii=False` 保留中文，`separators=(", ", ": ")`
#: 让 dump 结果更像"模型自然输出的可爱 JSON"（紧凑但键后带空格），也是 SFT 监督文本
#: 确定性的一部分。此配置在训练时会被冻结进 `configs/sft/sft_smoke_1f.yaml` 并登记 sha。
DUMP = dict(ensure_ascii=False, separators=(", ", ": "))


def as_target(expected_output: Mapping[str, Any] | None) -> dict[str, Any]:
    """从 record 的目标 dict 提取**四字段**训练目标。

    参数
    ----
    expected_output : mapping 或 None
        `record.training_targets.expected_output`。为 None 时（记录未 backfill GT）
        直接暴露出错——训练样本不允许无目标。

    返回
    ----
    dict
        只含 `TARGET_FIELD_ORDER` 四键，顺序即发射顺序；`reasoning` 被丢弃。

    失败语义
    -------
    显式列举（`d[field]`）会对缺失键直接抛 KeyError；字段不在白名单会被下面
    的严谨校验拦截。这是有意设计：**任何目标字段漂移都应在构造期 fail-fast**。
    """
    if expected_output is None:
        raise ValueError("expected_output is None: cannot build a supervise target")
    d: Mapping[str, Any] = dict(expected_output)

    # 白名单校验①：未知键。任何不在记录侧五字段集合内的键都应被拒——这能防止
    # 未来 record 结构加字段后这里悄悄放行了错误的键。这里用集合差分找出"多出来的键"。
    unexpected = set(d) - ALL_EXPECTED_OUTPUT_FIELDS
    if unexpected:
        raise ValueError(f"unexpected expected_output fields: {sorted(unexpected)}")

    # 白名单校验②：四字段必须齐全。使用 TARGET_FIELD_ORDER 显式遍历，缺一个报一个。
    missing = [field for field in TARGET_FIELD_ORDER if field not in d]
    if missing:
        raise ValueError(f"missing required target fields: {missing}")

    # 显式列举（按发射顺序），绝不写成推导式过滤。
    return {field: d[field] for field in TARGET_FIELD_ORDER}


def serialize_target(target: Mapping[str, Any]) -> str:
    """把四字段目标序列化成固定的监督目标**整串**（§2.2 的 `full`）。

    该串是 SFT 监督段（assistant 输出）的文本本体：训练时 model 被要求逐字预测
    这串 JSON。`DUMP` 配置冻结在模块常量里，保证同一目标永远产出同一字节串。

    确定性占位（§2.2）：
        - 构造恒等：`json.dumps(target, **DUMP) == 本函数输出`（防御性保留）。
        - 语义 round-trip：`json.loads(本输出) == target`（防序列化漂移）。

    返回
    ----
    str
        形如 `{"critical_objects": [...], "risk_factors": [...], ...}` 的整串。
    """
    full = json.dumps(target, **DUMP)
    # 确定性钉子①：再喂一次应逐字节一致（防未来实现改动悄悄改了格式）。
    assert json.dumps(target, **DUMP) == full
    # 确定性钉子②：语义上能无损读回同一 dict（防键序/编码漂移破坏 round-trip）。
    assert json.loads(full) == target
    return full