"""S11 回归防护单测：LoRA `target_modules` 必须把视觉塔排除在外（S11_bug_log B7）。

背景（静默缺陷）
----------------
`configs/sft/sft_smoke_1f.yaml` 早期把 `lora.target_modules` 写成裸后缀列表
`[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`。peft 对 **list** 的
匹配是 `key in list` 或 `key.endswith(f".{name}")`，于是视觉塔里同名的
`visual.blocks.*.mlp.{gate,up,down}_proj` 被一并命中：adapter 里多出 192 张量
（696 而非 504），训练照常、loss 照常下降，**全程不报错**。修复方式是把配置改成
"路径限定正则字符串"——peft 对 **str** 走 `re.fullmatch`，可把范围钉死在 `model.layers.<n>.`。

本测试直接读**冻结配置里的真实 pattern**，用 peft 自己的匹配函数校验：
    - 命中语言解码层的 7 类模块；
    - 不命中视觉塔的任何模块（`visual.*`）；
    - 不命中 `embed_tokens` / `lm_head` / `norm` 等非目标模块。
若有人把配置改回裸后缀 list，本测试立刻失败——这是防复发的主要闸门。

不依赖 GPU / 模型权重（只用 peft 的匹配函数 + 架构实测常量）。

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_lora_targets.py -q``
"""

from __future__ import annotations

from pathlib import Path

import yaml
from peft import LoraConfig
from peft.tuners.tuners_utils import check_target_module_exists

#: tests/unit/test_*.py → parents[2] = DriveAlign/，配置在其 configs/ 下。
CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "sft" / "sft_smoke_1f.yaml"
)

#: Qwen2.5-VL-3B 模块全路径形态（由 `named_modules()` 在 meta device 上实测得到，
#: 非推测）：语言解码层是 `model.layers.<0..35>.`，视觉塔是 `visual.blocks.<0..31>.`。
_LANGUAGE_KEYS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.35.self_attn.q_proj",
    "model.layers.35.mlp.down_proj",
]

#: 视觉塔模块：修复后必须一个都不命中。前 3 个是"裸后缀"最易误伤的（B7 的实际泄漏项）。
_VISION_KEYS = [
    "visual.blocks.0.mlp.gate_proj",
    "visual.blocks.0.mlp.up_proj",
    "visual.blocks.0.mlp.down_proj",
    "visual.blocks.31.mlp.down_proj",
    "visual.blocks.0.attn.qkv",
    "visual.blocks.0.attn.proj",
    "visual.merger.mlp.0",
    "visual.merger.mlp.2",
    "visual.merger.ln_q",
]

#: 语言侧但不在目标之内的模块（防止 pattern 写得太宽）。
_NON_TARGET_KEYS = [
    "model.embed_tokens",
    "lm_head",
    "model.norm",
]


def _lora_config_from_frozen_yaml() -> LoraConfig:
    """按冻结配置里的 `lora.target_modules` 造一个 LoraConfig（其余字段取配置值）。"""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )


def test_frozen_config_target_modules_is_path_scoped_string():
    # 必须是字符串（peft 才会走 re.fullmatch）。写成 list 会退化成裸后缀匹配 → B7 复发。
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(cfg["lora"]["target_modules"], str), (
        "lora.target_modules 必须是路径限定正则字符串；写成 list 会因 peft 的裸后缀"
        "匹配把视觉塔的 MLP 一并命中（S11_bug_log B7）"
    )


def test_pattern_matches_all_seven_language_module_types():
    lora = _lora_config_from_frozen_yaml()
    missed = [key for key in _LANGUAGE_KEYS if not check_target_module_exists(lora, key)]
    assert missed == [], f"以下语言解码层模块未被命中：{missed}"


def test_pattern_never_matches_vision_or_non_target_modules():
    lora = _lora_config_from_frozen_yaml()
    hit = [
        key
        for key in _VISION_KEYS + _NON_TARGET_KEYS
        if check_target_module_exists(lora, key)
    ]
    assert hit == [], f"以下非语言解码层模块被误命中：{hit}"


def test_bare_suffix_list_would_have_leaked_into_vision():
    # 固化根因（B7）：裸后缀 list 会命中视觉塔的 MLP。此断言同时证明上面那个
    # "不命中"用例并非恒真——它确实在区分两种配置。
    leaky = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    assert check_target_module_exists(leaky, "visual.blocks.0.mlp.gate_proj")
    assert not check_target_module_exists(leaky, "visual.blocks.0.attn.qkv")
