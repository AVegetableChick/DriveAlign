# S09 备案：未来推理加速手段

> 状态：**备案**（2026-10-03），S09 冻结配置不动；加速属未来"推理基建 v2"立项范围。
> 关联：`configs/benchmark/base_1f.yaml`（frozen）、`S09_benchmark_full_run.md`（性能实测）、S12 评测编排。

## 性能画像（pilot 实测，32 anchors）

- latency p50 5.13s / p90 7.37s / 均值 5.59s；显存峰值 7.94 GB；
- input_tokens p50 2,335，output_tokens p50 ~174；
- 瓶颈在 **decode**：~174 输出 token 的逐 token 解码受显存带宽限制，prefill（2,335 token）占比小；
- flash_attention_2 与 bfloat16 已在 base_1f.yaml 启用（低垂果实已摘）；
- 外推：全量 5,468 anchors ≈ 8.5h；反事实 2×200 ≈ 37min。

## 纪律约束

base_1f.yaml 为预注册冻结配置（greedy / dtype / 分辨率钉死）。**本轮 benchmark 不得中途更换推理路径**，否则毁掉预注册与 S09↔S12 可比性。任何加速手段落地前必须：

1. **parity gate**：抽 N 个 anchor 验证新旧路径输出逐值一致（JSON 层面）；
2. **预注册**：新推理配置（batch 大小 / 引擎版本 / 内核选项）作为独立配置文件登记 hash 后方可用于正式评测。

## 候选路线（按收益/侵入性排序）

| 路线 | 预期收益 | 侵入性 | 备注 |
|---|---|---|---|
| batch 推理（4–8 帧/批） | 2–4× | 中 | HF left-padding + 变长图像处理；显存余量充足（现峰值 7.9GB）；decode 为带宽受限，batch 摊薄权重读取即主要收益来源 |
| vLLM / SGLang 推理 | 5–10× | 大 | Qwen2.5-VL 已支持；continuous batching + PagedAttention；greedy 语义等价但数值路径不同，必须过 parity gate |
| int4 量化（AWQ/GPTQ） | 2–3× | 改输出 | 与冻结 benchmark 不兼容（模型权重变化即输出变化），仅限非基准场景，不进主线 |
| 降分辨率 / 砍 prompt | — | 改模型输入 | 改变模型输入分布，毁可比性，不做 |
| speculative decoding | 视 draft 模型 | 大 | 3B 目标模型收益比差，暂不列 |
| prompt 前缀缓存 | 有限 | 小 | 各 anchor prompt 仅 speed 行与图像不同，共享前缀短，收益有限 |

## 建议节奏

- S09/S12 v1 全部按现栈跑完（总 GPU 时 ≈ 9h 量级，可过夜）；
- 若 S12 之后出现规模化评测需求（多系统对比 × 反事实扩量），以"推理基建 v2"立项：batch 化优先（侵入最小），vLLM 作第二阶段，均带 parity gate 与预注册。
