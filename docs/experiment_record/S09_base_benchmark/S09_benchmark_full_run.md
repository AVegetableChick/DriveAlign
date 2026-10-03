# S09 Benchmark 记录：1F 全量推理（test 5,468）

> 状态：**运行中/待回填**。本文档预置于推理启动前，`结果记录`区在跑完后由 agent 核验回填。
> 配置：`configs/benchmark/base_1f.yaml`（frozen，sha256 登记见规划 §4）；contract v5-face；greedy 解码。

## 命令（tmux 内直接执行）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
cd /root/autodl-tmp/drivealign_workspace && \
set -o pipefail && \
PYTHONPATH=DriveAlign/src python -m drivealign.cli.base_benchmark \
  --config DriveAlign/configs/benchmark/base_1f.yaml \
  --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \
  --dataset-root data/dataset_v4 --nuscenes-dataroot data/nuscenes/trainval \
  --resume \
  --out runs/S09_base_benchmark/predictions_full.jsonl 2>&1 | tee runs/S09_base_benchmark/full.log
```

## 目标边界

- 范围：`data/dataset_v4` anchor manifest 中 **test split 全部 5,468 anchors**，无 `--limit`、无 `--anchors-file`；
- 输入政策：ONE_FRAME（仅 frames[-1] + backward-difference ego speed）；checkpoint `models/Qwen2.5-VL-3B-Instruct`（未微调 Base）；
- 产出：`runs/S09_base_benchmark/predictions_full.jsonl`（每 anchor 一行：sample_token、request_hash_1f 重算值、raw_text、parse status、结构化预测、telemetry）。

## 执行约定

- 前置 gate：pilot（`--limit 32`）先跑通并确认吞吐/显存/parse 分布正常后再放全量；
- `--resume` 幂等：中断后重跑同一条命令即续跑（按文件内已有 token 跳过）；
- `| tee` 前必须 `set -o pipefail`；只跑本命令不连跑反事实（GPU 串行，反事实单独跑，见 `S09_benchmark_counterfactual.md`）；
- 输出文件与 pilot 文件严格分离（resume 语义按文件内 token 计）。

## 结果记录（跑完后回填）

- [ ] 起止时间 / 总耗时：
- [ ] 吞吐（anchors/s）与显存峰值：
- [ ] coverage 对账：anchors_scored = 5,468；missing_predictions / extra_predictions = 0（evaluator 核验）；
- [ ] parse 分布：parse_rate、parse_error_counts 按 S03 taxonomy 分列；
- [ ] telemetry 完整性：抽样 telemetry 非空、request_hash_1f 与 manifest 一致；
- [ ] 产物路径确认：predictions_full.jsonl 行数 = 5,468。

## 验收 gates（进入 Step 5 的前提）

1. 行数与 token 集合与 test manifest 逐一对应（无缺无多）；
2. 单样本失败未中止批处理（generation_failure 降级行存在则记录其数量）；
3. `--resume` 重跑一遍末尾 10 token 无新增行（幂等性抽查）。

## 性能观察与未来推理加速手段（2026-10-03 备案，不影响本轮冻结配置）

**pilot 实测**（32 anchors）：latency p50 5.13s / p90 7.37s / 均值 5.59s；显存峰值 7.94 GB；input_tokens p50 2,335，output_tokens p50 ~174。瓶颈在 **decode**（输出 ~174 token 的逐 token 解码，显存带宽受限），prefill 占比小；flash_attention_2 与 bfloat16 已在 base_1f.yaml 启用（低垂果实已摘）。按 5.6s/anchor 估算：全量 ≈ 8.5h，反事实 2×200 ≈ 37min。

**纪律约束**：base_1f.yaml 为预注册冻结配置（greedy/dtype/分辨率钉死），**本轮 benchmark 不得中途更换推理路径**，否则毁掉预注册与 S09↔S12 可比性。加速属"推理基建 v2"，立项前提是 parity gate（抽 N anchor 验证新旧路径输出逐值一致）+ 预注册。

**候选路线**（按收益/侵入性排序）：

| 路线 | 预期收益 | 侵入性 | 备注 |
|---|---|---|---|
| batch 推理（4–8 帧/批） | 2–4× | 中 | HF left-padding + 变长图像；显存余量充足（现峰值 7.9GB） |
| vLLM/SGLang 推理 | 5–10× | 大 | Qwen2.5-VL 已支持；continuous batching + PagedAttention；greedy 语义等价但数值路径不同，必须过 parity gate |
| int4 量化（AWQ/GPTQ） | 2–3× | 改输出 | 与冻结 benchmark 不兼容，仅限非基准场景，不进主线 |
| 降分辨率 / 砍 prompt | — | 改模型输入 | 毁可比性，不做 |
| speculative decoding | 视 draft 模型 | 大 | 对 3B 目标模型复杂度收益比差，暂不列 |
