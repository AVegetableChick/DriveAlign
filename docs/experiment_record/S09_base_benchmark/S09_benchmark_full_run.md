# S09 Benchmark 记录：1F 全量推理（test 5,468）

> 状态：**已完成（2026-10-04）**，结果区已回填；gates 1/2 PASS，gate 3 见下方说明。
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

## 结果记录（2026-10-04 回填）

- [x] 起止时间 / 总耗时：2026-10-03 晚 → 2026-10-04 08:35，`wall_seconds = 30,609.6`（≈8.5 h，与 pilot 估算一致）；
- [x] 吞吐与显存：5,468 / 30,609.6 s = **0.179 anchors/s**；latency p50 = 4.98 s / p90 = 7.87 s / mean = 5.52 s；峰值显存 **7.94 GiB**（input_tokens p50 = 2,335，output_tokens p50 = 164 / p90 = 266）；
- [x] coverage 对账：rows = 5,468，unique = 5,468，dup = 0；与 test manifest 逐 token 对账 **missing = 0 / extra = 0**；config sha256 = `1aab7db3…1332cb4` 与登记值一致；
- [x] parse 分布：**parse_ok = 5,449/5,468（99.65%）**；错误按 S03 taxonomy：`json_parse_failure` 14、`schema_failure` 4、`semantic_range_failure` 1、`generation_failure` 0；14 条 output_tokens 触顶 512，其中 13 条即截断型 json_parse 失败——截断是主导失败模式（0.24%），与 S03 教训一致（max_new_tokens=512 为冻结值，保持不动）；
- [x] telemetry 完整性：5,468/5,468 telemetry 非空；`request_hash_match` **5,468/5,468 全 true**（v5-face manifest 逐值一致）；`record_hash`/`request_hash` 均逐行落档；
- [x] 产物路径确认：`runs/S09_base_benchmark/predictions_full.jsonl`（24.8 MB）+ `run_summary_predictions_full.json`。

## 初步行为观察（非 gate，Base 先验画像）

- `speed_action`：DECELERATE 4,724（86.7%）/ KEEP_SPEED 716（13.1%）/ STOP 9（0.2%）/ **ACCELERATE 0**；
- `yield_required`：true 4,524（82.9%）；
- `risk_factors` 帧计数：oncoming_traffic 3,618（66%）占绝对主导，corridor_conflict 915、pedestrian_crossing 904、lead_vehicle_braking 154、stationary_obstacle 72、congestion 121、vehicle_merging 63；
- 对象数/帧 mean = 2.30，max_items 触顶率 2.50%。
- 解读：Base 极度保守（近全减速 + 82.9% 让行 + 零加速），`overconservative_*` 两个代理指标预计读数很高——这正是 S09 要建立的基线参照，方向符合"未对齐 Base"预期，留待 Step 5 正式报告量化。

### 感知-动作解耦的行为学证据（2026-10-05 补充，主跑内部一致性分析）

- **risk_factors 字段饱和**：5,449 帧 parse_ok 输出**全部**至少报一个风险因子（"无任何风险"组为空集）——该字段在 Base 输出中零区分度，"报风险"是格式默认而非判断；
- **自报风险不改变动作**：模型自报行人/走廊冲突风险的帧（1,765 帧）中 action 分布 88.9% DECELERATE / 10.9% KEEP_SPEED，与全体（86.7% / 13.1%）几乎无差——模型自己输出的"危险"不影响它自己的动作；
- **yield↔action 强耦合但属同源先验**：yield=true 的 4,524 帧中 99.8% 配 DECELERATE/STOP，yield=false 中 77.2% 配 KEEP_SPEED——内部一致性极高，但结合上两条，这是"恒开谨慎模式"的两个出口，而非"感知→风险→动作"推理链；
- **结论**：感知是装饰性的——模型读图（critical_objects 100% 随图变，见反事实记录）、每帧报风险，但动作决策绕过感知内容由先验驱动。反事实（action 83–85% 不随图变）与主跑内部一致性两条独立证据链互证。SFT 核心任务即把 action 从"先验恒开"变为"由感知条件化"；S12 可复查探针：risk_factors 是否仍饱和、自报风险帧的 action 分布是否分化、action_flip 是否上升。

## 验收 gates（进入 Step 5 的前提）

1. 行数与 token 集合与 test manifest 逐一对应（无缺无多）——**PASS**（missing=0/extra=0）；
2. 单样本失败未中止批处理——**PASS**（generation_failure = 0；19 条 parse 失败均降级记录且批处理继续）；
3. `--resume` 重跑末尾 10 token 无新增行（幂等性抽查）——**待办**：需 GPU 重载模型，合并到下次任何 resume 运行时顺带核验（观察 `anchors_done_before`/`anchors_new` 计数即可），不阻塞 Step 5。

