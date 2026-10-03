# S09 实施规划（单帧未微调 Base Benchmark，2026-10-03）

> 状态：规划草案，待用户拍板（§7）。本阶段依据 `docs/project_stages/09_single_frame_base_benchmark.md`。
> 链路裁定（已与用户确认）：执行顺序 S09 → S11（仅 1F 半边）→ S12；S10 与 S11-4F 后置，S09 承担的预注册义务见 §4 Step 2。
> 执行环境约定：沿用 S08——命令先 `conda activate autovla_codeclean`；长跑（GPU 推理）用 tmux 且 `| tee` 前必加 `set -o pipefail`；物理盘（/root/autodl-tmp/datasets）写入只能由用户执行，workspace 内 `runs/` 写入可由 agent sandbox 执行。

## 0. 前置快照（Step 0）

未提交：本规划文档。先提交一次规划快照再开工代码，回滚边界清晰：

```
git add docs/experiment_record/S09_base_benchmark
git commit -m "[Docs] S09 implementation plan (single-frame base benchmark)"
```

## 1. 现状基线（已核实，执行前无需重查）

- **推理栈已就绪**（S02/S03/S04）：`inference/structured_runner.py`——`SampleRequest(image, available_speed)` → 冻结 prompt → `generate_one` → 严格契约解析；批处理单样本失败永不中止（generation_failure 降级记录）；telemetry 含 input/output tokens、延迟、峰值显存。`make_blank_image` / `make_shuffled_image` 反事实助手已实现并有单测。
- **1F 输入政策已实现**（S06）：`records/serializer.py` `serialize(ModelInputs, InputPolicy.ONE_FRAME)` 只取 `frames[-1]`；`request_hash_1f` 可从 record 重算。
- **数据已冻结**（S08）：`data/dataset_v4`（contract v4，GT 全回填）。test = 5,468 anchors；迭代入口 `anchor_policy_manifest.json` 字段：token → split/shard/line/record_hash/request_hash_1f/request_hash_4f/scene_token/anchor_image_relpath（dataroot 相对路径，dataroot = `data/nuscenes/trainval`）。GT = record 内 `training_targets.expected_output`（critical_objects / risk_factors / reasoning / yield_required / speed_action），评测与训练同源同规则，M09 评分无需 nuScenes 原始表。
- **Base checkpoint**：`models/Qwen2.5-VL-3B-Instruct`（S01 加载链路可用）。
- **speed 字符串格式沿用现有约定**：`"5.200 m/s"` 样式（`%.3f`，见 `test_records_serializer.py`），写入 base_1f.yaml 冻结。
- **M09 evaluator 未构建**：`evaluation/` 仅有 `projection.py` + `box_render.py`；S08 实际交付为 GT 回填侧（`gt/` 包）。**M09 v1 是本阶段最大新增工作项（§2.2）**。
- **模型面裁定（2026-10-03 用户拍板）：prompt 必须与 7 项 schema 对齐，以 contract v5 承载，v4 冻结不动**。背景：v4 `prompt.txt` 逐字节继承 v3，risk_factors 措辞仍列 8 项（含 small_following_gap），v4 `output_schema.json` 仅 7 项——Base 被prompt教唆输出 8 项却被 v4 parser 拒绝，等于主动制造 parse 失败。落地机制：**不可原地改 v4 prompt**——request hash 覆盖 prompt 文本（`versions.py`：HASH_FACE_FILES 含 prompt.txt），原地改会使冻结 v4 资产的 request_hash 列永久不可复现（dataset_verify 必死）；v5 = v4 八文件 + 修正版 prompt（7 项），`MODEL_FACE_LINEAGE["v5"]="v5"`（自有模型面）、`DEFAULT_CONTRACT_VERSION="v5"`。连带后果：dataset_v4 记录的 GT/model_inputs 与 prompt 无关，**不重建**；冻结 v4 anchor manifest 的 request_hash 列成为 v3-face 历史值（S08 三 gate 的 PASS 记录不受影响——当时 v3/v4 同 face），S09 起 gate 1 改对 **v5-face manifest**（`drivealign.dataset.manifest` 从冻结 shards 派生；2026-10-03 用户裁定：**替换** `data/dataset_v4/anchor_policy_manifest.json` 为 v5-face 版，原 v3-face 版归档至 `runs/S08_gt_backfill/anchor_policy_manifest_v3face_frozen.json`——数据集目录承载最新权威入口表，face 时代以文件内 `config.contract_version` 字段与归档文件分账）；新增 prompt↔schema 枚举一致性 pin 测试，防同类漂移再发。

## 2. 模块拆分

> 步骤切分约定（2026-10-03 用户裁定）：**contract v5 落地独立成 Step 1、先于 M09 与运行器（Step 2）**（§4）。理由：v5 是本阶段唯一触碰全局 request hash 语义的变更（模型面 v3→v5 切换、prompt 字节级修正），先单独落地、单独验证、单独提交形成独立回滚边界，再叠加 M09 与运行器代码，合约变更风险与评测代码风险互不纠缠。

### 2.1 contract v5 落地（Step 1，先于 M09）

| 文件 | 职责 |
|---|---|
| `configs/contracts/v5/` | **prompt 修正代际**：v4 八文件复制 + `prompt.txt` 修正（risk_factors 措辞 8→7 项，删 small_following_gap 行；其余逐字节不变） |
| `src/drivealign/contracts/versions.py` | 增 v5：`AVAILABLE_CONTRACT_VERSIONS += ("v5",)`、`DEFAULT_CONTRACT_VERSION="v5"`、`MODEL_FACE_LINEAGE["v5"]="v5"` + docstring 沿革 |
| `data/dataset_v4/anchor_policy_manifest.json`（替换为 **v5-face**） | **v5-face anchor manifest**：以 `drivealign.dataset.manifest` 从冻结 shards 派生（DEFAULT=v5 自动 stamp v5 request hash），**替换**数据集内 v3-face 原版（原版归档 `runs/S08_gt_backfill/anchor_policy_manifest_v3face_frozen.json`；2026-10-03 用户裁定，物理盘替换由用户 tmux 执行）；S09 gate 1 的比对基准，派生日志与 scratch 留存于 `runs/S09_base_benchmark/`。派生需加载 nuScenes trainval（10–15GB RAM）：S08 有 sandbox 跑通先例，OOM 则转用户 tmux |
| pin 单测（§2.4 首条） | prompt↔schema 一致性 + v5 继承完整性 + v4 资产零改动，随 Step 1 交付 |

### 2.2 新增 M09 v1（`src/drivealign/evaluation/` 扩展，Step 2）

| 文件 | 职责 |
|---|---|
| `evaluation/matching.py` | normalized category + bbox IoU + **类内贪心**一对一匹配（2026-10-03 用户裁定，替代 stage 08 文档原拟 Hungarian；理由：≤8×8 规模下与全局最优数值差异可忽略、类内匹配天然禁跨类别使"类别错误=FP+FN"语义更干净、δ_cat 参数消除、无 scipy 依赖；决策台账留痕）。贪心顺序：IoU 降序 → bbox 字典序 tie-break（与预测输出顺序无关，确定性可复现）；IoU < τ_min 的对不允许配对；同 GT 只配一次，落选预测计 FP |
| `evaluation/metrics.py` | 逐 anchor 分数 + 聚合指标（§3 全表键名逐一实现）；所有分母显式落报告；含 `max_items_hit_rate`（S03-Q5 监控项）与 `overconservative_*` 两键（§3.1） |
| `evaluation/bootstrap.py` | scene-cluster paired bootstrap（cluster = scene_token；重采样次数 B 与种子预注册），输出 point estimate + 95% CI + scene 数 + denominator |
| `evaluation/evaluator.py` | M09 入口：读 predictions JSONL + 经 anchor manifest 定位 dataset_v4 shard records（GT 侧）→ 指标 → `m09_report.{json,md}` + `anchor_scores.jsonl`（按 scene 配对所需的逐 anchor 分数）；`m09_version` + config sha256 落档；全程只读 |
| `cli/m09_evaluate.py` | CLI 封装（`--predictions --anchor-manifest --dataset-root --config --out`） |

### 2.3 新增 benchmark 运行器与配置（Step 2）

| 文件 | 职责 |
|---|---|
| `cli/base_benchmark.py` | test manifest → 逐 anchor 读 shard record → `serialize(ONE_FRAME)` → `SampleRequest`（image = dataroot/anchor_image_relpath；available_speed 按冻结格式）→ 逐条推理 → `predictions.jsonl`：sample_token、request_hash_1f 重算值、input-policy metadata、raw_text、parse status + 错误分类、结构化预测、telemetry。支持 `--resume`（按 sample_token 幂等续跑）、`--anchors-file`（pilot 与反事实子集复用同一入口）、`--image-transform blank|shuffled`（反事实模式，产物写独立文件，不污染主预测） |
| `configs/benchmark/base_1f.yaml` | 冻结运行配置：checkpoint 路径、contract_version=v5、policy=ONE_FRAME、resolution、greedy decoding（do_sample=false、temperature=0、max_new_tokens=512——沿 S03 截断教训）、seed、dtype、speed 格式 |
| `configs/benchmark/base_4f.yaml` | **只预注册不运行**：与 base_1f 共享全部字段、仅 policy/输入政策不同；hash 在 Step 2 登记进实验记录，履行 stage 09 gate"Stage 10 配置 hash 已在本阶段运行前登记" |
| `configs/evaluation/m09_v1.yaml` | 指标口径 + 匹配参数 + bootstrap 参数；status provisional → frozen（Step 5 冻结后才有正式报告） |

### 2.4 新增单测（tests/unit/）

- **prompt↔schema 一致性 pin（Step 1）**：v5 prompt.txt 枚举的 risk_factors 项集合 == v5 output_schema.json risk enum == risk_taxonomy.json（7 项，无 small_following_gap）；v5 其余五文件与 v4 逐字节一致；v4 资产未被改动（防回归）。
- `test_base_benchmark_requests`（Step 2）：record → SampleRequest 的 1F 政策正确性（仅 frames[-1] + ego_speed）、speed 格式、request_hash_1f 重算值与 v5-face manifest 一致、future fingerprint 零命中（复用 `serializer.scan_future_fingerprints`）。
- `test_eval_matching`（Step 2）：空集/重复框/IoU 平手 tie-break/跨类别不配对/τ_min 边界/同 GT 只配一次。
- `test_eval_metrics`（Step 2）：合成预测 × 合成 GT 的逐指标数值与分母断言；over-conservative；maxItems 触顶计数；parse 失败样本的排除语义。
- `test_eval_bootstrap`（Step 2）：scene cluster 配对差 CI 的确定性（固定种子逐值断言）。
- 确定性（Step 2）：同一 predictions 两次评测，`m09_report.json` 逐字节一致。

## 3. M09 v1 指标口径（§7 待拍板）

主指标均报告 point estimate + 95% CI（scene-cluster paired bootstrap）+ scene 数 + denominator。GT 一律取自 record 的 `expected_output`；"parse_ok 集" = parse 成功 anchors。**命名约定：下表"报告键名"即 `m09_report.json` 中的 key，键名直接编码统计量与聚合方式（`<字段>_<统计量>[_<聚合>]`），名称与实现一一对应。**

| 报告键名 | 分母 | 计算方式 |
|---|---|---|
| `parse_rate`，`parse_error_counts` | 全部 5,468 | parse_ok 帧占比；错误按 S03 taxonomy 分列，`small_following_gap` 触发的 schema 违规仍单列观察（v5 prompt 已不再教唆该项，若仍出现则为 Base 先验行为） |
| `object_precision_micro@τ` / `object_recall_micro@τ` / `object_f1_micro@τ` | micro：parse_ok 全体预测框与全部 GT 池化 | 类内贪心匹配后计 TP/FP/FN，跨类别合并池化；τ ∈ {0.3, 0.5, 0.7} 全报，主指标 τ=0.5；跨类别不配对，类别错误即 FP+FN（§2.2） |
| `object_f1_macro@τ`（附 per-category F1 表） | macro：类别均值 | 上行匹配结果按类别分别计 F1 后等权平均；仅计入 GT 非空类别（类别数落报告） |
| `matched_iou_p50` / `matched_iou_p90` / `matched_iou_mean` | 配对集 | matched 对的 IoU 分布 |
| `motion_state_accuracy`，`motion_state_confusion` | 配对集 | matched 对上 motion_state 相等占比 + 四枚举混淆矩阵 |
| `speed_action_f1_macro`，`speed_action_confusion` | parse_ok | 4 类 per-class F1 等权平均（macro）+ 混淆矩阵 |
| `yield_required_f1` | parse_ok | 二类 F1，正类 = true |
| `risk_factors_f1_per_class`（7 键），`risk_factors_f1_macro` | parse_ok | 每类 one-vs-rest：GT 集与预测集都含该类 = TP、仅一方含 = FN/FP；7 类 F1 各自报告 + macro 汇总（set 语义与 S08 双向精确口径对齐） |
| `max_items_hit_rate` | parse_ok | 预测对象数 == maxItems(8) 的帧占比（S03-Q5 监控） |
| `overconservative_yield_rate` / `overconservative_stop_rate` | parse_ok | 见 §3.1；本阶段仅报告，退化阈值在 S12 预注册 |
| `visual_dependence_action_flip_blank` / `_shuffled`，`visual_dependence_output_change_blank` / `_shuffled` | 预注册反事实子集 | 见 §3.1；诊断不 gate（Base 内容质量低可接受） |
| `latency_p50` / `latency_p90`、`throughput_anchors_per_s`、`visual_tokens_p50`、`peak_cuda_mem_bytes`、`gpu_hours` | 全部 | 成本摘要（§3 表原"成本摘要"行拆键） |

### 3.1 两个诊断指标的定义与动机

**over-conservative（过度保守代理）**——防"用怂换分"。一个把什么都判成要刹车/要让行的模型在安全类指标上好看但对驾驶无用。两个比率：

- `overconservative_yield_rate`（假让行率）：GT `yield_required=false` 而预测 true 的帧占比；
- `overconservative_stop_rate`（假急停率）：GT `speed_action ∈ {KEEP_SPEED, ACCELERATE}` 而预测 STOP 的帧占比。

叫"代理"指它们不是权威基准的官方定义，而是"过度保守"这一失效模式的直接可算代理。背景：S08 冒烟显示 STOP 帧 85% 的 yield=false（停车与让行是独立行为）；SFT 最易学到的捷径即"拿不准就 STOP/让行"。S09 只建立基线读数；S12 gate"over-conservative 未越过退化阈值"以 S09 读数为参照，防 SFT 退化。

**visual dependence（视觉依赖）**——检验"输出是否真的看了图"。反事实法：预注册子集上把输入图替换为 blank（全灰，摧毁全部场景内容）与 shuffled（patch 打乱，保留纹理摧毁布局）两版再推理，与正常输出比较：

- `visual_dependence_action_flip_*`：换图后 speed_action 翻转的帧占比；
- `visual_dependence_output_change_*`：换图后结构化输出任一字段变化的帧占比。

解读：变化率 ≈ 0 → 模型未使用图像（纯文本先验 + 速度字段"背答案"）；变化率高 → 输出由图像内容驱动。Base 阶段仅记录；S12 gate"visual dependence 未越过退化阈值"防 SFT 把模型训得更不看图、更贴标签先验。沿袭 S03/S04 blank/shuffled sanity check 传统（`make_blank_image`/`make_shuffled_image` 已有实现与单测）。

不在 v1 范围：temporal dependence（1F 无历史输入，归 S10）；per-prediction observability/unscorable（GT 池已在 S08 构建时可见性门控，评测侧无该输入；如需挂回须评测时访问 nuScenes，留决策台账）。

## 4. 执行编排

| Step | 内容 | 执行者/环境 |
|---|---|---|
| 0 | git 提交本规划（§0） | agent（用户确认后） |
| 1 | **contract v5 落地（§2.1，独立 step，先于 M09；2026-10-03 用户裁定）**：configs/contracts/v5 八文件（v4 复制 + prompt 8→7 修正）+ versions.py（DEFAULT=v5、MODEL_FACE_LINEAGE v5→v5）+ prompt↔schema pin 单测 + **v5-face anchor manifest 派生并替换 dataset_v4 内 v3-face 版**（原版归档 `runs/S08_gt_backfill/anchor_policy_manifest_v3face_frozen.json`；物理盘替换由用户 tmux 执行）+ 单测全绿 → git commit（代码与单测，独立回滚边界） | agent，sandbox（manifest 替换：用户 tmux） |
| 2 | base_1f/base_4f 配置预注册（4f 只登记 hash）+ benchmark 运行器（§2.3）+ m09_v1 配置骨架（provisional）+ M09 包（§2.2）+ 单测全绿 → git commit | agent，sandbox |
| 3 | pilot：预注册反事实子集抽样器先产出子集清单 → 取前 32 anchors 试跑（GPU），报告吞吐/显存/parse 分布/telemetry 完整性 → **用户确认全量排期** | 用户 tmux（GPU） |
| 4 | 全量 test 推理（5,468）+ 反事实子集（blank/shuffled 各一遍，独立文件） | 用户 tmux |
| 5 | m09_v1 冻结（provisional → frozen）→ M09 评测 + bootstrap CI + 成本摘要 → gates（§5） | agent，sandbox |
| 6 | 实验记录落档（决策台账 + 参数冻结 + 阶段关闭）+ git commit | agent |

Step 1 与 Step 2 各自以 commit 收口：v5 属合约代际变更（全局 request hash 语义），与 M09/运行器代码分开提交，任一步出问题可独立回滚而不牵连另一边。

### Step 2 交付物与 hash 登记（2026-10-03，履行 gate 4 预注册义务）

| 产物 | sha256 | 状态 |
|---|---|---|
| `configs/benchmark/base_1f.yaml` | `1aab7db306b17f3688587e5d6006a98bb6a3fcc16326c26b8c8a2066d1332cb4` | frozen，Step 4 将以此运行 |
| `configs/benchmark/base_4f.yaml` | `793db1beee23f26b4dfc7ad92f425ae7c34e2908133b92156b51bdb5812d5e90` | **仅预注册，S09 全程不运行**（gate 4；Stage 10 起才可执行） |
| `configs/evaluation/m09_v1.yaml` | `e9312af942e71950e96688e4db1c6ae03b9ca208383aaa97cf02ee3825cf36a0` | provisional；Step 5 冻结（provisional→frozen）后**须重新登记 hash**，正式报告以冻结版 hash 为准 |

Step 2 新增代码：`evaluation/{matching,metrics,bootstrap,evaluator}.py`、`cli/{base_benchmark,m09_evaluate}.py`、单测 5 个文件（matching 13 / metrics 16 / bootstrap 8 / evaluator 5 / runner 9）。全量单测 250 passed（含既有 193 条全绿）。

运行器关键防线（实现于 `cli/base_benchmark.py`）：逐 anchor 断言重算 request hash == v5-face manifest 存储值，且 runner 侧 prompt 与 serializer 侧 prompt 逐字节一致——合约 face 错位会在推理时即刻报错而非静默产出错误基准。已登记注意点：M09 报告 `input_tokens_p50` 承载规划原 `visual_tokens_p50` 键（S02 telemetry 无图像 token 单独计数，拆分需动冻结栈，不为之破坏；决策台账 Step 6 复核）。

标准命令模板（Step 4 示例）：

```
tmux new-session -d -s s09_bench \
  'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
   cd /root/autodl-tmp/drivealign_workspace && \
   set -o pipefail && PYTHONPATH=DriveAlign/src python -m drivealign.cli.base_benchmark \
   --config DriveAlign/configs/benchmark/base_1f.yaml \
   --anchor-manifest data/dataset_v4/anchor_policy_manifest.json \
   --dataset-root data/dataset_v4 --nuscenes-dataroot data/nuscenes/trainval \
   --out runs/S09_base_benchmark/predictions.jsonl 2>&1 | tee runs/S09_base_benchmark/infer.log'
```

新程序 docstring 必须含 example launch command（沿 build_dataset.py 风格）。

## 5. Gate 清单（Step 5）

1. **一一对应**：predictions 与 test anchor manifest 逐 token 对应（5,468，零缺失零重复），且逐条 request_hash_1f 重算值 == **v5-face manifest**（`data/dataset_v4/anchor_policy_manifest.json`）存储值——证明推理用的正是冻结请求。原 v3-face 版已归档（`runs/S08_gt_backfill/anchor_policy_manifest_v3face_frozen.json`），不作为本 gate 基准（§1）。
2. **可复算**：固定脚本由 predictions.jsonl 重跑 M09，报告与正式报告逐字节一致（指标层确定性；生成层不要求逐字节复现，见 §8）。
3. **反事实控制**：blank/shuffled 结果可重现，产物与主预测文件物理分离。
4. **预注册义务**：base_4f.yaml hash 已在 Step 2 登记于实验记录，全程未运行。
5. **输入政策审计**：1F 请求仅含 frames[-1] + ego_speed；future fingerprint 扫描零命中。
6. **CI 完整性**：全部主指标带 point estimate + 95% CI + scene 数 + denominator。
7. **溯源**：checkpoint 路径与 commit、contract v5 模型面标识、m09_version、各 config sha256 落档。
8. **prompt↔schema 一致性**：v5 prompt 枚举与 schema/taxonomy 7 项一致（pin 测试通过）；v4 资产零改动（冻结完好）。

## 6. 交接（与 S11/S12）

- 冻结 `Base-1F` 配置、predictions、指标、成本到 `runs/S09_base_benchmark/`；S12 以其全部主指标为对比基线（paired CI 同 anchor 同 scene 配对）。
- S11/S12 一律运行 contract v5 模型面（训练侧 prompt 与 S09 评测 prompt 同源）；dataset_v4 记录仍是数据源（GT 与 prompt 无关）。
- S11 只执行 1F 半边（32 条 train/save/reload + 128 条 overfit）；4F smoke 后置至 S13 开工前补跑，S11 实验记录须标注"1F closed / 4F deferred"。
- S10 待 S12 后启动，只允许加载本阶段预注册的 base_4f.yaml。

## 7. 拍板记录（2026-10-03，用户已确认）

1. §0 规划快照提交：**已批准，本提交即快照**；
2. §3 指标口径整体批准：主报告 IoU=0.5、`overconservative_*`/`visual_dependence_*` 定义（§3.1）、类内贪心 + tie-break（IoU 降序 → bbox 字典序）、命名约定（键名 = 实现口径）；
3. 反事实子集：**N=200、blank+shuffled 各一遍、按 scene 分层固定种子**（用户在必要性讨论后确认维持原提案）；
4. pilot 规模：**32 anchors**；全量排期待 pilot 吞吐数据；
5. 产物目录：**`runs/S09_base_benchmark/`**；
6. prompt/schema gap：**用户裁定必须修正 prompt 与 7 项 schema 对齐**，以 contract v5 承载（机制与后果见 §1；v4 冻结不动、数据集不重建、S08 历史gate 记录不受影响）。
7. 执行编排切分（2026-10-03 追加裁定）：**contract v5 落地独立成 Step 1、先于 M09 与运行器（Step 2）**，两步各自 git commit 形成独立回滚边界。
8. v5-face manifest 位置（2026-10-03 追加裁定）：**替换** `data/dataset_v4/anchor_policy_manifest.json` 为 v5-face 版（用户裁定：数据集目录应承载最新权威入口表，拒绝"最新 manifest 在 runs/"的布局；物理盘替换由用户 tmux 执行）；原 v3-face 版归档 `runs/S08_gt_backfill/anchor_policy_manifest_v3face_frozen.json`；连带更新 §1/§2.1/§4/§5/§8。

## 8. 风险与已知限制

- **生成层非确定性**：greedy + 固定 seed + 固定 dtype 尽力保证；GPU kernel（attention/TF32）客观存在非确定性，硬 gate 定在指标可复算（Gate 2）而非生成逐字节复现；torch/cuda 版本随成本摘要落档。
- **吞吐未知**：5,468 × 反事实子集的墙钟时间由 pilot 实测后定排期，不预估。
- **M09 v1 指标缺口**：§3 "不在 v1 范围"两项，均已在决策台账留痕，非静默丢弃。
- **后置风险**：4F 显存可行性在 S11-4F 前未验证（已知后置风险，S12 前不阻塞）。
- **face 时代分账**：contract v5 生效后，request hash 存在两套现值加一套归档——v3-face：归档文件 `runs/S08_gt_backfill/anchor_policy_manifest_v3face_frozen.json`（S08 冻结版原样保留）与 dataset_v1 内 manifest（dataset 审计用）；v5-face：`data/dataset_v4/anchor_policy_manifest.json`（2026-10-03 起替换，benchmark/training 权威入口表，文件内 `config.contract_version="v5"` 自标识）。任何未来 parity 审计必须先声明 face 时代，禁止跨代比对 request hash。**已知后果**：dataset_v4 的逐字节确定性重跑（dataset_verify）基线由归档文件承载——若在 v4 时代 commit 下重跑，将在 anchor_policy_manifest.json 一项上报字节差（预期，非资产损坏；以归档文件对照即可恢复原审计结论）。v5 manifest 确定性可复现：重派生须钉住 `--builder-commit 41810e5`。
