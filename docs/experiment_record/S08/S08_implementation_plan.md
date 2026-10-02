# S08 实现规划（2026-10-02）

> 状态：规划定稿，待执行。规则来源 = [S08_pre_implementation_decisions.md](S08_pre_implementation_decisions.md)（P1–P4 已冻结；P1 剩余数值归 P1.2 train-only 标定）。本文件是执行编排的唯一入口：新会话按 §5 顺序执行，遇规则歧义以决策台账为准。
> 执行环境约定：所有命令先 `conda activate autovla_codeclean`；长跑（标定、全量重建）用 tmux；`| tee` 前必加 `set -o pipefail`；物理盘（/root/autodl-tmp/datasets）写入只能由用户在 tmux 执行，workspace 内 runs/ 写入可由 agent sandbox 执行。

## 0. 前置快照（Step 0）

未提交：`docs/experiment_record/S08/`（决策台账 + 本规划）+ `docs/codebase/05_DriveAlign_Project_Structure_and_Execution_Blueprint_v1.md`（observability 三态修订）。先提交一次决策快照再开工代码，回滚边界清晰：

```
git add docs/experiment_record/S08 docs/codebase/05_DriveAlign_Project_Structure_and_Execution_Blueprint_v1.md
git commit -m "[Docs] S08 pre-implementation decisions and implementation plan"
```

## 1. 现状基线（已核实，执行前无需重查）

- **代码**：`src/drivealign/records/`（record.py：`training_targets = {expected_output, language_reference}`，`expected_output: dict | None`；adapter.py：`build_record` 唯一构造路径）、`dataset/build_dataset.py`（S07 builder：scene manifests → 分片，8 码 quarantine，SHARD_SIZE=1000，MIN_PREVS=3）、`evaluation/projection.py`（8 角点投影，S04 规则可复用）、`contracts/versions.py`（`AVAILABLE_CONTRACT_VERSIONS = ("v1","v2","v3")`，DEFAULT="v3"）、`cli/dataset_verify.py`（确定性重跑验证，已参数化资产目录）。
- **Record 内容**：`oracle_only = {future_ego_poses: 6}`（2Hz ≈ 3.0s 窗）；`model_inputs = {frames: 4, ego_speed_mps}`。**oracle_only 无对象未来信息**——builder 须扩未来帧读取（§2.2）。
- **expected_output 契约 = v3 output_schema**：critical_objects（category / bbox_2d / motion_state，maxItems=8）、risk_factors（8 枚举，unique）、reasoning（minLength 1）、yield_required（bool）、speed_action（4 枚举）。v4 的 output_schema 与 v3 内容相同（原样继承）。
- **数据**：`data/dataset_v1` symlink → `/root/autodl-tmp/datasets/drivealign_dataset/v1`；valid = train 22,941 / val 2,531 / test 5,468 = 30,940，quarantine 659 不回填；迭代入口 `anchor_policy_manifest.json`（token→split/shard/line/record_hash/request_hash_1f/4f）。
- **台账冻结要点**：可见性门控在前、top-8 在后；视野外对象不产生任何 risk factor；corridor 族 = t0 双侧 CV 反事实外推 + buffer（禁实际轨迹）；crossing 纯运动学三带铺满；observability 三态（observable/inferable/unscorable）；P4 英文模板、无数字、risk↔短语双向精确、信号灯零提及、hash(sample_token+段salt) 选 3–5 变体。

## 2. 模块拆分

### 2.1 新增 `src/drivealign/gt/` 包

| 文件 | 职责 |
|---|---|
| `gt/config.py` | `gt_rule_config.json` 加载器；contract_version v4 联动；全部阈值只从此处读（冻结值，拒绝硬编码散落） |
| `gt/observability.py` | 三态门控：8 角点投影（复用 `evaluation/projection.py`）→ behind 检查（任一角点深度 ≤ ε → unscorable 整框丢弃）→ AABB clip 图像界 → observable（全在界内）/ inferable（越界但相交，bbox=clip 后）/ unscorable |
| `gt/pool.py` | 三步池构建：门控幸存者 ∧（前向锥 ±30° ∪ 全向近距盘 10m）→ 距离升序 top-8；距离平手按 sample_token 字典序决胜 |
| `gt/motion_state.py` | 判定顺序 stationary → crossing → oncoming → same_direction；三带对称铺满（θ 待标定）；t0 box velocity |
| `gt/risk.py` | 8 项规则。corridor 族（corridor_conflict / pedestrian_crossing / vehicle_merging）= t0 双侧 CV 反事实外推（ego 行驶带宽=车宽+余量；对象侧对称 CV）T_risk 秒内最近距离 < 每类 d_risk；ego 静止时 corridor 退化为自车框+buffer。small_following_gap 同 t0 双侧 CV 口径；lead_vehicle_braking / congestion / stationary_obstacle 用实际轨迹证据窗（直接观测量，无自毁陷阱）；oncoming_traffic 朝向规则不变 |
| `gt/action_yield.py` | speed_action（实际未来 ego 轨迹行为证据）+ yield_required（纯反事实冲突证据，见 §3） |
| `gt/render.py` | P4 渲染器：hash(sample_token + 段 salt) 从 3–5 变体确定性选句；消费 `enum_phrase_map.json`；零矛盾 gate（见 §6.4） |
| `gt/backfill.py` | 单记录组装：池 + motion + risk + action/yield + reasoning → `expected_output` dict → jsonschema（v4 output_schema）校验 |
| `cli/gt_calibration.py` | P1.2 标定实验（§4） |

### 2.2 修改现有文件

- `contracts/versions.py`：`AVAILABLE_CONTRACT_VERSIONS += ("v4",)`，`DEFAULT_CONTRACT_VERSION = "v4"`。
- `configs/contracts/v4/`：v3 五文件（prompt.txt / output_schema.json / category_vocab.json / motion_vocab.json / risk_taxonomy.json）**原样复制** + 三件新资产 `gt_rule_config.json` / `reasoning_templates.json` / `enum_phrase_map.json`。
- `dataset/build_dataset.py`：v4 分支在 `build_record` 后对 valid 记录调用 backfill（quarantine 保持 None）；**扩展未来帧读取**——沿 `sample.next` 链取固定 6 帧（3s@2Hz），只按池内对象 ann token 查询 boxes（≤8×6 次/帧，开销可控），供 braking/stationary 证据窗；ego 未来轨迹复用 oracle_only 已有 poses。**adapter.py 不动**——request_hash 生成路径零改动，parity 天然成立。
- `cli/dataset_verify.py`：指向 v4 资产目录重跑（确认参数化即可）。

### 2.3 新增单测（tests/unit/）

`test_gt_observability / test_gt_pool / test_gt_motion / test_gt_risk / test_gt_action_yield / test_gt_render / test_gt_backfill`。必含样例（对应 08 文档 §3.9"合成几何、空对象、重复预测、behavior 边界、动作边界"）：

- **自毁陷阱回归（核心）**：构造"ego 检测到横穿而减速"场景——反事实 CV corridor 必须仍检出冲突；corridor / gap / yield 三处各一个。
- behind 对象 → unscorable；越界框 → inferable + clip bbox；AABB 不相交 → unscorable。
- 距离平手 → sample_token tiebreaker；三带边界偏角 45°/135°；STOP/±δ 动作边界。
- 空对象帧 → critical_objects [] + 空态 reasoning 渲染非空。
- 确定性：同输入两次渲染结果逐字节一致。

## 3. P1.6 定稿提案（待用户批准）

**speed_action**（实际未来 ego 轨迹，W_act = 3.0s = 6 帧，与 oracle_only 对齐；判定优先级 STOP > DECELERATE > ACCELERATE > KEEP_SPEED）：

| 动作 | 规则 |
|---|---|
| STOP | 窗内最小速度 < 0.5 m/s |
| DECELERATE | v_min < v0 × (1−δ)，δ 标定（候选 0.15），未触发 STOP |
| ACCELERATE | v_max > v0 × (1+δ)，未触发上述 |
| KEEP_SPEED | 其余 |

**yield_required**（纯反事实）：池内任一对象触发 corridor 族冲突（corridor_conflict / pedestrian_crossing / vehicle_merging 任一成立）→ true。不掺入"ego 实际是否让"的行为证据（台账 P1.6 证据分工预注记）。speed_action 与 yield 的联动分布进报告、不进强制 gate。

## 4. 标定实验（train-only，P1.2）

`cli/gt_calibration.py`：一遍加载 train split nuScenes，全部 22,941 anchors 内存扫描参数网格，输出 `runs/S08_gt_backfill/calibration_report.{json,md}` + 推荐值表：

1. **池几何**：锥角 {20,25,30,35,40}° × 近距盘 {8,10,12,15}m × 类别阈值（LeapAD 锚点：车辆 20m/本车道 60m、行人 40m，±50% 网格）→ **截断帧占比 < 5% 达标区** + 池大小 P50/P90 + 不可见率（observable/inferable/unscorable）。
2. **corridor 族**：T_risk {2,3,4,6}s × d_risk（行人/车辆分设网格）→ 冲突检出率（稀疏性检查：过高=无区分度，过低=GT 饥饿）。
3. **motion**：θ {30,45,60}° × stationary {0.3,0.5,0.8} m/s → 四值分布。
4. **action**：W_act {2,3,4}s × δ {0.1,0.15,0.2} → 四动作占比（与 expert 行为分布比对）。
5. ~~**gap**：间距 {2,4,6,8}m / 时间间隔 {0.5,1,1.5,2}s → 检出率。~~（**已移除**：实测检出 ≤0.64%，饥饿标签，`small_following_gap` 从 taxonomy 全链路删除，见 §9）

硬约束：只读 train scene manifest 与 nuScenes，val/test 不进任何 fitted state；标定代码路径须可审计（§6.6）。

## 5. 执行编排

| Step | 内容 | 执行者/环境 |
|---|---|---|
| 0 | git 提交决策快照（§0） | agent（用户确认后） |
| 1 | v4 三件资产骨架（数值标 provisional）+ gt 包 + 单测 + 标定 CLI | agent，sandbox |
| 2 | train-only 标定跑 → 报告 → **用户拍板数值** → 冻结 gt_rule_config（provisional → frozen） | 用户 tmux 执行命令（10–15GB RAM）；agent 分析报告 |
| 3 | build_dataset v4 回填改造 + 冒烟（`--max-scenes 2 --out runs/S08_gt_backfill/smoke_out`）+ 冒烟级 parity 预验（冒烟 anchor token 与 v3 manifest 对比） | agent，sandbox |
| 4 | 全量 v4 重建：新物理目录 `/root/autodl-tmp/datasets/drivealign_dataset/v4` + 新 symlink `data/dataset_v4`（**命名裁定**：dataset 目录跟 contract 代际走，v4=contract v4 产物；`dataset_v1` 为历史错位——实为 contract v3 产物，本轮不改名，作为已知命名错位记录。v1 只读保留，parity diff 与回滚依赖） | **用户，tmux** |
| 5 | Gates 全套（§6）+ observability/coverage 按 split 报告 → `runs/S08_gt_backfill/` | agent，sandbox |
| 6 | 实验记录落 `docs/experiment_record/S08/` + git commit | agent |

标准命令模板（Step 2 示例，Step 4 同构换模块名与日志路径）：

```
tmux new-session -d -s s08_calib \
  'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
   cd /root/autodl-tmp/drivealign_workspace && \
   set -o pipefail && PYTHONPATH=DriveAlign/src python -m drivealign.cli.gt_calibration 2>&1 | tee runs/S08_gt_backfill/calibration.log'
```

新程序 docstring 必须含 example launch command（沿用 build_dataset.py 风格）。

## 6. Gate 清单（Step 5 全量重建后）

1. **request_hash parity**：v4 manifest 的 request_hash_1f/4f 与 v3 逐值相同（30,940 anchors × 2 策略）。
2. **anchor 集合不变**：token/split 集合与 v3 完全一致。
3. **GT 填全**：valid 100% 非空 + v4 output_schema jsonschema 校验通过；quarantine 恒 None。
4. **零矛盾 gate**（P4.5，逐条记录级，构建时 fail-fast + 报告汇总）：reasoning 提及对象（类别名）⊆ critical_objects 集合；risk_factors 每项以其固定短语出现于风险段且无多余短语（双向精确）；traffic light 词族正则零命中；reasoning 非空。
5. **确定性**：dataset_verify pinned-commit 重跑（读冻结 manifest 的 builder_commit 传入，git HEAD 移动不致假失败），全文件逐字节一致。
6. **train-only 证明**：标定代码只读 train manifest 的路径审计 + 报告记录。
7. record_hash 新代际 + dataset_manifest / anchor_policy_manifest 重派生指纹落档。

## 7. 待用户拍板项

1. §0 先提交决策快照；
2. §3 speed_action / yield_required 提案（规则本身已按台账推演，数值 δ/W_act 归标定）;
3. v4 目录形式：新 symlink `data/dataset_v4`（v1 只读保留）——采用此形式还是切换 `data/dataset_v1` 指向 v4。

## 8. 交接（完成后）

记录 `m09_version`、oracle/config hash、contract v4 指纹、GT 填全报告、标定报告路径与冻结数值表。Stage 09/10 配置须同时预注册，之后不得据单帧 test 结果改多帧评测条件。

## 9. Step 2 冻结记录（2026-10-02）

> 本节为摘要；完整的选择过程、备选方案否决理由与实测数据见 [S08_parameter_freeze.md](./S08_parameter_freeze.md)。

### 9.1 五族定值（用户拍板 + 标定证据）

| 参数族 | 冻结值 | 依据 |
|---|---|---|
| ① 池几何 | 锥半角 30°，近距盘 15 m，类别距离 ×0.75（vehicle 45 / pedestrian 30 / static 15 m） | 用户决策（方案 A：收距离上限替代加权打分）；加权打分方案因"静止前车速度项陷阱/权重无判据/截断语义劣化"被否决 |
| ② corridor 族 | T_risk 3.0 s，d_risk 车辆 3.0 m、行人 2.0 m | 用户决策（加大危险距离），标定网格实测 (3.0,3.0,2.0) 检出 25.25%/2.25%/23.86% 无饥饿 |
| ③ motion | θ 半带宽 45°，stationary 0.5 m/s | 用户决策（固定值） |
| ④ action | 窗口 3.0 s（6 帧），δ=0.15（**相对值**：窗内 min < 0.85·v0 → DECELERATE，max > 1.15·v0 → ACCELERATE，min < 0.5 → STOP） | 用户决策（冻结），δ 语义已澄清 |
| ⑤ gap | **移除** | 实测检出 ≤0.64%（饥饿）；taxonomy 8→7 项，output_schema/phrase_map/risk_taxonomy/config/CLI/测试全链路同步 |

### 9.2 截断豁免（visibility-first 已知限制）

池几何 (30°, 15 m, ×0.75) 的截断帧率 **11.33%**（2,655/23,425），超 5% 硬目标；cap=10 方案因牵动模型面 `maxItems` 契约被否决，改为补验证跑量化代价：

- 验证 CLI：`cli/gt_truncation_check.py`（train-only，复用冻结 corridor 参数）
- 全量结果（`runs/S08_gt_backfill/truncation_risk_report.{json,md}`，23,425 anchors）：
  - 被截断对象 9,249 个（截断帧均值 3.48），距离 P50=33 m / P90=42 m——确为最远者；
  - 仅 **4.53%**（419 个）触发 corridor 冲突（行人 38 / 车辆组 87 / 静态类 294）；
  - **漏检帧 26 / 23,425 = 0.111%**（截断帧内 0.98%）——即解除 top-8 上限后 `yield_required` 会翻 True 的全部帧。
- 裁定：0.111% 远低于 0.5% 预期线，作为 visibility-first 原则下的已知限制记录（与"视野外真实冲突不记录"同列），豁免成立。

### 9.3 资产状态

- `gt_rule_config.json`：`status` **provisional → frozen**，description 记录标定与验证来源；
- 单测 179 passed（含 gap 移除、v4 升级、池几何定值后的回归修正）；
- 标定/验证冒烟均在真实 trainval 2 场景跑通（`runs/S08_gt_backfill/*smoke*`）。
- 下一步：Step 3 build_dataset v4 回填改造（sandbox）→ Step 4 全量重建（用户 tmux，`/root/autodl-tmp/datasets/drivealign_dataset/v4` + symlink `data/dataset_v4`；命名裁定见 §5 Step 4 行）。

## 10. Step 3 执行记录（2026-10-02，agent sandbox）

### 10.1 改动清单

| 文件 | 改动 |
|---|---|
| `contracts/versions.py` | 新增 `MODEL_FACE_LINEAGE`（v4→v3）、`MODEL_FACE_FILES`、`HASH_FACE_FILES`、`model_face_version()`；docstring 记录 v4 模型面唯一 delta = gap 移除（output_schema/risk_taxonomy 8→7，不进 request hash） |
| `records/serializer.py` | `serialize()` 打戳改为 `contract_version=model_face_version(DEFAULT_CONTRACT_VERSION)`——**S08 gate 1 关键设计**：request hash 内嵌版本戳，v4 继承 v3 模型面（逐字节），故 v4 request hash 与 v3 manifest 存储值逐值相同；record 侧 provenance 仍为 v4 |
| `dataset/build_dataset.py` | 接入 GT 回填：`GtAssets`（rule_config/templates/phrase_map + sha256 溯源）、`_backfill_record`（消费 record 自身 `ego_speed_mps`/`future_ego_poses` + anchor 证据，`validate_record` fail-fast）、quarantine 恒 None、gate `gt_backfill_full`、config 记录 `gt_rule_config_sha256/status/version` |
| `cli/dataset_parity_check.py` | **新增**：v3↔v4 parity CLI（smoke/full 两模式），8 项 gates：anchor_subset / split_assignment_match / request_hash_parity（硬）/ gt_backfill_present（硬）/ record_hash_new_generation（硬）/ placement_integrity / anchor_set_equality（full 要求集合相等）/ all_pass；v3 侧 manifest sha256 校验后读取 |
| `dataset/gt_distribution.py` | **新增**：GT 分布累计器 + markdown 渲染——构建时 inline 计数（零额外 I/O pass），枚举词表全部读自冻结 v4 契约资产（零计数行显式保留）；9 张表：speed_action / yield_required / risk_factors（unique 语义）/ category / motion_state / category×motion / objects-per-frame 直方图 / speed_action×yield 交叉表 / reasoning 长度，per split + TOTAL |
| 单测 | 新增 `test_dataset_build_v4.py`（mini 集成 + GT 分布 4 例 + parity 纯函数 2 例 + CLI 合成资产对 3 例）、`test_records_serializer.py` 增模型面戳与 hash-face 逐字节断言；**全量 193 passed** |

### 10.2 关键设计决策：model-face 版本戳

request hash payload 含 `contract_version` 字段。DEFAULT 已切 v4 后 serialize 重算值与 v3 manifest 存储值字面不等——探针实证：钉回 v3 戳后逐字节复现。为满足用户硬 gate（"request_hash_1f/4f 逐值相同"）且不动 prompt/因果字段，引入 model-face lineage：**序列化请求打模型面版本（v4 继承 v3），记录打 record 侧版本（v4）**。v3→v4 资产差异精确刻画：prompt.txt / category_vocab.json / motion_vocab.json 逐字节 IDENTICAL；output_schema/risk_taxonomy 仅差 small_following_gap 移除且不进 hash face。runner（structured_runner）走独立 SampleRequest 不消费此戳，改动局部安全。

### 10.3 冒烟结果（2 场景 × 3 split，真实 trainval）

- 构建：train 74 候选 → 68 valid 全回填 + 6 quarantine（全 BAD_GAP）；val 75/75；test 74/74；gates 全 PASS（含 `gt_backfill_full`）。
- Parity 预验：**217 anchors 全 PASS**——request_hash_1f/4f 与冻结 v3 manifest 逐值相同（硬 gate 达成）、GT 5 字段全填、record hash 全部新代际（回填生效证据）、placement 完好。
- 抽查样例：5 字段齐全；reasoning 仅提及池内对象，叙事顺序 perception→risk→decision；risk_factors 空帧与 reasoning 零风险句一致。
- 产物：`runs/S08_gt_backfill/smoke_out/`、`smoke_reports/build_report.{json,md}`、`parity_report.{json,md}`。

### 10.4 Step 4 命令（用户 tmux 执行）

```bash
tmux new-session -d -s s08_build \
  'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
   cd /root/autodl-tmp/drivealign_workspace && \
   set -o pipefail && PYTHONPATH=DriveAlign/src python -m drivealign.dataset.build_dataset \
   --out /root/autodl-tmp/datasets/drivealign_dataset/v4 \
   2>&1 | tee runs/S08_gt_backfill/build_dataset.log'
```

构建完成后建 symlink（同物理盘模式）：

```bash
ln -s /root/autodl-tmp/datasets/drivealign_dataset/v4 /root/autodl-tmp/drivealign_workspace/data/dataset_v4
```

> 内存 10–15GB、预计 30–60 分钟（30,940 valid + 659 quarantine 全量回填）。构建结束自动产出 **GT 分布表**：`runs/S08_gt_backfill/reports/gt_distribution_report.{md,json}`（5 域 9 表，per split + TOTAL；冒烟预览见 `smoke_reports/` 同名文件）。v1 目录只读保留（parity diff 与回滚依赖）。agent 侧随后接 Step 5 gates：`dataset_parity_check --mode full`（30,940 anchors 集合相等 + 逐值 parity）+ `dataset_verify` 确定性重跑。

### 10.5 Step 4 全量构建结果（2026-10-03，用户 tmux 执行）

- 产物落盘：物理盘 `/root/autodl-tmp/datasets/drivealign_dataset/v4/`（32 shards + quarantine.jsonl + dataset_manifest.json），symlink `data/dataset_v4`（agent 建立并验证）。
- Manifest 抽验：contract v4 / gt_rule v4-rules-0.1.0 (frozen)；records 30,940 = train 22,941 / val 2,531 / test 5,468（与 v3 逐 split 一致）；quarantine 659（全 BAD_GAP）。
- **命名裁定**：dataset 目录跟 contract 代际走（`dataset_v4` = contract v4 产物）；`dataset_v1` 为已知历史命名错位（实为 contract v3 产物），永久保留不改名；未来 contract vN 重建 → `dataset_vN`。

### 10.6 Step 5 gates 结果（全 PASS）

| Gate | 结果 |
|---|---|
| Full parity（`dataset_parity_check --mode full`） | **PASS**：30,940/30,940 anchors 集合相等；request_hash_1f/4f 零 mismatch（硬 gate）；GT 5 字段零缺失；record hash 零未换代；placement 零问题。报告 `runs/S08_gt_backfill/parity_report.{json,md}` |
| Determinism（`dataset_verify`：pinned 47b5e0a 重建 + 逐字节比对） | **PASS**：39 文件全同（32 shards + dataset_manifest + quarantine.jsonl + anchor_policy_manifest + 4 份 report）。报告 `runs/S08_gt_backfill/reports/determinism_report.json` |
| GT 分布表 | 已产出（全量数字：KEEP_SPEED 40.8% / STOP 21.7% / ACCELERATE 18.8% / DECELERATE 18.7%；yield_required true 22.6%） |

**插曲与教训**：首跑 verify 失败暴露两个问题。(1) v4 资产缺 `anchor_policy_manifest.json`——S07 完整流程是 build 主 CLI + anchor manifest 派生 CLI 两条命令，Step 4 命令只含前者；修复：verify 重建时 scratch 内派生出 pinned-commit 同产物，用户拷入 v4 资产作为初始冻结版（等价于跑完整两步）。(2) verify 假设 scene manifests 在资产目录内（v1 布局）；修复：从冻结 manifest config 的 `scene_manifests_dir` 读取，`_diff_files` 对 frozen 侧缺失文件优雅报错（原先裸 traceback + conda run 缓冲吞输出造成 exit 0 假象；sandbox 内跑 verify 须用 `conda run --no-capture-output`）。S08 结论：**contract v4 数据集资产冻结，S08 主线关闭**。
