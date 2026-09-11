# DriveAlign 项目结构与端到端执行蓝图 v1

> 状态日期：2026-09-11  
> 目标执行平台：AutoDL 远程 Linux GPU 实例。  
> 项目目录：`/root/autodl-tmp/DriveAlign`。  
> 目标模型：`Qwen/Qwen2.5-VL-3B-Instruct`。  
> 外部实现预算：AutoVLA、TRL、nuScenes-devkit，恰好三个。

关联文档：

- [01_DriveAlign_Base_Plan_v5.md](./01_DriveAlign_Base_Plan_v5.md)：Phase I 科研设计；
- [02_DriveAlign_RL_Plan_v5.md](./02_DriveAlign_RL_Plan_v5.md)：Phase II–III 科研设计；
- [03_DriveAlign_Weekly_Knowledge_and_Papers_v5.md](./03_DriveAlign_Weekly_Knowledge_and_Papers_v5.md)：知识与源码学习顺序；
- [04_DriveAlign_Weekly_OpenSource_Repos_v5.md](./04_DriveAlign_Weekly_OpenSource_Repos_v5.md)：上游入口、复用边界与回退条件。

本蓝图只回答三个问题：代码放在哪里、模块如何连接、实验按什么顺序推进。详细科研论证以上述四份 v5 文档为准。

---

# 1. 项目目标与硬约束

DriveAlign 研究通用 VLM 经自动驾驶 Multimodal SFT、DPO 和 GRPO 后，能否在因果输入条件下改善关键对象 grounding、风险判断与驾驶动作，并通过独立自动评测验证改进。

全项目必须遵守：

1. 模型只接收当前或过去可获得的因果输入；
2. 未来轨迹、未来对象和 future-derived command 只能进入 target、reward 或 evaluation oracle；
3. train、validation、test 按完整 scene 分离；
4. M05 Training Verifier 与 M09 Evaluation Oracle 分离；
5. 上游能力优先复用，只编写 DriveAlign 特有的薄 Adapter；
6. 不自写 SFT、DPO、GRPO loss、nuScenes 表连接或基础几何库；
7. `YIELD` 是独立让行意图，不是第五种速度动作；
8. 人工只能查看自动选出的案例，不能改标签、阈值或排名；
9. 任一阶段失败都可以形成负结果，不增加第四个实现仓库；
10. Phase III 第一版不进行 trajectory GRPO。

---

# 2. AutoDL 工作区与 Git 边界

## 2.1 物理目录

~~~text
/root/autodl-tmp/
├── DriveAlign/                  # 本项目 Git checkout
├── DriveAlign-Upstream/         # 三个独立上游 Git checkout
│   ├── AutoVLA/
│   ├── trl/
│   └── nuscenes-devkit/
├── DriveAlign-Data/             # raw / processed / quarantine
├── DriveAlign-Models/           # Qwen、SFT、DPO、GRPO 权重
├── DriveAlign-Artifacts/        # runs、日志、预测、checkpoint
└── DriveAlign-Cache/            # Hugging Face、Torch 等缓存

/root/autodl-fs/
└── DriveAlign-Backup/           # 可选的关键产物备份
~~~

对应本机配置：

~~~yaml
paths:
  project_root: /root/autodl-tmp/DriveAlign
  upstream_root: /root/autodl-tmp/DriveAlign-Upstream
  data_root: /root/autodl-tmp/DriveAlign-Data
  model_root: /root/autodl-tmp/DriveAlign-Models
  artifact_root: /root/autodl-tmp/DriveAlign-Artifacts
  cache_root: /root/autodl-tmp/DriveAlign-Cache
  backup_root: /root/autodl-fs/DriveAlign-Backup
~~~

真实路径写入被忽略的 `configs/local.yaml`；仓库提交 `configs/local.example.yaml`。文件存储未启用时，`backup_root` 显式设为 `null`。

## 2.2 Git 所有权

DriveAlign 仓库提交：

- `src/`、`tests/`、`configs/`、`scripts/`；
- scene manifests、schema 和规则版本；
- 实验结果摘要与文档；
- 三个上游的完整 commit SHA 和许可证记录。

DriveAlign 仓库不提交：

- 原始或处理后数据；
- 模型权重和 checkpoint；
- predictions 全集、训练日志和缓存；
- AutoVLA、TRL、nuScenes-devkit 源码副本；
- 密钥、token、实例地址和本机私有配置。

代码位于无冗余保证的 `autodl-tmp`，因此私有 Git 远端是代码历史的事实源。每个有效提交应及时 push。数据盘只承担高 IO 工作副本，不承担唯一备份。

当前只有 GitHub 源码 ZIP，无法可靠恢复对应 commit。Gate 0 必须从实际 checkout 记录 `git rev-parse HEAD`，在三个完整 SHA 固定前不能开始正式实验。

---

# 3. 代码结构与模块映射

## 3.1 目标仓库结构

~~~text
DriveAlign/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── local.example.yaml
│   ├── model/
│   ├── data/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── references/
│   └── upstream.yaml
├── manifests/
│   └── nuscenes/
├── src/
│   └── drivealign/
│       ├── contracts/
│       ├── data/
│       ├── causal/
│       ├── inference/
│       ├── training/
│       ├── verifier/
│       ├── evaluation/
│       ├── trajectory/
│       └── cli/
├── tests/
│   ├── fixtures/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   └── smoke/
├── scripts/
└── docs/
~~~

只在对应阶段真正开始实现时创建目录和文件，不预建空模块。CLI 只负责参数解析和编排，核心逻辑必须位于可直接测试的包内函数。

## 3.2 M01–M09 映射

| 模块 | DriveAlign 位置 | 唯一外部来源 |
|---|---|---|
| M01 数据预处理 | `data/` | AutoVLA preprocessing |
| M02 因果隔离 | `causal/`、`contracts/` | 内部实现 |
| M03 Base/SFT | `inference/`、`training/` | AutoVLA SFT |
| M04 DPO | `training/` | TRL DPOTrainer |
| M05 Training Verifier | `verifier/` | 内部纯函数 |
| M06 RFT/GRPO | `training/` | AutoVLA GRPO |
| M07 Action Codebook | `trajectory/` | AutoVLA action tokenizer |
| M08 Trajectory Evaluation | `trajectory/` | AutoVLA nusc/NAVSIM evaluator |
| M09 对象/风险 Oracle | `evaluation/` | nuScenes-devkit |

同一 AutoVLA pinned checkout 同时服务 M01、M03、M06、M07、M08。TRL 只服务 M04；nuScenes-devkit 只提供官方数据、Box、标定、几何与 M09 基础能力。

---

# 4. 核心数据契约

## 4.1 DriveAlignRecord

~~~json
{
  "record_version": "drivealign-record-v1",
  "sample_token": "sample-token",
  "scene_token": "scene-token",
  "model_inputs": {
    "image": "samples/CAM_FRONT/example.jpg",
    "ego_speed": 5.8,
    "prompt": "fixed-schema-prompt"
  },
  "training_targets": {
    "structured_answer": {}
  },
  "oracle_only": {
    "future_ego_poses": [],
    "future_agent_boxes": [],
    "map_context": {}
  },
  "provenance": {
    "dataset_version": "v1.0-trainval",
    "source_fields": [],
    "rule_versions": {}
  }
}
~~~

模型 serializer 只允许复制：

~~~text
model_inputs.image
model_inputs.ego_speed
model_inputs.prompt
~~~

如果没有可靠的因果 speed，就省略该字段；不能使用未来位置估算。任何新增字段必须先声明 namespace，默认不可见。

## 4.2 输出 Schema

~~~json
{
  "critical_objects": [],
  "risk_factors": [],
  "reasoning": "",
  "yield_required": false,
  "speed_action": "KEEP_SPEED"
}
~~~

速度动作只能是：

~~~text
ACCELERATE / KEEP_SPEED / DECELERATE / STOP
~~~

对象项固定包含 `category`、`bbox_2d`、`coarse_position` 和 `behavior`。详细约束由版本化 JSON Schema 定义。

## 4.3 AutoVLA 接口隔离

AutoVLA 原生 prompt 使用三路前向相机、每路四帧、acceleration、driving command 和 action codebook；其 nuScenes preprocessing 中的 driving command 来自未来轨迹。

因此必须分为两条路径：

1. **Reference Smoke**：保持上游原格式，证明 preprocessing、loader、trainer 和 checkpoint 生命周期可以运行；
2. **DriveAlign Experiment**：通过薄 Adapter，只向模型传入当前 CAM_FRONT、可选因果 speed 和固定结构化 prompt。

Reference Smoke 的多相机、历史帧和 future-derived command 不能直接成为 DriveAlign 正式实验输入。

---

# 5. 配置、运行记录与 AutoDL 备份

## 5.1 配置原则

配置按以下顺序覆盖：

~~~text
版本化实验配置 < configs/local.yaml < 显式 CLI 参数
~~~

正式运行必须保存最终解析后的完整配置。模型、数据、seed、dtype、图像分辨率、解码、训练参数和输出路径都必须显式配置。

## 5.2 最小 Run 目录

~~~text
/root/autodl-tmp/DriveAlign-Artifacts/runs/<phase>/<run-id>/
├── resolved_config.yaml
├── metadata.json
├── metrics.json
├── predictions.jsonl          # 仅推理和评测运行需要
├── train.log                  # 仅训练运行需要
└── checkpoints/
    ├── best.ckpt
    └── last.ckpt
~~~

`metadata.json` 统一记录：

- DriveAlign commit 和三个上游 SHA；
- model/processor revision；
- data manifest 路径与 hash；
- seed、dtype、quantization、GPU 和核心包版本；
- 开始时间、结束时间、退出状态和峰值显存。

不为这些信息分别创建 `environment.json`、`upstream_revisions.json` 或 `input_manifest.json`。逐 rollout 的 `events.jsonl` 只有在 GRPO 阶段证明有实际需要时才引入。

## 5.3 AutoDL 运行规则

首次进入或迁移实例后检查：

~~~bash
df -h / /root/autodl-tmp /root/autodl-fs
nvidia-smi
du -sh /root/autodl-tmp/DriveAlign* 2>/dev/null || true
~~~

缓存写入数据盘：

~~~bash
export HF_HOME=/root/autodl-tmp/DriveAlign-Cache/huggingface
export TORCH_HOME=/root/autodl-tmp/DriveAlign-Cache/torch
~~~

训练、全量预处理和长评测必须在 `tmux` 或 `screen` 中运行。每个关键 Gate 完成后：

1. 校验配置、metadata、metrics 和 checkpoint；
2. push 代码、配置、manifest 和结果摘要；
3. 只备份 `best.ckpt`、必要的 `last.ckpt` 和最终报告；
4. 文件存储容量不足时使用实例外备份；
5. 确认无后续任务后从 AutoDL 控制台关机。

系统镜像只保存系统盘，不能代替数据盘备份。实例连续关机达到平台释放周期或主机下架时可能清空本地数据。

---

# 6. 端到端执行流程

流程采用逐级 Gate。前一 Gate 未通过时，不扩大数据，也不启动更昂贵的训练。

## Stage 0：AutoDL 仓库与环境基线

操作：

1. 在 `/root/autodl-tmp/DriveAlign` clone 或初始化私有 Git 仓库；
2. 建立 `.gitignore`，设置并验证 `origin`；
3. 将三个上游 clone 到 `/root/autodl-tmp/DriveAlign-Upstream`；
4. 记录 DriveAlign commit、上游完整 SHA 和许可证；
5. 创建 `configs/local.yaml` 并检查磁盘挂载；
6. 记录 Python、CUDA、GPU、PyTorch、Transformers 与 qwen-vl-utils；
7. 配置 Hugging Face/Torch cache；
8. 用 `tmux` 完成一次脱离和恢复 smoke；
9. 首次 commit 并 push。

Gate 0：

- DriveAlign 和三个上游是四个独立 Git checkout；
- 私有 `origin` 可 push；
- 三个完整上游 SHA 已固定；
- 数据、模型、缓存、密钥和 checkpoint 未进入 Git；
- `autodl-tmp` 容量满足当前 smoke；
- `backup_root` 已验证或显式为 `null`。

## Stage 1：Base 模型加载 Smoke

操作：

1. 将固定 revision 的 Qwen2.5-VL-3B 下载到 `DriveAlign-Models`；
2. 加载 `Qwen2_5_VLForConditionalGeneration` 和匹配的 `AutoProcessor`；
3. 默认使用 BF16，确认 device、dtype 和 CUDA 状态；
4. 记录加载耗时和峰值显存。

低显存回退顺序：降低 image max pixels、降低 max new tokens、保持 batch 1，最后才使用 4-bit inference。量化必须作为独立实验变量记录。

Gate 1：模型和 processor 可重复加载，没有 CUDA error、NaN 或意外 CPU fallback，峰值显存低于硬件安全上限。

## Stage 2：普通单图推理 Smoke

操作：

1. 直接调用基础 Qwen VLM，不调用 `AutoVLA.predict()`；
2. 输入一张道路图片和简单视觉描述问题；
3. 使用 `do_sample=false` 与固定 max new tokens；
4. 对相同输入运行三次；
5. 记录输入/输出 token、耗时、峰值显存和原始输出。

Gate 2：三次推理均完成，greedy 输出一致，结果可正常 decode，并确认图片被 processor 接收。

## Stage 3：结构化驾驶 Base 推理

操作：

1. 输入当前 CAM_FRONT、可选因果 ego speed 和固定 Schema prompt；
2. 保存模型原始文本，再运行严格 JSON parser；
3. 统计 parse、required keys、类型、bbox 和动作合法性；
4. 对比原图、blank image 和 shuffled image；
5. 在 `oracle_only` 注入 canary，检查 prompt 和 processor output。

Gate 3：预测和解析结果可追溯；解析失败不会丢样本或中止批处理；没有未来字段或 canary 泄漏。Base 内容指标低不属于管线失败。

## Stage 4：nuScenes-devkit Mini Smoke

操作：

1. 准备 nuScenes `v1.0-mini`；
2. 初始化 `NuScenes` 并读取一个 sample token；
3. 找到当前 CAM_FRONT、ego pose、calibration 和 annotation；
4. 验证图片路径；
5. 将至少一个可见 3D box 投影到图像并保存坐标或检查图。

Gate 4：sample table 链路完整，变换和投影值有限，至少一个可见 box 合法落在图像范围内。

AutoVLA preprocessing 固定使用 `splits.train/val`。如果 mini 输出 0 条，应先检查 scene split 选择，不能把零输出视为成功。

## Stage 5：M01 Reference Smoke 与 DriveAlignRecord

操作：

1. 用 mini 或受控 trainval 子集原样运行 AutoVLA preprocessing；
2. 确认输出能被 AutoVLA loader 读取；
3. 记录 raw sample 到上游 JSON 的字段变换；
4. 通过 DriveAlign Adapter 只选择当前 CAM_FRONT；
5. 使用当前与过去 ego pose 的 backward difference 计算因果 speed，或省略 speed；
6. 将 future trajectory、future agents、map 和 future-derived command 放入 target/oracle；
7. 坏样本写入 quarantine reason；
8. 相同输入重跑并比较 canonical record hash；
9. canary 检查扩展到 prompt、processor、batch 和 rollout。

Gate 5：upstream loader smoke 通过；所有样本归为 valid 或 quarantine；相同输入生成相同 record；每个字段属于明确 namespace；canary 0 次泄漏。

## Stage 6：正式数据准备与冻结划分

操作：

1. 准备 nuScenes `v1.0-trainval` 和 DriveLM `v1_1_train_nus.json`；
2. 以完整 scene 生成并冻结 train、validation、test manifest；
3. 按 sample token join DriveLM QA；
4. 分开保存 Hard GT、Derived GT 和 Language Reference；
5. 生成数据数量、类别、动作、yield、observability、coverage 和 quarantine 统计；
6. 保存原始数据 hash、builder commit 和 rule versions。

Gate 6：三个 split 的 scene 交集为空；100% raw candidates 可解释；测试集在训练前冻结；validation/test 未影响任何 fitted state。

## Stage 7：M09 Oracle 与冻结 Base Benchmark

操作：

1. 投影 CAM_FRONT GT box；
2. 标记 observable、inferable、privileged、unscorable；
3. 使用 Hungarian matching 一对一匹配；
4. 派生 action、yield、future corridor conflict 和 minimum future distance；
5. 运行视觉反事实测试；
6. 以 scene 为 cluster 计算 paired bootstrap CI；
7. 报告每个指标的 denominator。

主指标：JSON parse rate、object precision/recall/F1、matched IoU、speed action macro-F1、yield F1、corridor conflict recall、over-conservative rate、counterfactual consistency 和 unscorable rate。

Gate 7：projection、matching 和 action derivation tests 通过；Base 预测与评测规则冻结；M09 不 import M05 总 reward；所有主指标均有有效样本数。

## Stage 8：M03 SFT 计算 Smoke

操作：

1. 用 32 条 DriveAlignRecord 完成 forward/backward/save/reload；
2. 新进程加载 checkpoint 并完成推理；
3. 记录显存、step time 和 checkpoint 大小；
4. 用 128 条样本做 overfit smoke；
5. 比较真实、blank 和 shuffled image。

资源回退顺序：batch 1、gradient accumulation、gradient checkpointing、冻结视觉塔、LoRA、QLoRA。

Gate 8：训练无 NaN；32 样本 save/reload 成功；128 样本 loss 明显下降；真实图片内容表现优于 blank/shuffled；所有配置、seed、data hash 和 GPU 可追溯。

## Stage 9：正式 SFT

操作：

1. 使用 Gate 8 通过的资源配置训练；
2. 按 validation 指标选择 checkpoint；
3. 只对最终 checkpoint 运行冻结 test；
4. 与 Base 使用相同 prompt、processor、解码和 evaluator；
5. 报告指标、CI、样本数、显存和 GPU-hours。

Gate 9：SFT parse rate 和至少一个内容指标超过 Base；visual dependence 未退化；over-conservative rate 未越过预注册阈值；checkpoint 可恢复。失败时先报告原因，不进入 DPO/GRPO。

## Stage 10：M04 TRL Multimodal DPO

操作：

1. 从冻结 SFT 一次性生成多个候选；
2. 按 schema、grounding、risk/action、yield 和简洁性排序；
3. 过滤 margin 太小或 oracle 冲突的 pair；
4. 输出 `prompt/chosen/rejected/image`；
5. 使用 TRL `DPOTrainer` 完成 16–32 pair train/save/reload smoke；
6. 再扩大训练并与 Base/SFT 公平比较。

Gate 10：pair 可追溯到候选和 oracle 分量；损坏正确答案后排序下降；chosen/rejected 交换测试失败；DPO 至少改善一个预注册内容指标且未触发退化阈值。

## Stage 11：M05 Verifier 与 M06 GRPO

只有 Phase I 完成后才进入。

操作：

1. 实现确定性 `verify(...)` 纯函数；
2. 通过非法 JSON、边界值、确定性和 corruption ranking tests；
3. 用 8–32 prompts 完成 AutoVLA one-step RFT 和 save/resume；
4. 接入 DriveAlignRecord 与 M05 reward callback；
5. 正式运行 GRPO，并用独立 M09 监控。

Gate 11：相同输入和版本得到相同 reward；oracle-only 原始字段不返回 prompt；one-step/save/resume 通过；无 NaN、泄漏或持续 OOM；至少一个 M09 内容指标相对 SFT 改善。

## Stage 12：DPO、GRPO 与 Reward Ablation

所有条件固定同一 SFT 起点、scene manifests、基础模型、图像分辨率、输出 Schema 和 M09 evaluator。

主比较：

~~~text
Base
SFT
SFT + TRL DPO
SFT + AutoVLA GRPO
~~~

Reward ablation：

~~~text
full reward
without grounding
without risk/action consistency
format-only negative control
~~~

Gate 12：主表与消融表报告点估计、95% CI、scene 数、有效样本数、GPU-hours、峰值显存和失败恢复次数。

## Stage 13：Action Codebook 与 Trajectory SFT

只有 Phase III 独立 Gate 通过后才进入。

操作：

1. 冻结 waypoint 坐标系、频率、horizon 和 mask；
2. 只用 train scenes fit AutoVLA action codebook；
3. 验证 fit/cache/encode/decode 和 reconstruction error；
4. 单独建立 xy-bin baseline；
5. 训练 trajectory SFT；
6. 比较 constant velocity、xy-bin 和 codebook；
7. 运行 nuScenes ADE/FDE，官方 NAVSIM 可运行时追加 PDMS。

Gate 13：test trajectory 未参与 codebook fit；codebook 与 xy-bin 完全分离；future waypoint 未进入模型输入；shape、mask、有限值和坐标系正确；第一版没有 trajectory GRPO。

当前只执行 Stage 0–5。32 条 raw sample → DriveAlignRecord → processor/batch 闭环通过后，再准备全量数据和 Stage 6。

---

# 7. 关键风险与参考

## 7.1 AutoVLA 输入不满足因果约束

上游原生多相机、多帧和 future-derived command 只能用于 Reference Smoke。正式实验必须通过 DriveAlign Adapter 进入单张当前 CAM_FRONT 路径。

## 7.2 数据或 scene 泄漏

所有字段经过 namespace validator 和 canary；split 只能按 scene 生成。任何未来字段泄漏或 scene 交叉都会直接使 Gate 失败。

## 7.3 Training Verifier 与 Evaluation Oracle 同源

M05 与 M09 使用独立 package、config、version 和主指标。M09 禁止 import M05 的 `verify` 或 `total_reward`。

## 7.4 GPU 资源不足

先做 Compute Gate，再按已规定的回退顺序降配。每次精度、量化或 PEFT 变化都生成独立配置，不覆盖原实验。

## 7.5 AutoDL 本地数据丢失

`autodl-tmp` 没有冗余保证。代码及时 push；最佳 checkpoint 和最终报告同步到文件存储或实例外备份；系统镜像不能代替数据盘备份。

AutoDL 官方参考：

- [容器实例目录与磁盘说明](https://www.autodl.com/docs/env/)；
- [本地数据盘说明](https://www.autodl.com/docs/local_disk/)；
- [文件存储建议](https://www.autodl.com/docs/fs/)；
- [SSH 与 tmux/screen](https://www.autodl.com/docs/ssh/)；
- [实例数据保留规则](https://www.autodl.com/docs/instance_data/)；
- [保存镜像范围](https://www.autodl.com/docs/image/)。

Phase I–III 的 Definition of Done、论文依据和模块回退条件不在本蓝图重复，分别以开头链接的 01–04 v5 文档为准。
