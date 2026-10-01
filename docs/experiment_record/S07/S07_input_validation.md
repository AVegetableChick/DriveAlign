# S07 输入验证记录：nuScenes trainval + DriveLM v1.1（2026-10-01）

> 状态：S07 两个输入全部就绪并通过验证；本文档同时记录实测发现、已做决策与 S07 启动前待决策项。
> 数据不入 git；本目录只记录验证结论与决策。

## 1. nuScenes v1.0-trainval 就绪

### 获取方式
- 来源：AutoDL 公共数据 `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/`（10 个 blobs 分卷 + meta tgz）
- 脚本：`DriveAlign/scripts/extract_nuscenes_trainval.sh`（tmux 后台，流式读 ~350GB，落盘 32G）
- 白名单解压：全部 6 相机 `samples/CAM_*` keyframe + meta 全量；**sweeps / lidar / radar 未解**
- 布局（与 mini 同构）：实体在 `/root/autodl-tmp/datasets/nuscenes/trainval`，symlink 到 `data/nuscenes/trainval`

### 验证结果（全绿）
| 检查 | 结果 |
|---|---|
| 解压完整性 | 10/10 blobs，6 相机 × 34,149 张 keyframe（≈850 scenes × 40，吻合） |
| metadata | 13 张表全齐 |
| 表规模 | 850 scenes / 34,149 samples / 2,631,083 sample_data（官方 trainval 规模） |
| train split keyframe | 28,130（官方 700 scenes，用于 split 场景基数） |
| 端到端 | `load_nuscenes` → 4F 链 OK → `build_record` 成功，`hash=ff8ba9ecd8c6…` |
| 部分解压安全性 | `NuScenes()` 只读 json 表、不校验文件；缺图仅按需访问时报错，且 `nuscenes_io` 以 `MISSING_IMAGE` 隔离码兜底 |

- 意外收获：4 张基础 map png 随 blob 带出（5.6MB，无害）；`NuScenesMap` 基础功能可用。
- 工程教训（已修入脚本）：tmux 管道中 `tee` 只建文件不建父目录，启动命令需先 `mkdir -p`；脚本内验证段必须自带 conda 激活。

## 2. DriveLM `v1_1_train_nus.json` 就绪

- 位置：`data/drivelm/` → symlink 至 `/root/autodl-tmp/datasets/drivelm/`（193MB）
- 文件身份确认（四项检查全过）：

| 检查 | 结果 |
|---|---|
| 结构 | 官方 v1.1 格式：顶层 696 个 scene token → `scene_description`(str) + `key_frames`(dict) → 以 **sample token** 为键 |
| 帧级字段 | `key_object_infos`（Category/Status/Visual_description/**2d_bbox**）、`QA`（4 组：{Q,A,C,con_up,con_down,cluster,layer}）、`image_paths`（6 相机） |
| token 对齐 | 抽样 300 帧 → 300/300 命中 nuScenes 索引 |
| split 归属 | 300/300 全在 train split，0 在 val |

### 实测事实（超出此前认知，已同步修订 S03-Q3 文档）
1. **标注覆盖率 14.5%**：4,072 帧 / 28,130 个 train-split keyframe。→ `language_reference=None` 是常态而非异常，join 报告必须报告覆盖分布。
2. **key_object_infos 自带 2D bbox**：`"2d_bbox": [966.6, 403.3, 1224.1, 591.7]`。→ S03-Q3 资格清单第 3 条论据（"无 bbox 几何"）失效；join 时 bbox 来源需决策（见决策项 D1）。
3. **v1.1 存在 val 版标注**（`v1_1_val_nus.json`）。→ S03-Q3 第 4 条论据改写为"选择不引入"（见决策 D2）。
4. 顶层每 scene 附带 `scene_description` 全局描述。→ 默认不消费，保留观察。
5. QA 中确认存在 `<cN, CAM_FRONT, x, y>` 引用格式（join 解析目标）。
6. **格式深探（2026-10-01）**：`key_object_infos` 的键即 `<cN,CAM,x,y>` 引用串本身（cN 为帧内跨相机全局编号），QA 文本引用与键**精确字符串匹配**，无坐标解析歧义；全量 377,956 条 QA（perception 162,480 / prediction 123,436 / planning 87,968 / behavior 4,072，≈93 条/帧），不可能全收，须定筛选规则（D6）；QA 覆盖全部 6 相机，v1 仅 CAM_FRONT；每 scene 4-8 帧（144×4 / 166×5 / 165×6 / 92×7 / 129×8）；planning 组为对象级 yes/no 人判（"Is \<cN\> an object the ego should consider?"）；91.3%（3,719/4,072）帧有 CAM_FRONT 对象（vehicle 4,881 / pedestrian 943 / other 1,817）。

### D1 quick check：DriveLM 2d_bbox vs nuScenes 3D 投影重合度（2026-10-01）

方法：抽 100 个有 CAM_FRONT 对象的 keyframe（seed=0），将 nuScenes 3D GT 框（vehicle/pedestrian 类）投影到 CAM_FRONT 取外接 2D 框，与 DriveLM `2d_bbox` 算 IoU；中心点取自引用键。

| 指标 | 结果 |
|---|---|
| 可评估对象 | 158 个（vehicle/pedestrian；other 类 54 个跳过） |
| 中心点命中率 | **96.8%**（153 matched / 5 miss → ambiguous） |
| IoU 分布 | mean 0.791 / **median 0.825** / P25 0.734 / P75 0.878 / **P90 0.910** |
| IoU<0.5 占比 | 2.6% |
| 判定（median>0.7 且 P90>0.5） | **通过** |

miss 归因（5/5 复查）：全部为 40-80m 远距离小目标（bbox ~40×26px）的 Vehicle，中心点与最近 nuScenes 框偏移 55-285px——系 nuScenes 远处 3D GT 漏标/不对应（信号灯等 other 类走 skipped 通道，与 miss 无关；类别映射与投影逻辑无误）。→ join 报告的 ambiguous 需细分 `GT_MISS` 与 `CATEGORY_UNMAPPED` 两类原因。

## 3. 已做决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | **已决并修订（2026-10-01）：全部 bbox 一律用 nuScenes 3D 投影外接框，DriveLM `2d_bbox` 不进入任何输出层** | quick check 证明两源重合（IoU median 0.825 / P90 0.910，中心点命中 96.8%），换源数值零风险；但 DriveLM 框覆盖率仅 14.5% 且人判选择 ≠ 规则 top-K≤8 选择——language_reference 的对象集合与 expected_output 同一规则，bbox 必须同源，否则双重错位。此修订回归 S03-Q3 原推论"一切 bbox 几何 100% 来自 nuScenes GT"。DriveLM 在 join 中的贡献收缩为纯语言（Visual_description + QA → reasoning 素材），其"人判选择"倾向保留在 reasoning 文本层。quick check 存档保留作两源一致性证据；D1 早期版本（语言参考用 DriveLM bbox）作废 |
| D2 | **不下载** DriveLM val 版 | test 来源隔离由构造性保证（构建流水线输入物理不含该文件）；对 M09 结构化评测无用（GT 走 oracle_only 规则派生）；对未来语言质量评测仅可作 reference（与确定性评测哲学有张力），届时再评估。**2026-10-01 补充实测：公开 val 版仅含 question 无 answer（challenge 提交格式，答案在官方侧）**——critical_objects/risk_factors/yield_required 等答案侧信息均不存在，作为评测 GT/参考的资格彻底关闭；S03-Q3 第 1 条论据（规则可派生）在 val 上以最强形式成立 |
| D7 | **S07 纯规则构建，DriveLM 移出主链路**：降级为 S08 后可选的 reasoning 语言增强包（默认关闭），`drivelm.py`/`join.py` 不再是 S07 交付物 | 决定性论据：reasoning 挂靠依赖 S08 top-K 规则（骨架须与 critical_objects 一致），DriveLM join 天然后置于 S08，阻塞 S07 属人为耦合；评测矩阵对 reasoning 零权重（M05 Gate 刻意不判分）；纯规则覆盖 100% 且确定性最强。代价（reasoning 模板化、语言多样性损失）在当前 gate 体系权重为零。已做工作全部保留为增强包资产：数据在盘、格式探明、D1/D5/D6 设计冻结待用 |

## 4. S07 实现前待决策项汇总（D7 修订后）

- ~~D1~~：已决（见第 3 节）
- ~~D2~~：已决——不下载、不引入
- ~~D7~~：已决（2026-10-01）——S07 纯规则构建，DriveLM 移出主链路
- **D3**：全量 record 存储布局（~26k 4F records：JSONL 分片粒度、目录结构）
- **D4**：scene split 冻结规则（train/val 比例、划分排序规则、test 规模）
- ~~D5 / D6~~：**随增强包冻结待用（D7）**——启用条件：S08 top-K 规则定稿且需要自然语言 reasoning 实验；届时按已冻结设计执行（信号灯过滤 + QA 筛选规则链），无需重议
