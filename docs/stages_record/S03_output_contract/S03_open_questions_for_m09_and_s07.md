# S03 遗留问题:输出契约设计与 M09 GT / Stage 07 数据的衔接

> 状态:全部 **open**。来源:S03 收尾阶段(2026-09-18)对 schema v2 字段的监督信号分析(GT 有无、指标有无)、第 5 轮 sanity 的对象数触顶现象,以及 DriveLM 关键目标定义的文献核查。
> 结论前置:每条问题给出候选项与建议,均须在**所属阶段规则冻结前**决策;过期未决将触发契约版本升级并重跑对比实验。

---

## Q1 `critical_objects` 的 `maxItems: 8` 与 GT 数量的关系

**背景**:schema v2 限定模型最多输出 8 个关键对象。GT 数量由场景决定(空旷路 0 个,密集路口 15+ 个),"8"只是模型输出预算,不是配额。

**分析**:
- GT < 8(常见):无问题。模型硬凑满 8 会产生 false positive、拉低 precision——指标天然激励"只输出有把握的对象",这是想要的防刷分性质。
- GT > 8(密集场景):**recall 天花板**。模型完美也只能匹配 8 个,recall ≤ 8/N。M09 文档(08)当前 GT 侧定义为"全部 observable 投影框",无相关性过滤、无 GT 侧截断,天花板真实存在。
- 公平性:Base/SFT/DPO 共享同一天花板,**相对比较不受损**;绝对分被压缩,密集帧占比高的 split 更明显。

**候选项**:
| 方案 | 说明 | 评价 |
|---|---|---|
| A. GT 侧相关性过滤(推荐) | oracle 只对"决策相关"GT 计分:与 ego corridor 相交或距离 < 阈值的 observable 对象,按距离取最近 8 个 | 与字段语义(`critical_objects`)一致;与动作/风险 GT 的走廊派生逻辑同源;denominator 有界 |
| B. 提高模型上限(12/16) | 简单 | token 成本涨,"critical"语义稀释,治标 |
| C. 双口径报告 | 全 GT + 相关性子集各报一份 F1 | 信息最全,denominator 翻倍 |
| D. 不改,文档记录天花板 | 零成本 | 绝对分解读打折 |

**建议**:A。注意:相关性规则必须确定性、可版本化、进 oracle 不进 prompt(模型不需要知道 GT 侧怎么选);配合 observability 门控后实际可打分集合通常远小于 8。
**决策期限**:Stage 08 实现冻结前。

## Q2 M09 GT 侧规则的确定性翻译(承接 Q1 方案 A)

**背景**:DriveLM 官方标注流程([DriveLM 官网](https://opendrivelab.com/DriveLM/))对 key object 的定义是**标准而非数量**:"能够影响 ego vehicle 动作的对象"(信号灯、横穿行人、朝 ego 方向运动的车辆等),由标注员按此标准挑选,数量涌现、无上限。这是语义来源,但人判不可作评测 GT。

**确定性先例**:LeapAD([arXiv:2405.15324](https://arxiv.org/html/2405.15324))把关键目标定义为纯几何规则:前/左/右视场内,车辆与骑行者距 ego ≤ 20 m 或本车道 60 m 内;行人 ≤ 40 m;控制本车行驶方向的信号灯与停车标志。

**建议**:M09 GT 侧 = ego corridor 相交 **或** 距离 < 阈值的 observable 对象,按距离 top-K(K=8,与模型 maxItems 对齐)。阈值数值在 Stage 08 用 nuScenes 实测分布标定(参考 LeapAD 量级:车辆 20 m/同车道 60 m、行人 40 m)。规则版本化入 oracle。
**决策期限**:Stage 08 实现冻结前。

## Q3 DriveLM key objects 的分工确认(训练侧专用)

> **状态更新(2026-09-30)**:与用户两轮质询讨论后修正论证,原"三理由"版本作废,结论不变:key objects 只当老师(`training_targets` 语言参考),不当裁判(评测 GT)。

**背景**:Stage 07 计划 join DriveLM QA 作为数据源之一,其中 key objects 是核心资产。需明确它**只进训练管线,不进评测管线**。

**分工**:
| | 训练管线(Stage 07) | 评测管线(Stage 08 M09) |
|---|---|---|
| 关键目标来源 | DriveLM key objects(人判选择 + 描述语言) | nuScenes 3D GT 投影 + Q2 几何规则 |
| bbox 几何 | 全部来自 nuScenes GT(点→框匹配派生) | nuScenes GT |
| 用途 | 语言参考:教模型提到什么、怎么描述 | 判分 GT:P/R/F1、IoU |
| namespace | `training_targets` | `oracle_only` |

**推论**:DriveAlign 系统内一切 bbox 几何 100% 来自 nuScenes GT,训练侧与评测侧皆然;DriveLM 在 join 中的独特贡献只剩**选择**(哪些对象值得提)与**描述语言**。

**M09 评测 GT 资格清单(4 条,按硬度排序)与 DriveLM 对照**:
1. **版本化规则可派生**:GT 必须能从物理真值用规则程序重新生成。DriveLM ✗——固定的人判快照不可再生成(数据本身无随机性,"固定 ≠ 可派生",指控的是派生过程不存在);
2. **任意 anchor 帧可计算**:缺席只能是规则性缺席(unscorable,如 Stage 08 的风险/action 派生),不能是数据缺席。DriveLM ✗——仅 ~3.8k 个选定 keyframe 有标注,且无法扩展;
3. **bbox 几何**:DriveLM ✗——key object 格式为 `<cN, CAM_FRONT, x_pixel, y_pixel>` 像素中心点 + `Visual_description` 自由文本(代码证据:`third_party/AutoVLA/tools/preprocessing/nusc_sample_generation.py` L447-451 仅做字符串替换,全程无框几何);*(2026-10-01 修正:见下方事实更新,本条论据失效)*
4. **与训练监督构造性隔离**:DriveLM ✗(本项目处境下)——公开标注(`v1_1_train_nus.json`)仅覆盖 nuScenes train split,评测域 trainval 中不存在"DriveLM 已标注且未参与训练"的 held-out 区域(nuScenes val 无 DriveLM 标注,test 无公开 GT)。*(2026-10-01 修正:见下方事实更新,论据需改写,结论不变)*

**已否决/降级的论点(避免将来重提)**:
- 点匹配(模型 bbox 包含中心点即命中):激励画大框刷 recall,且救不了理由 1/2;
- "DriveLM 数据不确定/随机":不成立,它是固定数据;正确指控是"不可由规则派生";
- "同源即循环论证":降级——split 隔离的同源标注是标准 ML 实践;本项目否决同源路线的原因是 held-out 区域数据不存在,不是方法论缺陷。

**旁证**:DriveLM 官方 challenge 也未把 key objects 当几何 GT(评测用 GPT 语言打分 + nuScenes GT 轨迹指标)。

**一致性说明**:SFT 教模型关注"人判相关"的对象,M09 用几何规则打分;两者多数帧一致,分歧帧以几何规则为准——研究可归因性的必要代价。
**决策期限**:无需新决策,作为 Stage 06/07 实现时的执行约定;其中"test 从 nuScenes val split 划分"与"点→框匹配 join 机制"已写入 Stage 07 计划。

**事实更新(2026-10-01,DriveLM v1.1 文件实测后)**:
1. **v1.1 key_object_infos 自带 2D bbox**(`"2d_bbox": [x1, y1, x2, y2]` 像素框,附带 Category/Status/Visual_description)——上面资格清单**第 3 条论据失效**(v1.0 时代"只有中心点"的认知过时)。
2. **v1.1 提供 val split 标注**(`v1_1_val_nus.json` 存在于公开下载列表)——**第 4 条论据改写**:held-out 数据并非不存在,而是**本项目选择不引入**。
3. **结论不变**:评测 GT 资格由第 1 条(人工 QA 不可由版本化规则派生)与第 2 条(标注覆盖 14.5%,任意 anchor 不可计算)独立锁定,第 3/4 条失效不影响否决。key objects 仍为训练侧专用。
4. **训练侧 join 影响已决(2026-10-01,见 S07 输入验证记录 D1)**:全部 bbox(含 `language_reference`)一律用 nuScenes 3D 投影外接框,DriveLM `2d_bbox` 不进任何输出层——回归本 Q3 原推论"一切 bbox 几何 100% 来自 nuScenes GT"。DriveLM 贡献收缩为纯语言(Visual_description + QA → reasoning 素材),其"人判选择"倾向保留在 reasoning 文本层;两源重合度已抽样验证(IoU median 0.825)。
5. **val 版下载决策**:**不下载**。test 隔离由构造性保证(构建流水线输入中物理不含 DriveLM val 文件),强于"约定不使用";未来语言评测扩展阶段若需 reference 可再评估,届时以 scene manifest 做隔离。*(2026-10-01 补充实测:公开 val 版仅含 question 无 answer——challenge 提交格式,答案在官方侧。答案侧语义信息根本不存在,作为评测 GT/参考的资格彻底关闭;第 1 条论据在 val 上以最强形式成立。)*

## Q4 交通信号灯的词表缺口

**背景**:DriveLM key objects **包含交通信号灯**(且是官方示例的第一类),但 schema v2 的 `category` 词表为 nuScenes 十类(无信号灯类)。SFT 目标文本会出现信号灯描述,而模型输出契约表达不了它。

**候选项**:
1. 信号灯信息走场景层:进 `risk_factors`(如新增受控词 `traffic_signal_binding`,需 GT 派生规则)或仅存 reasoning(无监督);
2. Stage 07 join 时过滤:DriveLM 信号灯 QA 不进 `training_targets` 的对象字段;
3. 契约 v4 增加信号灯类(触发版本升级,需同步 GT 来源——nuScenes 检测 GT 无信号灯类,需从地图 API 派生,成本高)。

**建议**:近期选 1+2(信号灯不进对象词表,风险评估层处理),v4 是否加类等 Stage 07 实测 join 数据后再定。`unmapped_terms` 统计会暴露模型自发输出信号灯词的频率,作为决策输入。
**决策期限**:Stage 07 数据构建前。

## Q5 对象数触顶现象(第 5 轮 sanity 新观察)

**现象**:第 5 轮 original/shuffled 均输出**恰好 8 个**对象(满 `maxItems`),blank 为 2 个。提示 3B Base 可能把 prompt 中的"at most 8"当作目标数量而非上限。

**影响**:
- 若 SFT/RL 后仍触顶,precision 会被大量低置信对象稀释(取决于 Q1/Q2 的 GT 侧规则设计);
- 触顶本身也是可观测指标:正式数据统计 `objects == maxItems` 的帧比率,作为"模型是否理解 critical 语义"的代理指标;
- 风险缓解选项(若后续确认):prompt 措辞从 "at most 8" 调整为 "list only objects that affect the current driving decision";或在 M05 format 分量中对触顶帧降权(需 Stage 14 评估是否违反"防刷分"初衷)。

**决策期限**:无需立即决策;Stage 09(Base 评测)与 Stage 11(SFT 验证)监控该比率,数据驱动再定。

---

## 汇总

| # | 问题 | 影响阶段 | 决策期限 |
|---|---|---|---|
| Q1 | maxItems=8 与 GT 数量(recall 天花板) | 08 M09 | Stage 08 冻结前 |
| Q2 | GT 侧确定性几何规则(阈值标定) | 08 M09 | Stage 08 冻结前 |
| Q3 | DriveLM 分工:训练侧专用 | 06/07 | 执行约定 |
| Q4 | 信号灯词表缺口 | 07 | Stage 07 构建前 |
| Q5 | 对象数触顶监控 | 09/11 | 数据驱动 |

相关依据:S03 主记录 §6.2 事实 6–9;Stage 08 文档;Stage 07 文档;[DriveLM 官网标注流程](https://opendrivelab.com/DriveLM/);[LeapAD (arXiv:2405.15324)](https://arxiv.org/html/2405.15324)。
