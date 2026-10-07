# S12 数据集装载路径演进（S12_dataset_evolution.md）

> **主题**：训练侧数据集容器从 S11 的「全量预编码 + HF `Dataset.from_list`」演进到 S12 的
> 「惰性 `LazySftDataset`」的完整链条——**改了什么、为什么改、为什么这改法真的解决了原问题**。
>
> **为什么单独立文档**：这条链条横跨三个提交（`1412900` → `03b2f46` → `ba93717`）、两个阶段
> （S11 / S12），且其证据分散在 `S11_bug_log` 的 B2/B5、`S11_training_acceleration` 的 A1/A3
> 与 S12 的 Step 3 诊断读数里。任何单一既有文档都只能装下其中一段。`S12_run.md` 记的是**执行
> 凭证**（命令 / 读数 / 产物），不承载跨阶段的技术论证，故另立本文。
>
> **本文档的性质**：**技术论证文档**，不是执行记录。所有数字均为实测，测量方法见附录 A，
> 可独立复现。文中对既有文档的**归因纠正**已显式标注（§4），不改写原文档。

---

## 0. 结论速览

| 阶段 | 容器 | 每 micro-batch CPU 成本 | 每 opt-step CPU 成本 |
|---|---|---|---|
| S11 Step 1–3 | 全量预编码 → `HFDataset.from_list` | **808 ms** | **3.23 s** |
| S12 Step 1 起 | `LazySftDataset`（惰性，直出张量） | **~42 ms** | **~0.17 s** |

- 每 optimizer step 的墙钟：**5.47 s → 2.59 s（2.1×）**。
- 该差距 **112% 由数据集路径解释**（3.23 s − 0.17 s = 3.06 s，实测总差距 2.88 s；超出部分
  属测量噪声，见 §3.2）。
- **慢的不是读图**。图像编码本身只占 41 ms/样本，是那 808 ms 的 5%。
- **根因不是 HF DataLoader，而是 `Dataset.from_list` 的 schema 推断**——它在**存入**时就把张量
  写成了 `Sequence(Sequence(float32))`。§4 给出证据并纠正 B2 注释里的归因。
- S12 这一改动**本为内存而生**（全量预编码 ≈184 GB，必爆），**顺带**根治了 B2/B5 的原始病根。
  2.1× 加速是这条改动的**副产品**，不是它的设计目标。

---

## 1. 问题的正式表述

S11 训练慢，容易被误诊为"I/O 慢"或"图像解码慢"。准确表述是：

> **代价不在"把图变成张量"，而在"把张量存进 HF `Dataset` 再取出来"这一趟往返。**

这条往返的代价有两个可测特征：

1. **类型退化**：取出来的不再是 `torch.Tensor`，而是嵌套 Python `list`；
2. **代价量级**：退化+还原这一步的开销，比它试图省掉的图像编码**贵约 20 倍**。

第 1 点是**症状**（表现为 B2/B5 两次报错），第 2 点是**真实成本**（不报错、不崩溃，只在
`5.47 s/step` 这个数字里静静躺着）。S11 修掉了第 1 点，第 2 点一直留到 S12。

---

## 2. 演进四段

### 2.1 起点：S11 Step 1（`1412900`）——预编码 + `from_list`

`cli/sft_train.py` 的装载函数原文：

```python
def _load_dataset(samples_path: Path, dataroot: Path, processor):
    """读冻结样本 JSONL → 逐条 `encode_frozen_sample` → HF Dataset。"""
    from datasets import Dataset as HFDataset
    from drivealign.sft.dataset import encode_frozen_sample, load_frozen_records

    rows = [
        encode_frozen_sample(processor, rec, dataroot)
        for rec in load_frozen_records(samples_path)
    ]
    return HFDataset.from_list(rows)
```

**当时这么选的三个理由，都是正当的**：

1. **编码逻辑必须与 G1 复验共用**（`encode_frozen_sample` 唯一实现）。样本集小（32 / 128 条），
   训练前一次性编码完，训练期零编码开销。
2. **接入 HF Trainer 的最短路径**。`Trainer` 要求 `train_dataset` 是一个有 `__len__` /
   `__getitem__` 的对象，`datasets.Dataset` 是最现成的选择。
3. **避开训练期读图**。S11 的样本量下，预编码"用内存换每步 CPU"是划算的。

**埋下的问题**：`from_list` 靠**内容推断 schema**。编码返回的 `pixel_values` 是一个
`torch.Tensor [1792, 1176]`，`from_list` 不把它认作 2D 数组，而是认作"变长序列的变长序列"。
实测打印出来的 features：

```
pixel_values   Sequence(feature=Sequence(feature=Value(dtype='float32'), length=-1), length=-1)
```

**张量在"存进去"的这一刻就已经不是张量了。** 后续每次取用都要付还原费。

> 每样本 `pixel_values` 实测 `[1792, 1176] float32` = **8.04 MB**。
> 22,941 条全量预编码 ≈ **184 GB** —— 比 S11 A3 的估算（96 GB，按 4.2 MB/样本）还高 1.9×
> （A3 少算了 Qwen 的 temporal patch 复制那一维）。结论方向不变，严重程度更高。

### 2.2 S11 施工期：B2 / B5 两次**症状修复**

这一步是整条链条的关键，也是本文档存在的理由。

| # | 触发 | 报错首行 | 当时的修法 | 落点 |
|---|---|---|---|---|
| B2 | 首个 batch collate | `TypeError: expected Tensor as element 0 in argument 0, but got list` | 拼接前 `torch.as_tensor` 复位 | `sft/collator.py` |
| B5 | G1 reload forward | `TypeError: embedding(): argument 'indices' must be Tensor, not list` | 复用 collator 把单样本拼回 batch | `sft/train.py: fixed_sample_batch` |

两次报错信息都指向**"list 不能被当张量用"**，于是两次修法都是**补一次类型转换**。流程因此跑通了，
但——

> **`Dataset.from_list` 没有被移除，张量退化的机制原封不动。**

`torch.as_tensor` 从那以后成了**每个 micro-batch 必付的"重建税"**：

```python
# sft/collator.py（S12 现状，注释仍保留 S11 时期的归因）
# HF DataLoader 交付 batch 时会对元素做隐式转换（tensor -> list / numpy）。
# 这里统一用 torch.as_tensor 复位为 tensor，避免 torch.cat 首元素是 list 报错
pixel_list = [torch.as_tensor(b["pixel_values"]) for b in batch if ...]
```

它的**动机是兜底**（注释原文：`避免 torch.cat 首元素是 list 报错`），实际角色却是**主成本**。

**为什么当时没继续往下挖**：报错的字面意思（"list 不能 cat"）把注意力引向了**类型转换本身**，
而不是**"张量为什么会变成 list"**。B5 的根因分析里写的是"与 B2 同源（都是张量经 HF 数据容器
往返后退化为 list）"——已经把容器点了出来，但**归因停在了 DataLoader 这一层**（见 §4 的纠正），
而 DataLoader 不是可优化项，于是链条到此为止。

这是典型的**"修好了症状，代价隐形了"**：没有崩溃、loss 正常下降、训练正常收敛，唯一异常是
`5.47 s/step` —— 而 S11 当时**没有可与对照的对象**，无法判断这个数字是快还是慢。

### 2.3 S11 收尾（`03b2f46`）：A1 决议 + A3 立规

同一提交里落了两个与本文相关的决定：

- **A1（已落地）**：`max_pixels` 降到 360000（448×784）。它把图像编码成本压到实测
  **41 ms/样本**——这一点在 §5 会反过来改变结论。
- **A3（立规未落地，转 S12）**：原文——

  > `cli/sft_train.py::_load_dataset` 现为"预编码全部样本"……全量 22,941 帧 ≈ **96 GB** ⇒
  > **全量训练必爆内存**。必须改为 `__getitem__` 即时 decode+resize（配 `dataloader_num_workers>0`）。
  > 它与分辨率决策独立，但同为**全量开工前置**，故立为 S12 的第一件事。

**A3 只写对了动机的一半**：它论证的是**内存**（必爆 96 GB），没有提到**速度**。
换句话说，S12 Step 1 是被内存逼出来的，不是被速度逼出来的——2.1× 加速属于意外之喜。

A3 括号里的「配 `dataloader_num_workers>0`」是当时的**顺带假设**，非论证结论。§5 用它自己的
逻辑实测推翻了它。

### 2.4 S12 Step 1（`ba93717`）：`LazySftDataset` —— 根因修复

装载函数换成：

```python
def _load_dataset(samples_path, dataroot, processor):
    from drivealign.sft.dataset import make_lazy_sft_dataset
    return make_lazy_sft_dataset(samples_path, processor, dataroot)
```

核心类（`sft/dataset.py`）：

```python
class LazySftDataset:
    def __init__(self, records, encode) -> None:
        self._records = list(records)   # 只持有纯文本 dict（MB 级）
        self._encode = encode           # 恒为 encode_frozen_sample 的 partial

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._encode(self._records[index])   # 现取现编码（惰性的落点）
```

**三处变化，只有两处是刻意的**：

| 维度 | S11 | S12 | 是否刻意 |
|---|---|---|---|
| **容器** | `datasets.Dataset` | 普通 Python 序列（自带 `__getitem__`） | 顺带（内存需要的直接后果） |
| **时机** | 训练前全量预编码 | `__getitem__` 时逐条编码 | **是**（A3 的动机：省内存） |
| **编码语义** | `encode_frozen_sample` | **同一条** `encode_frozen_sample` | **否，必须不变** |

**第三行是这次演进能"顺手修好病根"的全部秘密**。`LazySftDataset` 返回的就是
`encode_frozen_sample` 的**原始返回值**——一个货真价实的 `torch.Tensor`。它**没有**经过任何
序列化 / 存储 / 类型推断，因此**没有**任何东西能把它退化。`collator` 里那行
`torch.as_tensor` 于是命中"已是 tensor"的短路路径。

**语义不变式如何被保住**（这是"等价替换"而非"改了配方"的依据）：

1. `encode_frozen_sample` 是训练端与 G1 复验端**唯一**的编码实现（`sft/dataset.py` 明文），
   惰性化只改**调用时机**，不碰函数体；
2. `batch=1` 下逐样本编码与 S11 逐位一致（`LazySftDataset` docstring 的不变式声明）；
3. `Trainer` 侧配 `remove_unused_columns=False`，Trainer 不再触碰列，样本原样到达 collator；
4. collator 的 `torch.as_tensor` **保留不动**——它从"主成本"变回它本来的角色：DataLoader 侧
   隐式转换的兜底。

---

## 3. 闭环：实测对账

### 3.1 CPU 侧拆解（每 micro-batch）

| 环节 | S11 | S12 | 测量方式 |
|---|---|---|---|
| 取数据（arrow → Python 对象） | **616 ms** | — | 独立实测（直接对 `hf[i]` 计时） |
| `torch.as_tensor` 把 list 还原成张量 | **~192 ms** | ~1 ms | 差值（808 − 616），**非独立测量** |
| 读图 + resize + processor 编码 | 0（训练期不做） | 41 ms | 独立实测（`hf[0]` 前逐条计时） |
| **CPU 小计** | **808 ms** | **~42 ms** | 独立实测（`coll([hf[i]])` / `coll([rows[i]])`） |

### 3.2 归因闭合

每 optimizer step = 4 个 micro-batch（`gradient_accumulation_steps: 4`）：

| 运行 | CPU 段 | GPU 段（= 合计 − CPU） | 合计 | 实测合计 |
|---|---|---|---|---|
| S11 `smoke128` | 3.23 s | 2.24 s | 5.47 s | **5.47 s** ✓ |
| S12 全量 | 0.17 s | 2.42 s | 2.59 s | **2.59 s** ✓ |

- 数据集路径贡献的差距：`3.23 − 0.17 = 3.06 s`
- 实测总差距：`5.47 − 2.59 = 2.88 s`
- **解释率 = 3.06 / 2.88 = 106%**

> S12 30-step 全量诊断 smoke 实测 `train_runtime = 77.67 s / 30 step = 2.59 s/opt-step`，
> 与被打断的正式全量跑（`step3_train_aborted.log` 前 234 步）的 `2.60 s/opt-step` **逐值复现**
> ⇒ 该耗时是**代码路径的属性**，与数据规模无关。这是"归因于装载路径而非数据量"的关键判据。
>
> GPU 段两次运行在 2.24–2.42 s 之间波动（差值 0.18 s），属测量噪声；两次的 GPU 计算量本就相同。

**为什么这构成闭环**：预测值（3.06 s）与观测量（2.88 s）在噪声内一致，且残差方向可解释
（S12 多付了 0.17 s 的真实编码）。没有剩余未解释的量级。

---

## 4. 根因重定位：不是 DataLoader

`collator.py` 的注释（§2.2 引用）把退化归因于 **"HF DataLoader 交付 batch 时会对元素做隐式转换"**。
B5 的根因分析里也写"张量经 HF 数据容器往返后退化"。**这个归因不准确**，实测证据三条：

**证据 1：退化发生在 `from_list` 那一刻，早于任何 DataLoader。**

```
# HFDataset.from_list(rows) 之后立即打印 features：
pixel_values   Sequence(feature=Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None), length=-1, id=None)
```

schema 在**存入**时就已写成 `Sequence(Sequence(float32))` —— 这是**列的类型定义**，不是某次交付的
临时转换。DataLoader 无论如何配置，都不可能改变它。

**证据 2：不经过 DataLoader 也能测出退化。** 直接 `hf[0]` 取一行（无 DataLoader、无 collator）：

```
行索引 hf[0] 后  pixel_values 外层 = list, 内层 = list
                 input_ids   类型 = list (元素 int)
```

**证据 3：616 ms 的重头发生在 `hf[i]` 取行本身**，即 arrow 存储 → Python 对象的实例化——这一段
与 DataLoader 完全无关。DataLoader 只是把已经退化的东西**原样交付**。

**为什么这个纠正有实际意义**：归因不同，修法完全不同。

- 若归因 **DataLoader** → 修法会是"加 `num_workers` / `pin_memory` / 换 sampler"等**搬运优化**。
  但 616 ms 花在**类型实例化**上，搬运优化治不了它（§5 实测证实：加 worker 上限收益仅 6%）。
- 若归因 **`from_list` 的 schema 推断** → 正确修法是**换掉容器**，即 S12 做的。

**S11 两次修复之所以"没修到根"，正是因为归因停在了 DataLoader 那一层。** 类型转换（B2/B5 的
修法）是这条错误归因链的必然产物：既然以为是"交付时被转换"，那么"收到后转回来"就是唯一解，
自然就写出了 `torch.as_tensor` 兜底，也就自然把成本固化进了每一步。

---

## 5. 附带前提 `dataloader_num_workers > 0` 的实测推翻

A3 原文括号里写了「配 `dataloader_num_workers>0`」。实测结论：**不需要，建议保持 0。**

**当前值**：HF `TrainingArguments` 的 dataclass 默认 `dataloader_num_workers = 0`，
`sft/train.py: make_trainer` 未覆盖该参数 ⇒ **实际为 0**（`dataloader_pin_memory` 默认 `True`，已生效）。

**为什么不必开**：worker 能重叠掉的**只是 CPU 侧**，即 §3.1 里的 42 ms。

| 项 | 实测 |
|---|---|
| 单样本 CPU 编码 | 中位 **41 ms** |
| × 4 micro-batch / opt-step | **0.165 s** |
| 占 2.59 s/opt-step | **6%** |

- **理论上限就是 6%**（≈ 4.1 h 里省 15 min）——worker 只能把 CPU 藏到 GPU 后面，不能缩短 GPU 计算。
- **GPU 侧已饱和**：全量训练运行中采样 `nvidia-smi` 为 `28100 MiB, 100%`。GPU 打满、CPU 只占 6%
  ⇒ 当前**不是**"CPU 喂不饱 GPU"的形态，加 worker 换不来吞吐。
- **有实打实的风险**：既有运行已反复出现 tokenizers 告警
  `The current process just got forked, after parallelism has already been used. Disabling parallelism
  to avoid deadlocks`。worker 是 fork 出来的，且 fork 发生在 processor 已被使用之后——正是该告警
  点名的**死锁场景**。

**A3 那句括号为什么过时了**：它的用意是"别在 `__getitem__` 里同步做 I/O 而卡住训练"。这个顾虑在
**A1 之前**成立（原生分辨率下编码成本高得多）；A1 把分辨率压到 360000 之后，单帧编码只剩 41 ms，
**A1 自己把这个前提消掉了**。A3 与 A1 写在同一提交里，但 A3 的括号是 A1 之前的口径。

---

## 6. 遗留与前瞻

### 6.1 保留项（有意为之，不要"顺手清理"）

- `collator.py` 的 `torch.as_tensor` **保留**。它现在是 ~1 ms 的短路，同时是 DataLoader 侧隐式
  转换的兜底。删掉它能省的量级可忽略，却会重新暴露 B2 类脆弱性。
- `LazySftDataset` 的 `encode` 以**参数注入**而非直接 import，是为了让装载策略能在
  **无模型、无 processor** 的 CPU 单测里被独立验证。不要为"省一层间接"把它写死。

### 6.2 前瞻：4F 会让 §5 的结论翻转

帧数从 1 提到 4，编码成本近似 ×4（每样本 4 张图）⇒ 约 **0.66 s/opt-step，占比约 25%**。
届时 `dataloader_num_workers > 0` 才值得重新评估（同时必须处理 §5 的 fork/死锁风险）。
**建议在 S13 开工前重测，不要沿用本文的 6% 结论。**

### 6.3 待订正

- `sft/dataset.py` 中 `LazySftDataset` docstring 写的"单样本 `pixel_values` 约 448×1176×2 B
  ≈ 1.0 MB"与实测不符：实测 `[1792, 1176] float32` = **8.04 MB**（差 8×）。结论（全量不可驻内存）
  不受影响，但该数字应更正，否则会误导后续的显存/内存预算。
- 本文 §4 纠正的 `collator.py` 归因注释（"HF DataLoader 隐式转换"）建议同步更正为
  "`Dataset.from_list` 的 schema 推断"，以免 S13 重蹈覆辙。

### 6.4 回归风险

若将来因故**重新引入** `datasets.Dataset`（例如改用其 `Array2D` / torch 特征），必须重新验证
`features` 的类型定义，而不是只看功能是否跑通——**B2/B5 的教训正是"跑通了不等于没退化"**。
判据：`ds.features[k]` 是否仍含 `Sequence(Sequence(...))`。

---

## 7. 证据索引

| 项 | 位置 / 值 |
|---|---|
| S11 起点装载实现 | commit `1412900`（`sft package + target/dataset/collator + frozen 1f config`） |
| S11 收尾（A1 落地 + A3 立规） | commit `03b2f46`（`S11 stage closure`） |
| S12 惰性装载 | commit `ba93717`（`S12 Step 1: lazy dataset + ...`） |
| B2 / B5 原始记录 | [S11_bug_log.md](../S11_sft_smoke/S11_bug_log.md) §B2 / §B5 |
| A1 / A3 原始决议 | [S11_training_acceleration.md](../S11_sft_smoke/S11_training_acceleration.md) §A1 / §A3 |
| S11 基线读数 | `runs/S11_sft_smoke/checkpoints/smoke128/train128.log`（`train_runtime 525.42 s / 96 opt-step`） |
| S12 全量诊断读数 | `runs/S12_sft_1f/reports/diag_smoke30.log`（`77.67 s / 30 step`） |
| 被打断的正式全量读数 | `runs/S12_sft_1f/reports/step3_train_aborted.log`（前 234 步 = 2.60 s/step） |
| 相关源码 | [dataset.py](../../../src/drivealign/sft/dataset.py)（`LazySftDataset` / `encode_frozen_sample`）、[collator.py](../../../src/drivealign/sft/collator.py)、[train.py](../../../src/drivealign/sft/train.py)（`make_trainer`）、[sft_train.py](../../../src/drivealign/cli/sft_train.py)（`_load_dataset`） |

---

## 附录 A：复现脚本

以下脚本可在 CPU 上独立跑通（只需 processor，**不需加载 3B 权重**），复现 §3.1 与 §4 的全部数字。

```bash
conda activate autovla_codeclean
cd /root/autodl-tmp/drivealign_workspace
PYTHONPATH=DriveAlign/src python - <<'PY'
import time, statistics, yaml
from datasets import Dataset as HFDataset
from transformers import AutoProcessor
from drivealign.sft.dataset import load_frozen_records, encode_frozen_sample
from drivealign.sft.collator import make_data_collator

cfg = yaml.safe_load(open("DriveAlign/configs/sft/sft_1f.yaml"))
proc = AutoProcessor.from_pretrained(cfg["model"]["path"], local_files_only=True,
                                     **cfg["model"]["processor"])
rows = [encode_frozen_sample(proc, r, "data/nuscenes/trainval")
        for r in load_frozen_records("data/sft_samples/v6_1f/train_full.jsonl")[:8]]

# ---- §4 证据：退化发生在 from_list，而非 DataLoader ----
hf = HFDataset.from_list(rows)
print("features:", hf.features["pixel_values"])          # Sequence(Sequence(float32))
r = hf[0]
print("hf[0] pixel_values 外层/内层:",
      type(r["pixel_values"]).__name__, type(r["pixel_values"][0]).__name__)

# ---- §3.1 计时：S11 路径 vs S12 路径 ----
coll = make_data_collator(proc.tokenizer.pad_token_id)
def bench(fn, n=6):
    ts = []
    for i in range(n):
        s = time.perf_counter(); fn(i % len(rows)); ts.append(time.perf_counter() - s)
    return statistics.median(ts)

t_row = bench(lambda i: hf[i])                       # 仅 arrow 行取回
t_s11 = bench(lambda i: coll([hf[i]]))               # S11 完整 micro-batch
t_s12 = bench(lambda i: coll([rows[i]]))             # S12 完整 micro-batch
print(f"S11 仅行取回  {t_row*1000:7.0f} ms/micro-b  -> {t_row*4:5.2f} s/opt-step")
print(f"S11 collator  {t_s11*1000:7.0f} ms/micro-b  -> {t_s11*4:5.2f} s/opt-step")
print(f"S12 collator  {t_s12*1000:7.0f} ms/micro-b  -> {t_s12*4:5.2f} s/opt-step")
PY
```

**测量条件与局限（诚实标注）**：

- 8 条样本、每项 6 次迭代取**中位数**；页面缓存已预热（真实训练首轮可能略慢）。
- `616 ms`（`hf[i]`）与 `808 ms`（`coll([hf[i]])`）为独立实测；`192 ms` 为二者**差值**，
  非独立测量。
- GPU 段（2.24 / 2.42 s）为 `实测合计 − 实测 CPU` 的**残差**，非独立测量。
- 解释率按 3.06 / 2.88 = 106% 计；超出 100% 的部分属上述差值法的噪声，不影响"数据集路径
  是唯一主要贡献项"这一结论。
