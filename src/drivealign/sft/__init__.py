"""S11 SFT 计算 Smoke 的训练侧包（`src/drivealign/sft/`，Step 1 交付）。

本包的职责与既有体育逻辑保持同源：

- `targets.py`  —— 训练目标构造（四字段显式列举 + 整串 dump + 确定性断言）。
- `dataset.py`  —— record → 训练样本：读 shard record，用 1F v6 面序列化，
                   做"训练面 == 评测面"parity 断言，产出逐样本 sha256。
- `collator.py` —— 逐样本 processor 处理后的批融合（文本右 pad、图像沿 patch 维拼接）。
- `gate.py`    —— G1 判定纯逻辑（硬层状态保真：逐张量/step/NaN-Inf；软层功能冒烟：
                 四字段语义相等），CPU 可测。
- `train.py`   —— LoRA 训练入口与新进程 reload 的模型管道（判定逻辑在 `gate.py`）。

`targets`/`dataset`/`collator`/`gate` 均不依赖本地 GPU，可由 CPU 单测独立验证；
训练入口见 `cli/sft_train.py`，G1 复验入口见 `cli/sft_reload_verify.py`。
"""

from drivealign.sft.targets import TARGET_FIELD_ORDER, as_target, serialize_target
from drivealign.sft.dataset import TrainingSample, build_training_sample

__all__ = [
    "TARGET_FIELD_ORDER",
    "as_target",
    "serialize_target",
    "TrainingSample",
    "build_training_sample",
]