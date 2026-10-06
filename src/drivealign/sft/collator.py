"""批次融合 collator（§2.4）：把逐样本的 processor 输出合成一个训练 batch。

分工（承接 `dataset.encode_chat`）
--------------------------------
`dataset.encode_chat` 已产出**单样本**张量：`input_ids`/`labels`（整条文本序列）与
`pixel_values`/`image_grid_thw`（图像张量）。本模块只做"批内对齐"，不改单个样本
的 supervisor 语义：

    - 文本侧：`input_ids` 与 `labels` 先按文本长度**右对齐** -> 右 pad。pad 处
      `input_ids` 填 `tokenizer.pad_token_id`（或 0），`labels` 填 -100（不监督）。
      `attention_mask` 同步（1=真实/0=pad），语义由 HF Trainer 消费。
    - 视觉侧：Qwen2.5-VL 一条样本可能含多张图（4F 时才多张；1F 只有 1 张）。每张图
      的 `pixel_values` 沿 patch 首维拼接，`image_grid_thw` 按 (图, v,h,w) 堆叠。
    - `remove_unused_columns=False`：collator 收到的是我们显式构造的 dict，不需要
      HF 从 Dataset 列里删列。

为什么右 pad 而不是左 pad
------------------------
因果 LM 只关注前缀。右 pad（把 pad 放在序列尾部）与 Qwen2.5-VL 训练惯例一致，
且 `labels` 尾部 pad=-100 不会参与 loss，不影响监督段。左 pad 反而会破坏注意力对齐。

本模块不引入模型；单测可用合成 tensor 验证 pad 与拼接形状（无需 GPU）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def _pad_1d(sequences: Sequence[list[int]], pad_id: int, label: bool) -> torch.Tensor:
    """把一维整型序列右 pad 成 (B, max_len) 的 tensor。

    - 非 label：pad 处填 `pad_id`。
    - label：pad 处填 -100（天然不监督，标准 SFT 约定）。
    """
    max_len = max(len(seq) for seq in sequences)
    rows = []
    pad_fill = -100 if label else pad_id
    for seq in sequences:
        amount = max_len - len(seq)
        rows.append(list(seq) + [pad_fill] * amount)
    return torch.tensor(rows, dtype=torch.long)


def collate_fn(batch: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """把一批 "encode_chat 单样本 dict" 融合成一个模型可用的 batch。

    参数
    ----
    batch : 每个元素是 `dataset.encode_chat` 的返回 dict（含 input_ids/labels/
            pixel_values/image_grid_thw，均来自同一 processor 与同一图像尺寸）。

    返回
    ----
    dict
        - `input_ids`、`labels`、`attention_mask`：文本侧 (B, max_len) tensor。
        - `pixel_values`：所有样本图像的 patch 拼接后的整体 tensor。
        - `image_grid_thw`：按样本/图堆叠后的 (N_images, 3) 长整 tensor。

    注意：返回 dict 的 key 必须**全部**是模型 forward 能接收的入参。HF Trainer 在
    `remove_unused_columns=False` 下会把 collator 输出整份透传给 `model(**inputs)`，
    任何多余 key 都会触发 `forward() got an unexpected keyword argument`。此前的
    `seq_len_max`（logging 辅助字段）就是因此导致训练首个 step 崩溃，已移除。
    """
    # --- 文本侧：分别垫 input_ids 与 labels，并据此生成 attention_mask ---
    input_ids_raw = [b["input_ids"] for b in batch]
    labels_raw = [b["labels"] for b in batch]
    input_ids_t = _pad_1d(input_ids_raw, pad_id=pad_token_id, label=False)
    labels_t = _pad_1d(labels_raw, pad_id=pad_token_id, label=True)
    # attention_mask：1=真实 token，0=右 pad 位。
    attention_mask_t = (input_ids_t != pad_token_id).long()

    # --- 视觉侧：拼接所有图像的 pixel_values，堆叠全部 image_grid_thw ---
    # HF DataLoader 交付 batch 时会对元素做隐式转换（tensor -> list / numpy）。
    # 这里统一用 torch.as_tensor 复位为 tensor，避免 torch.cat 首元素是 list 报错
    # （S11 Step 3 实测：encoder 单样本 pixel_values 已是 "(patch, 1176)" 无批维张量，
    #  但经 DataLoader 后变 list）。
    pixel_list = [torch.as_tensor(b["pixel_values"]) for b in batch if b["pixel_values"] is not None]
    # 沿 patch 首维拼接：每张图在内维度再叠，最终一张大 tensor。
    pixel_values = torch.cat(pixel_list, dim=0) if pixel_list else torch.zeros(0)
    grid_list = [torch.as_tensor(b["image_grid_thw"]) for b in batch if b["image_grid_thw"] is not None]
    image_grid_thw = torch.cat(grid_list, dim=0) if grid_list else torch.zeros(0)

    return {
        "input_ids": input_ids_t,
        "labels": labels_t,
        "attention_mask": attention_mask_t,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def make_data_collator(pad_token_id: int):
    """返回一个训练器可直接用的 collator（绑定 pad_token_id）。

    HF Trainer 的 data collator 签名是 ``callable(batch)``；本工厂把
    ``pad_token_id`` 闭包进 `collate_fn`，避免每次手动传。用法：
    ``Trainer(data_collator=make_data_collator(tokenizer.pad_token_id), ...)``
    """

    def data_collator(batch):
        return collate_fn(batch, pad_token_id=pad_token_id)

    return data_collator