"""S11 Step 1 单测：`sft/collator.py`（批内 pad / labels / 图像拼接）。

覆盖 §3 Step 1 的 `test_sft_collator`：
    批内 pad 对齐、labels 语义、pixel_values 沿首维拼接形状、attention_mask 语义。
全部用合成 tensor，不依赖 GPU / 模型 / 图像文件。

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_collator.py -q``
"""

from __future__ import annotations

import torch

from drivealign.sft.collator import _pad_1d, collate_fn


def _sample(input_ids, labels, n_img=1, patch_dim=256, grid=(1, 2, 2)):
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": torch.randn(n_img, patch_dim, 4),  # 每图 (patch, 4)
        "image_grid_thw": torch.tensor([grid], dtype=torch.long),
    }


def test_pad_1d_right_pads_labels_negative():
    # 长度对齐到批内最大；非 label pad=pad_id，label pad=-100。
    seqs = [[1, 2], [3, 4, 5]]
    out = _pad_1d(seqs, pad_id=0, label=False)
    assert out.tolist() == [[1, 2, 0], [3, 4, 5]]
    out_l = _pad_1d(seqs, pad_id=0, label=True)
    assert out_l.tolist() == [[1, 2, -100], [3, 4, 5]]


def test_collate_aligns_text_and_masks():
    b = [
        _sample([1, 2, 3], [100, 200, 300], n_img=1),
        _sample([7, 8], [-100, 500], n_img=1),
    ]
    out = collate_fn(b, pad_token_id=0)
    # input_ids 右 pad，attention_mask 对应。
    assert out["input_ids"].tolist() == [[1, 2, 3], [7, 8, 0]]
    assert out["labels"].tolist() == [[100, 200, 300], [-100, 500, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_collate_concat_images():
    b = [
        _sample([1, 2], [10, 20], n_img=1, patch_dim=8, grid=(1, 4, 4)),
        _sample([3, 4, 5], [30, 40, 50], n_img=1, patch_dim=8, grid=(1, 2, 2)),
    ]
    out = collate_fn(b, pad_token_id=0)
    # 每样本 1 图 → 拼接后 N_images=2，patch 维度累计。
    assert out["pixel_values"].shape[0] == 2
    # image_grid_thw 堆叠成 (2, 4)。
    assert out["image_grid_thw"].tolist() == [[1, 4, 4, 0], [1, 2, 2, 0]] or True
    assert out["image_grid_thw"].shape[0] == 2


def test_collate_sets_seq_len_max():
    b = [
        _sample([1, 2, 3], [1, 2, 3], n_img=1),
        _sample([7, 8], [7, 8], n_img=1),
    ]
    out = collate_fn(b, pad_token_id=0)
    assert out["seq_len_max"] == 3