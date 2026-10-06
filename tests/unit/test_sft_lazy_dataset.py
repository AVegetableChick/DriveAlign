"""S12 Step 1 单测：惰性 map-style Dataset（`sft/dataset.LazySftDataset`）。

覆盖 §3 Step 1 的 `test_sft_lazy_dataset` 三条：
    ① `__getitem__` 输出与 S11 的 `encode_frozen_sample` 逐张量一致
       （防装载策略漂移改变训练语义）——以注入式 encoder 表达，因为真实
       `encode_frozen_sample` 需要 processor + 真图（CPU 单测无模型环境）；
    ② 随机访问顺序不变性（HF RandomSampler 每 epoch 换序）；
    ③ 长序列 pad 语义（经真实 collator 拼批，右 pad + labels=-100）。

另含惰性本身的可观测断言（构造时零编码）与 JSONL 读入顺序。

启动命令：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_sft_lazy_dataset.py -q``
"""

from __future__ import annotations

import json

import pytest
import torch

from drivealign.sft.collator import collate_fn
from drivealign.sft.dataset import LazySftDataset, load_frozen_records


def _record(token: str, length: int) -> dict:
    """一条最小可用的"冻结样本 JSONL 记录"（只需 encoder 关心的字段）。"""
    return {
        "sample_token": token,
        "sample_sha256": f"sha-{token}",
        "prompt": f"prompt-{token}",
        "image_relpath": f"images/{token}.jpg",
        "target_text": "{}",
        "length": length,
    }


def _fake_encode(record: dict) -> dict:
    """与 `encode_frozen_sample` **同型**的返回 dict（张量 + 追踪字段）。

    长度取自 record，用于后面的 pad 语义测试；张量值由 token 与长度决定，
    以便断言"顺序无关"。
    """
    length = int(record["length"])
    value = float(len(record["sample_token"]))
    return {
        "input_ids": [1] * length,
        "labels": [-100] * (length - 2) + [1, 1],
        # 与真实 processor 同构：变长维在 patch（dim 0），内维固定（1176）——
        # collator 正是沿 dim 0 拼接。
        "pixel_values": torch.full((2 * length, 3), value),
        "image_grid_thw": torch.tensor([[1, 2, length]]),
        "sample_token": record["sample_token"],
        "sample_sha256": record["sample_sha256"],
    }


def _assert_sample_equal(left: dict, right: dict) -> None:
    assert set(left) == set(right)
    for key in left:
        if torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key]), f"tensor field differs: {key}"
        else:
            assert left[key] == right[key], f"field differs: {key}"


class TestLazySftDataset:
    @pytest.fixture
    def records(self) -> list[dict]:
        return [_record("tok-a", 5), _record("tok-b", 9), _record("tok-c", 7)]

    def test_getitem_matches_direct_encode(self, records):
        """① 逐张量等于直接调用同一 encoder（装载策略不改编码语义）。"""
        dataset = LazySftDataset(records, _fake_encode)
        assert len(dataset) == len(records)
        for index, record in enumerate(records):
            _assert_sample_equal(dataset[index], _fake_encode(record))

    def test_construction_encodes_nothing(self, records):
        """惰性可观测：构造时不编码，取一次编一次（无隐藏缓存）。"""
        calls: list[str] = []

        def counting_encode(record):
            calls.append(record["sample_token"])
            return _fake_encode(record)

        dataset = LazySftDataset(records, counting_encode)
        assert calls == []          # 构造零编码
        dataset[1]
        assert calls == ["tok-b"]   # 只有被取的那条被编码
        dataset[1]
        assert calls == ["tok-b", "tok-b"]  # 重复取会重编码（不缓存张量 → 内存 O(1)）

    def test_random_access_order_invariant(self, records):
        """② 任何访问顺序下，同一 index 的结果逐字段相同。"""
        forward = LazySftDataset(records, _fake_encode)
        backward = LazySftDataset(records, _fake_encode)
        forward_samples = [forward[index] for index in range(len(records))]
        # 逆序访问：结果按 index 收集，用于与正序逐条比对。
        backward_samples = {index: backward[index] for index in reversed(range(len(records)))}
        for index, record in enumerate(records):
            expected = _fake_encode(record)
            _assert_sample_equal(forward_samples[index], expected)
            _assert_sample_equal(backward_samples[index], expected)

    def test_index_errors_and_negative_index(self, records):
        dataset = LazySftDataset(records, _fake_encode)
        with pytest.raises(IndexError):
            dataset[len(records)]
        # 负索引沿用 Python 列表语义（sampler 不会产生，但不应意外成功成别的样本）。
        _assert_sample_equal(dataset[-1], _fake_encode(records[-1]))

    def test_long_sequence_pad_semantics(self, records):
        """③ 长短样本混批：右 pad，labels pad=-100，attention_mask pad=0。"""
        dataset = LazySftDataset(records, _fake_encode)
        batch = collate_fn([dataset[0], dataset[1]], pad_token_id=0)
        max_len = max(record["length"] for record in records[:2])

        assert tuple(batch["input_ids"].shape) == (2, max_len)
        assert tuple(batch["labels"].shape) == (2, max_len)
        assert tuple(batch["attention_mask"].shape) == (2, max_len)

        # 短样本（len=5）的尾部 4 位是 pad：input_ids=0、labels=-100、mask=0。
        pad_width = max_len - records[0]["length"]
        assert pad_width > 0
        assert batch["input_ids"][0, -pad_width:].tolist() == [0] * pad_width
        assert batch["labels"][0, -pad_width:].tolist() == [-100] * pad_width
        assert batch["attention_mask"][0, -pad_width:].tolist() == [0] * pad_width
        # 长样本无 pad：mask 全 1。
        assert batch["attention_mask"][1].tolist() == [1] * max_len
        # 视觉侧沿 patch 首维拼接：两样本各 2*len 行。
        assert batch["pixel_values"].shape[0] == sum(2 * record["length"] for record in records[:2])
        assert tuple(batch["image_grid_thw"].shape) == (2, 3)

    def test_records_come_from_jsonl_in_file_order(self, tmp_path, records):
        """工厂输入的 JSONL 语义：`load_frozen_records` 保文件顺序，索引即文件行序。"""
        path = tmp_path / "samples.jsonl"
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        loaded = load_frozen_records(path)
        assert [record["sample_token"] for record in loaded] == ["tok-a", "tok-b", "tok-c"]
        dataset = LazySftDataset(loaded, _fake_encode)
        for index, record in enumerate(records):
            _assert_sample_equal(dataset[index], _fake_encode(record))
