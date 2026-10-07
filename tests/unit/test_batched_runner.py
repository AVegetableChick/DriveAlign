"""Unit tests for the S12 Step 6b batched runner (batched_v1 path).

Purpose:
    Freeze the batched-path behaviors that can be verified without GPU:
    deterministic chunking, EOS-truncated output-token accounting (bs=1
    semantics), the never-abort contract (a failing batch falls back to the
    frozen single-sample path), and batch-amortized telemetry bookkeeping.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_batched_runner.py -q``
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import drivealign.inference.batched_runner as batched_runner_module
from drivealign.contracts.output import OutputErrorCategory
from drivealign.inference.batched_runner import (
    _chunked,
    _first_eos_index,
    _per_sample_output_tokens,
    run_structured_inference_batched,
)
from drivealign.inference.structured_runner import SampleRequest


VALID_TEXT = (
    "```json\n"
    + json.dumps(
        {
            "critical_objects": [
                {
                    "category": "car",
                    "bbox_2d": [120, 300, 480, 560],
                    "motion_state": "same_direction",
                }
            ],
            "risk_factors": [],
            "yield_required": False,
            "speed_action": "KEEP_SPEED",
        }
    )
    + "\n```"
)


class TestChunked:
    def test_splits_in_order_with_remainder_tail(self):
        chunks = list(_chunked(list(range(7)), 3))
        assert [list(chunk) for chunk in chunks] == [[0, 1, 2], [3, 4, 5], [6]]

    def test_exact_multiple_yields_full_batches(self):
        chunks = list(_chunked(list(range(6)), 3))
        assert [len(chunk) for chunk in chunks] == [3, 3]

    def test_empty_sequence_yields_nothing(self):
        assert list(_chunked([], 4)) == []

    def test_rejects_non_positive_size(self):
        with pytest.raises(ValueError):
            list(_chunked([1], 0))


class TestEosAccounting:
    def test_first_eos_index_found(self):
        assert _first_eos_index([5, 6, 7, 8], {7}) == 2

    def test_first_eos_index_prefers_earliest(self):
        assert _first_eos_index([7, 5, 7], {7}) == 0

    def test_first_eos_index_missing(self):
        assert _first_eos_index([5, 6], {7}) is None

    def test_per_sample_lengths_count_eos_and_stop_at_it(self):
        # 行内 [生成..., EOS, pad, pad] → 计数含 EOS、不含 pad。
        ids = torch.tensor([[10, 11, 99, 0, 0], [20, 21, 22, 23, 24], [99, 0, 0, 0, 0]])
        assert _per_sample_output_tokens(ids, {99}) == [3, 5, 1]


def _loaded_stub() -> SimpleNamespace:
    """LoadedModel 形状的桩：generate_batch 会被 monkeypatch，不触网。"""
    return SimpleNamespace(
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=99)),
        processor=SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left")),
        model_path="/stub",
        dtype=torch.bfloat16,
    )


def _requests(n: int) -> list[SampleRequest]:
    return [
        SampleRequest(sample_id=f"tok_{i}", image=f"/img/{i}.jpg", contract_version="v6")
        for i in range(n)
    ]


class TestRunStructuredInferenceBatched:
    def test_success_path_returns_aligned_records_with_batch_telemetry(
        self, monkeypatch
    ):
        requests = _requests(3)

        def fake_generate_batch(loaded, images, prompts, *, generation_config=None):
            assert len(images) == len(prompts) == 3
            return [
                SimpleNamespace(
                    text=VALID_TEXT,
                    input_tokens=100,
                    output_tokens=10,
                    preprocess_seconds=0.3 / 3,
                    generation_seconds=6.0 / 3,
                    peak_cuda_memory_bytes=1234,
                )
                for _ in range(3)
            ]

        monkeypatch.setattr(
            batched_runner_module, "generate_batch", fake_generate_batch
        )
        records = run_structured_inference_batched(
            _loaded_stub(), requests, batch_size=3
        )
        assert [record.sample_id for record in records] == [
            request.sample_id for request in requests
        ]
        assert all(record.parse_ok for record in records)
        telemetry = records[0].telemetry
        assert telemetry["batch_size"] == 3
        assert telemetry["generation_seconds"] == pytest.approx(2.0)
        assert records[0].telemetry["peak_cuda_memory_bytes"] == 1234

    def test_batch_failure_falls_back_to_single_sample_path(self, monkeypatch):
        requests = _requests(2)

        def exploding_generate_batch(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        calls: list[str] = []

        def fake_single(loaded, request, *, generation_config=None):
            calls.append(request.sample_id)
            return SimpleNamespace(
                sample_id=request.sample_id,
                status="ok",
                parse_ok=True,
            )

        monkeypatch.setattr(
            batched_runner_module, "generate_batch", exploding_generate_batch
        )
        monkeypatch.setattr(
            batched_runner_module, "run_structured_inference", fake_single
        )
        records = run_structured_inference_batched(
            _loaded_stub(), requests, batch_size=2
        )
        # 整批 OOM → 冻结面单样本路径逐样本兜底，顺序保持。
        assert calls == ["tok_0", "tok_1"]
        assert [record.sample_id for record in records] == ["tok_0", "tok_1"]

    def test_single_sample_failure_becomes_generation_failure_record(
        self, monkeypatch
    ):
        requests = _requests(2)

        def fake_generate_batch(loaded, images, prompts, *, generation_config=None):
            return [
                SimpleNamespace(
                    text=VALID_TEXT,
                    input_tokens=100,
                    output_tokens=10,
                    preprocess_seconds=0.1,
                    generation_seconds=1.0,
                    peak_cuda_memory_bytes=None,
                )
                for _ in range(2)
            ]

        def failing_parse(text, **kwargs):
            raise ValueError("parser exploded")

        monkeypatch.setattr(
            batched_runner_module, "generate_batch", fake_generate_batch
        )
        monkeypatch.setattr(
            batched_runner_module, "parse_structured_output", failing_parse
        )
        records = run_structured_inference_batched(
            _loaded_stub(), requests, batch_size=2
        )
        assert all(
            record.status == OutputErrorCategory.GENERATION_FAILURE.value
            for record in records
        )
        assert all("parser exploded" in (record.generation_error or "") for record in records)
