"""Unit tests for the Stage 03 structured runner batch behavior.

Purpose:
    Freeze the Stage 03 Gate cases for the runner: one bad sample never
    aborts a batch, generation problems become generation_failure records,
    raw text and errors stay traceable per sample id, and records serialize
    to JSON for per-sample error reports.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit -q``
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import drivealign.inference.structured_runner as structured_runner_module
from drivealign.contracts.output import OutputErrorCategory
from drivealign.inference.structured_runner import (
    OK_STATUS,
    SampleRequest,
    run_structured_batch,
    run_structured_inference,
)


VALID_SAMPLE = {
    "critical_objects": [
        {
            "category": "car",
            "bbox_2d": [120, 300, 480, 560],
            "motion_state": "same_direction",
        }
    ],
    "risk_factors": [],
    "reasoning": "Car ahead is moving; keep speed and monitor.",
    "yield_required": False,
    "speed_action": "KEEP_SPEED",
}

LEGACY_V1_SAMPLE = {
    "critical_objects": [
        {
            "category": "car",
            "bbox_2d": [120, 300, 480, 560],
            "coarse_position": "front center",
            "behavior": "going ahead",
        }
    ],
    "risk_factors": [],
    "reasoning": "Car ahead is moving; keep speed and monitor.",
    "yield_required": False,
    "speed_action": "KEEP_SPEED",
}


def _generation(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        input_tokens=42,
        output_tokens=99,
        preprocess_seconds=0.1,
        generation_seconds=1.5,
        peak_cuda_memory_bytes=12345678,
    )


@pytest.fixture()
def stub_generate(monkeypatch):
    """Stub the Stage 02 runner with per-image scripted outcomes."""

    def _generate(loaded, image, prompt, generation_config=None):
        name = str(image)
        if "fail" in name:
            raise RuntimeError("CUDA out of memory")
        if "schema" in name:
            bad = dict(VALID_SAMPLE, speed_action="YIELD")
            return _generation(json.dumps(bad))
        if "json" in name:
            return _generation("```json\nnot really json\n```")
        if "fence" in name:
            return _generation("```json\n" + json.dumps(VALID_SAMPLE) + "\n```")
        if "semantic" in name:
            bad = json.loads(json.dumps(VALID_SAMPLE))
            bad["critical_objects"][0]["bbox_2d"] = [480, 300, 120, 560]
            return _generation(json.dumps(bad))
        if "legacy" in name:
            return _generation(json.dumps(LEGACY_V1_SAMPLE))
        return _generation(json.dumps(VALID_SAMPLE))

    monkeypatch.setattr(structured_runner_module, "generate_one", _generate)
    return _generate


@pytest.mark.usefixtures("stub_generate")
def test_mixed_batch_maps_all_error_categories():
    requests = [
        SampleRequest("ok-1", "img_ok.jpg"),
        SampleRequest("gen-fail-1", "img_fail.jpg"),
        SampleRequest("schema-fail-1", "img_schema.jpg"),
        SampleRequest("json-fail-1", "img_json.jpg"),
        SampleRequest("semantic-fail-1", "img_semantic.jpg", image_size=(1280, 720)),
    ]
    records = run_structured_batch(None, requests)
    assert [record.status for record in records] == [
        OK_STATUS,
        OutputErrorCategory.GENERATION_FAILURE.value,
        OutputErrorCategory.SCHEMA_FAILURE.value,
        OutputErrorCategory.JSON_PARSE_FAILURE.value,
        OutputErrorCategory.SEMANTIC_RANGE_FAILURE.value,
    ]
    assert [record.sample_id for record in records] == [
        request.sample_id for request in requests
    ]


@pytest.mark.usefixtures("stub_generate")
def test_generation_failure_keeps_raw_none_and_error_message():
    record = run_structured_inference(None, SampleRequest("gen-1", "img_fail.jpg"))
    assert record.status == OutputErrorCategory.GENERATION_FAILURE.value
    assert record.raw_text is None and record.output is None
    assert record.generation_error == "RuntimeError: CUDA out of memory"
    assert record.errors[0].category == OutputErrorCategory.GENERATION_FAILURE


@pytest.mark.usefixtures("stub_generate")
def test_ok_record_keeps_raw_text_and_telemetry():
    record = run_structured_inference(None, SampleRequest("ok-1", "img_ok.jpg"))
    assert record.parse_ok and record.telemetry["output_tokens"] == 99
    assert json.loads(record.raw_text) == VALID_SAMPLE


@pytest.mark.usefixtures("stub_generate")
def test_schema_error_field_is_traced():
    record = run_structured_inference(None, SampleRequest("s-1", "img_schema.jpg"))
    assert record.errors[0].field == "$.speed_action"


@pytest.mark.usefixtures("stub_generate")
def test_speed_request_enters_prompt_and_record():
    record = run_structured_inference(
        None,
        SampleRequest("speed-1", "img_ok.jpg", available_speed="current 8.3 m/s"),
    )
    assert record.prompt.startswith("Currently available ego speed: current 8.3 m/s")
    assert record.available_speed == "current 8.3 m/s"


@pytest.mark.usefixtures("stub_generate")
def test_record_to_dict_is_json_serializable():
    records = run_structured_batch(
        None,
        [
            SampleRequest("ok-1", "img_ok.jpg"),
            SampleRequest("fail-1", "img_fail.jpg"),
        ],
    )
    blob = json.dumps([structured_runner_module.record_to_dict(r) for r in records])
    restored = json.loads(blob)
    assert restored[0]["status"] == OK_STATUS
    assert restored[0]["output"]["critical_objects"][0]["bbox_2d"] == [
        120.0,
        300.0,
        480.0,
        560.0,
    ]
    assert restored[1]["errors"][0]["category"] == "generation_failure"


@pytest.mark.usefixtures("stub_generate")
def test_fenced_output_is_extracted_with_flag():
    record = run_structured_inference(None, SampleRequest("fence-1", "img_fence.jpg"))
    assert record.status == OK_STATUS
    assert record.extracted_from_fences is True
    assert record.output.speed_action == "KEEP_SPEED"


@pytest.mark.usefixtures("stub_generate")
def test_default_record_carries_contract_v5():
    record = run_structured_inference(None, SampleRequest("ok-1", "img_ok.jpg"))
    assert record.contract_version == "v5"  # S09: prompt aligned to 7-item schema


@pytest.mark.usefixtures("stub_generate")
def test_contract_v2_reproduces_recorded_s03_pairing():
    record = run_structured_inference(
        None,
        SampleRequest("legacy-1", "img_legacy.jpg", contract_version="v2"),
    )
    assert record.status == OK_STATUS
    assert record.contract_version == "v2"
    assert record.output.critical_objects[0].coarse_position == "front center"


def test_missing_image_becomes_generation_failure():
    record = run_structured_inference(None, SampleRequest("missing", "no_such.jpg"))
    assert record.status == OutputErrorCategory.GENERATION_FAILURE.value
    assert "FileNotFoundError" in record.generation_error


def test_batch_of_failures_still_returns_every_sample():
    records = run_structured_batch(
        None,
        [
            SampleRequest("missing-1", "a_missing.jpg"),
            SampleRequest("missing-2", "b_missing.jpg"),
        ],
    )
    assert len(records) == 2
    assert all(
        record.status == OutputErrorCategory.GENERATION_FAILURE.value
        for record in records
    )
