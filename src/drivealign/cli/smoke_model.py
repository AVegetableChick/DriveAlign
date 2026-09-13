"""Stage 01 smoke test for loading the local Qwen2.5-VL checkpoint.

Purpose:
    Load the configured local model and processor, then print loading metadata
    and CUDA memory telemetry without duplicating loader logic.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.smoke_model --config DriveAlign/configs/model/qwen25_vl_3b.yaml``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from drivealign.inference.model_loader import load_model_and_processor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    loaded = load_model_and_processor(config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    metadata = {
        "model_path": str(loaded.model_path),
        "model_class": loaded.model.__class__.__name__,
        "processor_class": loaded.processor.__class__.__name__,
        "dtype": str(loaded.dtype),
        "device": str(next(loaded.model.parameters()).device),
        "load_seconds": round(time.perf_counter() - started, 3),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()