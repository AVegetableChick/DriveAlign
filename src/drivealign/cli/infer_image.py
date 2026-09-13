"""Run deterministic Qwen2.5-VL inference on one local image.

Purpose:
    Load the configured local checkpoint, run the same image and prompt three
    times with greedy decoding, and print raw outputs plus telemetry as JSON.
    The default image is the configured nuScenes-mini CAM_FRONT sample.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.infer_image --prompt "请描述图像中的道路场景。"``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from drivealign.inference.base_runner import GenerationResult, generate_one
from drivealign.inference.model_loader import load_model_and_processor


DEFAULT_IMAGE = (
    "data/nuscenes/mini/samples/CAM_FRONT/"
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
)
DEFAULT_CONFIG = "DriveAlign/configs/model/qwen25_vl_3b.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def _result_metadata(result: GenerationResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "preprocess_seconds": round(result.preprocess_seconds, 4),
        "generation_seconds": round(result.generation_seconds, 4),
        "peak_cuda_memory_bytes": result.peak_cuda_memory_bytes,
    }


def main() -> None:
    args = _parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    config = yaml.safe_load(args.config.read_text())
    loaded = load_model_and_processor(config)
    generation_config = {"max_new_tokens": args.max_new_tokens}
    results = [
        generate_one(loaded, args.image, args.prompt, generation_config)
        for _ in range(args.runs)
    ]
    texts = [result.text for result in results]
    print(
        json.dumps(
            {
                "image": str(args.image),
                "prompt": args.prompt,
                "model_path": str(loaded.model_path),
                "model_class": loaded.model.__class__.__name__,
                "processor_class": loaded.processor.__class__.__name__,
                "generation_config": {"do_sample": False, **generation_config},
                "runs": [_result_metadata(result) for result in results],
                "greedy_outputs_identical": len(set(texts)) == 1,
                "cuda_available": torch.cuda.is_available(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()