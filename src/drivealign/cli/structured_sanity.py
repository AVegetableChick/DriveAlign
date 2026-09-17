"""Run the Stage 03 structured-output sanity check on one image.

Purpose:
    Run the structured pipeline on one image and two destroyed variants
    (blank and patch-shuffled). This verifies pipeline sensitivity only and
    never proves driving ability. Writes per-sample records (records.jsonl),
    a summary (summary.json), and the two variant images into --out.

Startup command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m drivealign.cli.structured_sanity \
    --image data/nuscenes/mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg \
    --out artifacts/model/qwen25_vl_3b/structured_sanity``
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from drivealign.inference.model_loader import load_model_and_processor
from drivealign.inference.structured_runner import (
    SampleRequest,
    make_blank_image,
    make_shuffled_image,
    record_to_dict,
    run_structured_batch,
)


DEFAULT_IMAGE = (
    "data/nuscenes/mini/samples/CAM_FRONT/"
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
)
DEFAULT_CONFIG = "DriveAlign/configs/model/qwen25_vl_3b.yaml"
DEFAULT_OUT = "runs/structured_infer_sanity_check"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _image_size(image: Path) -> tuple[int, int]:
    with Image.open(image) as opened:
        return opened.size


def main() -> None:
    args = _parse_args()
    image_path = args.image.expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    args.out.expanduser().mkdir(parents=True, exist_ok=True)

    original_path = args.out / "original.jpg"
    shutil.copyfile(image_path, original_path)
    blank_path = make_blank_image(image_path, args.out / "blank.jpg")
    shuffled_path = make_shuffled_image(
        image_path,
        args.out / "shuffled.jpg",
        seed=args.seed,
    )
    width, height = _image_size(image_path)

    config = yaml.safe_load(args.config.read_text())
    loaded = load_model_and_processor(config)
    generation_config = {"max_new_tokens": args.max_new_tokens}

    requests = [
        SampleRequest("original", image_path, image_size=(width, height)),
        SampleRequest("blank", blank_path, image_size=(width, height)),
        SampleRequest("shuffled", shuffled_path, image_size=(width, height)),
    ]
    records = run_structured_batch(loaded, requests, generation_config=generation_config)

    records_path = args.out / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_dict(record), ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "image": str(image_path),
        "model_path": str(loaded.model_path),
        "generation_config": {"do_sample": False, **generation_config},
        "shuffle_seed": args.seed,
        "image_size": [width, height],
        "cuda_available": torch.cuda.is_available(),
        "variants": [
            {
                "sample_id": record.sample_id,
                "status": record.status,
                "speed_action": (
                    record.output.speed_action if record.output is not None else None
                ),
                "critical_objects_count": (
                    len(record.output.critical_objects)
                    if record.output is not None
                    else None
                ),
                "error_categories": sorted(
                    {error.category.value for error in record.errors}
                ),
            }
            for record in records
        ],
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"records": str(records_path), "summary": str(summary_path)}))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
