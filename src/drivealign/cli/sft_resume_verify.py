"""S12 Step 4 save/resume 验证 CLI（`cli/sft_resume_verify.py`，**新进程**，GPU）。

职责（`S12_implementation_plan.md` §2.9，G 拍板）
-----------------------------------------------
蓝图要求"确认 optimizer、scheduler、global step 和随机状态恢复"。GPU 续训逐位复现
不现实，故**预注册为状态序列化层验证**（沿 S11 G1 的分层思路，两层）：

- **硬层（状态保真，纯 CPU 可判）**：对 Trainer 落下的中间 ckpt 目录逐项断言——
  1. 必需 artifact 齐备（`trainer_state.json` / `optimizer.pt` / `scheduler.pt` /
     `rng_state.pth` / adapter 权重）；
  2. `optimizer.pt` 的 state 条目数与 adapter 参数张量数**相等**（LoRA 全参数
     都有梯度 → 都应进 optimizer），且每个 state 条目的张量 shape 与某个 adapter
     参数 shape 逐一对齐（多重集相等，判别的是"state 与权重对不上号"）；
  3. 每个 state 条目引用的参数 id 出现在某个 `param_groups[*].params` 里（
     optimizer 内部自洽）；
  4. `scheduler.pt` 的 `last_epoch` == `trainer_state.json` 的 `global_step`
     （证明调度器与步数同批落盘、未脱节）；
  5. `rng_state.pth` 含 torch / cuda / numpy / python 四段，且 torch 段能被
     `torch.Generator().set_state()` 成功载入（可恢复性）。
- **功能冒烟层（生成合法输出，容忍 kernel 非确定性）**：新进程用
  `sft.train.load_adapter_model` 从 ckpt 目录重建 PeftModel，在固定样本上贪心
  生成一次，要求 `parse_structured_output`（v6）可解析。**不比字节**。

退出码：PASS → 0；FAIL → 1（便于 tmux/CI 直接判）。软层未提供 `--samples` /
`--dataroot` 时只报硬层结论，并在报告里显式标 `evaluated: false`（不假装跑过）。

启动命令：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.sft_resume_verify \
    --config DriveAlign/configs/sft/sft_1f.yaml \
    --checkpoint runs/S12_sft_1f/checkpoints/sft_1f/checkpoint-500 \
    --samples data/sft_samples/v6_1f/train_full.jsonl \
    --dataroot data/nuscenes/trainval \
    --report runs/S12_sft_1f/reports/resume_verify.json``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

#: Trainer 中间 ckpt 目录里硬层验证所需的最小 artifact 集合。
REQUIRED_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
#: RNG 状态必须覆盖的四段（torch 可 load 性单独判）。
REQUIRED_RNG_KEYS = ("torch", "cuda", "numpy", "python")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify S12 save/resume (hard + smoke)")
    parser.add_argument("--config", required=True, help="sft config YAML")
    parser.add_argument(
        "--checkpoint", required=True, help="Trainer checkpoint dir (contains optimizer.pt etc.)"
    )
    parser.add_argument("--samples", default=None, help="frozen samples JSONL for the smoke layer")
    parser.add_argument("--dataroot", default=None, help="nuScenes dataroot for the smoke layer")
    parser.add_argument("--report", default=None, help="optional path to write the JSON report")
    return parser.parse_args(argv)


def _adapter_shapes(checkpoint: Path) -> dict[str, list[int]]:
    """读 ckpt 目录里的 adapter 权重 → 参数名 -> shape。

    优先 safetensors（peft 默认落盘格式），回退 `.bin`（旧格式/torch.save）。
    """
    safetensors_path = checkpoint / "adapter_model.safetensors"
    bin_path = checkpoint / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        tensors: Mapping[str, Any] = load_file(str(safetensors_path))
    elif bin_path.is_file():
        tensors = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"no adapter weights in {checkpoint} (looked for adapter_model.safetensors/.bin)"
        )
    return {name: list(tensor.shape) for name, tensor in tensors.items()}


def _optimizer_state_shapes(optimizer: Mapping[str, Any]) -> dict[int, list[int]]:
    """每个 optimizer state 条目的代表 shape（取 exp_avg，其次任一非标量张量）。"""
    shapes: dict[int, list[int]] = {}
    for index, entry in dict(optimizer.get("state", {})).items():
        shape: list[int] | None = None
        for key in ("exp_avg", "exp_avg_sq"):
            value = entry.get(key) if isinstance(entry, Mapping) else None
            if torch.is_tensor(value):
                shape = list(value.shape)
                break
        if shape is None:
            for value in entry.values() if isinstance(entry, Mapping) else []:
                if torch.is_tensor(value) and value.ndim > 0:
                    shape = list(value.shape)
                    break
        if shape is not None:
            shapes[int(index)] = shape
    return shapes


def verify_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """硬层验证（纯 CPU、无模型）：返回结构化报告 + `passed`。

    逐项结论都写进返回 dict，便于 Step 4 回填 `S12_run.md` 时直接引用，
    也便于"哪一条挂了"可定位（不是只给一个 bool）。
    """
    checkpoint = Path(checkpoint)
    missing_files = [name for name in REQUIRED_FILES if not (checkpoint / name).is_file()]

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "missing_files": missing_files,
    }
    if missing_files:
        # 文件不全时后续断言无法进行：直接判 FAIL，避免"缺文件但误判通过"。
        report["passed"] = False
        return report

    adapter_shapes = _adapter_shapes(checkpoint)
    adapter_shape_list = sorted(tuple(shape) for shape in adapter_shapes.values())

    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu")
    state_shapes = _optimizer_state_shapes(optimizer)
    state_shape_list = sorted(tuple(shape) for shape in state_shapes.values())

    # optimizer 内部自洽：每个有 state 的参数 id 必须出现在某个 param_group。
    grouped_ids: set[int] = set()
    for group in optimizer.get("param_groups", []):
        grouped_ids.update(int(index) for index in group.get("params", []))
    orphan_state_ids = sorted(set(state_shapes) - grouped_ids)

    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu")
    rng_state = torch.load(checkpoint / "rng_state.pth", map_location="cpu")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))

    global_step = int(trainer_state["global_step"])
    last_epoch = int(scheduler["last_epoch"])
    missing_rng_keys = [key for key in REQUIRED_RNG_KEYS if key not in rng_state]
    try:
        generator = torch.Generator()
        generator.set_state(rng_state["torch"])
        torch_rng_loadable = True
    except Exception:  # pragma: no cover - 仅在 rng_state 损坏时触发
        torch_rng_loadable = False

    checks = {
        "adapter_tensor_count": len(adapter_shapes),
        "optimizer_state_count": len(state_shapes),
        "counts_match": len(adapter_shapes) == len(state_shapes),
        "shapes_match": adapter_shape_list == state_shape_list,
        "orphan_state_ids": orphan_state_ids,
        "param_groups_consistent": not orphan_state_ids,
        "global_step": global_step,
        "scheduler_last_epoch": last_epoch,
        "step_and_epoch_match": last_epoch == global_step,
        "missing_rng_keys": missing_rng_keys,
        "torch_rng_loadable": torch_rng_loadable,
    }
    passed = bool(
        checks["counts_match"]
        and checks["shapes_match"]
        and checks["param_groups_consistent"]
        and checks["step_and_epoch_match"]
        and not missing_rng_keys
        and torch_rng_loadable
    )
    report.update({"passed": passed, "checks": checks})
    return report


def functional_smoke(config: dict, checkpoint: Path, samples: str, dataroot: str) -> dict[str, Any]:
    """软层：新进程重建 PeftModel → 固定样本贪心生成 → v6 严格解析。"""
    from drivealign.contracts.output import parse_structured_output
    from drivealign.inference.base_runner import generate_one
    from drivealign.sft.dataset import load_frozen_records
    from drivealign.sft.train import load_adapter_model

    record = load_frozen_records(samples)[0]
    loaded = load_adapter_model(config, checkpoint)
    generation = generate_one(
        loaded,
        Path(dataroot) / record["image_relpath"],
        record["prompt"],
        generation_config=config.get("generation"),
    )
    parsed = parse_structured_output(
        generation.text, contract_version=str(config["contract_version"])
    )
    return {
        "evaluated": True,
        "passed": bool(parsed.ok),
        "sample_token": record["sample_token"],
        "parse_ok": bool(parsed.ok),
        "output_tokens": int(generation.output_tokens),
        "generated_text": generation.text,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = Path(args.checkpoint)

    report = verify_checkpoint(checkpoint)

    # 软层：给了 samples + dataroot 才跑（缺则显式标未评估，不假装跑过）。
    if args.samples and args.dataroot and report.get("missing_files") == []:
        report["functional_layer"] = functional_smoke(
            config, checkpoint, args.samples, args.dataroot
        )
    else:
        report["functional_layer"] = {
            "evaluated": False,
            "reason": "provide --samples and --dataroot to run the generation smoke layer",
        }

    # 总判定 = 硬层 PASS 且（软层未评估 或 软层 PASS）。
    hard_passed = bool(report.get("passed"))
    smoke = report["functional_layer"]
    passed = bool(hard_passed and (smoke.get("passed") is not False))
    report["functional_layer_evaluated"] = bool(smoke.get("evaluated"))
    report["overall_passed"] = passed

    print(json.dumps(report, sort_keys=True, ensure_ascii=False), flush=True)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
