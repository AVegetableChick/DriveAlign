"""S11 G1 复验 CLI（`cli/sft_reload_verify.py`，**新进程**，GPU 执行）。

职责（`S11_implementation_plan.md` §7-I）
----------------------------------------
在**训练进程之外**，仅凭磁盘 artifact（配置基座 + adapter 目录 + 冻结样本）复现
并判定 G1 的两层：

- **硬层（状态保真，确定性）**
    1. 逐张量比对：新进程重载的可训练权重 vs `train_state_ref.pt` 的参照
       （shape/dtype 同构，`max|Δ| < 1e-6`；见 `sft/gate.compare_state_dicts`）。
    2. global step：参照记录的步数 == 由配置 + 样本数推导的期望步数
       （`sft/gate.expected_global_step`），证明训练按注册调度跑满、参照是跑完那刻落盘。
    3. 无 NaN/Inf：权重侧 `isnan | isinf`（`sft/gate.nonfinite_parameter_names`）。
- **软层（功能冒烟，容忍 token 级非确定）**
    4. 固定输入（train32 第一条，经与训练**同一个 collator** 拼 batch）forward 的
       logits 有限。
    5. 同一 1F 输入贪心生成 → `parse_structured_output`（v6）可解析；
       四字段与**训练端参照输出**语义相等（`sft/gate.four_fields_equal`）。

软层基准的由来（用户 2026-10-06 拍板）
---------------------------------------
软层判的是 `save adapter → 新进程 reload` 后**模型功能不变**，因此比对对象必须是
"训练端在同一固定样本上的输出"（参照里 `generation` 段），而**不是**冻结 GT：
训练 32/128 条样本本就不可能学会全部 GT，拿 GT 当基准会把"训练效果不足"误判成
"G1 复验失败"。与 GT 的一致率降为报告里的信息项 `soft_layer.vs_gt`（只做记录，
不参与 `passed`）。

退出码：全部 PASS → 0；任一 FAIL → 1（便于 tmux/CI 直接判）。

先训练（另起进程落盘 adapter + 参照）：
    ``... python -m drivealign.cli.sft_train --config ... --samples ... --out ... --dataroot ...``
再复验：
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.sft_reload_verify \
    --config DriveAlign/configs/sft/sft_smoke_1f.yaml \
    --adapter runs/S11_sft_smoke/checkpoints/smoke32 \
    --samples runs/S11_sft_smoke/sft_samples/train32.jsonl \
    --dataroot data/nuscenes/trainval``

本 CLI 依赖本地模型，无法在 CPU 无模型环境跑通，不参与 CPU 单测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify S11 G1 (hard + soft layers)")
    parser.add_argument("--config", required=True, help="sft config YAML")
    parser.add_argument("--adapter", required=True, help="adapter dir saved by sft_train")
    parser.add_argument("--samples", required=True, help="frozen samples JSONL used for training")
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot for images")
    parser.add_argument("--overfit", action="store_true", help="expect overfit128 schedule")
    parser.add_argument(
        "--ref",
        default=None,
        help="state reference file (default: <adapter>/train_state_ref.pt)",
    )
    parser.add_argument("--report", default=None, help="optional path to write JSON report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    adapter_dir = Path(args.adapter)
    ref_path = Path(args.ref) if args.ref else adapter_dir / "train_state_ref.pt"

    from drivealign.contracts.output import parse_structured_output
    from drivealign.inference.base_runner import generate_one
    from drivealign.sft.dataset import load_frozen_records
    from drivealign.sft.gate import (
        adapter_state_dict,
        compare_state_dicts,
        expected_global_step,
        four_fields_equal,
        nonfinite_parameter_names,
    )
    from drivealign.sft.train import (
        fixed_sample_batch,
        forward_fixed_batch,
        load_adapter_model,
    )

    records = load_frozen_records(args.samples)

    # ---------- 新进程重载（仅从磁盘 artifact 重建）----------
    loaded = load_adapter_model(config, adapter_dir)
    reference = torch.load(ref_path, map_location="cpu")

    # ---------- 硬层 ----------
    reloaded_state = adapter_state_dict(loaded.model)
    comparison = compare_state_dicts(reference["adapter_state"], reloaded_state)
    nonfinite = nonfinite_parameter_names(reloaded_state)

    reference_step = int(reference["global_step"])
    expected_step = expected_global_step(config, len(records), overfit=args.overfit)
    step_equal = reference_step == expected_step

    hard_pass = bool(comparison["passed"]) and not nonfinite and step_equal

    # ---------- 软层①：固定 batch forward 的 logits 有限 ----------
    # 固定输入走 training 端共用的 `fixed_sample_batch`（编码 + collator 同一实现），
    # 保证与训练时逐位一致。
    record, batch = fixed_sample_batch(
        loaded.processor, Path(args.samples), Path(args.dataroot)
    )
    logits = forward_fixed_batch(loaded.model, batch)
    logits_finite = bool(torch.isfinite(logits).all())
    logits_shape = list(logits.shape)
    # logits 是 [1, seq, vocab] 的大张量（vocab≈15万）；只保留判定结论后立即释放，
    # 给紧随其后的生成腾出显存（单卡 32GB 上 3B + KV 已接近上限）。
    del logits

    # ---------- 软层②：生成 → 解析 → 与**训练端参照输出**四字段语义相等 ----------
    # 参照是训练端在同一固定样本上的贪心输出（`train_state_ref.pt` 的 `generation` 段）。
    # 比对语义 = "save→reload 后模型功能不变"；与 GT 的一致率另记为信息项。
    reference_generation = reference.get("generation")
    if not reference_generation:
        raise ValueError(
            f"state reference {ref_path} has no 'generation' section; retrain with the "
            "updated cli/sft_train so it records the training-side generation baseline "
            "(soft layer compares reload output vs training output)"
        )

    contract_version = str(config["contract_version"])
    image_path = Path(args.dataroot) / record["image_relpath"]
    generation = generate_one(
        loaded,
        image_path,
        record["prompt"],
        generation_config=config.get("generation"),
    )
    parsed = parse_structured_output(generation.text, contract_version=contract_version)
    # 参照文本在训练端已落盘，这里同样按 v6 严格解析；解析失败视为软层 FAIL。
    ref_parsed = parse_structured_output(
        str(reference_generation["text"]), contract_version=contract_version
    )

    # 参照必须是同一条固定样本（防御：参照与 --samples 对不上号时不能算通过）。
    ref_sample_token = reference_generation.get("sample_token")
    sample_token_match = ref_sample_token == record["sample_token"]

    if parsed.ok and ref_parsed.ok:
        fields_equal, field_diffs = four_fields_equal(
            parsed.output.to_dict(), ref_parsed.output.to_dict()
        )
    else:
        fields_equal, field_diffs = False, ["<unparseable>"]

    # 与冻结 GT 的一致率：**信息项**（G1 判状态保真，不判训练效果），不参与 passed。
    gt_fields = json.loads(record["target_text"])
    if parsed.ok:
        gt_equal, gt_field_diffs = four_fields_equal(parsed.output.to_dict(), gt_fields)
    else:
        gt_equal, gt_field_diffs = False, ["<unparseable>"]

    soft_pass = bool(
        logits_finite
        and sample_token_match
        and parsed.ok
        and ref_parsed.ok
        and fields_equal
    )
    passed = bool(hard_pass and soft_pass)

    report = {
        "passed": passed,
        "config": str(args.config),
        "adapter_dir": str(adapter_dir),
        "state_reference": str(ref_path),
        "sample_token": record["sample_token"],
        "n_samples": len(records),
        "hard_layer": {
            "passed": hard_pass,
            "n_adapter_tensors": len(reloaded_state),
            "worst_abs_diff": comparison["worst_abs_diff"],
            "atol": comparison["atol"],
            "missing": comparison["missing"],
            "unexpected": comparison["unexpected"],
            "shape_mismatch": comparison["shape_mismatch"],
            "dtype_mismatch": comparison["dtype_mismatch"],
            "nonfinite_parameters": nonfinite,
            "reference_global_step": reference_step,
            "expected_global_step": expected_step,
            "global_step_equal": step_equal,
        },
        "soft_layer": {
            "passed": soft_pass,
            "logits_shape": logits_shape,
            "logits_finite": logits_finite,
            "reference_sample_token": ref_sample_token,
            "sample_token_match": sample_token_match,
            "parse_ok": bool(parsed.ok),
            "parse_errors": [
                {"category": err.category.value, "message": err.message, "field": err.field}
                for err in parsed.errors
            ],
            "reference_parse_ok": bool(ref_parsed.ok),
            # 判定项：reload 输出 vs 训练端参照输出（四字段语义）。
            "four_fields_equal_to_reference": bool(fields_equal),
            "field_diffs_vs_reference": field_diffs,
            "generated_text": generation.text,
            "generated_tokens": generation.output_tokens,
            "reference_text": str(reference_generation["text"]),
            "reference_tokens": reference_generation.get("output_tokens"),
            # 信息项：与冻结 GT 的一致率（不参与 passed，仅记录训练效果）。
            "vs_gt": {
                "four_fields_equal": bool(gt_equal),
                "field_diffs": gt_field_diffs,
            },
        },
    }

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
