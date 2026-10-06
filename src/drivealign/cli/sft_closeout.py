"""S11 Step 5 收尾：剩余 Gate 判定 CLI（`cli/sft_closeout.py`）。

背景：Gate 范围收缩（2026-10-07 用户拍板，见 `S11_decision_log.md` D7）
--------------------------------------------------------------------
原规划 §5 列了 G0.x–G7 八个 Gate。收尾复核后判定：G2/G3/G4 的真实信息已被
**更早环节**覆盖（"构造期强制"或"G1 的更强证据"），G5 本就是"仅记录"。因此只保留
两个需要**实跑**的 Gate，其余降级为凭证引用或读数记录：

    | 原 Gate | 处置 | 凭证 / 落点 |
    |---|---|---|
    | G2 loss 下降 | **降级为读数记录**（不再判 PASS/FAIL） | 本 CLI `disk` 子命令解析训练日志 |
    | G3 面 parity | **构造期已强制**，引结论 | `sft/dataset.build_training_sample` 的 ValueError 断言 |
    | G4 target 断言 + 确定性 | 断言半 by-construction；确定性半由本 CLI `disk` 实跑 | 重建 JSONL 逐字节比 sha256 |
    | G5 maxItems 触顶率 | **折入 G6 报告**的一个记录字段 | 本 CLI `model` 子命令 |
    | G6 自推理诊断 | **保留，实跑** | 本 CLI `model` 子命令 |
    | G7 视觉依赖抽查 | **保留，实跑** | 本 CLI `model` 子命令 |

为什么不给 G2 判 PASS/FAIL：规划 §7-D 的阈值（最终 ≤ 初始 × 0.3）在实测曲线上
会判 FAIL（首点 0.9882 → 末点 0.4606，仅降 53%），而同一 run 的 G1 双层判定 PASS、
且 `vs_gt.four_fields_equal=true`（128×3 epoch 已把固定样本过拟合到 GT）——即
"阈值判 FAIL 但证据表明模型学得很好"。一个健康 run 会因技术性阈值判失败的 Gate
产出的是噪声，故只留读数。

两个子命令
----------
``disk``（纯 CPU，不需要模型）
    - **G4 确定性**：把冻结样本按 `cli/sft_build_samples.py serialize` 的同一条路径
      **重打一份**到临时目录，逐文件比 sha256 是否与冻结 `samples_manifest.json`
      登记值**逐字节一致**。
    - **G3 面 parity**：重建过程本身走 `build_training_sample`（不等即抛错）；另外
      逐样本独立复核 `request_hash_1f == manifest.request_hash_1f` 与
      `record_hash` 的一致性，逐条落档，不靠"没报错"当证据。
    - **G2 读数**：解析训练日志的 loss 序列与总览行，落成**记录**（不做判定）。

``model``（GPU，需本地模型）
    - **G6 自推理诊断**：对 overfit128 的 128 条训练样本逐条贪心生成（解码沿 S09
      冻结口径），严格解析 v6，统计 parse_rate 与截断率，并落"四字段抽检表"。
    - **G5 记录**：在 G6 的解析结果上统计 `maxItems` 触顶率（critical_objects /
      risk_factors 各 8 项上限），仅记录不决策。
    - **G7 视觉依赖抽查**：对前 N 条（默认 32，即 train32 集合）做 blank-image
      反事实（复用 `structured_runner.make_blank_image`），比较**离散字段**
      （speed_action / yield_required / critical_objects 集合）是否至少 1 项改变。

用法
----
``# disk（CPU）``
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.sft_closeout disk \
    --dataset-root data/dataset_v4 \
    --anchor-manifest data/face_manifests/v6/anchor_policy_manifest.json \
    --subsets-dir data/sft_subsets \
    --samples-dir runs/S11_sft_smoke/sft_samples \
    --train-log runs/S11_sft_smoke/checkpoints/smoke128/train128.log \
    --train-log runs/S11_sft_smoke/checkpoints/smoke32/train32.log \
    --report runs/S11_sft_smoke/reports/g34_disk.json``

``# model（GPU，tmux + tee）``
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    drivealign.cli.sft_closeout model \
    --config DriveAlign/configs/sft/sft_smoke_1f.yaml \
    --adapter runs/S11_sft_smoke/checkpoints/smoke128 \
    --samples runs/S11_sft_smoke/sft_samples/overfit128.jsonl \
    --dataroot data/nuscenes/trainval \
    --report runs/S11_sft_smoke/reports/g67_model.json``

退出码：``disk`` 全部判定 PASS → 0，否则 1；``model`` 同理（便于 tmux/CI 直接判）。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

#: 日志里 HF `Trainer` 每个 logging 窗口输出的扁平 dict（`{'loss': ..., 'epoch': ...}`）。
#: 用 `[^}]*` 即可——该行没有嵌套 dict；`'train_loss'` 不会被误匹配（前缀是 `train_`）。
_LOSS_ENTRY = re.compile(r"\{'loss':[^}]*\}")
#: 总览行里的标量字段（总览行含嵌套 dict，故逐字段取值而非整行 literal_eval）。
_SUMMARY_FIELDS = {
    "train_runtime": r"'train_runtime':\s*([0-9.]+)",
    "train_loss": r"'train_loss':\s*([0-9.]+)",
    "train_samples_per_second": r"'train_samples_per_second':\s*([0-9.]+)",
    "train_steps_per_second": r"'train_steps_per_second':\s*([0-9.]+)",
    "global_step": r"'global_step':\s*([0-9]+)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S11 Step 5 close-out gates (disk: G3/G4 + G2 record; model: G5/G6/G7)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- disk：纯 CPU，重建冻结样本 + 解析训练日志 ---
    p_disk = sub.add_parser("disk", help="G4 determinism + G3 parity + G2 loss readout (CPU)")
    p_disk.add_argument("--dataset-root", required=True, help="dataset root with shards")
    p_disk.add_argument(
        "--anchor-manifest", required=True, help="v6-face anchor policy manifest"
    )
    p_disk.add_argument("--subsets-dir", required=True, help="A-layer token subset dir")
    p_disk.add_argument("--samples-dir", required=True, help="frozen B-layer samples dir")
    p_disk.add_argument(
        "--train-log",
        action="append",
        required=True,
        default=[],
        help="training log to read the loss curve from (repeatable)",
    )
    p_disk.add_argument("--report", required=True, help="output JSON report path")

    # --- model：GPU，自推理诊断 + blank-image 反事实 ---
    p_model = sub.add_parser("model", help="G6 self-inference + G5 record + G7 counterfactual (GPU)")
    p_model.add_argument("--config", required=True, help="sft config YAML")
    p_model.add_argument("--adapter", required=True, help="overfit adapter dir")
    p_model.add_argument("--samples", required=True, help="frozen samples JSONL (overfit128)")
    p_model.add_argument("--dataroot", required=True, help="nuScenes dataroot for images")
    p_model.add_argument(
        "--cf-count",
        type=int,
        default=32,
        help="G7 blank-image counterfactual subset size (first N samples)",
    )
    p_model.add_argument("--report", required=True, help="output JSON report path")
    p_model.add_argument(
        "--table",
        default=None,
        help="per-sample four-field table path (default: <report stem>_table.md)",
    )
    return parser.parse_args(argv)


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# G2 读数记录：训练日志解析（不做 PASS/FAIL）
# ---------------------------------------------------------------------------


def parse_training_log(path: str | Path) -> dict[str, Any]:
    """从 HF `Trainer` 日志里抽出 loss 曲线与总览读数（G2 记录的原料）。

    取的是 `logging_steps` 粒度的**窗口均值**（`logging_steps=10` → 每点 = 10 个
    optimizer step 的均值），不是逐步 loss——报告里以 `log_window_granularity` 显式
    声明，避免把"相邻点抬升"误读成"逐步不单调"。

    返回：`losses`（时间序）、`init_loss` / `final_loss` / `mean_loss` / `rises`
    （相邻点抬升次数）以及总览标量（`train_loss` / `train_runtime` / `global_step` 等）。
    """
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8", errors="replace")

    losses = [
        float(ast.literal_eval(match.group(0))["loss"])
        for match in _LOSS_ENTRY.finditer(text)
    ]
    summary: dict[str, Any] = {}
    for key, pattern in _SUMMARY_FIELDS.items():
        found = re.search(pattern, text)
        if found:
            summary[key] = float(found.group(1)) if "." in found.group(1) else int(found.group(1))

    rises = [
        index for index in range(len(losses) - 1) if losses[index + 1] > losses[index]
    ]
    record: dict[str, Any] = {
        "log_path": str(log_path),
        "log_window_granularity": "每点 = logging_steps 个 optimizer step 的均值",
        "n_points": len(losses),
        "losses": losses,
        "rises": len(rises),
        "rise_positions": rises,
        "summary": summary,
    }
    if losses:
        record["init_loss"] = losses[0]
        record["final_loss"] = losses[-1]
        record["mean_loss"] = sum(losses) / len(losses)
        record["drop_ratio"] = 1.0 - losses[-1] / losses[0] if losses[0] else None
    return record


# ---------------------------------------------------------------------------
# disk 子命令：G4 确定性 + G3 parity + G2 读数
# ---------------------------------------------------------------------------


def cmd_disk(args: argparse.Namespace) -> int:
    """重建冻结样本比对 sha（G4）、逐样本复核面 parity（G3）、记录 loss 曲线（G2）。"""
    import tempfile

    from drivealign.cli.sft_build_samples import cmd_serialize

    samples_dir = Path(args.samples_dir)
    frozen_manifest = json.loads(
        (samples_dir / "samples_manifest.json").read_text(encoding="utf-8")
    )

    # ---------- G4：用同一条序列化路径重打一份，比逐字节 sha ----------
    # 复用 `sft_build_samples.cmd_serialize`（而不是重写一遍），保证"被验证的"就是
    # "当初冻结时跑的那段代码"；重写一份会让这条证据失去意义。
    with tempfile.TemporaryDirectory(prefix="s11_g4_rebuild_") as tmp:
        rebuild_dir = Path(tmp)
        cmd_serialize(
            argparse.Namespace(
                dataset_root=args.dataset_root,
                anchor_manifest=args.anchor_manifest,
                subsets_dir=args.subsets_dir,
                out_dir=str(rebuild_dir),
            )
        )
        artifacts: dict[str, Any] = {}
        determinism_ok = True
        for name, entry in frozen_manifest.get("artifacts", {}).items():
            frozen_file = samples_dir / f"{name}.jsonl"
            rebuilt_file = rebuild_dir / f"{name}.jsonl"
            frozen_sha = (
                _sha256_of_file(frozen_file) if frozen_file.is_file() else None
            )
            rebuilt_sha = _sha256_of_file(rebuilt_file) if rebuilt_file.is_file() else None
            registered_sha = entry.get("file_sha256")
            matched = (
                rebuilt_sha is not None
                and rebuilt_sha == frozen_sha
                and rebuilt_sha == registered_sha
            )
            determinism_ok = determinism_ok and matched
            artifacts[name] = {
                "frozen_sha256": frozen_sha,
                "registered_sha256": registered_sha,
                "rebuilt_sha256": rebuilt_sha,
                "byte_identical": bool(matched),
            }

        # ---------- G3：逐样本复核面 parity（独立重读 manifest，不靠"没报错"） ----------
        anchors = json.loads(
            Path(args.anchor_manifest).read_text(encoding="utf-8")
        )["anchors"]
        parity_mismatches: list[dict[str, Any]] = []
        n_checked = 0
        for name in frozen_manifest.get("artifacts", {}):
            for line in (samples_dir / f"{name}.jsonl").read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                token = row["sample_token"]
                entry = anchors.get(token)
                n_checked += 1
                if entry is None:
                    parity_mismatches.append({"sample_token": token, "reason": "不在 v6 manifest 中"})
                    continue
                if row["request_hash_1f"] != entry["request_hash_1f"]:
                    parity_mismatches.append(
                        {
                            "sample_token": token,
                            "reason": "request_hash_1f != v6 manifest 值",
                            "sample": row["request_hash_1f"],
                            "manifest": entry["request_hash_1f"],
                        }
                    )
                if row["manifest_request_hash"] != row["request_hash_1f"]:
                    parity_mismatches.append(
                        {"sample_token": token, "reason": "样本内两份 hash 字段不一致"}
                    )
                if row["record_hash"] != entry["record_hash"]:
                    parity_mismatches.append(
                        {"sample_token": token, "reason": "record_hash 与 manifest 不一致"}
                    )

    # ---------- G2：loss 曲线读数（记录，不判定） ----------
    loss_records = {}
    for log in args.train_log:
        record = parse_training_log(log)
        # 键用 "<checkpoint 目录>/<日志名>"（如 smoke128/train128.log），便于对号入座。
        log_path = Path(log)
        loss_records[f"{log_path.parent.name}/{log_path.name}"] = record

    report = {
        "kind": "s11_closeout_disk",
        "gate_g4_determinism": {
            "passed": bool(determinism_ok),
            "method": "以 cli/sft_build_samples.py serialize 的同一条路径重打到临时目录，逐字节比 sha256",
            "artifacts": artifacts,
            "contract_version": frozen_manifest.get("contract_version"),
            "git_commit_at_freeze": frozen_manifest.get("git_commit"),
            "note": "git_commit 不参与比对：重建发生在更晚的 commit，逐字节一致反而证明内容与 commit 无关地可复现",
        },
        "gate_g3_face_parity": {
            "passed": not parity_mismatches,
            "method": "逐样本独立复核 request_hash_1f vs v6-face manifest（并顺带核 record_hash）",
            "n_samples_checked": n_checked,
            "mismatches": parity_mismatches,
            "by_construction_note": "sft/dataset.build_training_sample 在构造期即以 ValueError 强制该等式，本项为独立复核",
        },
        "g2_loss_curve_record": {
            "verdict": "记录项（2026-10-07 收缩，不判 PASS/FAIL）",
            "reason": "§7-D 阈值在本 run 实测曲线上会判 FAIL，而 G1 双层 PASS 且 vs_gt 四字段全等；阈值判失败而证据表明学得好 ⇒ 该项只留读数",
            "logs": loss_records,
        },
    }
    report["passed"] = bool(report["gate_g4_determinism"]["passed"] and report["gate_g3_face_parity"]["passed"])

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False), flush=True)
    print(
        f"[disk] G4 determinism={'PASS' if report['gate_g4_determinism']['passed'] else 'FAIL'} "
        f"G3 parity={'PASS' if report['gate_g3_face_parity']['passed'] else 'FAIL'} "
        f"({n_checked} samples) -> {report_path}",
        flush=True,
    )
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------------------
# model 子命令：G6 自推理诊断 + G5 记录 + G7 blank-image 反事实
# ---------------------------------------------------------------------------


def _summary_of_parsed(parsed: Any) -> dict[str, Any]:
    """把一次解析结果压成抽检表/统计需要的最小字段集。"""
    if not parsed.ok or parsed.output is None:
        return {
            "parse_ok": False,
            "error_categories": sorted({err.category.value for err in parsed.errors}),
            "error_fields": [err.field for err in parsed.errors][:3],
        }
    out = parsed.output
    return {
        "parse_ok": True,
        "n_critical_objects": len(out.critical_objects),
        "n_risk_factors": len(out.risk_factors),
        "speed_action": out.speed_action,
        "yield_required": out.yield_required,
        "risk_factors": list(out.risk_factors),
        "critical_object_keys": [
            [obj.category, obj.motion_state] for obj in out.critical_objects
        ],
    }


def cmd_model(args: argparse.Namespace) -> int:
    """跑 G6（128 条自推理诊断）+ G5（触顶率记录）+ G7（blank-image 反事实）。"""
    import yaml

    from drivealign.contracts.output import parse_structured_output
    from drivealign.inference.base_runner import generate_one
    from drivealign.inference.model_loader import load_model_and_processor
    from drivealign.inference.structured_runner import make_blank_image
    from drivealign.sft.dataset import load_frozen_records
    from drivealign.sft.gate import discrete_fields_changed, semantic_fields

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    contract_version = str(config["contract_version"])
    generation_config = config.get("generation") or {}
    max_new_tokens = int(generation_config.get("max_new_tokens", 512))

    records = load_frozen_records(args.samples)
    dataroot = Path(args.dataroot)

    # 新进程从磁盘重建 base + adapter（与 G1 复验同一条路，只从这里读 artifact）。
    from peft import PeftModel

    loaded = load_model_and_processor(config)
    loaded.model = PeftModel.from_pretrained(loaded.model, str(args.adapter))
    loaded.model.eval()

    # ---------- G6：逐条自推理诊断（含 G5 触顶率记录） ----------
    per_sample: list[dict[str, Any]] = []
    original_fields: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        result = generate_one(
            loaded,
            dataroot / record["image_relpath"],
            record["prompt"],
            generation_config=generation_config,
        )
        parsed = parse_structured_output(result.text, contract_version=contract_version)
        # 截断判定 = 生成预算被用尽（严格口径）；"文本未以 } 收尾"另记为诊断项，
        # 避免把"提前停下但格式合法"误算成截断。
        truncated = result.output_tokens >= max_new_tokens
        row: dict[str, Any] = {
            "sample_token": record["sample_token"],
            "output_tokens": int(result.output_tokens),
            "input_tokens": int(result.input_tokens),
            "truncated": bool(truncated),
            "unterminated_text": bool(not result.text.rstrip().endswith("}")),
        }
        row.update(_summary_of_parsed(parsed))
        per_sample.append(row)
        original_fields.append(
            parsed.output.to_dict() if parsed.ok and parsed.output else {}
        )
        if index % 16 == 0 or index == len(records):
            print(
                f"[model] G6 {index}/{len(records)} 条已生成；"
                f"当前 parse_ok={sum(1 for r in per_sample if r['parse_ok'])}",
                flush=True,
            )

    n_total = len(per_sample)
    n_parsed = sum(1 for row in per_sample if row["parse_ok"])
    n_truncated = sum(1 for row in per_sample if row["truncated"])
    parse_rate = n_parsed / n_total if n_total else 0.0
    truncation_rate = n_truncated / n_total if n_total else 0.0

    token_lengths = sorted(row["output_tokens"] for row in per_sample)
    length_stats = {
        "min": token_lengths[0] if token_lengths else None,
        "median": token_lengths[len(token_lengths) // 2] if token_lengths else None,
        "max": token_lengths[-1] if token_lengths else None,
    }

    # G5（记录项）：schema 的 maxItems=8。只在**解析成功**的样本上统计触顶率，
    # 解析失败的样本不计入分母（否则会把格式失败混进"输出太满"的读数）。
    parsed_rows = [row for row in per_sample if row["parse_ok"]]
    n_parsed_rows = len(parsed_rows)
    g5 = {
        "verdict": "记录项（仅记录不决策，S03-Q5 监控项）",
        "schema_max_items": 8,
        "n_parsed": n_parsed_rows,
        "critical_objects_at_cap": sum(
            1 for row in parsed_rows if row["n_critical_objects"] >= 8
        ),
        "risk_factors_at_cap": sum(
            1 for row in parsed_rows if row["n_risk_factors"] >= 8
        ),
        "critical_objects_at_cap_rate": (
            sum(1 for row in parsed_rows if row["n_critical_objects"] >= 8) / n_parsed_rows
            if n_parsed_rows
            else None
        ),
        "note": "触顶率只统计解析成功的样本；解析失败样本不计入分母",
    }

    g6_passed = bool(parse_rate >= 0.99 and truncation_rate <= 0.01)

    # ---------- G7：blank-image 反事实（前 cf-count 条） ----------
    scratch = Path(args.report).parent / "cf_blank_images"
    cf_rows: list[dict[str, Any]] = []
    n_changed = 0
    n_cf_parsed = 0
    for record, baseline in zip(records[: args.cf_count], original_fields[: args.cf_count]):
        blank = make_blank_image(
            dataroot / record["image_relpath"], scratch / f"{record['sample_token']}.jpg"
        )
        result = generate_one(
            loaded, blank, record["prompt"], generation_config=generation_config
        )
        parsed = parse_structured_output(result.text, contract_version=contract_version)
        if not parsed.ok or parsed.output is None or not baseline:
            cf_rows.append(
                {
                    "sample_token": record["sample_token"],
                    "usable": False,
                    "reason": "blank 侧或原图侧输出不可解析，无法比较",
                    "blank_parse_ok": bool(parsed.ok),
                }
            )
            continue
        n_cf_parsed += 1
        blank_fields = parsed.output.to_dict()
        changed = discrete_fields_changed(baseline, blank_fields)
        if changed:
            n_changed += 1
        cf_rows.append(
            {
                "sample_token": record["sample_token"],
                "usable": True,
                "changed_discrete_fields": changed,
                "baseline": {
                    key: (sorted(value) if isinstance(value, (set, frozenset)) else value)
                    for key, value in semantic_fields(baseline).items()
                },
                "blank": {
                    key: (sorted(value) if isinstance(value, (set, frozenset)) else value)
                    for key, value in semantic_fields(blank_fields).items()
                },
            }
        )
        print(
            f"[model] G7 {len(cf_rows)}/{min(args.cf_count, len(records))} 条反事实完成；"
            f"已改变 {n_changed} 条",
            flush=True,
        )

    g7_passed = bool(n_changed >= 1)

    report = {
        "kind": "s11_closeout_model",
        "config": str(args.config),
        "adapter_dir": str(args.adapter),
        "samples": str(args.samples),
        "n_samples": n_total,
        "gate_g6_self_inference": {
            "passed": g6_passed,
            "criteria": "parse_rate >= 0.99 且 truncation_rate <= 0.01（§7-E 冻结阈值）",
            "parse_rate": parse_rate,
            "n_parsed": n_parsed,
            "truncation_rate": truncation_rate,
            "n_truncated": n_truncated,
            "unterminated_text_count": sum(
                1 for row in per_sample if row["unterminated_text"]
            ),
            "output_token_stats": length_stats,
            "error_taxonomy": _count_by(
                row.get("error_categories", []) for row in per_sample if not row["parse_ok"]
            ),
        },
        "g5_max_items_record": g5,
        "gate_g7_visual_dependence": {
            "passed": g7_passed,
            "criteria": "blank-image 反事实下至少 1 条样本的离散字段（speed_action / yield_required / critical_objects 集合）改变（§7-J）",
            "cf_count": min(args.cf_count, len(records)),
            "n_usable": n_cf_parsed,
            "n_changed": n_changed,
            "subset_note": f"取 overfit128 的前 {args.cf_count} 条（该前缀即 train32 集合）",
            "per_sample": cf_rows,
        },
        "per_sample_g6": per_sample,
    }
    report["passed"] = bool(g6_passed and g7_passed)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    table_path = Path(args.table) if args.table else report_path.with_name(
        report_path.stem + "_table.md"
    )
    table_path.write_text(_render_table(per_sample), encoding="utf-8")

    print(
        json.dumps(
            {
                "passed": report["passed"],
                "g6": report["gate_g6_self_inference"],
                "g5": g5,
                "g7": {
                    "passed": g7_passed,
                    "n_changed": n_changed,
                    "n_usable": n_cf_parsed,
                },
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"[model] report -> {report_path}\n[model] table  -> {table_path}", flush=True)
    return 0 if report["passed"] else 1


def _count_by(groups: Sequence[Sequence[str]]) -> dict[str, int]:
    """扁平统计错误 taxonomy 的出现次数（用于 G6 失败归因）。"""
    counts: dict[str, int] = {}
    for group in groups:
        for item in group:
            counts[item] = counts.get(item, 0) + 1
    return counts


def _render_table(per_sample: Sequence[Mapping[str, Any]]) -> str:
    """渲染 G6 的"四字段抽检表"（Markdown，逐样本一行）。"""
    lines = [
        "# S11 G6 四字段抽检表（overfit128 自推理）",
        "",
        "> 由 `cli/sft_closeout.py model` 生成；bbox 数值不入表（本表按语义字段抽检）。",
        "",
        "| # | sample_token | parse | tokens | truncated | speed_action | yield | n_obj | risk_factors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(per_sample, start=1):
        if row["parse_ok"]:
            lines.append(
                f"| {index} | `{row['sample_token']}` | ok | {row['output_tokens']} | "
                f"{'是' if row['truncated'] else '否'} | {row['speed_action']} | "
                f"{row['yield_required']} | {row['n_critical_objects']} | "
                f"{', '.join(row['risk_factors']) or '—'} |"
            )
        else:
            lines.append(
                f"| {index} | `{row['sample_token']}` | **FAIL** | {row['output_tokens']} | "
                f"{'是' if row['truncated'] else '否'} | — | — | — | "
                f"{', '.join(row.get('error_categories', []))} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "disk":
        return cmd_disk(args)
    if args.command == "model":
        return cmd_model(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
