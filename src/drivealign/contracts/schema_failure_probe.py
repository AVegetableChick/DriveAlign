#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""schema_failure 系统化排查探针（contracts 层诊断工具，S12 Step 7 后续研究）。

Location:
    放在 ``drivealign.contracts`` 包内而非 ``scripts/``：本工具强依赖合同层的
    解析器与私有归一化助手（``output.parse_structured_output`` /
    ``_normalize_v2_payload``），本质是合同诊断工具，随包走、随包版本演进。

Purpose:
    对任意一份 predictions JSONL 复现 eval 链路的严格解析，把
    json_parse_failure / schema_failure / semantic_range_failure 分开计数，
    并对 schema_failure 做**两级下钻**：
      L1 定位问题字段：违规按 ``字段路径 × jsonschema 关键字`` 交叉聚合计数；
      L2 定位字段具体问题：对 enum / required / additionalProperties / maxItems
         四类高频关键字，从 validator 消息里提取**具体取值/键名**并排布分布。
    另附失败样本输出长度直方图与奇异长度簇的违规构成，用于判断
    "单一刻板坏模板"还是"散发噪声"。

Design notes:
    - 规则零复刻：分类阶段直接调用 eval 同款 ``parse_structured_output``
      （含 markdown fence 与 v6 同义词归一化），保证与 eval_report 的
      parse_error_counts 逐数可对账；
    - 细节阶段直接用 ``Draft202012Validator(load_output_schema(version))``
      重放 schema 校验，以拿到 jsonschema 的 ``validator`` 关键字与
      ``absolute_path``（parse_structured_output 对外只保留 message/field）；
    - 只读分析，不修改任何 predictions / eval 产物；报告 JSON 默认写到
      predictions 同目录，方便与该 run 一起归档。

Usage:
    conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace
    PYTHONPATH=DriveAlign/src python -m drivealign.contracts.schema_failure_probe \
        --predictions runs/S12_sft_1f/eval_v6_batched/predictions_bs16.jsonl \
        [--contract-version v6] [--out <report.json>] [--top 15]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .output import (
    OutputErrorCategory,
    _has_vocab_tables,
    _normalize_v2_payload,
    load_output_schema,
    parse_structured_output,
)
from .versions import DEFAULT_CONTRACT_VERSION


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def load_predictions(path: Path) -> list[dict[str, Any]]:
    """读 predictions JSONL，保留 sample_token / raw_text / output_tokens。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(
                {
                    "sample_token": r["sample_token"],
                    "raw_text": r["raw_text"],
                    "output_tokens": r.get("telemetry", {}).get("output_tokens"),
                }
            )
    return rows


def classify(rows: list[dict[str, Any]], version: str) -> dict[str, list[dict[str, Any]]]:
    """用 eval 同款解析器把每条 raw_text 分到四类，附错误列表。"""
    buckets: dict[str, list[dict[str, Any]]] = {
        "ok": [],
        "json_parse_failure": [],
        "schema_failure": [],
        "semantic_range_failure": [],
    }
    for row in rows:
        result = parse_structured_output(row["raw_text"], contract_version=version)
        if result.ok:
            buckets["ok"].append(row)
            continue
        # 取最重错误类别（一条文本可能同时给多条 schema 错误）
        cat = result.errors[0].category
        name = cat.value if isinstance(cat, OutputErrorCategory) else str(cat)
        buckets[name].append({**row, "errors": result.errors})
    return buckets


def replay_schema_errors(
    raw_text: str, version: str
) -> list[tuple[tuple[Any, ...], str, str]]:
    """重放 schema 校验，返回 (absolute_path, validator关键字, message) 列表。

    前处理与 parse_structured_output 完全一致：fence 提取 + 同义词归一化，
    保证这里看到的 payload 就是 eval 链路送进 validator 的那个。
    """
    payload = None
    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        fence = re.match(r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$", raw_text, re.DOTALL)
        if fence is not None:
            try:
                payload = json.loads(fence.group("body"))
            except (json.JSONDecodeError, TypeError, ValueError):
                return []
    if payload is None:
        return []
    if _has_vocab_tables(version):
        payload, _, _ = _normalize_v2_payload(version, payload)

    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load_output_schema(version))
    return [
        (tuple(err.absolute_path), str(err.validator), err.message)
        for err in validator.iter_errors(payload)
    ]


def _norm(parts: tuple[Any, ...]) -> str:
    """absolute_path 归一化：数组下标换成 *，得到可聚合的字段模式。"""
    field = "$"
    for part in parts:
        field += "[*]" if isinstance(part, int) else f".{part}"
    return field


# --------------------------------------------------------------------------- #
# L1/L2 聚合
# --------------------------------------------------------------------------- #
def aggregate_schema(schema_rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
    """对 schema_failure 样本做 L1（字段定位）与 L2（具体问题）两级聚合。"""
    field_counter: Counter = Counter()          # L1: 归一化字段路径
    keyword_counter: Counter = Counter()        # L1: jsonschema 关键字
    cross: Counter = Counter()                  # L1: 字段 × 关键字
    enum_values: dict[str, Counter] = defaultdict(Counter)   # L2: 字段 → 非法取值
    extra_keys: Counter = Counter()             # L2: additionalProperties 多余键
    missing_keys: Counter = Counter()           # L2: required 缺失键
    maxitems_hits: list[str] = []               # L2: 超数组上限的字段
    per_anchor: list[dict[str, Any]] = []       # 逐 anchor 明细（落盘用）

    for row in schema_rows:
        replayed = replay_schema_errors(row["raw_text"], version)
        for parts, keyword, message in replayed:
            field = _norm(parts)
            field_counter[field] += 1
            keyword_counter[keyword] += 1
            cross[f"{field} | {keyword}"] += 1
            if keyword == "enum":
                m = re.search(r"'([^']*)' is not one of", message)
                if m:
                    enum_values[field][m.group(1)] += 1
            elif keyword == "additionalProperties":
                m = re.search(r"\((.*?)\) wer", message) or re.search(
                    r"\((.+?)\) were? unexpected", message
                )
                if m:
                    for name in re.findall(r"'([^']+)'", m.group(1)):
                        extra_keys[f"{field} :: {name}"] += 1
                else:
                    extra_keys[f"{field} :: <未知>"] += 1
            elif keyword == "required":
                m = re.search(r"'([^']+)' is a required property", message)
                if m:
                    missing_keys[f"{field} :: {m.group(1)}"] += 1
            elif keyword == "maxItems":
                maxitems_hits.append(f"{field} :: {message}")
            per_anchor.append(
                {
                    "sample_token": row["sample_token"],
                    "output_tokens": row["output_tokens"],
                    "field": field,
                    "validator": keyword,
                    "message": message,
                }
            )
    return {
        "field_counter": field_counter,
        "keyword_counter": keyword_counter,
        "cross": cross,
        "enum_values": enum_values,
        "extra_keys": extra_keys,
        "missing_keys": missing_keys,
        "maxitems_hits": maxitems_hits,
        "per_anchor": per_anchor,
    }


def length_profile(schema_rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
    """失败样本输出长度画像 + 最长奇异簇的违规构成。"""
    lens = [r["output_tokens"] or 0 for r in schema_rows]
    hist = Counter(lens)
    # 找"奇异峰"：出现次数 ≥10 的同一长度
    peaks = [n for n, c in hist.items() if c >= 10]
    peak_detail = {}
    for n in sorted(peaks):
        members = [r for r in schema_rows if r["output_tokens"] == n]
        toks: Counter = Counter()
        for r in members:
            for parts, keyword, _ in replay_schema_errors(r["raw_text"], version):
                toks[f"{_norm(parts)} | {keyword}"] += 1
        peak_detail[n] = {"count": len(members), "violations": dict(toks.most_common(8))}
    return {"top_lengths": hist.most_common(12), "peaks": peak_detail}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="schema_failure 系统化排查：L1 定位问题字段 → L2 定位字段具体问题"
    )
    parser.add_argument("--predictions", required=True, help="predictions JSONL 路径")
    parser.add_argument(
        "--contract-version", default=DEFAULT_CONTRACT_VERSION, help="合同版本（默认取项目默认）"
    )
    parser.add_argument("--out", default=None, help="详细报告 JSON 输出路径（默认与 predictions 同目录）")
    parser.add_argument("--top", type=int, default=15, help="各分布展示前 N 行")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    rows = load_predictions(pred_path)
    buckets = classify(rows, args.contract_version)

    print("=" * 72)
    print(f"样本总数 {len(rows)} | 解析对账（应与 eval_report.parse_error_counts 一致）")
    for name in ("ok", "json_parse_failure", "schema_failure", "semantic_range_failure"):
        print(f"  {name:24s} {len(buckets[name]):5d}")

    agg = aggregate_schema(buckets["schema_failure"], args.contract_version)

    print("\n" + "=" * 72)
    print("L1-a 违规字段分布（归一化路径，数组下标 → *）")
    for field, n in agg["field_counter"].most_common(args.top):
        print(f"  {n:5d}  {field}")

    print("\nL1-b jsonschema 关键字分布")
    for kw, n in agg["keyword_counter"].most_common():
        print(f"  {n:5d}  {kw}")

    print("\nL1-c 字段 × 关键字 交叉表")
    for key, n in agg["cross"].most_common(args.top):
        print(f"  {n:5d}  {key}")

    print("\n" + "=" * 72)
    print("L2-a enum 非法取值分布（字段 → 取值 → 条数）")
    for field, values in sorted(
        agg["enum_values"].items(), key=lambda kv: -sum(kv[1].values())
    ):
        total = sum(values.values())
        detail = ", ".join(f"{v!r}×{c}" for v, c in values.most_common(args.top))
        print(f"  {field}（共 {total}）: {detail}")

    print("\nL2-b additionalProperties 多余键分布")
    for key, n in agg["extra_keys"].most_common(args.top):
        print(f"  {n:5d}  {key}")

    print("\nL2-c required 缺失键分布")
    for key, n in agg["missing_keys"].most_common(args.top):
        print(f"  {n:5d}  {key}")

    if agg["maxitems_hits"]:
        print(f"\nL2-d maxItems 超限（{len(agg['maxitems_hits'])} 条）")
        for msg in agg["maxitems_hits"][:5]:
            print(f"  {msg}")

    print("\n" + "=" * 72)
    profile = length_profile(buckets["schema_failure"], args.contract_version)
    print("失败样本 output_tokens Top12：", profile["top_lengths"])
    for n, detail in profile["peaks"].items():
        print(f"  奇异峰 out_tokens={n}（{detail['count']} 条）违规构成：")
        for key, c in detail["violations"].items():
            print(f"    {c:4d}  {key}")

    # 详细报告落盘（默认与 predictions 同目录，随 run 归档）
    out_path = Path(args.out) if args.out else pred_path.parent / "schema_probe_report.json"
    report = {
        "predictions": str(pred_path),
        "contract_version": args.contract_version,
        "counts": {k: len(v) for k, v in buckets.items()},
        "field_counter": dict(agg["field_counter"]),
        "keyword_counter": dict(agg["keyword_counter"]),
        "cross": dict(agg["cross"]),
        "enum_values": {k: dict(v) for k, v in agg["enum_values"].items()},
        "extra_keys": dict(agg["extra_keys"]),
        "missing_keys": dict(agg["missing_keys"]),
        "maxitems_hits": agg["maxitems_hits"],
        "length_profile": profile,
        "per_anchor": agg["per_anchor"],
        "json_parse_failure_tokens": [
            {"sample_token": r["sample_token"], "output_tokens": r["output_tokens"]}
            for r in buckets["json_parse_failure"]
        ],
    }
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n详细报告已落盘：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
