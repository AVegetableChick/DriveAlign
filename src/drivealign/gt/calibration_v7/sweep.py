"""S13 Stage B：标定网格扫描 B0–B5（v7 calibration plan §4–§6）。

Purpose:
    消费 Stage A 的中间量 parquet（``features/shard_*_{objects,anchors}.
    parquet``），执行六个扫描块并把全部必报项落表 + 落图。**零 nuscenes
    依赖**——扫描全部是中间量上的向量化比较（plan §1.3 的设计：五个标定
    槽位都是对象级中间量的后验比较阈值）：

    - **B0** v6 基线统计：v6 各因子检出率 / yield 率（对照列）、congestion
      残核计数、static 残余集合识别（v6 corridor_conflict ∧ 所有 v6-firing
      对象均为 static ∧ v6 stationary=false）。v6 因子用
      :func:`rebuild_v6_factors` 逐 anchor 重建（与 G3 同代码路径，全量
      G3 通过后与冻结 GT 等价）。
    - **B1** base 网格（81 组合含分裂假设）：firing ⟺ 组内 min_clearance
      < base（vru 组用 base_vru，vehicle+static 组用 base）。产出对角线
      视图（单假设曲线）+ 分裂热力图 + min_clearance 分族直方图。
    - **B2** thr 网格（base 对角 9 × thr 13 × 两族）：crossing = firing ∧
      统一横向 gate（``v_toward > thr``，带内 |v_lat| / 带外 −sign(y_c)·
      v_lat）。产出热力图 + 边际曲线 + firing 对象 v_toward 分布 + gate
      与 motion_state=crossing 的一致率诊断。
    - **B3** r 网格（9 点）：object_near 检出率-r 曲线（5–15% 目标带）+
      与 stationary 的共现率-(r, Y_st) 矩阵（>80% 护栏材料）。
    - **B4** Y_brake 网格（11 点）：braking 检出率曲线（1.185 锚点）+
      每档新增前车的 |y_c| 分向（左/右）构成——邻道吸入风险量化。
    - **B5** Y_st 网格（19 点）：stationary 检出率曲线 + static 残余
      覆盖率曲线（Y_st 终值主要依据）+ 覆盖对象类别构成。

    Y 扫描的公共技巧：in-path 条件 ``x_c > 0 ∧ |y_c| ≤ Y`` 中与 Y 无关的
    判据（core：窗口证据 / 速度条件 / x_c > 0）先在对象级预计算为布尔，
    Y 网格只做 ``|y_c| <= Y`` 的广播比较。

网格（plan §2，审查可调）：
    base 1.7→2.5 步 0.1（9）；thr 0.3→1.5 步 0.1（13）；r 3.0→7.0 步
    0.5（9）；Y_brake = 1.185 锚点 + 1.3→2.2 步 0.1（11）；Y_st = 1.185
    锚点 + 1.3→3.0 步 0.1（19）。

Example launch command（分钟级，直接前台跑）::

    PYTHONPATH=DriveAlign/src python -m drivealign.gt.calibration_v7.sweep \
        --features-dir runs/S13_v7_calibration/features \
        --out runs/S13_v7_calibration/sweep_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from drivealign.gt.calibration_v7.extract import rebuild_v6_factors
from drivealign.gt.legacy_v4 import load_v4_rule_config
from drivealign.gt.risk import (
    VEHICLE_CROSSING_CATEGORIES,
    VRU_CATEGORIES,
)

# --- 网格常量（plan §2） -------------------------------------------------------

BASE_GRID = tuple(round(1.7 + 0.1 * i, 1) for i in range(9))          # 1.7..2.5
THR_GRID = tuple(round(0.3 + 0.1 * i, 1) for i in range(13))          # 0.3..1.5
R_GRID = tuple(round(3.0 + 0.5 * i, 1) for i in range(9))             # 3.0..7.0
Y_BRAKE_GRID = (1.185,) + tuple(round(1.3 + 0.1 * i, 1) for i in range(10))   # 11 点
Y_ST_GRID = (1.185,) + tuple(round(1.3 + 0.1 * i, 1) for i in range(18))      # 19 点

# gate 带（band_half_width_m，v7 config 确定性值，非标定参数）。
GATE_BAND_M = 1.185

# static 类别（v7 static 组；v6 firing 全 static 判定共用）。
STATIC_CATEGORIES = frozenset({"barrier", "traffic_cone"})
# braking 族（v7 BRAKING_AND_ONCOMING_CATEGORIES = vehicle 五类 ∪ 两轮车）。
BRAKING_FAMILY = frozenset(
    {"car", "truck", "bus", "trailer", "construction_vehicle", "bicycle", "motorcycle"}
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=Path("runs/S13_v7_calibration/features")
    )
    parser.add_argument("--out", type=Path, default=Path("runs/S13_v7_calibration/sweep_results"))
    parser.add_argument(
        "--max-anchors", type=int, default=None, help="冒烟：只取前 N 个 anchor"
    )
    return parser.parse_args(argv)


def _load_features(features_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """合并全部 shard 的对象表 / anchor 表（按 sample_token 保序）。"""
    obj_paths = sorted(features_dir.glob("shard_*_objects.parquet"))
    anc_paths = sorted(features_dir.glob("shard_*_anchors.parquet"))
    if not obj_paths or not anc_paths:
        raise FileNotFoundError(f"no shard parquet under {features_dir}")
    objects = pd.concat([pd.read_parquet(p) for p in obj_paths], ignore_index=True)
    anchors = pd.concat([pd.read_parquet(p) for p in anc_paths], ignore_index=True)
    # 池序稳定：对象表按 (sample_token, pool_idx) 排序（B0 重建需要池序）。
    objects = objects.sort_values(["sample_token", "pool_idx"], kind="mergesort").reset_index(drop=True)
    anchors = anchors.sort_values("sample_token", kind="mergesort").reset_index(drop=True)
    return objects, anchors


# --- B0：v6 基线 ---------------------------------------------------------------


def _run_b0(objects: pd.DataFrame, anchors: pd.DataFrame) -> dict:
    """v6 基线统计（检出率 / yield 率 / congestion 残核 / static 残余集合）。

    v6 因子逐 anchor 重建（G3 同代码路径）；static 残余集合的判定材料
    （corridor firing 对象的类别）由重建返回的 ``fired_tokens`` 提供。
    """
    v4cfg = load_v4_rule_config()
    obj_grouped = {k: v for k, v in objects.groupby("sample_token", sort=False)}
    factor_counts: Counter = Counter()
    n_yield = 0
    static_residual: List[str] = []
    gt_factor_counts: Counter = Counter()
    n_gt_yield = 0

    for row in anchors.itertuples(index=False):
        obj_rows = obj_grouped.get(row.sample_token)
        if obj_rows is None:
            continue
        rows = obj_rows.to_dict("records")
        factors, yld, fired = rebuild_v6_factors(rows, v4cfg)
        factor_counts.update(factors)
        n_yield += int(yld)
        # static 残余：走廊 firing 非空 ∧ firing 对象全部 static ∧ 无 stationary。
        if fired and "stationary_obstacle" not in factors:
            cat_of = {r["ann_token"]: r["category"] for r in rows}
            if all(cat_of.get(t, "?") in STATIC_CATEGORIES for t in fired):
                static_residual.append(row.sample_token)
        # GT 侧对照（G3 通过后应与重建计数一致；此处交叉核验）。
        gt_factors = [f for f in row.v6_risk_factors.split(",") if f]
        gt_factor_counts.update(gt_factors)
        n_gt_yield += int(row.v6_yield_required)

    n = len(anchors)
    return {
        "anchors": n,
        "rebuild_factor_counts": dict(factor_counts),
        "rebuild_factor_rates": {k: v / n for k, v in factor_counts.items()},
        "rebuild_yield_rate": n_yield / n,
        "gt_factor_counts": dict(gt_factor_counts),
        "gt_yield_rate": n_gt_yield / n,
        "static_residual_count": len(static_residual),
        "static_residual_tokens": static_residual,
    }


# --- 向量化基建 ---------------------------------------------------------------


def _anchor_index(objects: pd.DataFrame, anchors: pd.DataFrame) -> np.ndarray:
    """对象行 → anchor 行号（与 anchors 表顺序对齐），供 groupby 聚合。"""
    pos = {t: i for i, t in enumerate(anchors["sample_token"])}
    return objects["sample_token"].map(pos).to_numpy()


def _any_per_anchor(obj_mask: np.ndarray, anchor_idx: np.ndarray, n_anchors: int) -> np.ndarray:
    """对象级布尔 → anchor 级 any（np.add.at 直方图法，无 pandas 开销）。"""
    hit = np.zeros(n_anchors, dtype=bool)
    idx = anchor_idx[obj_mask]
    if idx.size:
        cnt = np.bincount(idx, minlength=n_anchors)
        hit = cnt > 0
    return hit


# --- B1：base 网格 -------------------------------------------------------------


def _run_b1(objects: pd.DataFrame, anchor_idx: np.ndarray, n: int, out_dir: Path) -> dict:
    """81 组合（含分裂假设）+ 对角线视图 + min_clearance 分族直方图。

    组口径：vru 组（ped/bicycle/motorcycle）用 base_vru；vehicle+static
    组用 base（v7 firing_base_for 的 split 语义，static 归 vehicle 口径）。
    无该组对象的 anchor 该组 min_clearance = +inf（不触发）。
    """
    is_vru = objects["category"].isin(VRU_CATEGORIES).to_numpy()
    is_static = objects["category"].isin(STATIC_CATEGORIES).to_numpy()
    clearance = objects["min_clearance"].to_numpy()

    # 每 anchor 的组内 min_clearance（inf 表示组内无对象）。
    vru_min = np.full(n, np.inf)
    veh_min = np.full(n, np.inf)
    for i, c in zip(anchor_idx[is_vru], clearance[is_vru]):
        if c < vru_min[i]:
            vru_min[i] = c
    for i, c in zip(anchor_idx[~is_vru], clearance[~is_vru]):
        if c < veh_min[i]:
            veh_min[i] = c

    rows = []
    for base_vru in BASE_GRID:
        for base_veh in BASE_GRID:
            vru_fire = vru_min < base_vru
            veh_fire = veh_min < base_veh
            rows.append(
                {
                    "base_vru_m": base_vru,
                    "base_vehicle_m": base_veh,
                    "vru_firing_rate": float(vru_fire.mean()),
                    "vehicle_firing_rate": float(veh_fire.mean()),
                    "any_firing_rate": float((vru_fire | veh_fire).mean()),  # v7 yield 率
                }
            )
    grid = pd.DataFrame(rows)
    grid.to_parquet(out_dir / "b1_base_grid.parquet", index=False)

    # 对角线视图（单假设 base）。
    diag = grid[grid["base_vru_m"] == grid["base_vehicle_m"]].reset_index(drop=True)
    diag.to_csv(out_dir / "b1_diagonal.csv", index=False)

    # min_clearance 分族直方图（0.1 m 桶，[−5, 15] 区间；base 取值依据与
    # 擦肩残余规模）。三桶：vru 组 / vehicle+static 组（firing 分组口径）/
    # 仅 static 类别（B0 残余诊断）。
    hist = {}
    groups = {
        "vru": objects.loc[is_vru, "min_clearance"],
        "vehicle_static": objects.loc[~is_vru, "min_clearance"],
        "static_only": objects.loc[is_static, "min_clearance"],
    }
    for name, s in groups.items():
        counts, edges = np.histogram(s.to_numpy(), bins=np.arange(-5.0, 15.01, 0.1))
        hist[name] = {"counts": counts.tolist(), "bin_edges_m": edges.tolist()}
    with (out_dir / "b1_clearance_histograms.json").open("w") as f:
        json.dump(hist, f)

    return {"grid_rows": len(grid), "diagonal": diag.to_dict("records")}


# --- B2：thr 网格 ---------------------------------------------------------------


def _v_toward(objects: pd.DataFrame) -> np.ndarray:
    """crossing 趋近速度（risk.crossing_kinematics 同式，逐对象）。"""
    y_c = objects["y_c"].to_numpy()
    v_lat = objects["v_lat"].to_numpy()
    return np.where(np.abs(y_c) <= GATE_BAND_M, np.abs(v_lat), -np.sign(y_c) * v_lat)


def _run_b2(
    objects: pd.DataFrame, anchor_idx: np.ndarray, n: int, out_dir: Path, base_diag
) -> dict:
    """thr 扫描（base 对角 × thr 13 × 两族）+ v_toward 分布 + 一致率诊断。"""
    v_toward = _v_toward(objects)
    thr = np.asarray(THR_GRID)[None, :]                       # (1, 13)
    gate = v_toward[:, None] > thr                            # (n_obj, 13)

    # firing（base 对角，按组分 base）。
    is_vru = objects["category"].isin(VRU_CATEGORIES).to_numpy()
    clearance = objects["min_clearance"].to_numpy()
    base_arr = np.asarray([r["base_vehicle_m"] for r in base_diag])  # 对角 = base
    firing = clearance[:, None] < base_arr[None, :]           # (n_obj, 9)

    rows = []
    consistent = []
    for b_pos, base in enumerate(BASE_GRID):
        for t_pos, thr_v in enumerate(THR_GRID):
            fg = firing[:, b_pos] & gate[:, t_pos]
            vru_hit = _any_per_anchor(fg & is_vru, anchor_idx, n)
            veh_hit = _any_per_anchor(fg & objects["category"].isin(VEHICLE_CROSSING_CATEGORIES).to_numpy(), anchor_idx, n)
            rows.append(
                {
                    "base_m": base,
                    "thr_mps": thr_v,
                    "vru_crossing_rate": float(vru_hit.mean()),
                    "vehicle_crossing_rate": float(veh_hit.mean()),
                }
            )
            # 诊断列：firing ∧ gate 的对象中 gate 判定与 motion_state=crossing 的一致率。
            fg_obj = fg
            if fg_obj.any():
                agree = (fg_obj == objects["motion_state"].eq("crossing").to_numpy())[fg_obj]
                consistent.append(
                    {"base_m": base, "thr_mps": thr_v, "agreement_rate": float(agree.mean())}
                )
    grid = pd.DataFrame(rows)
    grid.to_parquet(out_dir / "b2_thr_grid.parquet", index=False)
    pd.DataFrame(consistent).to_csv(out_dir / "b2_gate_motion_agreement.csv", index=False)

    # firing 对象（对角 base、thr=0.3 最宽 gate）的 v_toward 分族分布直方图。
    fired_wide = firing[:, -1]
    hist = {}
    for name, mask in (
        ("vru", is_vru),
        ("vehicle", objects["category"].isin(VEHICLE_CROSSING_CATEGORIES).to_numpy()),
        ("static", objects["category"].isin(STATIC_CATEGORIES).to_numpy()),
    ):
        vals = v_toward[fired_wide & mask]
        counts, edges = np.histogram(vals, bins=np.arange(0.0, 8.01, 0.1))
        hist[name] = {"counts": counts.tolist(), "bin_edges_mps": edges.tolist()}
    with (out_dir / "b2_v_toward_histograms.json").open("w") as f:
        json.dump(hist, f)

    return {"grid_rows": len(grid)}


# --- B3：object_near 半径 -------------------------------------------------------


def _stationary_core(objects: pd.DataFrame, max_speed_mps: float = 0.5) -> np.ndarray:
    """stationary 的 Y 无关判据：t0 停 ∧ 窗（回退 t0）停 ∧ x_c > 0。

    窗回退语义与正式引擎一致：无 track / 空窗 → 仅 t0 速度判。
    """
    speed = objects["speed_mps"].to_numpy()
    has_win = (objects["has_track"] & objects["win_nonempty"]).to_numpy()
    max_win = objects["max_win_speed"].to_numpy()
    effective = np.where(has_win, max_win, speed)
    return (speed <= max_speed_mps) & (effective <= max_speed_mps) & (objects["x_c"].to_numpy() > 0.0)


def _run_b3(objects: pd.DataFrame, anchor_idx: np.ndarray, n: int, out_dir: Path) -> dict:
    """r 网格：检出率-r 曲线 + 与 stationary 的共现率-(r, Y_st) 矩阵。"""
    distance = objects["distance_m"].to_numpy()
    r_arr = np.asarray(R_GRID)[None, :]
    near_obj = distance[:, None] <= r_arr                     # (n_obj, 9)
    near_any = np.stack([_any_per_anchor(near_obj[:, j], anchor_idx, n) for j in range(len(R_GRID))], axis=1)

    # stationary 的 Y 扫描（共现矩阵需要）。
    core = _stationary_core(objects)
    y_abs = np.abs(objects["y_c"].to_numpy())
    y_arr = np.asarray(Y_ST_GRID)[None, :]
    stat_obj = core[:, None] & (y_abs[:, None] <= y_arr)      # (n_obj, 19)
    stat_any = np.stack([_any_per_anchor(stat_obj[:, j], anchor_idx, n) for j in range(len(Y_ST_GRID))], axis=1)

    rows = [{"radius_m": r, "object_near_rate": float(near_any[:, j].mean())} for j, r in enumerate(R_GRID)]
    pd.DataFrame(rows).to_parquet(out_dir / "b3_radius.parquet", index=False)

    # 共现率矩阵：P(object_near ∧ stationary)（anchor 级联合）。
    co = np.zeros((len(R_GRID), len(Y_ST_GRID)))
    for j in range(len(R_GRID)):
        for k in range(len(Y_ST_GRID)):
            co[j, k] = float((near_any[:, j] & stat_any[:, k]).mean())
    co_df = pd.DataFrame(co, index=list(R_GRID), columns=list(Y_ST_GRID))
    co_df.index.name = "radius_m"
    co_df.columns.name = "y_st_m"
    co_df.to_csv(out_dir / "b3_cooccurrence_matrix.csv")

    # 共现条件率：P(stationary | object_near)——护栏口径（>80% 触发互斥变体）。
    cond = np.zeros_like(co)
    for j in range(len(R_GRID)):
        denom = near_any[:, j].sum()
        for k in range(len(Y_ST_GRID)):
            cond[j, k] = float((near_any[:, j] & stat_any[:, k]).sum() / denom) if denom else 0.0
    cond_df = pd.DataFrame(cond, index=list(R_GRID), columns=list(Y_ST_GRID))
    cond_df.index.name = "radius_m"
    cond_df.columns.name = "y_st_m"
    cond_df.to_csv(out_dir / "b3_cooccurrence_conditional.csv")
    return {"rates": rows}


# --- B4：Y_brake ----------------------------------------------------------------


def _braking_core(objects: pd.DataFrame, speed_drop_mps: float = 0.5, stationary_speed_mps: float = 0.5) -> np.ndarray:
    """braking 的 Y 无关判据：族内 ∧ x_c>0 ∧ 非静止 ∧ 有 track ∧ 窗最小 ≤ t0 − drop。

    族 = BRAKING_FAMILY（vehicle 五类 ∪ bicycle/motorcycle，与 v7 risk.py
    的类别集合一致）。无 track / 空窗 → 不判（减速需正向证据）。
    """
    speed = objects["speed_mps"].to_numpy()
    min_win = objects["min_win_speed"].to_numpy()
    has_win = (objects["has_track"] & objects["win_nonempty"]).to_numpy()
    return (
        objects["category"].isin(BRAKING_FAMILY).to_numpy()
        & (objects["x_c"].to_numpy() > 0.0)
        & (speed > stationary_speed_mps)
        & has_win
        & (min_win <= speed - speed_drop_mps)
    )


def _run_b4(objects: pd.DataFrame, anchor_idx: np.ndarray, n: int, out_dir: Path) -> dict:
    """Y_brake 网格：检出率曲线 + 每档新增前车的 |y_c| 分向构成。"""
    core = _braking_core(objects)
    y = objects["y_c"].to_numpy()
    y_abs = np.abs(y)

    rows = []
    new_leads_rows = []
    prev_mask = np.zeros(len(objects), dtype=bool)
    for y_b in Y_BRAKE_GRID:
        hit_obj = core & (y_abs <= y_b)
        hit_anchor = _any_per_anchor(hit_obj, anchor_idx, n)
        rows.append(
            {
                "y_brake_m": y_b,
                "braking_rate": float(hit_anchor.mean()),
                "hit_objects": int(hit_obj.sum()),
            }
        )
        fresh = hit_obj & ~prev_mask  # 相对前一档新增的前车对象
        if fresh.any() or y_b == Y_BRAKE_GRID[0]:
            left = int((fresh & (y > 0)).sum())
            right = int((fresh & (y < 0)).sum())
            zero = int((fresh & (y == 0)).sum())
            new_leads_rows.append(
                {
                    "y_brake_m": y_b,
                    "new_objects": int(fresh.sum()),
                    "new_left_y_gt0": left,   # y>0 = ego 左侧（含对向压线方向）
                    "new_right_y_lt0": right,
                    "new_zero": zero,
                    "categories": dict(Counter(objects.loc[fresh, "category"])),
                }
            )
        prev_mask = hit_obj
    pd.DataFrame(rows).to_parquet(out_dir / "b4_ybrake.parquet", index=False)
    pd.DataFrame(new_leads_rows).to_json(out_dir / "b4_new_leads.json", orient="records", indent=1, force_ascii=False)
    return {"curve": rows, "new_leads": new_leads_rows}


# --- B5：Y_st -------------------------------------------------------------------


def _run_b5(objects: pd.DataFrame, anchors: pd.DataFrame, anchor_idx: np.ndarray, n: int, out_dir: Path, b0: dict) -> dict:
    """Y_st 网格：检出率 + static 残余覆盖率 + 覆盖对象类别构成。"""
    core = _stationary_core(objects)
    y_abs = np.abs(objects["y_c"].to_numpy())
    is_static = objects["category"].isin(STATIC_CATEGORIES).to_numpy()

    rows = []
    coverage_rows = []
    residual_set = set(b0["static_residual_tokens"])
    is_residual_anchor = anchors["sample_token"].isin(residual_set).to_numpy()

    # 残余 anchor 的 corridor-firing static 对象（吸收残余的具体对象）。
    v4cfg = load_v4_rule_config()
    d_risk_static = v4cfg.corridor.d_risk_vehicle_m  # static 用 vehicle 阈值
    if residual_set:
        residual_obj_mask = (
            is_static
            & objects["sample_token"].isin(residual_set).to_numpy()
            & (objects["min_d_raw"].to_numpy() < d_risk_static)
        )
    else:
        residual_obj_mask = np.zeros(len(objects), dtype=bool)

    covered_category_counter: Counter = Counter()
    for y_st in Y_ST_GRID:
        hit_obj = core & (y_abs <= y_st)
        hit_anchor = _any_per_anchor(hit_obj, anchor_idx, n)
        rows.append(
            {
                "y_st_m": y_st,
                "stationary_rate": float(hit_anchor.mean()),
                "vs_v6_stationary_rate": float(hit_anchor.mean()) - b0["rebuild_factor_rates"].get("stationary_obstacle", 0.0),
            }
        )
        # 残余覆盖率：残余 anchor 中存在 firing-static 对象 |y_c| ≤ Y_st。
        cov_obj = residual_obj_mask & (y_abs <= y_st)
        if residual_set:
            cov_anchor = _any_per_anchor(cov_obj, anchor_idx, n) & is_residual_anchor
            coverage = float(cov_anchor.sum() / len(residual_set))
            for cat, cnt in Counter(objects.loc[cov_obj, "category"]).items():
                covered_category_counter[cat] += cnt
        else:
            coverage = 0.0
        coverage_rows.append({"y_st_m": y_st, "residual_coverage": coverage})

    pd.DataFrame(rows).to_parquet(out_dir / "b5_yst.parquet", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "b5_residual_coverage.csv", index=False)
    with (out_dir / "b5_covered_categories_cumulative.json").open("w") as f:
        json.dump(dict(covered_category_counter), f, ensure_ascii=False, indent=1)
    return {"curve": rows, "coverage": coverage_rows}


# --- 图表（matplotlib，标签用英文避免中文字体缺失） -----------------------------


def _make_figures(out_dir: Path, b1_diag: List[dict], b1_grid: pd.DataFrame, b2_grid: pd.DataFrame, b3: dict, b4: dict, b5: dict) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    produced = []

    def _save(name):
        path = fig_dir / f"{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        produced.append(str(path))

    # F1 B1 对角线：两族 firing 率 + yield 率 vs base。
    base = [r["base_vru_m"] for r in b1_diag]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(base, [r["vru_firing_rate"] for r in b1_diag], marker="o", label="vru firing")
    ax.plot(base, [r["vehicle_firing_rate"] for r in b1_diag], marker="s", label="vehicle+static firing")
    ax.plot(base, [r["any_firing_rate"] for r in b1_diag], marker="^", label="any firing (yield)")
    ax.set_xlabel("base (m), single-assumption diagonal")
    ax.set_ylabel("anchor rate")
    ax.set_title("B1: firing / yield rate vs base (diagonal)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save("b1_diagonal_rates")

    # F2/F3 B1 分裂热力图（vru / vehicle firing rate over (base_vru, base_veh)）。
    for col, name in (("vru_firing_rate", "b1_heatmap_vru"), ("vehicle_firing_rate", "b1_heatmap_vehicle")):
        piv = b1_grid.pivot(index="base_vru_m", columns="base_vehicle_m", values=col)
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), [f"{c:.1f}" for c in piv.columns], rotation=45)
        ax.set_yticks(range(len(piv.index)), [f"{i:.1f}" for i in piv.index])
        ax.set_xlabel("base_vehicle (m)")
        ax.set_ylabel("base_vru (m)")
        ax.set_title(f"B1 split heatmap: {col}")
        fig.colorbar(im, ax=ax, label="anchor rate")
        _save(name)

    # F5/F6 B2 边际曲线：固定 base 档（1.7 / 2.0 / 2.5）的检出率-thr。
    for col, name in (
        ("vru_crossing_rate", "b2_vru_crossing_vs_thr"),
        ("vehicle_crossing_rate", "b2_vehicle_crossing_vs_thr"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for b in (1.7, 2.0, 2.5):
            sub = b2_grid[b2_grid["base_m"] == b]
            ax.plot(sub["thr_mps"], sub[col], marker="o", label=f"base={b}")
        ax.set_xlabel("lateral gate thr (m/s)")
        ax.set_ylabel("anchor detection rate")
        ax.set_title(f"B2: {col} vs thr")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(name)

    # F8 B3 检出率-r（5–15% 目标带）。
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["radius_m"] for r in b3["rates"]], [r["object_near_rate"] for r in b3["rates"]], marker="o")
    ax.axhspan(0.05, 0.15, alpha=0.15, color="green", label="target band 5-15%")
    ax.set_xlabel("object_near radius (m)")
    ax.set_ylabel("anchor detection rate")
    ax.set_title("B3: object_near detection rate vs radius")
    ax.grid(alpha=0.3)
    ax.legend()
    _save("b3_radius_rate")

    # F10 B4 braking 检出率-Y_brake（1.185 锚点竖线）。
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["y_brake_m"] for r in b4["curve"]], [r["braking_rate"] for r in b4["curve"]], marker="o")
    ax.axvline(1.185, color="red", ls="--", alpha=0.6, label="v6 anchor 1.185")
    ax.set_xlabel("Y_brake (m)")
    ax.set_ylabel("anchor detection rate")
    ax.set_title("B4: lead_vehicle_braking rate vs Y_brake")
    ax.grid(alpha=0.3)
    ax.legend()
    _save("b4_braking_rate")

    # F11 B4 新增前车分向堆叠条形。
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ys = [r["y_brake_m"] for r in b4["new_leads"]]
    left = [r["new_left_y_gt0"] for r in b4["new_leads"]]
    right = [-(r["new_right_y_lt0"]) for r in b4["new_leads"]]  # 向下条形表示右侧
    ax.bar(ys, left, label="left (y>0)")
    ax.bar(ys, right, label="right (y<0)")
    ax.axvline(1.185, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("Y_brake (m)")
    ax.set_ylabel("newly in-path lead objects (per 0.1 m step)")
    ax.set_title("B4: newly absorbed leads by side (down = right side)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save("b4_new_leads_sides")

    # F12 B5 stationary 检出率 + static 残余覆盖率双线。
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["y_st_m"] for r in b5["curve"]], [r["stationary_rate"] for r in b5["curve"]], marker="o", label="stationary rate")
    ax.plot([r["y_st_m"] for r in b5["coverage"]], [r["residual_coverage"] for r in b5["coverage"]], marker="s", label="static residual coverage")
    ax.axvline(1.185, color="red", ls="--", alpha=0.6, label="v6 anchor 1.185")
    ax.set_xlabel("Y_st (m)")
    ax.set_ylabel("rate")
    ax.set_title("B5: stationary detection & residual coverage vs Y_st")
    ax.grid(alpha=0.3)
    ax.legend()
    _save("b5_yst_rates")
    return produced


# --- 主流程 ---------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    objects, anchors = _load_features(args.features_dir.expanduser())
    if args.max_anchors is not None:
        keep = set(anchors["sample_token"].head(args.max_anchors))
        anchors = anchors[anchors["sample_token"].isin(keep)].reset_index(drop=True)
        objects = objects[objects["sample_token"].isin(keep)].reset_index(drop=True)
    n = len(anchors)
    anchor_idx = _anchor_index(objects, anchors)
    print(f"anchors={n} object_rows={len(objects)} ({time.time() - t0:.0f}s)", flush=True)

    b0 = _run_b0(objects, anchors)
    (out_dir / "b0_baseline.json").write_text(json.dumps(b0, indent=1, ensure_ascii=False) + "\n")
    print(f"B0 done: residual={b0['static_residual_count']} ({time.time() - t0:.0f}s)", flush=True)

    b1_summary = _run_b1(objects, anchor_idx, n, out_dir)
    print(f"B1 done ({time.time() - t0:.0f}s)", flush=True)
    b1_grid = pd.read_parquet(out_dir / "b1_base_grid.parquet")

    _run_b2(objects, anchor_idx, n, out_dir, b1_summary["diagonal"])
    print(f"B2 done ({time.time() - t0:.0f}s)", flush=True)

    b3 = _run_b3(objects, anchor_idx, n, out_dir)
    print(f"B3 done ({time.time() - t0:.0f}s)", flush=True)

    b4 = _run_b4(objects, anchor_idx, n, out_dir)
    print(f"B4 done ({time.time() - t0:.0f}s)", flush=True)

    b5 = _run_b5(objects, anchors, anchor_idx, n, out_dir, b0)
    print(f"B5 done ({time.time() - t0:.0f}s)", flush=True)

    figures = _make_figures(out_dir, b1_summary["diagonal"], b1_grid, pd.read_parquet(out_dir / "b2_thr_grid.parquet"), b3, b4, b5)

    summary = {
        "anchors": n,
        "object_rows": len(objects),
        "grids": {
            "base": list(BASE_GRID),
            "thr": list(THR_GRID),
            "r": list(R_GRID),
            "y_brake": list(Y_BRAKE_GRID),
            "y_st": list(Y_ST_GRID),
        },
        "b0_static_residual_count": b0["static_residual_count"],
        "b0_yield_rate_v6": b0["gt_yield_rate"],
        "figures": figures,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(f"SWEEP DONE {json.dumps({k: summary[k] for k in ('anchors', 'elapsed_s')})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
