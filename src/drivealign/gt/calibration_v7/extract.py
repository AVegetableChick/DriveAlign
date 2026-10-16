"""S13 Stage A：参数无关中间量提取 + G3 闸门（v7 标定扫描，calibration plan §3）。

Purpose:
    一次性提取全部对象级中间量（plan §3.1）与 anchor 级 v6 基线 join
    （§3.2），并在提取时同步执行 G3 闸门（§3.4）：用中间量按 v6 公式
    重建 7 因子 + yield，与 dataset_v4 冻结 GT 逐 anchor 对比。设计核心：

    1. **参数无关**：五个标定槽位（firing.base / gate.thr / object_near.r /
       braking.y_brake / stationary.y_st）都不进提取——CV 外推、支撑函数
       投影、证据窗统计与参数无关，任何参数组合下的因子判定都退化为
       中间量上的轻量数值比较（Stage B sweep.py 的全部输入）。

    2. **提取域 = dataset_v4 train 的全部记录**（22,941 = train manifest
       的 4F anchors 23,425 − 484 个 BAD_GAP quarantine）：train-only 红线
       （val/test 不进任何统计）；G3 join 逐 anchor 完备；与 dataset_v7
       未来构建域一致。

    3. **v6 池复现的关键事实**（G3 可行性的根据）：v7 池与 v4 池的逐类别
       有效锥距完全相同——v4 中 bicycle/motorcycle 属 vehicle 组（45 m）
       == v7 的 per-class 覆盖（45 m）；pedestrian 30 == vru 30；static
       15 == 15。因此池成员与排序 tie-break ``(round(distance,3),
       ann_token)`` 一致，v6 因子重建只需中间量即可逐字节复现。

    4. **同源保证**：``ego_speed_mps`` 直接取 dataset_v4 record 的冻结值
       （GT 构建时的后向差分结果），不重算；motion_state 口径（0.5 m/s /
       45 deg）v4 与 v7 config 相同；track 速度序列同在 ``round(s, 3)``
       之后取 min/max（与正式引擎同舍入口径）。

对象级中间量列（``shard_*_objects.parquet``，一行 = 一个池对象，池序 =
``pool_idx``）：

    ================  =====================================================
    sample_token      anchor 标识（join 键）
    pool_idx          池序下标 0..7（top-8 距离升序，tie-break 同 v4）
    ann_token         标注 token（归因/诊断键）
    category          canonical 10 类
    group_v7          v7 三组归属（vru / vehicle / static）
    x_c, y_c          t0 ego 系中心坐标（ego_frame_xy 原语）
    v_lat             相对速度 ego 系横向分量（risk.crossing_kinematics 同式）
    speed_mps         t0 平面速度
    distance_m        t0 传感器系 3D 中心距
    motion_state      P1.4 四分类
    min_d_raw         min_k d_k（13 CV 采样点带距，不含 h —— v6 口径）
    min_clearance     min_k (d_k − h_k)（d_k=0 样本记 0 —— v7 口径）
    has_track         未来轨迹存在
    win_nonempty      3.0 s 证据窗速度序列非空
    min_win_speed     窗内速度 min（无 track / 空窗为 NaN）
    max_win_speed     窗内速度 max（无 track / 空窗为 NaN）
    ================  =====================================================

anchor 级列（``shard_*_anchors.parquet``，一行 = 一个 anchor）：

    ==================  =============================================
    sample_token        anchor 标识
    scene_token         场景标识（shard 分派 / 诊断分组）
    ego_speed_mps       dataset_v4 冻结 t0 后向差分速度
    n_pool              池对象数
    v6_risk_factors     dataset_v4 冻结 7 因子（逗号 join 字符串）
    v6_yield_required   dataset_v4 冻结 yield_required
    ==================  =============================================

G3 闸门（同步执行，plan §3.4）：
    :func:`rebuild_v6_factors` 内嵌 v6 判定公式（全部常量来自冻结的 v4
    config，经 :func:`drivealign.gt.legacy_v4.load_v4_rule_config` 加载；
    v5/v6 的 gt_rule_config 与 v4 逐字节相同），从对象行重建 7 因子 +
    yield，与 anchor 行的 v6 GT 对比；任何不一致写 mismatches jsonl 并
    在 shard 汇总打印。**全量一致率 100% 才进入 Stage B**。

两个等价性声明（重建公式的正确性依据，均已用 min/any 性质核对）：

    1. ``min_d_raw < d_risk`` ⟺ v6 的 ``any(d_k < d_risk)``（最小值性质，
       浮点逐采样比较与最小值比较逐位一致——d_k 序列本身不落盘、比较
       在同一 float64 序列上进行）；
    2. braking/stationary/oncoming 的"首中即止 break"只影响循环早停，
       不影响命中布尔（any 语义），故重建无需复现 break 位置；congestion
       的 ``leads[0]``（最近前车）= 池内距离最小者，与中间量排序一致。

Example（冒烟 20 anchors，先验证 G3 再谈全量）::

    PYTHONPATH=DriveAlign/src python -m drivealign.gt.calibration_v7.extract \
        --max-anchors 20 --out runs/S13_v7_calibration/smoke

Example（全量 tmux，8 shard 顺序执行；shard 按 scene 模分派，可独立重跑）::

    tmux new-session -d -s s13_extract \
    'source /root/miniconda3/etc/profile.d/conda.sh && conda activate autovla_codeclean && \
     cd /root/autodl-tmp/drivealign_workspace && set -o pipefail && \
     for i in 0 1 2 3 4 5 6 7; do \
       PYTHONPATH=DriveAlign/src python -m drivealign.gt.calibration_v7.extract \
         --num-shards 8 --shard-i $i 2>&1 | tee runs/S13_v7_calibration/extract_s$i.log; \
     done'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from drivealign.data.nuscenes_io import load_nuscenes
from drivealign.evaluation.projection import project_box
from drivealign.gt.backfill import collect_anchor_evidence
from drivealign.gt.config import RuleConfig, canonical_category, load_rule_config
from drivealign.gt.legacy_v4 import RuleConfig as V4RuleConfig
from drivealign.gt.legacy_v4 import load_v4_rule_config
from drivealign.gt.motion_state import classify_motion_state, relative_angle_deg
from drivealign.gt.observability import INFERABLE, OBSERVABLE, classify_observability
from drivealign.gt.pool import PoolCandidate, azimuth_deg_from_center, select_objects
from drivealign.gt.risk import EgoState, ObjectState, clearance_curve, ego_frame_xy

# v6 taxonomy 顺序（重建输出按此序；与 dataset_v4 GT 的因子顺序一致）。
V6_TAXONOMY_ORDER = (
    "pedestrian_crossing",
    "vehicle_merging",
    "lead_vehicle_braking",
    "corridor_conflict",
    "stationary_obstacle",
    "congestion",
    "oncoming_traffic",
)

# v6 yield 口径（v6 action_yield.CORRIDOR_FAMILY）：走廊族任一命中即 true。
V6_CORRIDOR_FAMILY = ("corridor_conflict", "pedestrian_crossing", "vehicle_merging")

# dataset_v4 train shards 的物理位置（相对 workspace 根；提取域的唯一来源）。
DEFAULT_TRAIN_SHARDS = "data/dataset_v4/train"

# 对象表 / anchor 表的列序（parquet 落盘顺序，稳定 schema 便于 sweep 消费）。
OBJECT_COLUMNS = (
    "sample_token",
    "pool_idx",
    "ann_token",
    "category",
    "group_v7",
    "x_c",
    "y_c",
    "v_lat",
    "speed_mps",
    "distance_m",
    "motion_state",
    "min_d_raw",
    "min_clearance",
    "has_track",
    "win_nonempty",
    "min_win_speed",
    "max_win_speed",
)
ANCHOR_COLUMNS = (
    "sample_token",
    "scene_token",
    "ego_speed_mps",
    "n_pool",
    "v6_risk_factors",
    "v6_yield_required",
)


# --- v6 公式重建（G3 闸门 + Stage B B0 共用） ---------------------------------


def rebuild_v6_factors(objs: Sequence[dict], v4cfg: V4RuleConfig) -> Tuple[Tuple[str, ...], bool, set]:
    """从对象行按 v6 公式重建 7 因子 + yield（G3 与 B0 共用，plan §3.4）。

    ``objs`` 必须是**池序**（top-8 距离升序）的对象行列表；每行至少含
    ``ann_token / category / x_c / y_c / speed_mps / distance_m /
    motion_state / min_d_raw / has_track / win_nonempty / min_win_speed /
    max_win_speed``。``v4cfg`` 为冻结 v4 规则配置（v6 GT 的全部常量来源）。

    返回 ``(factors, yield_required, corridor_firing_tokens)``：
    ``factors`` 为 taxonomy 序 7 因子元组；``corridor_firing_tokens`` 为
    v6 走廊口径 firing 的对象 token 集合（B0 static 残余识别需要"所有
    v6-firing 对象均为 static"的判定材料）。

    逐因子对照 v6 版 ``risk.derive_risk_factors``（git HEAD 47b5e0a 版）：

    - corridor 三因子：v6 逐采样 ``any(d_k < d_risk)`` ⟺ ``min_d_raw <
      d_risk``；d_risk 按类别（pedestrian 2.0，其余一律 3.0，含 static
      与两轮车）；``vehicle_merging`` 的族口径 = v4 ``vehicle`` 组
      （含 bicycle/motorcycle）。
    - lead_vehicle_braking：v4 vehicle 组 ∧ 在走廊列（x_c>0 ∧ |y_c|≤
      1.185）∧ 有 track ∧ 窗非空 ∧ t0 非静止 ∧ 窗最小速度 ≤ t0 − 0.5。
    - stationary_obstacle：全类别 ∧ 在走廊列 ∧ t0 停 ∧ 窗最大速度（无
      track / 空窗回退 t0）≤ 0.5。
    - congestion：走廊列内 v4 vehicle 组对象 ≥ 3 且**最近**对象的窗最大
      速度（回退 t0）≤ 2.0。
    - oncoming_traffic：v4 vehicle 组 ∧ motion_state == oncoming ∧
      distance < 20。
    - yield：走廊族三因子任一（v6 ``derive_yield_required``）。
    """
    corr = v4cfg.corridor
    motion_stationary = v4cfg.motion.stationary_speed_mps  # 0.5

    def _d_risk(category: str) -> float:
        # v6 _d_risk_for：仅 pedestrian 走行人阈值，其余（含 static 与
        # 两轮车）一律 vehicle 阈值。
        return (
            corr.d_risk_pedestrian_m
            if category == "pedestrian"
            else corr.d_risk_vehicle_m
        )

    def _is_v4_vehicle(category: str) -> bool:
        # v4 组口径（bicycle/motorcycle 属 vehicle 组）。
        return v4cfg.pool.group_of(category) == "vehicle"

    band = (corr.ego_width_m + corr.width_margin_m) / 2.0  # 1.185

    hits = set()

    # --- 走廊三因子（min_d_raw < d_risk ⟺ 逐采样 any） -------------------
    firing_tokens = {
        o["ann_token"] for o in objs if o["min_d_raw"] < _d_risk(o["category"])
    }
    if firing_tokens:
        hits.add("corridor_conflict")
    if any(o["category"] == "pedestrian" for o in objs if o["ann_token"] in firing_tokens):
        hits.add("pedestrian_crossing")
    if any(_is_v4_vehicle(o["category"]) for o in objs if o["ann_token"] in firing_tokens):
        hits.add("vehicle_merging")

    # --- leads：v4 vehicle 组 ∧ 走廊列（x_c>0 ∧ |y_c|≤1.185），距离升序 ---
    leads = sorted(
        (
            o
            for o in objs
            if _is_v4_vehicle(o["category"]) and o["x_c"] > 0.0 and abs(o["y_c"]) <= band
        ),
        key=lambda o: (round(o["distance_m"], 3), o["ann_token"]),
    )

    # --- lead_vehicle_braking（有 track 才查；窗最小速度判减速） ----------
    brake_cfg = v4cfg.braking
    for lead in leads:
        if not (lead["has_track"] and lead["win_nonempty"]):
            continue  # 无未来轨迹 / 空窗：减速需正向证据，跳过
        if lead["speed_mps"] <= motion_stationary:
            continue  # 静止"前车"不是制动
        if lead["min_win_speed"] <= lead["speed_mps"] - brake_cfg.speed_drop_mps:
            hits.add("lead_vehicle_braking")
            break  # 首中即止（any 语义，重建无需复现 break 位置）

    # --- stationary_obstacle（全类别；无 track/空窗回退仅 t0） ------------
    so_cfg = v4cfg.stationary_obstacle
    for o in objs:  # 池序（objs 保持池序传入）
        if not (o["x_c"] > 0.0 and abs(o["y_c"]) <= band):
            continue
        if o["speed_mps"] > so_cfg.max_speed_mps:
            continue
        max_speed = (
            o["max_win_speed"]
            if (o["has_track"] and o["win_nonempty"])
            else o["speed_mps"]
        )
        if max_speed <= so_cfg.max_speed_mps:
            hits.add("stationary_obstacle")
            break

    # --- congestion（≥3 队列 + 最近前车慢行，回退 t0 速度） ---------------
    cong_cfg = v4cfg.congestion
    if len(leads) >= cong_cfg.min_queue_vehicles:
        nearest = leads[0]
        lead_speed = (
            nearest["max_win_speed"]
            if (nearest["has_track"] and nearest["win_nonempty"])
            else nearest["speed_mps"]
        )
        if lead_speed <= cong_cfg.lead_speed_mps:
            hits.add("congestion")

    # --- oncoming_traffic（heading 规则，严格 < range） -------------------
    for o in objs:
        if (
            _is_v4_vehicle(o["category"])
            and o["motion_state"] == "oncoming"
            and o["distance_m"] < v4cfg.oncoming.range_m
        ):
            hits.add("oncoming_traffic")
            break

    factors = tuple(r for r in V6_TAXONOMY_ORDER if r in hits)
    yield_required = any(r in V6_CORRIDOR_FAMILY for r in factors)
    return factors, yield_required, firing_tokens


# --- 单 anchor 提取 -----------------------------------------------------------


def _build_candidates(evidence, v7cfg: RuleConfig) -> List[PoolCandidate]:
    """Observability gate survivors（与 backfill.build_expected_output 步骤 1 逐行同款）。

    唯一差异：不构造 bbox_2d 以外的 GT 面材料——本函数只服务标定提取，
    池判定消费的字段与 backfill 完全一致（category / distance / azimuth /
    visibility / 全局位姿速度 / v7 盒几何），保证池成员与正式 GT 构建相同。
    """
    candidates: List[PoolCandidate] = []
    for box in evidence.camera_boxes:
        category = canonical_category(box.name)
        if category is None:
            continue  # 10 类封闭词表之外
        ann = evidence.ann_records.get(box.token)
        if ann is None:
            continue
        projection = project_box(box, evidence.intrinsics)
        gated = classify_observability(
            projection.corners_2d,
            projection.depth,
            evidence.image_size,
            v7cfg.observability.min_depth_m,
            v7cfg.observability.min_visible_area_frac,
        )
        if gated.state not in (OBSERVABLE, INFERABLE):
            continue
        candidates.append(
            PoolCandidate(
                ann_token=box.token,
                instance_token=ann.instance_token,
                category=category,
                distance_m=projection.distance_m,
                azimuth_deg=azimuth_deg_from_center(float(box.center[0]), float(box.center[2])),
                bbox_2d=gated.bbox_2d,
                visibility_state=gated.state,
                global_x=ann.x,
                global_y=ann.y,
                velocity_x_mps=ann.velocity_x_mps,
                velocity_y_mps=ann.velocity_y_mps,
                box_length_m=ann.size_l_m,
                box_width_m=ann.size_w_m,
                box_yaw_ego_deg=math.degrees(ann.yaw_global_rad - evidence.ego_yaw),
            )
        )
    return candidates


def _build_tracks(evidence, pool: Sequence[PoolCandidate]) -> Dict[str, object]:
    """未来证据窗轨迹（与 backfill.build_expected_output 步骤 4 同款）。

    速度序列在 round(s, 3) 之后使用（与正式引擎同舍入口径）；键 =
    ann_token（池对象标识）。为避免 import 循环，返回值仍是
    drivealign.gt.risk.ObjectTrack（类型注解用 object 规避）。
    """
    from drivealign.gt.risk import ObjectTrack  # 局部 import：与 risk 无循环依赖

    tracks: Dict[str, ObjectTrack] = {}
    for cand in pool:
        times, speeds, positions = [], [], []
        for frame in evidence.future_frames:
            ann_token = frame.ann_by_instance.get(cand.instance_token)
            if ann_token is None:
                continue
            ann = evidence.ann_records[ann_token]
            times.append((frame.timestamp_us - evidence.ego_timestamp_us) / 1e6)
            speeds.append(math.hypot(ann.velocity_x_mps, ann.velocity_y_mps))
            positions.append((ann.x, ann.y))
        if times:
            tracks[cand.ann_token] = ObjectTrack(
                rel_times_s=tuple(times),
                speeds_mps=tuple(round(s, 3) for s in speeds),
                positions_m=tuple(positions),
            )
    return tracks


def extract_anchor(
    nusc,
    sample_token: str,
    record: dict,
    v7cfg: RuleConfig,
    v4cfg: V4RuleConfig,
) -> Tuple[List[dict], dict, Optional[dict]]:
    """提取一个 anchor 的中间量并同步执行 G3 重建对比。

    ``record`` 是 dataset_v4 的冻结记录（dict）： ego_speed_mps 取
    ``model_inputs.ego_speed_mps``，v6 GT 取
    ``training_targets.expected_output`` 的 risk_factors / yield_required。

    返回 ``(obj_rows, anchor_row, mismatch)``；``mismatch`` 为 None 表示
    G3 对比一致，否则为含 expected/got 明细的 dict（写 mismatches jsonl）。
    """
    expected = record["training_targets"]["expected_output"]
    ego_speed_mps = float(record["model_inputs"]["ego_speed_mps"])

    evidence = collect_anchor_evidence(nusc, sample_token)
    ego_state = EgoState(
        x=evidence.ego_x,
        y=evidence.ego_y,
        yaw=evidence.ego_yaw,
        speed_mps=ego_speed_mps,
        length_m=v7cfg.corridor.ego_length_m,
        width_m=v7cfg.corridor.ego_width_m,
    )

    candidates = _build_candidates(evidence, v7cfg)
    pool = select_objects(candidates, v7cfg.pool)

    motion_by_token = {
        c.ann_token: classify_motion_state(
            c.speed_mps,
            relative_angle_deg(c.velocity_x_mps, c.velocity_y_mps, evidence.ego_yaw),
            v7cfg.motion.stationary_speed_mps,
            v7cfg.motion.band_half_width_deg,
        )
        for c in pool
    }
    tracks = _build_tracks(evidence, pool)

    # v6/v7 证据窗统一 3.0 s（v4 braking/stationary 窗相等，congestion 用
    # brake 窗；v7 两窗亦同值）——单窗统计即可覆盖全部因子重建与扫描。
    evidence_window_s = v4cfg.braking.evidence_window_s
    assert v4cfg.stationary_obstacle.evidence_window_s == evidence_window_s, (
        "v4 braking/stationary 窗不等，单窗统计口径失效，需拆列"
    )

    cos_y = math.cos(ego_state.yaw)
    sin_y = math.sin(ego_state.yaw)

    obj_rows: List[dict] = []
    for idx, c in enumerate(pool):
        obj_state = ObjectState(
            ann_token=c.ann_token,
            category=c.category,
            x=c.global_x,
            y=c.global_y,
            velocity_x_mps=c.velocity_x_mps,
            velocity_y_mps=c.velocity_y_mps,
            distance_m=c.distance_m,
            azimuth_deg=c.azimuth_deg,
            box_length_m=c.box_length_m,
            box_width_m=c.box_width_m,
            box_yaw_ego_deg=c.box_yaw_ego_deg,
        )
        # 13 个 CV 采样 (t_k, d_k, h_k)：d_k 与 v6 走廊曲线同公式同序列；
        # h_k 为 v7 支撑半延展（clearance 口径，零标定）。
        samples = clearance_curve(ego_state, obj_state, v7cfg.corridor)
        min_d_raw = min(d for _, d, _ in samples)
        min_clearance = min((d - h) if d > 0.0 else 0.0 for _, d, h in samples)

        x_c, y_c = ego_frame_xy(ego_state, c.global_x, c.global_y)
        # v_lat：与 risk.crossing_kinematics 内部同式——相对速度的 ego 系横向分量。
        dvx = c.velocity_x_mps - ego_speed_mps * cos_y
        dvy = c.velocity_y_mps - ego_speed_mps * sin_y
        v_lat = -sin_y * dvx + cos_y * dvy

        track = tracks.get(c.ann_token)
        win_speeds = (
            track.window(evidence_window_s).speeds_mps if track is not None else ()
        )
        has_track = track is not None
        win_nonempty = bool(win_speeds)

        obj_rows.append(
            {
                "sample_token": sample_token,
                "pool_idx": idx,
                "ann_token": c.ann_token,
                "category": c.category,
                # v7 三组归属（vru/vehicle/static；两轮车属 vru——与 v4 组
                # 不同，扫描期 firing 分裂与族口径都按 v7 语义消费）。
                "group_v7": v7cfg.pool.group_of(c.category),
                "x_c": x_c,
                "y_c": y_c,
                "v_lat": v_lat,
                "speed_mps": c.speed_mps,
                "distance_m": c.distance_m,
                "motion_state": motion_by_token[c.ann_token],
                "min_d_raw": min_d_raw,
                "min_clearance": min_clearance,
                "has_track": has_track,
                "win_nonempty": win_nonempty,
                "min_win_speed": min(win_speeds) if win_nonempty else float("nan"),
                "max_win_speed": max(win_speeds) if win_nonempty else float("nan"),
            }
        )

    anchor_row = {
        "sample_token": sample_token,
        "scene_token": record["scene_token"],
        "ego_speed_mps": ego_speed_mps,
        "n_pool": len(pool),
        "v6_risk_factors": ",".join(expected["risk_factors"]),
        "v6_yield_required": bool(expected["yield_required"]),
    }

    # --- G3：中间量按 v6 公式重建，与 dataset_v4 冻结 GT 对比 -------------
    rebuilt_factors, rebuilt_yield, _ = rebuild_v6_factors(obj_rows, v4cfg)
    gt_factors = tuple(expected["risk_factors"])
    mismatch = None
    if rebuilt_factors != gt_factors or rebuilt_yield != bool(expected["yield_required"]):
        mismatch = {
            "sample_token": sample_token,
            "gt_risk_factors": list(gt_factors),
            "rebuilt_risk_factors": list(rebuilt_factors),
            "gt_yield_required": bool(expected["yield_required"]),
            "rebuilt_yield_required": rebuilt_yield,
        }

    return obj_rows, anchor_row, mismatch


# --- 数据装配与 shard 主流程 ---------------------------------------------------


def load_train_records(train_dir: Path) -> Dict[str, dict]:
    """读 dataset_v4 train 全部 shards，建立 sample_token -> record map。

    只保留提取消费的字段引用（整条 record 的 dict 原样保留即可，
    22,941 条的内存占用可忽略）。train shards 只含 train split 记录
    （dataset_v4 构建时按 manifest 分列），train-only 红线由此保证。
    """
    records: Dict[str, dict] = {}
    shard_paths = sorted(train_dir.glob("shard-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard-*.jsonl under {train_dir}")
    for path in shard_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["sample_token"]] = rec
    return records


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes/trainval"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--train-shards", type=Path, default=Path(DEFAULT_TRAIN_SHARDS))
    parser.add_argument("--out", type=Path, default=Path("runs/S13_v7_calibration"))
    parser.add_argument("--num-shards", type=int, default=1, help="scene 模分派的 shard 总数")
    parser.add_argument("--shard-i", type=int, default=0, help="本进程处理的 shard 序号")
    parser.add_argument("--max-anchors", type=int, default=None, help="冒烟：本 shard 只跑前 N 个 anchor")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()

    v7cfg = load_rule_config()          # v7 池/观测/motion/走廊几何（逐类别锥距 == v4）
    v4cfg = load_v4_rule_config()       # v6 重建常量（冻结 v4 config，与 v6 逐字节相同）
    if v7cfg.corridor.ego_length_m != v4cfg.corridor.ego_length_m or \
            v7cfg.corridor.ego_width_m != v4cfg.corridor.ego_width_m or \
            v7cfg.corridor.width_margin_m != v4cfg.corridor.width_margin_m:
        raise ValueError("v7/v4 ego 走廊几何不一致，d_k 序列不同源，G3 失效")

    records = load_train_records(args.train_shards.expanduser())
    by_scene: Dict[str, List[str]] = defaultdict(list)
    for token, rec in records.items():
        by_scene[rec["scene_token"]].append(token)
    scenes = sorted(by_scene)
    shard_scenes = scenes[args.shard_i :: args.num_shards]
    shard_tokens = [t for s in shard_scenes for t in sorted(by_scene[s])]
    if args.max_anchors is not None:
        shard_tokens = shard_tokens[: args.max_anchors]

    features_dir = args.out.expanduser() / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{args.shard_i:03d}"

    print(
        f"[{stem}] train records={len(records)} scenes={len(scenes)} "
        f"shard scenes={len(shard_scenes)} anchors={len(shard_tokens)}",
        flush=True,
    )
    print(f"[{stem}] loading nuscenes {args.version} from {args.dataroot} ...", flush=True)
    nusc = load_nuscenes(args.dataroot.expanduser(), args.version)
    print(f"[{stem}] nuscenes loaded ({time.time() - t0:.0f}s)", flush=True)

    all_obj_rows: List[dict] = []
    all_anchor_rows: List[dict] = []
    mismatches: List[dict] = []

    for i, token in enumerate(shard_tokens, start=1):
        obj_rows, anchor_row, mismatch = extract_anchor(
            nusc, token, records[token], v7cfg, v4cfg
        )
        all_obj_rows.extend(obj_rows)
        all_anchor_rows.append(anchor_row)
        if mismatch is not None:
            mismatches.append(mismatch)
        if i % 200 == 0 or i == len(shard_tokens):
            rate = i / max(time.time() - t0, 1e-6)
            print(
                f"[{stem}] anchors {i}/{len(shard_tokens)} "
                f"({rate:.1f}/s, elapsed {time.time() - t0:.0f}s) "
                f"G3 mismatches so far: {len(mismatches)}",
                flush=True,
            )

    objects_df = pd.DataFrame(all_obj_rows, columns=list(OBJECT_COLUMNS))
    anchors_df = pd.DataFrame(all_anchor_rows, columns=list(ANCHOR_COLUMNS))
    objects_path = features_dir / f"{stem}_objects.parquet"
    anchors_path = features_dir / f"{stem}_anchors.parquet"
    objects_df.to_parquet(objects_path, index=False)
    anchors_df.to_parquet(anchors_path, index=False)

    mismatch_path = features_dir / f"{stem}_mismatches.jsonl"
    with mismatch_path.open("w", encoding="utf-8") as f:
        for m in mismatches:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    n = len(all_anchor_rows)
    meta = {
        "shard_i": args.shard_i,
        "num_shards": args.num_shards,
        "anchors": n,
        "object_rows": len(all_obj_rows),
        "g3_mismatches": len(mismatches),
        "g3_consistency": (n - len(mismatches)) / n if n else None,
        "max_anchors": args.max_anchors,
        "train_shards": str(args.train_shards),
        "elapsed_s": round(time.time() - t0, 1),
        "v7_rule_version": v7cfg.rule_version,
        "v4_rule_version": v4cfg.rule_version,
    }
    (features_dir / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[{stem}] DONE {json.dumps(meta, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
