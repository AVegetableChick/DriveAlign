"""Unit pins for the S13 v7 calibration extract layer (Stage A).

Purpose:
    The calibration sweep (Stage B) trusts the extracted intermediate
    features blindly — so this file pins the three foundations that make
    the G3 gate meaningful:

    1. **Pool equivalence**: the v7 pool and the v4 pool have the same
       per-category effective cone range (bicycle/motorcycle live in the
       v4 ``vehicle`` group at 45 m == the v7 per-class override at 45 m).
       This is WHY the v6 factors can be rebuilt from v7-pool features at
       all; if a future config edit breaks it, G3 loses its basis.
    2. **Rebuild semantics**: ``rebuild_v6_factors`` reproduces the v6
       rule engine (git HEAD 47b5e0a ``risk.py``) from feature rows —
       corridor trio via the ``min_d_raw < d_risk`` equivalence, the
       braking/stationary/congestion window logic with the no-track
       fallback asymmetry, the oncoming heading rule, and the corridor
       family yield. Synthetic rows follow the decision-ledger semantics
       (S13_contract_v7_decision_log.md §3.4).
    3. **Column contract**: the parquet schemas consumed by sweep.py are
       pinned by column order.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -m pytest \
    tests/unit/test_calibration_v7_extract.py -q``
"""

import math

import pytest

from drivealign.gt.calibration_v7.extract import (
    ANCHOR_COLUMNS,
    OBJECT_COLUMNS,
    V6_TAXONOMY_ORDER,
    rebuild_v6_factors,
)
from drivealign.gt.config import NUSCENES_CATEGORY_MAP, load_rule_config
from drivealign.gt.legacy_v4 import load_v4_rule_config


@pytest.fixture(scope="module")
def v4cfg():
    return load_v4_rule_config()


def _o(
    ann_token: str = "a",
    category: str = "car",
    x_c: float = 30.0,
    y_c: float = 5.0,
    speed_mps: float = 5.0,
    distance_m: float = 30.5,
    motion_state: str = "same_direction",
    min_d_raw: float = 10.0,
    has_track: bool = False,
    win_nonempty: bool = False,
    min_win_speed: float = math.nan,
    max_win_speed: float = math.nan,
) -> dict:
    """一个中性池对象行：默认远离一切触发（走廊外 / 快速 / 远距）。

    各测试用例只覆写触达判据的字段；默认值保证"不触发"基线。
    """
    return {
        "ann_token": ann_token,
        "category": category,
        "x_c": x_c,
        "y_c": y_c,
        "speed_mps": speed_mps,
        "distance_m": distance_m,
        "motion_state": motion_state,
        "min_d_raw": min_d_raw,
        "has_track": has_track,
        "win_nonempty": win_nonempty,
        "min_win_speed": min_win_speed,
        "max_win_speed": max_win_speed,
    }


# --- 1. pool equivalence（G3 可行性根基） -------------------------------------


def test_v7_pool_matches_v4_pool_per_category():
    """v7 池与 v4 池的逐类别有效锥距完全相同（含 per-class 覆盖）。"""
    v7pool = load_rule_config().pool
    v4pool = load_v4_rule_config().pool
    cats = sorted({c for c in NUSCENES_CATEGORY_MAP.values() if c})
    for cat in cats:
        assert v7pool.cone_range_for(cat) == v4pool.cone_range_by_group_m[
            v4pool.group_of(cat)
        ], f"cone range drifted for {cat}"


def test_v7_v4_corridor_geometry_identical():
    """v7/v4 ego 走廊几何一致（d_k 序列同源的加载期前提）。"""
    v7 = load_rule_config().corridor
    v4 = load_v4_rule_config().corridor
    assert (v7.ego_length_m, v7.ego_width_m, v7.width_margin_m) == (
        v4.ego_length_m,
        v4.ego_width_m,
        v4.width_margin_m,
    )
    assert (v7.t_risk_s, v7.time_step_s) == (v4.t_risk_s, v4.time_step_s)


def test_v4_window_configs_equal():
    """v4 braking/stationary 证据窗相等——单窗统计（3.0 s）口径的前提。"""
    v4 = load_v4_rule_config()
    assert v4.braking.evidence_window_s == v4.stationary_obstacle.evidence_window_s


# --- 2. rebuild semantics（v6 引擎复现） --------------------------------------


def test_rebuild_v6_corridor_trio_and_yield(v4cfg):
    """走廊三因子的分组口径：ped 2.0 / 其余 3.0；yield = 走廊族任一。"""
    # pedestrian min_d_raw=1.5 → firing(<2.0)；car 2.5 → firing(<3.0)。
    objs = [
        _o("p", category="pedestrian", min_d_raw=1.5),
        _o("c", category="car", min_d_raw=2.5),
    ]
    factors, yld, fired = rebuild_v6_factors(objs, v4cfg)
    assert set(fired) == {"p", "c"}
    assert "corridor_conflict" in factors
    assert "pedestrian_crossing" in factors
    assert "vehicle_merging" in factors
    assert yld is True


def test_rebuild_v6_bicycle_is_v4_vehicle_group(v4cfg):
    """bicycle 在 v4 组口径下属 vehicle（merging 族），v7 组概念不影响重建。"""
    objs = [_o("b", category="bicycle", min_d_raw=2.5)]  # < 3.0 firing
    factors, _, fired = rebuild_v6_factors(objs, v4cfg)
    assert fired == {"b"}
    assert "vehicle_merging" in factors
    assert "pedestrian_crossing" not in factors


def test_rebuild_v6_static_firing_no_crossing_label(v4cfg):
    """static 对象 firing 只给 corridor_conflict（无 crossing 标签）但喂 yield。"""
    objs = [_o("bar", category="barrier", min_d_raw=2.9)]
    factors, yld, fired = rebuild_v6_factors(objs, v4cfg)
    assert factors == ("corridor_conflict",)
    assert yld is True
    assert fired == {"bar"}


def test_rebuild_v6_min_any_equivalence_boundary(v4cfg):
    """min_d_raw == d_risk 恰等不触发（v6 逐采样严格 <；min 性质逐位一致）。"""
    objs = [_o("c", category="car", min_d_raw=3.0)]  # 恰等 3.0
    factors, yld, fired = rebuild_v6_factors(objs, v4cfg)
    assert fired == set()
    assert factors == ()
    assert yld is False


def test_rebuild_v6_braking_window_logic(v4cfg):
    """braking：列内 + 非静止 + 有 track + 窗最小速度 ≤ t0 − 0.5。"""
    # 命中：7.2 ≤ 8 − 0.5。
    hit = _o("c", x_c=10.0, y_c=0.5, speed_mps=8.0,
             has_track=True, win_nonempty=True, min_win_speed=7.2, max_win_speed=8.0)
    factors, _, _ = rebuild_v6_factors([hit], v4cfg)
    assert "lead_vehicle_braking" in factors

    # 不命中：窗最小 7.6 > 7.5。
    miss = _o("c", x_c=10.0, y_c=0.5, speed_mps=8.0,
              has_track=True, win_nonempty=True, min_win_speed=7.6, max_win_speed=8.0)
    factors, _, _ = rebuild_v6_factors([miss], v4cfg)
    assert "lead_vehicle_braking" not in factors

    # 无 track / 空窗跳过（减速需正向证据）。
    for no_evidence in (
        _o("c", x_c=10.0, y_c=0.5, speed_mps=8.0),  # has_track=False
        _o("c", x_c=10.0, y_c=0.5, speed_mps=8.0, has_track=True, win_nonempty=False),
    ):
        factors, _, _ = rebuild_v6_factors([no_evidence], v4cfg)
        assert "lead_vehicle_braking" not in factors

    # 静止"前车"不是制动；列外（|y_c| > 1.185）也不判。
    factors, _, _ = rebuild_v6_factors(
        [_o("c", x_c=10.0, y_c=0.5, speed_mps=0.4,
            has_track=True, win_nonempty=True, min_win_speed=0.0, max_win_speed=0.3)], v4cfg
    )
    assert "lead_vehicle_braking" not in factors
    factors, _, _ = rebuild_v6_factors(
        [_o("c", x_c=10.0, y_c=1.3, speed_mps=8.0,
            has_track=True, win_nonempty=True, min_win_speed=7.2, max_win_speed=8.0)], v4cfg
    )
    assert "lead_vehicle_braking" not in factors


def test_rebuild_v6_stationary_fallback_and_band(v4cfg):
    """stationary：全类别；无 track 回退仅 t0；窗最大速度判"继续停"。"""
    # 锥桶列内停着（无 track → 回退 t0 0.2 ≤ 0.5）。
    objs = [_o("cone", category="traffic_cone", x_c=8.0, y_c=0.3, speed_mps=0.2)]
    factors, _, _ = rebuild_v6_factors(objs, v4cfg)
    assert "stationary_obstacle" in factors

    # 有 track 但窗内动起来（max 0.6 > 0.5）→ 不判。
    objs = [_o("cone", category="traffic_cone", x_c=8.0, y_c=0.3, speed_mps=0.2,
               has_track=True, win_nonempty=True, min_win_speed=0.1, max_win_speed=0.6)]
    factors, _, _ = rebuild_v6_factors(objs, v4cfg)
    assert "stationary_obstacle" not in factors

    # 列外（|y_c| = 1.3 > 1.185）→ 不判；恰在 1.185 边界（≤）→ 判。
    objs = [_o("c", x_c=8.0, y_c=1.3, speed_mps=0.2)]
    assert "stationary_obstacle" not in rebuild_v6_factors(objs, v4cfg)[0]
    objs = [_o("c", x_c=8.0, y_c=1.185, speed_mps=0.2)]
    assert "stationary_obstacle" in rebuild_v6_factors(objs, v4cfg)[0]


def test_rebuild_v6_congestion_queue_and_nearest(v4cfg):
    """congestion：列内 vehicle 组 ≥3 + 最近前车窗最大速度 ≤ 2.0。"""

    def _queue(nearest_max_win: float) -> list:
        rows = []
        for i in range(3):
            rows.append(
                _o(f"c{i}", x_c=6.0 + 5.0 * i, y_c=0.2, speed_mps=1.0,
                   has_track=True, win_nonempty=True,
                   min_win_speed=1.0, max_win_speed=nearest_max_win if i == 0 else 1.5)
            )
        return rows

    assert "congestion" in rebuild_v6_factors(_queue(1.5), v4cfg)[0]
    # 队列不足 3。
    assert "congestion" not in rebuild_v6_factors(_queue(1.5)[:2], v4cfg)[0]
    # 最近前车窗最大 2.5 > 2.0（即使回退也按此判）。
    assert "congestion" not in rebuild_v6_factors(_queue(2.5), v4cfg)[0]
    # 最近前车无 track → 回退 t0 速度。
    rows = _queue(1.5)
    rows[0] = _o("c0", x_c=6.0, y_c=0.2, speed_mps=1.5)
    assert "congestion" in rebuild_v6_factors(rows, v4cfg)[0]
    rows[0] = _o("c0", x_c=6.0, y_c=0.2, speed_mps=2.5)
    assert "congestion" not in rebuild_v6_factors(rows, v4cfg)[0]


def test_rebuild_v6_oncoming_rule(v4cfg):
    """oncoming：v4 vehicle 组 ∧ motion_state == oncoming ∧ distance < 20。"""
    hit = _o("c", category="car", distance_m=15.0, motion_state="oncoming")
    assert "oncoming_traffic" in rebuild_v6_factors([hit], v4cfg)[0]
    # 严格 < range：20.0 恰等不触发。
    edge = _o("c", category="car", distance_m=20.0, motion_state="oncoming")
    assert "oncoming_traffic" not in rebuild_v6_factors([edge], v4cfg)[0]
    # v4 组外（pedestrian）不触发；列内静止不改变 oncoming 判定。
    ped = _o("p", category="pedestrian", distance_m=15.0, motion_state="oncoming")
    assert "oncoming_traffic" not in rebuild_v6_factors([ped], v4cfg)[0]


def test_rebuild_v6_yield_excludes_window_factors(v4cfg):
    """yield（v6）只看走廊族：仅 braking / stationary 命中 → yield False。"""
    braking_only = [
        _o("c", x_c=10.0, y_c=0.5, speed_mps=8.0,
           has_track=True, win_nonempty=True, min_win_speed=7.2, max_win_speed=8.0)
    ]
    factors, yld, _ = rebuild_v6_factors(braking_only, v4cfg)
    assert factors == ("lead_vehicle_braking",) and yld is False

    stationary_only = [
        _o("cone", category="traffic_cone", x_c=8.0, y_c=0.3, speed_mps=0.2)
    ]
    factors, yld, _ = rebuild_v6_factors(stationary_only, v4cfg)
    assert factors == ("stationary_obstacle",) and yld is False


def test_rebuild_v6_taxonomy_order(v4cfg):
    """重建因子输出按 v6 taxonomy 顺序（与 dataset_v4 GT 的因子序一致）。"""
    objs = [
        _o("p", category="pedestrian", min_d_raw=1.5),   # pedestrian_crossing
        _o("c", x_c=10.0, y_c=0.5, speed_mps=8.0,        # lead_vehicle_braking
           has_track=True, win_nonempty=True, min_win_speed=7.2, max_win_speed=8.0),
        _o("cone", category="traffic_cone", x_c=8.0, y_c=0.3, speed_mps=0.2,
           distance_m=8.0),                              # stationary_obstacle
        _o("on", category="car", distance_m=15.0, motion_state="oncoming"),
    ]
    factors, _, _ = rebuild_v6_factors(objs, v4cfg)
    assert factors == (
        "pedestrian_crossing",
        "lead_vehicle_braking",
        "corridor_conflict",
        "stationary_obstacle",
        "oncoming_traffic",
    )
    # V6_TAXONOMY_ORDER 覆盖且仅覆盖 v6 的 7 项（顺序钉子）。
    assert V6_TAXONOMY_ORDER == (
        "pedestrian_crossing",
        "vehicle_merging",
        "lead_vehicle_braking",
        "corridor_conflict",
        "stationary_obstacle",
        "congestion",
        "oncoming_traffic",
    )


# --- 3. parquet 列契约（sweep.py 消费面） -------------------------------------


def test_object_columns_contract():
    """对象表列序钉子（sweep 按名读取，顺序变化须显式改此 pin）。"""
    assert OBJECT_COLUMNS == (
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


def test_anchor_columns_contract():
    """anchor 表列序钉子。"""
    assert ANCHOR_COLUMNS == (
        "sample_token",
        "scene_token",
        "ego_speed_mps",
        "n_pool",
        "v6_risk_factors",
        "v6_yield_required",
    )
