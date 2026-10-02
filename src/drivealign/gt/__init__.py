"""GT-side rule engine for expected_output backfill (Stage 08).

Purpose:
    Deterministic derivation of the ``training_targets.expected_output`` GT
    from nuScenes evidence, implementing the frozen S08 decision ledger
    (``docs/experiment_record/S08/S08_pre_implementation_decisions.md``):

    - ``observability``: three-state gating (observable / inferable /
      unscorable) from 8-corner projection with a behind check FIRST; only
      gate survivors may enter the candidate pool (visibility-first
      principle: objects outside CAM_FRONT never produce risk factors).
    - ``pool``: geometric relevance filter (forward cone per class group +
      omnidirectional proximity disc) then distance-ascending top-8 with a
      sample-token tie-break.
    - ``motion_state``: pure kinematics at t0 (stationary -> crossing ->
      oncoming -> same_direction; three symmetric bands fill [0, 180]).
    - ``risk``: the 8-item closed taxonomy. The corridor family
      (corridor_conflict / pedestrian_crossing / vehicle_merging) uses the
      t0 bilateral constant-velocity counterfactual extrapolation (never the
      actual future trajectories, which would self-defeat when ego reacts);
      braking / stationary / congestion use actual-trajectory evidence
      windows; small_following_gap was removed by the P1.2 calibration;
      oncoming_traffic stays a heading rule.
    - ``action_yield``: speed_action from the actual future ego trajectory
      (behavior evidence); yield_required from the counterfactual corridor
      family only (conflict evidence, P1.6 evidence split).
    - ``render``: deterministic English reasoning from frozen templates and
      the shared enum phrase map, with the zero-contradiction gate.
    - ``backfill``: single-record assembly + jsonschema (v4 output_schema)
      validation + gate enforcement.

    All thresholds come from ``configs/contracts/v4/gt_rule_config.json``
    (loaded by :mod:`drivealign.gt.config`) — never hardcoded in modules.
    Values are FROZEN (``status`` = ``frozen``) since the P1.2 train-only
    calibration and the 2026-10-02 truncation-risk verification.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.config import load_rule_config; \
    cfg = load_rule_config(); print(cfg.rule_version, cfg.status, cfg.pool.cone_half_angle_deg)"``
"""

from drivealign.gt.config import (
    RuleConfig,
    load_phrase_map,
    load_rule_config,
    load_templates,
)

__all__ = [
    "RuleConfig",
    "load_rule_config",
    "load_phrase_map",
    "load_templates",
]
