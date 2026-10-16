"""GT-side rule engine for expected_output backfill (S13 contract v7).

Purpose:
    Deterministic derivation of the ``training_targets.expected_output`` GT
    from nuScenes evidence, implementing the v7 rules doc
    (``S13_v7_risk_factor_rules.md``) and BEV GT rules doc:

    - ``observability``: three-state gating (observable / inferable /
      unscorable) from 8-corner projection with a behind check FIRST; only
      gate survivors may enter the candidate pool (visibility-first
      principle: objects outside CAM_FRONT never produce risk factors).
    - ``pool``: geometric relevance filter (forward cone with per-class
      range override + omnidirectional proximity disc) then
      distance-ascending top-8 with an annotation-token tie-break.
    - ``motion_state``: pure kinematics at t0 (stationary -> crossing ->
      oncoming -> same_direction); v7 uses it ONLY for the oncoming_traffic
      heading rule and as a model-visible feature — no longer as a crossing
      label gate.
    - ``risk``: the v7 six-item taxonomy driven by the firing engine
      (bilateral CV clearance with the deterministic support-function box
      projection, rules §0.4-0.5) + the unified lateral gate (§0.7).
      Returns ``(factors, any_firing)``; the firing union feeds yield.
    - ``action_yield``: speed_action from the actual future ego trajectory
      (behavior evidence); yield_required from the firing union (conflict
      evidence, P1.6 evidence split).
    - ``backfill``: single-record assembly + BEV nearest-point GT +
      jsonschema (v7 output_schema, FOUR fields — the reasoning render chain
      is retired) validation.
    - ``legacy_v4``: frozen S08 v4 reproduction layer, consumed ONLY by the
      two S08 audit CLIs — do not extend.

    All thresholds come from ``configs/contracts/v7/gt_rule_config.json``
    (loaded by :mod:`drivealign.gt.config`) — never hardcoded in modules.
    The five calibration slots stay provisional until the S13 train-only
    calibration sweep freezes them.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.gt.config import load_rule_config; \
    cfg = load_rule_config(); print(cfg.rule_version, cfg.status, cfg.firing.base_m)"``
"""

from drivealign.gt.config import RuleConfig, load_rule_config

__all__ = [
    "RuleConfig",
    "load_rule_config",
]
