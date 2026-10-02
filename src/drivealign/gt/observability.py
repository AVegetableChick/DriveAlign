"""Three-state observability gating for CAM_FRONT boxes (S08 P1.3).

Purpose:
    Decide, for one GT box projected onto the CAM_FRONT image plane, whether
    it can appear in GT at all (evaluation denominator qualification +
    trainability filter). Exact three states (decision ledger 2026-10-02,
    the former privileged state is gone):

    - ``unscorable``: any corner depth <= min_depth_m ("behind" — degenerate
      projections across the camera plane are discarded wholesale), or the
      AABB does not intersect the image, or the clipped visible area falls
      below ``min_visible_area_frac`` (anti-degeneration, 0 disables).
    - ``observable``: all corners in front and the AABB fully inside the
      image; bbox_2d = the raw AABB.
    - ``inferable``: all corners in front, AABB crossing the image border
      but intersecting it; bbox_2d = the CLIPPED AABB (the same clipped box
      is used for the expected_output bbox and the M09 IoU scoring — GT and
      scoring share one implementation, so no double bookkeeping).

    Gating runs BEFORE pool selection (visibility-first principle): only
    observable / inferable boxes may enter the candidate pool, and objects
    outside CAM_FRONT never produce risk factors.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "import numpy as np; \
    from drivealign.gt.observability import classify_observability; \
    c = np.array([[10., 10.], [50., 10.], [50., 50.], [10., 50.]] * 2); \
    d = np.full(8, 20.0); \
    print(classify_observability(c, d, (1600, 900)))"``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

OBSERVABLE = "observable"
INFERABLE = "inferable"
UNSCORABLE = "unscorable"

# Unscorable sub-reasons (audit trail for the invisibility report).
REASON_BEHIND = "behind"
REASON_NO_OVERLAP = "no_overlap"
REASON_DEGENERATE = "degenerate"


@dataclass(frozen=True)
class ObservabilityResult:
    """Gate decision for one box; bbox_2d is None exactly when unscorable."""

    state: str
    bbox_2d: Optional[Tuple[float, float, float, float]]
    reason: Optional[str]  # None unless unscorable


def classify_observability(
    corners_2d: np.ndarray,
    depth: np.ndarray,
    image_size: Tuple[int, int],
    min_depth_m: float = 0.1,
    min_visible_area_frac: float = 0.0,
) -> ObservabilityResult:
    """Gate one box from its 8 projected corners and per-corner depths.

    ``corners_2d`` is 8x2 pixel coordinates (may be non-finite), ``depth``
    the 8 sensor-frame z coordinates; ``image_size`` is (width, height).
    """
    if np.any(np.asarray(depth) <= min_depth_m):
        return ObservabilityResult(UNSCORABLE, None, REASON_BEHIND)

    width, height = image_size
    x = np.asarray(corners_2d)[:, 0]
    y = np.asarray(corners_2d)[:, 1]
    ax0, ay0 = float(x.min()), float(y.min())
    ax1, ay1 = float(x.max()), float(y.max())

    # Clip to the image bounds [0, W] x [0, H].
    cx0 = max(ax0, 0.0)
    cy0 = max(ay0, 0.0)
    cx1 = min(ax1, float(width))
    cy1 = min(ay1, float(height))
    full_area = (ax1 - ax0) * (ay1 - ay0)
    if cx1 <= cx0 or cy1 <= cy0:
        if full_area <= 0.0 and cx0 == cx1 and cy0 == cy1:
            # Edge-on box landing on the image: the clip is empty only
            # because the AABB has no extent -> degenerate, not a miss.
            return ObservabilityResult(UNSCORABLE, None, REASON_DEGENERATE)
        return ObservabilityResult(UNSCORABLE, None, REASON_NO_OVERLAP)

    visible_area = (cx1 - cx0) * (cy1 - cy0)
    if min_visible_area_frac > 0.0 and full_area > 0.0:
        if visible_area / full_area < min_visible_area_frac:
            return ObservabilityResult(UNSCORABLE, None, REASON_DEGENERATE)

    # Round for a stable canonical representation (1 decimal of a pixel).
    def _r(v: float) -> float:
        return round(float(v), 1)

    inside = ax0 >= 0.0 and ay0 >= 0.0 and ax1 <= width and ay1 <= height
    if inside:
        return ObservabilityResult(
            OBSERVABLE, (_r(ax0), _r(ay0), _r(ax1), _r(ay1)), None
        )
    return ObservabilityResult(
        INFERABLE, (_r(cx0), _r(cy0), _r(cx1), _r(cy1)), None
    )
