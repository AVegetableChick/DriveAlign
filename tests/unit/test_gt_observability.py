"""Unit tests for the S08 three-state observability gating.

Purpose:
    Freeze the P1.3 gate cases: behind corners discard the whole box, a
    fully-inside AABB is observable, a border-crossing AABB is inferable
    with the CLIPPED bbox (the box Eval IoU scoring must reuse), a
    non-intersecting AABB is unscorable, and the anti-degeneration area
    fraction is enforced when enabled.

Startup command:
    ``conda activate autovla_codeclean && PYTHONPATH=DriveAlign/src python -m \
    pytest DriveAlign/tests/unit/test_gt_observability.py -q``
"""

from __future__ import annotations

import numpy as np
import pytest

from drivealign.gt.observability import (
    INFERABLE,
    OBSERVABLE,
    REASON_BEHIND,
    REASON_DEGENERATE,
    REASON_NO_OVERLAP,
    UNSCORABLE,
    classify_observability,
)

IMAGE = (800, 600)


def _corners(x0, y0, x1, y1):
    # One corner per quadrant of the AABB, duplicated to 8 entries.
    quad = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return np.array(quad * 2, dtype=float)


def _depth(value):
    return np.full(8, float(value))


def test_behind_corner_discards_whole_box():
    corners = _corners(100, 100, 200, 200)
    depth = _depth(20.0)
    depth[0] = 0.05  # one corner at/behind the camera plane
    result = classify_observability(corners, depth, IMAGE)
    assert result.state == UNSCORABLE
    assert result.reason == REASON_BEHIND
    assert result.bbox_2d is None


def test_fully_inside_is_observable_with_raw_aabb():
    corners = _corners(100, 100, 200, 200)
    result = classify_observability(corners, _depth(20.0), IMAGE)
    assert result.state == OBSERVABLE
    assert result.bbox_2d == (100.0, 100.0, 200.0, 200.0)
    assert result.reason is None


def test_border_crossing_is_inferable_with_clipped_bbox():
    # AABB extends past the right and bottom image borders.
    corners = _corners(700, 500, 900, 700)
    result = classify_observability(corners, _depth(20.0), IMAGE)
    assert result.state == INFERABLE
    assert result.bbox_2d == (700.0, 500.0, 800.0, 600.0)


def test_no_overlap_is_unscorable():
    corners = _corners(900, 700, 1000, 800)  # fully outside the image
    result = classify_observability(corners, _depth(20.0), IMAGE)
    assert result.state == UNSCORABLE
    assert result.reason == REASON_NO_OVERLAP
    assert result.bbox_2d is None


def test_min_visible_area_frac_guards_degenerate_clips():
    # 90% of the box lies outside the image; a 0.2 fraction gate rejects it.
    corners = _corners(750, 550, 1150, 950)
    enabled = classify_observability(
        corners, _depth(20.0), IMAGE, min_visible_area_frac=0.2
    )
    assert enabled.state == UNSCORABLE
    assert enabled.reason == REASON_DEGENERATE
    disabled = classify_observability(
        corners, _depth(20.0), IMAGE, min_visible_area_frac=0.0
    )
    assert disabled.state == INFERABLE  # gate disabled by default


def test_zero_area_aabb_inside_image_is_degenerate():
    corners = _corners(100, 100, 100, 100)  # edge-on box
    result = classify_observability(corners, _depth(20.0), IMAGE)
    assert result.state == UNSCORABLE
    assert result.reason == REASON_DEGENERATE


@pytest.mark.parametrize("depth", [0.0, -1.0, 0.1, 0.0999])
def test_depth_boundary_values_are_behind(depth):
    result = classify_observability(_corners(0, 0, 10, 10), _depth(depth), IMAGE)
    assert result.state == UNSCORABLE
