"""Project nuScenes GT 3D boxes into the CAM_FRONT image plane.

Purpose:
    Stage 04 projection helpers. ``camera_boxes_for_frame`` wraps the devkit
    ``NuScenes.get_sample_data``, which returns GT boxes ALREADY transformed
    into the current sensor frame — callers must never apply the ego/sensor
    transforms a second time. ``project_box`` maps the 8 box corners through
    the camera intrinsics, and ``classify_visibility`` mirrors the devkit
    ``box_in_image`` semantics with an out-of-image margin.

Example launch command:
    ``conda activate autovla_codeclean && cd /root/autodl-tmp/drivealign_workspace && \
    PYTHONPATH=DriveAlign/src python -c \
    "from drivealign.data.nuscenes_io import load_nuscenes; \
    from drivealign.evaluation.projection import camera_boxes_for_frame; \
    nusc = load_nuscenes(Path('data/nuscenes/mini')); \
    sd = nusc.get('sample', nusc.scene[0]['first_sample_token'])['data']['CAM_FRONT']; \
    data = camera_boxes_for_frame(nusc, sd); print(len(data.boxes))"``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

# Corners closer to the camera than this depth (m) are treated as behind /
# on the camera plane; devkit uses 0.1 in its image-plane checks.
MIN_DEPTH_M = 0.1


@dataclass(frozen=True)
class CameraFrameData:
    """All GT boxes of one keyframe, already in the sensor frame."""

    image_path: Path
    boxes: tuple[Box, ...]
    intrinsics: np.ndarray  # 3x3 camera_intrinsic


@dataclass(frozen=True)
class ProjectionResult:
    """Projection of one GT box onto the image plane."""

    corners_2d: np.ndarray  # 8x2 pixel coordinates (may be non-finite)
    depth: np.ndarray  # 8 positive-or-negative depths in meters
    distance_m: float  # euclidean norm of the box center in sensor frame


def camera_boxes_for_frame(nusc: NuScenes, sample_data_token: str) -> CameraFrameData:
    """Return image path, GT boxes (sensor frame) and intrinsics for a frame.

    Uses BoxVisibility.NONE so the report contains every annotation and the
    visibility decision is made by classify_visibility below.
    """
    data_path, boxes, cam_intrinsic = nusc.get_sample_data(
        sample_data_token, box_vis_level=BoxVisibility.NONE
    )
    return CameraFrameData(
        image_path=Path(data_path),
        boxes=tuple(boxes),
        intrinsics=np.asarray(cam_intrinsic, dtype=float),
    )


def project_box(box: Box, intrinsics: np.ndarray) -> ProjectionResult:
    """Project the 8 corners of a sensor-frame box via the 3x3 intrinsics."""
    corners_3d = box.corners()  # 3x8, sensor frame
    distance_m = float(np.linalg.norm(box.center))
    with np.errstate(divide="ignore", invalid="ignore"):
        points = view_points(corners_3d, intrinsics, normalize=True)  # 3x8
    # view_points(normalize=True) makes its third row all ones, so the real
    # per-corner depth is the z coordinate in the sensor frame.
    depth = corners_3d[2, :].astype(float)
    return ProjectionResult(
        corners_2d=points[:2, :].T,
        depth=depth,
        distance_m=distance_m,
    )


def classify_visibility(
    corners_2d: np.ndarray,
    depth: np.ndarray,
    image_size: tuple[int, int],
    margin: float = 0.5,
) -> str:
    """Mirror devkit box_in_image semantics; return one of four decisions.

    visible: all 8 corners in front of the camera and inside the image.
    clipped: some corners inside, some outside (box crosses the border).
    out:     every corner outside the extended image bounds.
    behind:  at least one corner at or behind the camera plane (depth <=
             MIN_DEPTH_M); projection coordinates are meaningless then.
    """
    if np.any(depth <= MIN_DEPTH_M):
        return "behind"
    width, height = image_size
    x = corners_2d[:, 0]
    y = corners_2d[:, 1]
    in_x = (x >= -margin * width) & (x <= width * (1 + margin))
    in_y = (y >= -margin * height) & (y <= height * (1 + margin))
    in_bounds = int(np.count_nonzero(in_x & in_y))
    if in_bounds == 0:
        return "out"
    if in_bounds == len(corners_2d):
        return "visible"
    return "clipped"
