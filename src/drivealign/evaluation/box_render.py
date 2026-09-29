"""PIL-based rendering of projected GT boxes and the four-frame grid.

Purpose:
    Stage 04 visualization. Draws 3D box wireframes (12 edges + front-face
    indication line) on CAM_FRONT frames with PIL only — no matplotlib, no
    system fonts, no wall-clock content — so the JPEG output is deterministic
    for a fixed PIL version. Also composes the 2x2 history grid.

Example launch command:
    (called from drivealign.cli.nuscenes_mini_smoke)
    ``PYTHONPATH=DriveAlign/src python -m drivealign.cli.nuscenes_mini_smoke``
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from drivealign.evaluation.projection import MIN_DEPTH_M

# nuScenes 10-class detection vocabulary -> fixed RGB (short category name).
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (238, 61, 61),
    "truck": (240, 130, 40),
    "construction_vehicle": (155, 90, 210),
    "bus": (235, 200, 40),
    "trailer": (70, 120, 220),
    "barrier": (150, 150, 150),
    "motorcycle": (210, 60, 200),
    "bicycle": (60, 190, 190),
    "pedestrian": (70, 190, 80),
    "traffic_cone": (170, 120, 70),
}

_CAPTION_BAND_PX = 30
_GAP_PX = 6
_BG_RGB = (20, 20, 20)


def short_category(category: str) -> str:
    """'vehicle.car' -> 'car'."""
    return category.split(".")[-1]


def stable_color(category: str) -> tuple[int, int, int]:
    """Fixed palette hit, else a deterministic md5-derived RGB."""
    short = short_category(category)
    if short in CATEGORY_COLORS:
        return CATEGORY_COLORS[short]
    digest = hashlib.md5(category.encode("utf-8")).hexdigest()
    return tuple(int(digest[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _draw_rect(draw: ImageDraw.ImageDraw, corners: Sequence[Sequence[float]],
               color: tuple[int, int, int], width: int) -> None:
    """Closed polyline through the given 2D corners (devkit-style)."""
    prev = corners[-1]
    for corner in corners:
        draw.line([prev, corner], fill=color, width=width)
        prev = corner


def draw_box(
    image: Image.Image,
    corners_2d,  # 8x2 np array; typed loosely to avoid a numpy import here
    depth,  # 8 depths in meters
    category: str,
) -> None:
    """Draw one projected 3D box wireframe (no label; see draw_box_label).

    Devkit corner layout: 0-3 front/bottom face, 4-7 rear/top face, side i
    connects to i+4, and the front orientation line joins corner 0 -> 5.
    """
    color = stable_color(category)
    draw = ImageDraw.Draw(image)
    pts = [(float(x), float(y)) for x, y in corners_2d]
    for i in range(4):  # side edges
        draw.line([pts[i], pts[i + 4]], fill=color, width=2)
    _draw_rect(draw, pts[:4], color, width=2)
    _draw_rect(draw, pts[4:], color, width=2)
    draw.line([pts[0], pts[5]], fill=color, width=3)  # orientation line


def draw_box_label(
    image: Image.Image,
    corners_2d,
    category: str,
    distance_m: float,
) -> None:
    """Draw 'category distance' near the topmost projected corner."""
    text = f"{short_category(category)} {distance_m:.1f}m"
    top = min(corners_2d, key=lambda p: float(p[1]))
    x = min(max(float(top[0]) + 4, 2), image.width - 90)
    y = max(float(top[1]) - 14, _CAPTION_BAND_PX + 2)
    draw = ImageDraw.Draw(image)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # black outline
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=stable_color(category))


def render_frame(
    image_path: Path,
    drawn_boxes: Sequence[tuple],  # (corners_2d, depth, category, distance_m)
    caption: str,
) -> Image.Image:
    """Open a frame, draw its GT boxes (far first) and the caption band."""
    image = Image.open(image_path).convert("RGB")
    ordered = sorted(drawn_boxes, key=lambda item: -item[3])  # far -> near
    for corners_2d, depth, category, distance_m in ordered:
        draw_box(image, corners_2d, depth, category)
    for corners_2d, depth, category, distance_m in ordered:
        if float(min(depth)) > MIN_DEPTH_M:  # behind-plane boxes: draw only
            draw_box_label(image, corners_2d, category, distance_m)

    band = ImageDraw.Draw(image)
    band.rectangle([0, 0, image.width - 1, _CAPTION_BAND_PX - 1], fill=(0, 0, 0))
    band.text((6, 9), caption, fill=(255, 255, 255))
    return image


def compose_grid(panels: Sequence[Image.Image]) -> Image.Image:
    """Compose four same-size panels into a 2x2 grid (TL, TR, BL, BR)."""
    if len(panels) != 4:
        raise ValueError(f"Expected 4 panels, got {len(panels)}")
    width, height = panels[0].size
    out = Image.new(
        "RGB",
        (width * 2 + _GAP_PX * 3, height * 2 + _GAP_PX * 3),
        color=_BG_RGB,
    )
    slots = (
        (_GAP_PX, _GAP_PX),
        (width + _GAP_PX * 2, _GAP_PX),
        (_GAP_PX, height + _GAP_PX * 2),
        (width + _GAP_PX * 2, height + _GAP_PX * 2),
    )
    for panel, (x, y) in zip(panels, slots):
        out.paste(panel, (x, y))
    return out
