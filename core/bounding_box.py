"""
core/bounding_box.py
3D Bounding Box Computation, Dimension Extraction, and Procedural Cage Geometry for UniView.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import numpy as np
import glm

from core.scene_loader import SceneData
from core.math_utils import AABB, glm_to_numpy


@dataclass
class BoundingBoxInfo:
    """Consolidated 3D bounding box metrics and spatial bounds."""
    min_pt: glm.vec3
    max_pt: glm.vec3
    center: glm.vec3
    size: glm.vec3          # (width=X, height=Y, depth=Z)
    diagonal: float
    volume: float
    is_valid: bool = True

    @property
    def width(self) -> float:
        return float(self.size.x)

    @property
    def height(self) -> float:
        return float(self.size.y)

    @property
    def depth(self) -> float:
        return float(self.size.z)


def compute_scene_bounds(
    scene: Optional[SceneData],
    model_matrix: Optional[glm.mat4] = None,
) -> BoundingBoxInfo:
    """
    Compute oriented/transformed world-space bounding box for the active 3D model.
    Accounts for model matrix, up-vector rotations, and submesh node transforms.
    """
    if scene is None or not scene.scene_aabb.is_valid:
        default_min = glm.vec3(-0.5, -0.5, -0.5)
        default_max = glm.vec3(0.5, 0.5, 0.5)
        size = default_max - default_min
        return BoundingBoxInfo(
            min_pt=default_min,
            max_pt=default_max,
            center=glm.vec3(0.0),
            size=size,
            diagonal=float(glm.length(size)),
            volume=float(size.x * size.y * size.z),
            is_valid=False,
        )

    m_mat = model_matrix if model_matrix is not None else glm.mat4(1.0)

    # Transform 8 corners of the scene AABB
    orig_min = scene.scene_aabb.min_pt
    orig_max = scene.scene_aabb.max_pt

    corners = [
        glm.vec3(orig_min.x, orig_min.y, orig_min.z),
        glm.vec3(orig_max.x, orig_min.y, orig_min.z),
        glm.vec3(orig_max.x, orig_min.y, orig_max.z),
        glm.vec3(orig_min.x, orig_min.y, orig_max.z),
        glm.vec3(orig_min.x, orig_max.y, orig_min.z),
        glm.vec3(orig_max.x, orig_max.y, orig_min.z),
        glm.vec3(orig_max.x, orig_max.y, orig_max.z),
        glm.vec3(orig_min.x, orig_max.y, orig_max.z),
    ]

    t_min = glm.vec3(float("inf"))
    t_max = glm.vec3(float("-inf"))

    for c in corners:
        tc = glm.vec3(m_mat * glm.vec4(c, 1.0))
        t_min = glm.min(t_min, tc)
        t_max = glm.max(t_max, tc)

    size = t_max - t_min
    center = (t_min + t_max) * 0.5
    diag = float(glm.length(size))
    vol = float(max(size.x, 1e-4) * max(size.y, 1e-4) * max(size.z, 1e-4))

    return BoundingBoxInfo(
        min_pt=t_min,
        max_pt=t_max,
        center=center,
        size=size,
        diagonal=diag,
        volume=vol,
        is_valid=True,
    )


def generate_bbox_cage_lines(
    bbox: BoundingBoxInfo,
    edge_color: Tuple[float, float, float] = (0.25, 0.82, 0.95),
    corner_color: Tuple[float, float, float] = (1.0, 0.85, 0.25),
    corner_tick_ratio: float = 0.18,
) -> np.ndarray:
    """
    Generate 3D line segment geometry for an architectural / CAD style bounding cage.
    Includes:
    - 12 primary bounding box edges in edge_color
    - Highlighting corner brackets (ticks) at each of the 8 corners in corner_color
    Returns float32 ndarray of shape (N * 2, 6) -> [px, py, pz, r, g, b].
    """
    if not bbox.is_valid:
        return np.zeros((0, 6), dtype=np.float32)

    p0 = bbox.min_pt
    p1 = bbox.max_pt

    # 8 box corners:
    # Bottom 4 (y = p0.y):
    c0 = glm.vec3(p0.x, p0.y, p0.z)
    c1 = glm.vec3(p1.x, p0.y, p0.z)
    c2 = glm.vec3(p1.x, p0.y, p1.z)
    c3 = glm.vec3(p0.x, p0.y, p1.z)

    # Top 4 (y = p1.y):
    c4 = glm.vec3(p0.x, p1.y, p0.z)
    c5 = glm.vec3(p1.x, p1.y, p0.z)
    c6 = glm.vec3(p1.x, p1.y, p1.z)
    c7 = glm.vec3(p0.x, p1.y, p1.z)

    lines: List[List[float]] = []

    def add_line(a: glm.vec3, b: glm.vec3, col: Tuple[float, float, float]):
        lines.append([a.x, a.y, a.z, col[0], col[1], col[2]])
        lines.append([b.x, b.y, b.z, col[0], col[1], col[2]])

    # 1. 12 Main Box Edges (Semi-subtle cage lines)
    # Bottom ring
    add_line(c0, c1, edge_color)
    add_line(c1, c2, edge_color)
    add_line(c2, c3, edge_color)
    add_line(c3, c0, edge_color)

    # Top ring
    add_line(c4, c5, edge_color)
    add_line(c5, c6, edge_color)
    add_line(c6, c7, edge_color)
    add_line(c7, c4, edge_color)

    # Vertical pillars
    add_line(c0, c4, edge_color)
    add_line(c1, c5, edge_color)
    add_line(c2, c6, edge_color)
    add_line(c3, c7, edge_color)

    # 2. Sleek Corner Accent Brackets (at all 8 corners)
    min_dim = min(float(bbox.size.x), float(bbox.size.y), float(bbox.size.z))
    tick_len = max(min_dim * corner_tick_ratio, 0.04)

    # Each corner has 3 outgoing directions (+/- X, +/- Y, +/- Z)
    corner_defs = [
        (c0, glm.vec3(1, 0, 0), glm.vec3(0, 1, 0), glm.vec3(0, 0, 1)),
        (c1, glm.vec3(-1, 0, 0), glm.vec3(0, 1, 0), glm.vec3(0, 0, 1)),
        (c2, glm.vec3(-1, 0, 0), glm.vec3(0, 1, 0), glm.vec3(0, 0, -1)),
        (c3, glm.vec3(1, 0, 0), glm.vec3(0, 1, 0), glm.vec3(0, 0, -1)),
        (c4, glm.vec3(1, 0, 0), glm.vec3(0, -1, 0), glm.vec3(0, 0, 1)),
        (c5, glm.vec3(-1, 0, 0), glm.vec3(0, -1, 0), glm.vec3(0, 0, 1)),
        (c6, glm.vec3(-1, 0, 0), glm.vec3(0, -1, 0), glm.vec3(0, 0, -1)),
        (c7, glm.vec3(1, 0, 0), glm.vec3(0, -1, 0), glm.vec3(0, 0, -1)),
    ]

    for pt, dx, dy, dz in corner_defs:
        add_line(pt, pt + dx * tick_len, corner_color)
        add_line(pt, pt + dy * tick_len, corner_color)
        add_line(pt, pt + dz * tick_len, corner_color)

    return np.array(lines, dtype=np.float32)


def format_dimension_text(width: float, height: float, depth: float, unit: str = "m") -> str:
    """Format dimension readouts cleanly with units."""
    return f"W: {width:.2f} {unit}  H: {height:.2f} {unit}  D: {depth:.2f} {unit}"
