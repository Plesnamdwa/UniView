"""
core/gizmo.py
Interactive 3D Corner Orientation XYZ Gizmo Hit Testing and View-Angle Mapping.
Used to click on X/Y/Z axes to smoothly snap the camera to Orthographic / Standard views.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, Any
import glm

GIZMO_MARGIN = 15.0
GIZMO_SIZE = 80.0
GIZMO_HALF_EXTENT = 1.4  # Matches glm.ortho(-1.4, 1.4, -1.4, 1.4)

# Standard axis presets: (Label, World Dir, Target Yaw, Target Pitch, View Name)
AXIS_PRESETS = [
    ("+X", glm.vec3(1.0, 0.0, 0.0), 90.0, 0.0, "Right View (+X)"),
    ("-X", glm.vec3(-1.0, 0.0, 0.0), -90.0, 0.0, "Left View (-X)"),
    ("+Y", glm.vec3(0.0, 1.0, 0.0), 0.0, 89.0, "Top View (+Y)"),
    ("-Y", glm.vec3(0.0, -1.0, 0.0), 0.0, -89.0, "Bottom View (-Y)"),
    ("+Z", glm.vec3(0.0, 0.0, 1.0), 0.0, 0.0, "Front View (+Z)"),
    ("-Z", glm.vec3(0.0, 0.0, -1.0), 180.0, 0.0, "Back View (-Z)"),
]


def is_pos_in_gizmo_region(mouse_x: float, mouse_y: float, viewport_h: float) -> bool:
    """Return True if (mouse_x, mouse_y) in widget coords is inside the gizmo bounding box."""
    min_x = GIZMO_MARGIN
    max_x = GIZMO_MARGIN + GIZMO_SIZE
    min_y = float(viewport_h) - GIZMO_MARGIN - GIZMO_SIZE
    max_y = float(viewport_h) - GIZMO_MARGIN
    return (min_x <= mouse_x <= max_x) and (min_y <= mouse_y <= max_y)


def hit_test_gizmo_axis(
    mouse_x: float,
    mouse_y: float,
    viewport_h: float,
    camera: Any,
    hit_radius: float = 16.0,
) -> Optional[Tuple[str, float, float, str]]:
    """
    Hit-test mouse click against projected 3D gizmo axis endpoints.
    Returns (axis_label, target_yaw, target_pitch, view_name) or None.
    If multiple axes overlap in 2D, selects the one in front (largest view-space Z).
    """
    if not is_pos_in_gizmo_region(mouse_x, mouse_y, viewport_h):
        return None

    cx = GIZMO_MARGIN + GIZMO_SIZE * 0.5
    cy = float(viewport_h) - GIZMO_MARGIN - GIZMO_SIZE * 0.5
    scale_px = (GIZMO_SIZE * 0.5) / GIZMO_HALF_EXTENT

    rot_view = camera.get_orientation_matrix()

    best_match: Optional[Tuple[str, float, float, str]] = None
    best_dist = float("inf")
    best_depth = -float("inf")

    for label, world_dir, yaw, pitch, name in AXIS_PRESETS:
        # Transform direction into view space
        v_view = rot_view * glm.vec4(world_dir, 0.0)
        sx = cx + float(v_view.x) * scale_px
        sy = cy - float(v_view.y) * scale_px  # Qt Y goes downward
        depth = float(v_view.z)

        dist = math.hypot(mouse_x - sx, mouse_y - sy)
        if dist <= hit_radius:
            # Depth ordering: in OpenGL view space, larger Z (closer to 0) is foreground
            if depth > best_depth or (abs(depth - best_depth) < 0.2 and dist < best_dist):
                best_depth = depth
                best_dist = dist
                best_match = (label, yaw, pitch, name)

    return best_match
