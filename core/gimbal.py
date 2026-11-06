"""
core/gimbal.py
3D Viewport Rotation Gimbal for Interactive FK Posing in UniView.
Provides:
- Procedural generation of 3 orthogonal rotation rings (X=Pitch/Red, Y=Yaw/Green, Z=Roll/Blue)
- Extraction of bone-aligned orthonormal gimbal matrices in world space
- Screen-space projection and robust hit-testing of gimbal rings
- Natural angular delta computation during mouse dragging with angle snapping and precision modifiers
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict
import numpy as np
import glm


def get_bone_gimbal_matrix(
    bone: any,
    model_matrix: Optional[glm.mat4] = None,
    gimbal_radius: float = 1.0,
) -> Tuple[glm.vec3, glm.mat4, glm.mat4]:
    """
    Extract the 3D world-space pivot, orthonormal orientation basis, and combined
    render transform for the interactive FK gimbal rings of a bone.
    
    Returns:
      (world_pivot, rot_matrix, full_transform)
      - world_pivot: glm.vec3 position of joint in world space
      - rot_matrix: 4x4 matrix whose columns are normalized local X, Y, Z axes of the bone in world space
      - full_transform: Combined T * R * S(gimbal_radius) ready for direct upload to shader u_model
    """
    model_m = model_matrix if model_matrix is not None else glm.mat4(1.0)
    M = model_m * bone.world_matrix

    # Joint pivot in world coordinates
    world_pivot = glm.vec3(M[3].x, M[3].y, M[3].z)

    # Orthonormal orientation basis from column vectors
    v0 = glm.vec3(M[0].x, M[0].y, M[0].z)
    v1 = glm.vec3(M[1].x, M[1].y, M[1].z)
    v2 = glm.vec3(M[2].x, M[2].y, M[2].z)

    l0 = float(glm.length(v0))
    l1 = float(glm.length(v1))
    l2 = float(glm.length(v2))

    col0 = (v0 / l0) if l0 > 1e-6 else glm.vec3(1.0, 0.0, 0.0)
    col1 = (v1 / l1) if l1 > 1e-6 else glm.vec3(0.0, 1.0, 0.0)
    col2 = (v2 / l2) if l2 > 1e-6 else glm.vec3(0.0, 0.0, 1.0)

    rot_matrix = glm.mat4(
        glm.vec4(col0, 0.0),
        glm.vec4(col1, 0.0),
        glm.vec4(col2, 0.0),
        glm.vec4(0.0, 0.0, 0.0, 1.0),
    )

    full_transform = (
        glm.translate(glm.mat4(1.0), world_pivot)
        * rot_matrix
        * glm.scale(glm.mat4(1.0), glm.vec3(gimbal_radius))
    )

    return world_pivot, rot_matrix, full_transform


def generate_single_ring_mesh(
    axis: str,
    radius: float = 1.0,
    tube_radius: float = 0.045,
    num_segments: int = 48,
    cross_segments: int = 8,
) -> np.ndarray:
    """
    Generate 3D tubular torus geometry for a single axis rotation ring.
    - axis == 'X': Ring in YZ plane (normal = +X, Pitch)
    - axis == 'Y': Ring in XZ plane (normal = +Y, Yaw)
    - axis == 'Z': Ring in XY plane (normal = +Z, Roll)
    Returns float32 ndarray of shape (N, 6) -> [px, py, pz, nx, ny, nz].
    """
    axis = axis.upper()
    vertices: List[List[float]] = []

    for i in range(num_segments):
        a1 = (i / num_segments) * 2.0 * math.pi
        a2 = ((i + 1) / num_segments) * 2.0 * math.pi

        for j in range(cross_segments):
            b1 = (j / cross_segments) * 2.0 * math.pi
            b2 = ((j + 1) / cross_segments) * 2.0 * math.pi

            def pt(a: float, b: float) -> List[float]:
                ca, sa = math.cos(a), math.sin(a)
                cb, sb = math.cos(b), math.sin(b)
                r = radius + tube_radius * cb
                h = tube_radius * sb

                if axis == "X":
                    # Lies in YZ plane, normal is +X
                    p = [h, r * ca, r * sa]
                    n = [sb, cb * ca, cb * sa]
                elif axis == "Y":
                    # Lies in XZ plane, normal is +Y
                    p = [r * ca, h, r * sa]
                    n = [cb * ca, sb, cb * sa]
                else:  # 'Z'
                    # Lies in XY plane, normal is +Z
                    p = [r * ca, r * sa, h]
                    n = [cb * ca, cb * sa, sb]
                return p + n

            p00 = pt(a1, b1)
            p10 = pt(a2, b1)
            p11 = pt(a2, b2)
            p01 = pt(a1, b2)

            # Two triangles per quad
            vertices.extend([p00, p10, p11, p00, p11, p01])

    return np.array(vertices, dtype=np.float32)


def project_world_to_screen(
    world_pos: glm.vec3,
    view_mat: glm.mat4,
    proj_mat: glm.mat4,
    viewport_w: float,
    viewport_h: float,
) -> Tuple[float, float, float, bool]:
    """
    Project a 3D world coordinate into 2D viewport screen coordinates (pixels).
    Returns (screen_x, screen_y, ndc_z, is_in_front).
    Qt screen coordinates have origin (0, 0) at top-left.
    """
    clip = proj_mat * view_mat * glm.vec4(world_pos, 1.0)
    w = float(clip.w)
    if w <= 1e-5:
        return 0.0, 0.0, 1.0, False

    ndc_x = float(clip.x) / w
    ndc_y = float(clip.y) / w
    ndc_z = float(clip.z) / w

    # Map NDC [-1, 1] to screen pixel coords
    screen_x = (ndc_x + 1.0) * 0.5 * viewport_w
    screen_y = (1.0 - ndc_y) * 0.5 * viewport_h  # Invert Y for Qt

    return screen_x, screen_y, ndc_z, True


def hit_test_gimbal_rings(
    mouse_x: float,
    mouse_y: float,
    viewport_w: float,
    viewport_h: float,
    camera: any,
    gimbal_center: glm.vec3,
    gimbal_radius: float,
    rot_matrix: Optional[glm.mat4] = None,
    hit_tolerance_px: float = 16.0,
    num_samples: int = 48,
) -> Optional[str]:
    """
    Robust screen-space hit testing against the 3 rotation rings (X, Y, Z).
    Transforms sampled ring circles by the bone's orientation rot_matrix and center,
    projects them into screen space, and returns the closest hit axis ('X', 'Y', 'Z')
    in front of the camera, or None.
    """
    if gimbal_radius <= 1e-4:
        return None

    view_mat = camera.get_view_matrix()
    proj_mat = camera.get_projection_matrix()
    r_mat = rot_matrix if rot_matrix is not None else glm.mat4(1.0)

    # Check if gimbal center is in front of camera
    _, _, center_ndc_z, center_visible = project_world_to_screen(
        gimbal_center, view_mat, proj_mat, viewport_w, viewport_h
    )
    if not center_visible or center_ndc_z >= 1.0 or center_ndc_z <= -1.0:
        return None

    best_axis: Optional[str] = None
    best_dist = hit_tolerance_px
    best_depth = float("inf")

    axes = ["X", "Y", "Z"]

    for axis_name in axes:
        for s in range(num_samples):
            theta = (s / num_samples) * 2.0 * math.pi
            c_val = math.cos(theta) * gimbal_radius
            s_val = math.sin(theta) * gimbal_radius

            if axis_name == "X":
                local_pt = glm.vec3(0.0, c_val, s_val)
            elif axis_name == "Y":
                local_pt = glm.vec3(c_val, 0.0, s_val)
            else:
                local_pt = glm.vec3(c_val, s_val, 0.0)

            # Transform into world space using bone orientation rot_matrix
            world_pt = gimbal_center + glm.vec3(r_mat * glm.vec4(local_pt, 0.0))
            sx, sy, sz, in_front = project_world_to_screen(
                world_pt, view_mat, proj_mat, viewport_w, viewport_h
            )

            if not in_front:
                continue

            dist = math.hypot(mouse_x - sx, mouse_y - sy)
            if dist < hit_tolerance_px:
                # Prioritize foreground depth (smaller NDC z in OpenGL is closer to camera)
                if (dist < best_dist and sz <= best_depth + 0.05) or (sz < best_depth - 0.05):
                    best_dist = dist
                    best_depth = sz
                    best_axis = axis_name

    return best_axis


def compute_gimbal_drag_angle(
    start_mouse: Tuple[float, float],
    current_mouse: Tuple[float, float],
    gimbal_center: glm.vec3,
    axis: str,
    camera: any,
    viewport_w: float,
    viewport_h: float,
    rot_matrix: Optional[glm.mat4] = None,
) -> float:
    """
    Compute rotation angle delta (in degrees) from screen-space mouse drag around the gimbal center.
    Properly handles Qt screen Y-inversion and camera viewing angle relative to the active ring normal.
    """
    view_mat = camera.get_view_matrix()
    proj_mat = camera.get_projection_matrix()
    cx, cy, _, in_front = project_world_to_screen(
        gimbal_center, view_mat, proj_mat, viewport_w, viewport_h
    )
    if not in_front:
        return 0.0

    # 2D screen vectors from projected center
    dx0 = start_mouse[0] - cx
    dy0 = start_mouse[1] - cy
    dx1 = current_mouse[0] - cx
    dy1 = current_mouse[1] - cy

    angle0 = math.atan2(dy0, dx0)
    angle1 = math.atan2(dy1, dx1)

    delta_rad = angle1 - angle0
    # Normalize to [-pi, pi]
    while delta_rad > math.pi:
        delta_rad -= 2.0 * math.pi
    while delta_rad < -math.pi:
        delta_rad += 2.0 * math.pi

    # Check orientation of the active ring normal relative to the camera vector
    r_mat = rot_matrix if rot_matrix is not None else glm.mat4(1.0)
    if axis.upper() == "X":
        axis_dir = glm.vec3(r_mat[0].x, r_mat[0].y, r_mat[0].z)
    elif axis.upper() == "Y":
        axis_dir = glm.vec3(r_mat[1].x, r_mat[1].y, r_mat[1].z)
    else:
        axis_dir = glm.vec3(r_mat[2].x, r_mat[2].y, r_mat[2].z)

    # Vector from gimbal center towards camera position
    view_dir = glm.normalize(camera.position - gimbal_center)
    dot = glm.dot(axis_dir, view_dir)

    # In Qt screen coordinates (Y down), moving clockwise increases atan2(dy, dx),
    # which is opposite to standard 3D right-hand rule when axis faces camera (dot >= 0).
    # When axis faces away from camera (dot < 0), viewing angle is inverted.
    sign = -1.0 if dot >= 0.0 else 1.0

    delta_deg = math.degrees(delta_rad) * sign
    return float(delta_deg)
