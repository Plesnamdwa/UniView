"""
core/gimbal.py
3D Viewport Rotation Gimbal for Interactive FK Posing in UniView.
Provides:
- Procedural generation of 3 orthogonal rotation rings (X=Pitch/Red, Y=Yaw/Green, Z=Roll/Blue)
- Extraction of bone-aligned orthonormal gimbal matrices in world space
- Screen-space constant pixel sizing (zoom-independent) and distance threshold culling
- Unreal Engine / Blender style tangential linear mouse drag projection
- Screen-space projection and robust hit-testing of gimbal rings
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict
import numpy as np
import glm


def compute_screen_constant_radius(
    camera_pos: glm.vec3,
    gimbal_center: glm.vec3,
    fov_deg: float,
    viewport_h: float,
    target_pixel_radius: float = 70.0,
) -> float:
    """
    Compute 3D world-space radius that produces a constant pixel radius on screen.
    Guarantees the gimbal ring stays comfortably sized regardless of camera zoom distance.
    """
    cam_dist = float(glm.length(camera_pos - gimbal_center))
    fov_rad = math.radians(max(fov_deg, 1.0))
    vp_h = max(float(viewport_h), 10.0)
    world_radius = cam_dist * math.tan(fov_rad * 0.5) * (2.0 * target_pixel_radius / vp_h)
    return max(world_radius, 0.001)


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
    hit_tolerance_px: float = 18.0,
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


def compute_tangent_drag_delta(
    start_mouse: Tuple[float, float],
    current_mouse: Tuple[float, float],
    gimbal_center: glm.vec3,
    axis: str,
    rot_matrix: glm.mat4,
    camera: any,
    viewport_w: float,
    viewport_h: float,
    pixel_radius: float = 70.0,
) -> float:
    """
    Unreal Engine / Blender style rotation drag.
    Projects mouse drag vector onto the 2D screen tangent of the rotation arc.
    Allows user to drag linearly in any natural direction (left/right, up/down);
    extracts the rotational component along the active ring's axis without requiring circular motions.
    """
    view_mat = camera.get_view_matrix()
    proj_mat = camera.get_projection_matrix()

    # 1. World rotation axis vector
    axis = axis.upper()
    if axis == "X":
        axis_vec = glm.vec3(rot_matrix[0].x, rot_matrix[0].y, rot_matrix[0].z)
    elif axis == "Y":
        axis_vec = glm.vec3(rot_matrix[1].x, rot_matrix[1].y, rot_matrix[1].z)
    else:
        axis_vec = glm.vec3(rot_matrix[2].x, rot_matrix[2].y, rot_matrix[2].z)

    # 2. Find closest point on the 3D ring to start_mouse to establish the tangent
    best_dist = float("inf")
    best_pt_3d = gimbal_center
    gimbal_rad = compute_screen_constant_radius(
        camera.position, gimbal_center, camera.fov_deg, viewport_h, pixel_radius
    )

    for i in range(24):
        theta = (i / 24.0) * 2.0 * math.pi
        c_val = math.cos(theta) * gimbal_rad
        s_val = math.sin(theta) * gimbal_rad

        if axis == "X":
            loc_p = glm.vec3(0.0, c_val, s_val)
        elif axis == "Y":
            loc_p = glm.vec3(c_val, 0.0, s_val)
        else:
            loc_p = glm.vec3(c_val, s_val, 0.0)

        world_p = gimbal_center + glm.vec3(rot_matrix * glm.vec4(loc_p, 0.0))
        sx, sy, _, in_front = project_world_to_screen(world_p, view_mat, proj_mat, viewport_w, viewport_h)
        if in_front:
            d = math.hypot(start_mouse[0] - sx, start_mouse[1] - sy)
            if d < best_dist:
                best_dist = d
                best_pt_3d = world_p

    # 3. 3D Tangent vector at the hit point: T = normalize(Axis x Radial)
    radial_vec = best_pt_3d - gimbal_center
    rad_len = float(glm.length(radial_vec))
    if rad_len > 1e-6:
        r_unit = radial_vec / rad_len
    else:
        r_unit = glm.vec3(1.0, 0.0, 0.0)

    tangent_3d = glm.normalize(glm.cross(axis_vec, r_unit))

    # 4. Project tangent into 2D screen space
    p0_x, p0_y, _, _ = project_world_to_screen(best_pt_3d, view_mat, proj_mat, viewport_w, viewport_h)
    p1_x, p1_y, _, _ = project_world_to_screen(
        best_pt_3d + tangent_3d * (gimbal_rad * 0.25), view_mat, proj_mat, viewport_w, viewport_h
    )

    tx = p1_x - p0_x
    ty = p1_y - p0_y
    t_len = math.hypot(tx, ty)
    if t_len > 1e-4:
        tx /= t_len
        ty /= t_len
    else:
        tx, ty = 1.0, 0.0

    # 5. Mouse delta vector from start of drag
    dmx = current_mouse[0] - start_mouse[0]
    dmy = current_mouse[1] - start_mouse[1]

    # Project mouse delta onto 2D screen tangent
    tangent_pixels = dmx * tx + dmy * ty

    # Sensitivity: ~0.65 degrees per pixel (matches Unreal Engine / Blender feeling)
    delta_deg = tangent_pixels * 0.65
    return float(delta_deg)


# Backward compatibility alias
compute_gimbal_drag_angle = compute_tangent_drag_delta
