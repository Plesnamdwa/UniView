"""
core/raycast.py
Fast 3D Screen-to-World Raycaster with Vectorized Möller-Trumbore Triangle Intersection.
Used by UniView 3D Measurement Tool to accurately pick points on model surfaces.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import glm

from core.camera import ArcballCamera
from core.scene_loader import SceneData
from core.math_utils import AABB


def get_screen_ray(
    screen_x: float,
    screen_y: float,
    viewport_w: int,
    viewport_h: int,
    camera: ArcballCamera,
) -> Tuple[glm.vec3, glm.vec3]:
    """
    Unproject 2D screen coordinate to 3D world space ray (origin, direction).
    Screen coords: (0,0) is top-left, (viewport_w, viewport_h) is bottom-right.
    """
    if viewport_w <= 0 or viewport_h <= 0:
        return glm.vec3(0.0), glm.vec3(0.0, 0.0, -1.0)

    # Convert screen to Normalized Device Coordinates (NDC) [-1, 1]
    ndc_x = (2.0 * screen_x) / float(viewport_w) - 1.0
    ndc_y = 1.0 - (2.0 * screen_y) / float(viewport_h)  # OpenGL Y is up

    inv_view_proj = glm.inverse(camera.get_projection_matrix() * camera.get_view_matrix())

    # Unproject near (z = -1.0) and far (z = 1.0) NDC coordinates
    p_near_clip = inv_view_proj * glm.vec4(ndc_x, ndc_y, -1.0, 1.0)
    p_far_clip = inv_view_proj * glm.vec4(ndc_x, ndc_y, 1.0, 1.0)

    p_near = glm.vec3(p_near_clip) / (p_near_clip.w if abs(p_near_clip.w) > 1e-6 else 1.0)
    p_far = glm.vec3(p_far_clip) / (p_far_clip.w if abs(p_far_clip.w) > 1e-6 else 1.0)

    ray_origin = p_near
    ray_dir = glm.normalize(p_far - p_near)
    return ray_origin, ray_dir


def ray_intersect_aabb(
    ray_origin: glm.vec3,
    ray_dir: glm.vec3,
    aabb: AABB,
) -> Optional[float]:
    """Ray-AABB slab intersection test. Returns entry distance t if hit, or None."""
    if not aabb.is_valid:
        return None

    t_min = -float("inf")
    t_max = float("inf")

    orig = [ray_origin.x, ray_origin.y, ray_origin.z]
    d = [ray_dir.x, ray_dir.y, ray_dir.z]
    b_min = [aabb.min_pt.x, aabb.min_pt.y, aabb.min_pt.z]
    b_max = [aabb.max_pt.x, aabb.max_pt.y, aabb.max_pt.z]

    for i in range(3):
        if abs(d[i]) < 1e-8:
            if orig[i] < b_min[i] or orig[i] > b_max[i]:
                return None
        else:
            inv_d = 1.0 / d[i]
            t1 = (b_min[i] - orig[i]) * inv_d
            t2 = (b_max[i] - orig[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None

    return t_min if t_max >= 0.0 else None


def ray_intersect_triangles_vectorized(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    vertices: np.ndarray,
    indices: Optional[np.ndarray],
) -> Optional[float]:
    """
    Vectorized Möller-Trumbore ray-triangle intersection over entire mesh using NumPy.
    Runs in < 5ms for tens of thousands of triangles.
    Returns closest positive hit distance t or None.
    """
    if len(vertices) < 3:
        return None

    # Triangle vertex arrays (N, 3)
    if indices is not None and len(indices) >= 3:
        idx = indices.reshape(-1, 3)
        v0 = vertices[idx[:, 0]]
        v1 = vertices[idx[:, 1]]
        v2 = vertices[idx[:, 2]]
    else:
        num_tris = len(vertices) // 3
        v0 = vertices[0 : num_tris * 3 : 3]
        v1 = vertices[1 : num_tris * 3 : 3]
        v2 = vertices[2 : num_tris * 3 : 3]

    if len(v0) == 0:
        return None

    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(ray_dir, e2)
    det = np.einsum("ij,ij->i", e1, pvec)

    # Filter out parallel rays
    valid_det = np.abs(det) > 1e-7
    if not np.any(valid_det):
        return None

    e1 = e1[valid_det]
    e2 = e2[valid_det]
    pvec = pvec[valid_det]
    det = det[valid_det]
    v0_v = v0[valid_det]

    inv_det = 1.0 / det
    tvec = ray_origin - v0_v
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det

    # Filter u in [0, 1]
    valid_u = (u >= 0.0) & (u <= 1.0)
    if not np.any(valid_u):
        return None

    e1 = e1[valid_u]
    e2 = e2[valid_u]
    tvec = tvec[valid_u]
    inv_det = inv_det[valid_u]
    u = u[valid_u]

    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,j->i", qvec, ray_dir) * inv_det

    # Filter v in [0, 1] and u + v <= 1
    valid_v = (v >= 0.0) & (u + v <= 1.0)
    if not np.any(valid_v):
        return None

    e2 = e2[valid_v]
    qvec = qvec[valid_v]
    inv_det = inv_det[valid_v]

    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    pos_t = t[t > 1e-4]
    if len(pos_t) == 0:
        return None

    return float(np.min(pos_t))


def raycast_scene(
    screen_x: float,
    screen_y: float,
    viewport_w: int,
    viewport_h: int,
    camera: ArcballCamera,
    scene: Optional[SceneData],
    model_matrix: Optional[glm.mat4] = None,
) -> Optional[glm.vec3]:
    """
    Raycast from 2D viewport coordinates onto 3D scene geometry.
    Returns exact 3D world hit point glm.vec3 or fallback ground plane intersection.
    """
    ray_origin, ray_dir = get_screen_ray(screen_x, screen_y, viewport_w, viewport_h, camera)

    if model_matrix is None:
        model_matrix = glm.mat4(1.0)

    closest_hit_world: Optional[glm.vec3] = None
    min_dist_to_cam = float("inf")

    if scene and scene.submeshes:
        for sub in scene.submeshes:
            if not sub.visible or sub.vertex_count == 0:
                continue

            # Full submesh transformation into world space
            sub_world_m = model_matrix * sub.node_transform
            inv_sub_m = glm.inverse(sub_world_m)

            # Transform world ray into submesh local space
            local_orig_v4 = inv_sub_m * glm.vec4(ray_origin, 1.0)
            local_orig = glm.vec3(local_orig_v4) / (local_orig_v4.w if abs(local_orig_v4.w) > 1e-6 else 1.0)
            local_dir_v4 = inv_sub_m * glm.vec4(ray_dir, 0.0)
            local_dir = glm.normalize(glm.vec3(local_dir_v4))

            # Fast AABB rejection
            t_box = ray_intersect_aabb(local_orig, local_dir, sub.aabb)
            if t_box is None:
                continue

            # sub.positions is already a clean (N, 3) float32 array
            verts_np = sub.positions
            if verts_np is None or len(verts_np) < 3:
                continue
            idx_np = sub.indices if (sub.indices is not None and len(sub.indices) > 0) else None

            # Vectorized triangle intersection in local space
            orig_np = np.array([local_orig.x, local_orig.y, local_orig.z], dtype=np.float32)
            dir_np = np.array([local_dir.x, local_dir.y, local_dir.z], dtype=np.float32)

            t_hit = ray_intersect_triangles_vectorized(orig_np, dir_np, verts_np, idx_np)
            if t_hit is not None:
                local_hit = local_orig + local_dir * t_hit
                world_hit_v4 = sub_world_m * glm.vec4(local_hit, 1.0)
                world_hit = glm.vec3(world_hit_v4) / (world_hit_v4.w if abs(world_hit_v4.w) > 1e-6 else 1.0)

                dist_to_cam = glm.length(world_hit - camera.position)
                if dist_to_cam < min_dist_to_cam:
                    min_dist_to_cam = dist_to_cam
                    closest_hit_world = world_hit

    if closest_hit_world is not None:
        return closest_hit_world

    # Fallback: Raycast against ground horizontal studio floor (Y = 0)
    if abs(ray_dir.y) > 1e-6:
        t_ground = -ray_origin.y / ray_dir.y
        if t_ground > 0.0:
            ground_hit = ray_origin + ray_dir * t_ground
            # Bound within reasonable studio radius
            radius = scene.scene_aabb.radius * 4.0 if (scene and scene.scene_aabb.is_valid) else 100.0
            if math.hypot(ground_hit.x, ground_hit.z) <= radius:
                return ground_hit

    return None
