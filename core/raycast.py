"""
core/raycast.py
Fast 3D Screen-to-World Raycaster with Vectorized Möller-Trumbore Triangle Intersection.
Used by UniView 3D Measurement Tool to accurately pick points on model surfaces.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
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


@dataclass
class RaycastHit:
    """Represents a 3D raycast hit point on model surface or ground grid."""
    point: glm.vec3
    is_vertex_snapped: bool = False

    def __iter__(self):
        yield self.point
        yield self.is_vertex_snapped


def ray_intersect_triangles_vectorized(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    vertices: np.ndarray,
    indices: Optional[np.ndarray],
    snap_to_vertex: bool = False,
) -> Optional[Tuple[float, Optional[np.ndarray]]]:
    """
    Vectorized Möller-Trumbore ray-triangle intersection over entire mesh using NumPy.
    Runs in < 5ms for tens of thousands of triangles.
    Returns (closest_t, closest_vertex_coords or None).
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
    v0_u = v0_v[valid_u]

    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,j->i", qvec, ray_dir) * inv_det

    # Filter v in [0, 1] and u + v <= 1
    valid_v = (v >= 0.0) & (u + v <= 1.0)
    if not np.any(valid_v):
        return None

    e1 = e1[valid_v]
    e2 = e2[valid_v]
    qvec = qvec[valid_v]
    inv_det = inv_det[valid_v]
    v0_final = v0_u[valid_v]

    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    pos_mask = t > 1e-4
    if not np.any(pos_mask):
        return None

    pos_t = t[pos_mask]
    min_i = int(np.argmin(pos_t))
    best_t = float(pos_t[min_i])

    if not snap_to_vertex:
        return best_t, None

    # Retrieve 3 vertices of hit triangle
    hit_v0 = v0_final[pos_mask][min_i]
    hit_v1 = hit_v0 + e1[pos_mask][min_i]
    hit_v2 = hit_v0 + e2[pos_mask][min_i]

    hit_pt = ray_origin + ray_dir * best_t
    d0 = float(np.sum((hit_pt - hit_v0) ** 2))
    d1 = float(np.sum((hit_pt - hit_v1) ** 2))
    d2 = float(np.sum((hit_pt - hit_v2) ** 2))

    cand_verts = [hit_v0, hit_v1, hit_v2]
    best_v = cand_verts[int(np.argmin([d0, d1, d2]))]
    return best_t, best_v


def get_deformed_submesh_positions(sub: any, skeleton: Optional[any]) -> np.ndarray:
    """Return vertex positions, deformed by skeletal skinning if skeleton is present."""
    if (
        sub.positions is None
        or not getattr(sub, "has_bones", False)
        or skeleton is None
        or not getattr(skeleton, "has_bones", False)
        or getattr(sub, "bone_ids", None) is None
        or getattr(sub, "bone_weights", None) is None
    ):
        return sub.positions

    pose_version = getattr(skeleton, "pose_version", 0)
    cached_version = getattr(sub, "_cached_pose_version", -1)
    cached_pos = getattr(sub, "_cached_skinned_pos", None)
    if cached_version == pose_version and cached_pos is not None:
        return cached_pos

    # Linear blend skinning on CPU (matches GPU vertex shader)
    N = len(sub.positions)
    pos_h = np.ones((N, 4), dtype=np.float32)
    pos_h[:, :3] = sub.positions
    mats = skeleton.skinning_matrices

    skinned_pos = np.zeros((N, 3), dtype=np.float32)
    for slot in range(4):
        w = sub.bone_weights[:, slot]
        mask = w > 1e-5
        if not np.any(mask):
            continue
        b_ids = sub.bone_ids[mask, slot]
        slot_mats = mats[b_ids]
        transformed = np.einsum("ij,ijk->ik", pos_h[mask], slot_mats)
        skinned_pos[mask] += transformed[:, :3] * w[mask, np.newaxis]

    sub._cached_pose_version = pose_version
    sub._cached_skinned_pos = skinned_pos
    return skinned_pos


def raycast_scene(
    screen_x: float,
    screen_y: float,
    viewport_w: int,
    viewport_h: int,
    camera: ArcballCamera,
    scene: Optional[SceneData],
    model_matrix: Optional[glm.mat4] = None,
    snap_to_vertex: bool = False,
    allow_ground_fallback: bool = False,
) -> Optional[RaycastHit]:
    """
    Raycast from 2D viewport coordinates onto 3D scene geometry.
    If `snap_to_vertex=True`, snaps the hit point to the closest triangle vertex.
    Returns RaycastHit(point, is_vertex_snapped) or None (no ground fallback unless explicit).
    """
    ray_origin, ray_dir = get_screen_ray(screen_x, screen_y, viewport_w, viewport_h, camera)

    if model_matrix is None:
        model_matrix = glm.mat4(1.0)

    closest_hit_world: Optional[glm.vec3] = None
    closest_hit_snapped: bool = False
    min_dist_to_cam = float("inf")

    if scene and scene.submeshes:
        skeleton = scene.skeleton if hasattr(scene, "skeleton") else None
        is_posed = skeleton and skeleton.has_bones and any(
            (abs(b.slider_euler_deg.x) > 1e-4 or abs(b.slider_euler_deg.y) > 1e-4 or abs(b.slider_euler_deg.z) > 1e-4)
            for b in skeleton.bones
        )

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

            # Fast AABB rejection (only when in rest pose; when posed, test directly)
            if not is_posed:
                t_box = ray_intersect_aabb(local_orig, local_dir, sub.aabb)
                if t_box is None:
                    continue

            # Deformed or bind vertex positions
            verts_np = get_deformed_submesh_positions(sub, skeleton)
            if verts_np is None or len(verts_np) < 3:
                continue
            idx_np = sub.indices if (sub.indices is not None and len(sub.indices) > 0) else None

            # Vectorized triangle intersection in local space
            orig_np = np.array([local_orig.x, local_orig.y, local_orig.z], dtype=np.float32)
            dir_np = np.array([local_dir.x, local_dir.y, local_dir.z], dtype=np.float32)

            res = ray_intersect_triangles_vectorized(
                orig_np,
                dir_np,
                verts_np,
                idx_np,
                snap_to_vertex=snap_to_vertex,
            )
            if res is not None:
                t_hit, snap_v = res
                if snap_to_vertex and snap_v is not None:
                    local_hit = glm.vec3(float(snap_v[0]), float(snap_v[1]), float(snap_v[2]))
                    is_snapped = True
                else:
                    local_hit = local_orig + local_dir * t_hit
                    is_snapped = False

                world_hit_v4 = sub_world_m * glm.vec4(local_hit, 1.0)
                world_hit = glm.vec3(world_hit_v4) / (world_hit_v4.w if abs(world_hit_v4.w) > 1e-6 else 1.0)

                dist_to_cam = glm.length(world_hit - camera.position)
                if dist_to_cam < min_dist_to_cam:
                    min_dist_to_cam = dist_to_cam
                    closest_hit_world = world_hit
                    closest_hit_snapped = is_snapped

    if closest_hit_world is not None:
        return RaycastHit(point=closest_hit_world, is_vertex_snapped=closest_hit_snapped)

    # Optional Fallback against ground horizontal studio floor (Y = 0)
    if allow_ground_fallback and abs(ray_dir.y) > 1e-6:
        t_ground = -ray_origin.y / ray_dir.y
        if t_ground > 0.0:
            ground_hit = ray_origin + ray_dir * t_ground
            radius = scene.scene_aabb.radius * 4.0 if (scene and scene.scene_aabb.is_valid) else 100.0
            if math.hypot(ground_hit.x, ground_hit.z) <= radius:
                return RaycastHit(point=ground_hit, is_vertex_snapped=False)

    return None


# -------------------------------------------------------------------------
# Ray-vs-Skeleton Picking (Joint Spheres & Bone Segment Capsules)
# -------------------------------------------------------------------------

_EPS = 1e-9


def ray_intersect_sphere(
    orig: glm.vec3,
    dir: glm.vec3,
    center: glm.vec3,
    radius: float,
) -> Optional[float]:
    """Smallest positive hit distance t where ray R(t) = orig + t * dir reaches sphere, else None."""
    if radius <= 0.0:
        return None
    oc = center - orig
    a = float(glm.dot(dir, dir))
    if a <= 0.0:
        return None
    b = float(glm.dot(oc, dir))
    c = float(glm.dot(oc, oc)) - radius * radius
    disc = b * b - a * c
    if disc < -1e-6:
        return None
    if disc < 0.0:
        disc = 0.0
    sq = math.sqrt(disc)
    t_near = (b - sq) / a
    t_far = (b + sq) / a
    if t_far <= _EPS:
        return None
    if t_near > _EPS:
        return t_near
    return _EPS


def ray_intersect_capsule(
    orig: glm.vec3,
    dir: glm.vec3,
    p1: glm.vec3,
    p2: glm.vec3,
    radius: float,
) -> Optional[float]:
    """Smallest positive hit distance t where ray intersects capsule around segment p1->p2."""
    if radius <= 0.0:
        return None
    a_dir = float(glm.dot(dir, dir))
    if a_dir <= 0.0:
        return None

    axis = p2 - p1
    l2 = float(glm.dot(axis, axis))
    if l2 <= _EPS * _EPS:
        return ray_intersect_sphere(orig, dir, p1, radius)

    l = math.sqrt(l2)
    u = axis * (1.0 / l)

    def sphere_interval(c: glm.vec3) -> Optional[Tuple[float, float]]:
        oc = c - orig
        b = float(glm.dot(oc, dir))
        c_val = float(glm.dot(oc, oc)) - radius * radius
        disc = b * b - a_dir * c_val
        if disc < -1e-6:
            return None
        if disc < 0.0:
            disc = 0.0
        sq = math.sqrt(disc)
        tn, tf = (b - sq) / a_dir, (b + sq) / a_dir
        if tf <= _EPS:
            return None
        return (tn, tf)

    s1 = sphere_interval(p1)
    s2 = sphere_interval(p2)

    o1 = orig - p1
    du1 = float(glm.dot(o1, u))
    du = float(glm.dot(dir, u))
    w = float(glm.dot(o1, o1)) - du1 * du1 - radius * radius
    x = float(glm.dot(dir, o1)) - du1 * du
    y = a_dir - du * du

    b_iv: Optional[Tuple[float, float]] = None
    if y > _EPS:
        disc = x * x - y * w
        if disc >= -1e-6:
            if disc < 0.0:
                disc = 0.0
            sq = math.sqrt(disc)
            b_iv = ((-x - sq) / y, (-x + sq) / y)
    else:
        if abs(x) <= _EPS:
            b_iv = None if w > 0.0 else (-float("inf"), float("inf"))
        else:
            tc = -w / (2.0 * x)
            b_iv = (-float("inf"), tc) if x > 0.0 else (tc, float("inf"))

    if b_iv is not None:
        lo, hi = b_iv
        if du > _EPS:
            lo = max(lo, -du1 / du)
            hi = min(hi, (l - du1) / du)
        elif du < -_EPS:
            lo = max(lo, (l - du1) / du)
            hi = min(hi, -du1 / du)
        else:
            if not (0.0 <= du1 <= l):
                b_iv = None
        if b_iv is not None and lo <= hi:
            b_iv = (lo, hi)
        else:
            b_iv = None

    pieces = [iv for iv in (s1, s2, b_iv) if iv is not None]
    if not pieces:
        return None
    a = min(iv[0] for iv in pieces)
    b = max(iv[1] for iv in pieces)
    if b <= 0.0:
        return None
    if a > _EPS:
        return a
    return _EPS


def raycast_skeleton(
    orig: glm.vec3,
    dir: glm.vec3,
    joints: List[Any],
    segments: List[Any],
    padding_scale: float = 1.4,
    default_radius: float = 0.05,
) -> Optional[Tuple[int, float]]:
    """
    Test 3D ray against skeleton joint spheres and bone segment capsules.
    Returns (bone_id, hit_distance_t) for closest hit, or None.
    """
    padded: Dict[int, float] = {}
    best_t: Optional[float] = None
    best_bone: int = -1

    # 1. Test Joint Spheres
    for item in joints:
        bone_id = item[0]
        pos = item[1]
        radius = float(item[2])
        r = max(radius * padding_scale, default_radius * 0.4)
        padded[bone_id] = r
        t = ray_intersect_sphere(orig, dir, pos, r)
        if t is not None and (best_t is None or t < best_t):
            best_t, best_bone = t, bone_id

    # 2. Test Bone Segments (Capsules)
    for seg in segments:
        bone_id = seg.bone_id
        r = padded.get(bone_id, default_radius)
        seg_len = float(glm.length(seg.tail_pos - seg.head_pos))
        cap_r = max(r, seg_len * 0.12, default_radius * 0.6)
        t = ray_intersect_capsule(orig, dir, seg.head_pos, seg.tail_pos, cap_r)
        if t is not None and (best_t is None or t < best_t):
            best_t, best_bone = t, bone_id

    if best_t is None:
        return None
    return best_bone, best_t


def pick_bone_at_screen(
    screen_x: float,
    screen_y: float,
    viewport_w: int,
    viewport_h: int,
    camera: ArcballCamera,
    skeleton: Any,
    model_matrix: Optional[glm.mat4] = None,
    model_scale_hint: float = 1.0,
    padding_scale: float = 1.5,
) -> Optional[int]:
    """
    Project 2D screen coordinate to 3D ray and hit-test against skeleton.
    Returns clicked bone_id (int) or None.
    """
    if skeleton is None or not getattr(skeleton, "has_bones", False):
        return None

    ray_orig, ray_dir = get_screen_ray(screen_x, screen_y, viewport_w, viewport_h, camera)

    # Transform ray into model local coordinate space (reversing Up Vector rotation)
    if model_matrix is not None:
        inv_m = glm.inverse(model_matrix)
        local_orig_v4 = inv_m * glm.vec4(ray_orig, 1.0)
        local_orig = glm.vec3(local_orig_v4) / (local_orig_v4.w if abs(local_orig_v4.w) > 1e-6 else 1.0)
        local_dir = glm.normalize(glm.vec3(inv_m * glm.vec4(ray_dir, 0.0)))
    else:
        local_orig = ray_orig
        local_dir = ray_dir

    segments, joints = skeleton.extract_visual_primitives(
        model_scale_hint=model_scale_hint,
        camera_distance=camera.distance,
    )

    default_r = max(float(model_scale_hint) * 0.025, 0.02)
    hit = raycast_skeleton(
        local_orig,
        local_dir,
        joints,
        segments,
        padding_scale=padding_scale,
        default_radius=default_r,
    )

    if hit is not None:
        return hit[0]
    return None
