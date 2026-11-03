"""
core/math_utils.py
Mathematical utilities for UniView:
- Axis-Aligned Bounding Box (AABB) computation and merging
- Matrix conversions between NumPy, PyAssimp (row-major), and PyGLM / OpenGL (column-major)
- TRS (Translation, Rotation, Scale) decomposition and composition
- Euler (XYZ degrees) <-> Quaternion (WXYZ) conversions
- Procedural 3D geometry generation for octahedral bones and joint spheres
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import numpy as np
import glm


@dataclass
class AABB:
    """Axis-Aligned Bounding Box in 3D space."""
    min_pt: glm.vec3
    max_pt: glm.vec3

    @classmethod
    def empty(cls) -> "AABB":
        return cls(
            min_pt=glm.vec3(float("inf"), float("inf"), float("inf")),
            max_pt=glm.vec3(float("-inf"), float("-inf"), float("-inf")),
        )

    @classmethod
    def from_vertices(cls, vertices: np.ndarray, transform: Optional[glm.mat4] = None) -> "AABB":
        """Compute AABB from an (N, 3) vertex array, optionally transformed by a 4x4 matrix."""
        if vertices is None or len(vertices) == 0:
            return cls(glm.vec3(-0.5), glm.vec3(0.5))

        verts = np.asarray(vertices, dtype=np.float32)
        if verts.ndim == 1:
            verts = verts.reshape(-1, 3)

        if transform is not None and transform != glm.mat4(1.0):
            # Transform the 8 corners of the local AABB or all vertices
            # Transforming all vertices is exact:
            mat_np = glm_to_numpy(transform)  # 4x4 column-major or standard matrix
            # In column-major glm, v' = M * v. In row-vector numpy: v' = v @ M^T
            ones = np.ones((verts.shape[0], 1), dtype=np.float32)
            hom = np.hstack([verts, ones])
            transformed = hom @ mat_np.T
            verts = transformed[:, :3]

        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        return cls(
            min_pt=glm.vec3(float(vmin[0]), float(vmin[1]), float(vmin[2])),
            max_pt=glm.vec3(float(vmax[0]), float(vmax[1]), float(vmax[2])),
        )

    @property
    def is_valid(self) -> bool:
        return (
            self.min_pt.x <= self.max_pt.x
            and self.min_pt.y <= self.max_pt.y
            and self.min_pt.z <= self.max_pt.z
            and not math.isinf(self.min_pt.x)
        )

    @property
    def center(self) -> glm.vec3:
        if not self.is_valid:
            return glm.vec3(0.0, 0.0, 0.0)
        return (self.min_pt + self.max_pt) * 0.5

    @property
    def size(self) -> glm.vec3:
        if not self.is_valid:
            return glm.vec3(1.0, 1.0, 1.0)
        return self.max_pt - self.min_pt

    @property
    def extents(self) -> glm.vec3:
        return self.size * 0.5

    @property
    def max_extent(self) -> float:
        s = self.size
        return max(float(s.x), float(s.y), float(s.z), 1e-4)

    @property
    def radius(self) -> float:
        if not self.is_valid:
            return 1.0
        r = glm.length(self.extents)
        return max(float(r), 1e-4)

    def merge(self, other: "AABB") -> "AABB":
        if not other.is_valid:
            return self
        if not self.is_valid:
            return AABB(glm.vec3(other.min_pt), glm.vec3(other.max_pt))
        return AABB(
            min_pt=glm.min(self.min_pt, other.min_pt),
            max_pt=glm.max(self.max_pt, other.max_pt),
        )


def assimp_mat_to_glm(mat) -> glm.mat4:
    """
    Convert an Assimp 4x4 row-major matrix (or 4x4 numpy array) into a PyGLM column-major mat4.
    In Assimp, translation is stored at mat[0][3], mat[1][3], mat[2][3].
    In PyGLM, mat4[col][row] has translation at mat[3][0], mat[3][1], mat[3][2].
    """
    arr = np.asarray(mat, dtype=np.float32)
    if arr.shape == (16,):
        arr = arr.reshape((4, 4))
    elif arr.shape != (4, 4):
        return glm.mat4(1.0)

    # Construct column by column: column j of GLM is column j of mathematical matrix (arr[:, j])
    m = glm.mat4(1.0)
    for col in range(4):
        for row in range(4):
            m[col][row] = float(arr[row, col])
    return m


def glm_to_numpy(m: glm.mat4) -> np.ndarray:
    """
    Convert a PyGLM mat4 to a standard 4x4 mathematical NumPy array (row-major representation,
    where translation is in [:, 3]), suitable for `hom_verts @ mat.T`.
    """
    arr = np.zeros((4, 4), dtype=np.float32)
    for col in range(4):
        for row in range(4):
            arr[row, col] = m[col][row]
    return arr


def glm_to_bytes(m: glm.mat4) -> bytes:
    """Return column-major float32 bytes for direct ModernGL uniform upload."""
    # PyGLM stores data in column-major order natively
    return m.to_bytes()


def euler_deg_to_quat(euler_deg: glm.vec3) -> glm.quat:
    """Convert Euler XYZ angles in degrees to a normalized Quaternion (W, X, Y, Z)."""
    rad = glm.radians(euler_deg)
    qx = glm.angleAxis(rad.x, glm.vec3(1.0, 0.0, 0.0))
    qy = glm.angleAxis(rad.y, glm.vec3(0.0, 1.0, 0.0))
    qz = glm.angleAxis(rad.z, glm.vec3(0.0, 0.0, 1.0))
    # Intrinsic XYZ / Extrinsic ZYX standard rotation order
    return glm.normalize(qz * qy * qx)


def quat_to_euler_deg(q: glm.quat) -> glm.vec3:
    """Convert a Quaternion to Euler XYZ angles in degrees."""
    q_norm = glm.normalize(q)
    euler_rad = glm.eulerAngles(q_norm)
    return glm.degrees(euler_rad)


def compose_trs(pos: glm.vec3, rot: glm.quat, scale: glm.vec3) -> glm.mat4:
    """Compose a 4x4 transformation matrix from Translation, Rotation (quat), and Scale."""
    t_mat = glm.translate(glm.mat4(1.0), pos)
    r_mat = glm.mat4_cast(glm.normalize(rot))
    s_mat = glm.scale(glm.mat4(1.0), scale)
    return t_mat * r_mat * s_mat


def decompose_trs(m: glm.mat4) -> Tuple[glm.vec3, glm.quat, glm.vec3]:
    """
    Safely decompose a 4x4 affine matrix into (position: vec3, rotation: quat, scale: vec3).
    """
    pos = glm.vec3(m[3][0], m[3][1], m[3][2])

    col0 = glm.vec3(m[0][0], m[0][1], m[0][2])
    col1 = glm.vec3(m[1][0], m[1][1], m[1][2])
    col2 = glm.vec3(m[2][0], m[2][1], m[2][2])

    sx = glm.length(col0)
    sy = glm.length(col1)
    sz = glm.length(col2)

    # Detect negative scale / reflection
    det = glm.determinant(glm.mat3(col0, col1, col2))
    if det < 0:
        sx = -sx

    scale = glm.vec3(
        sx if abs(sx) > 1e-6 else 1.0,
        sy if abs(sy) > 1e-6 else 1.0,
        sz if abs(sz) > 1e-6 else 1.0,
    )

    rot_mat = glm.mat3(
        col0 / scale.x,
        col1 / scale.y,
        col2 / scale.z,
    )
    rot = glm.normalize(glm.quat_cast(rot_mat))
    return pos, rot, scale


def generate_octahedral_bone_mesh(waist_ratio: float = 0.2, radius: float = 0.10) -> np.ndarray:
    """
    Generate a 3D Octahedral Bone mesh oriented along +Y from (0,0,0) [Head] to (0,1,0) [Tail].
    Returns an interleaved float32 array of shape (24, 6) -> [px, py, pz, nx, ny, nz]
    representing 8 triangles (24 vertices) with flat face normals for crisp 3D shading.
    """
    head = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tail = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 4 waist vertices in CCW order around +Y
    w0 = np.array([radius, waist_ratio, radius], dtype=np.float32)
    w1 = np.array([-radius, waist_ratio, radius], dtype=np.float32)
    w2 = np.array([-radius, waist_ratio, -radius], dtype=np.float32)
    w3 = np.array([radius, waist_ratio, -radius], dtype=np.float32)

    # 8 triangular faces (4 bottom pyramid from head, 4 top pyramid to tail)
    faces = [
        # Bottom pyramid (head to waist)
        (head, w1, w0),
        (head, w2, w1),
        (head, w3, w2),
        (head, w0, w3),
        # Top pyramid (waist to tail)
        (w0, w1, tail),
        (w1, w2, tail),
        (w2, w3, tail),
        (w3, w0, tail),
    ]

    vertex_data = []
    for v0, v1, v2 in faces:
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        norm_len = np.linalg.norm(n)
        if norm_len > 1e-6:
            n = n / norm_len
        else:
            n = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        for v in (v0, v1, v2):
            vertex_data.append([v[0], v[1], v[2], n[0], n[1], n[2]])

    return np.array(vertex_data, dtype=np.float32)


def generate_bone_align_matrix(
    head_pos: glm.vec3,
    tail_pos: glm.vec3,
    max_thickness_len: Optional[float] = None,
) -> glm.mat4:
    """
    Compute the 4x4 world matrix that maps a unit bone along +Y ((0,0,0) -> (0,1,0))
    to start at `head_pos` and end at `tail_pos`.
    If `max_thickness_len` is given, clamps lateral X/Z thickness so long bones stay sleek.
    """
    diff = tail_pos - head_pos
    length = glm.length(diff)
    if length < 1e-5:
        return glm.translate(glm.mat4(1.0), head_pos) * glm.scale(glm.mat4(1.0), glm.vec3(0.01))

    direction = diff / length
    up = glm.vec3(0.0, 1.0, 0.0)

    # Find rotation from +Y to direction
    dot_val = float(glm.clamp(glm.dot(up, direction), -1.0, 1.0))
    if abs(dot_val - 1.0) < 1e-5:
        rot_mat = glm.mat4(1.0)
    elif abs(dot_val + 1.0) < 1e-5:
        rot_mat = glm.rotate(glm.mat4(1.0), math.pi, glm.vec3(1.0, 0.0, 0.0))
    else:
        axis = glm.normalize(glm.cross(up, direction))
        angle = math.acos(dot_val)
        rot_mat = glm.rotate(glm.mat4(1.0), angle, axis)

    xz_scale = min(length, max_thickness_len) if (max_thickness_len and max_thickness_len > 0.0) else length
    scale_mat = glm.scale(glm.mat4(1.0), glm.vec3(xz_scale, length, xz_scale))
    trans_mat = glm.translate(glm.mat4(1.0), head_pos)
    return trans_mat * rot_mat * scale_mat


def generate_joint_sphere_mesh(radius: float = 1.0, rings: int = 8, sectors: int = 12) -> np.ndarray:
    """
    Generate a 3D UV-sphere mesh for rendering skeletal joint points.
    Returns interleaved float32 array of shape (N, 6) -> [px, py, pz, nx, ny, nz].
    """
    vertices = []
    for r in range(rings):
        theta1 = (r / rings) * math.pi
        theta2 = ((r + 1) / rings) * math.pi

        for s in range(sectors):
            phi1 = (s / sectors) * 2.0 * math.pi
            phi2 = ((s + 1) / sectors) * 2.0 * math.pi

            def pt(t, p):
                nx = math.sin(t) * math.cos(p)
                ny = math.cos(t)
                nz = math.sin(t) * math.sin(p)
                return [nx * radius, ny * radius, nz * radius, nx, ny, nz]

            p00 = pt(theta1, phi1)
            p10 = pt(theta2, phi1)
            p11 = pt(theta2, phi2)
            p01 = pt(theta1, phi2)

            vertices.extend([p00, p10, p11, p00, p11, p01])

    return np.array(vertices, dtype=np.float32)
