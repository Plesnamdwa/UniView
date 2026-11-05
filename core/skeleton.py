"""
core/skeleton.py
Skeletal Hierarchy, Forward Kinematics (FK), Transform Inspector, and GPU Skinning Manager.
Supports:
- Hierarchical BoneNode tree (Mixamo, Blender FBX, Unreal Engine 5, GLTF skins)
- Local Space and World Space Position (XYZ), Rotation (Euler XYZ deg & Quat WXYZ), Scale (XYZ)
- Interactive Euler XYZ delta rotation sliders with real-time Forward Kinematics propagation
- Reset Bind Pose (restoring rest pose across all bones)
- GPU skinning matrices (`world_transform * inverse_bind_matrix`) for real-time mesh deformation
- 3D Octahedral bone segment & joint extraction for viewport overlay rendering
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import glm
from core.math_utils import (
    AABB,
    decompose_trs,
    compose_trs,
    euler_deg_to_quat,
    quat_to_euler_deg,
)

MAX_BONES = 128


@dataclass
class BoneTransformData:
    """Structured transform readout for the UI Transform Inspector (Local or World Space)."""
    position: glm.vec3
    euler_deg: glm.vec3
    quaternion: glm.quat  # (w, x, y, z) accessed via .w, .x, .y, .z
    scale: glm.vec3


@dataclass
class BoneSegment:
    """Represents a single 3D octahedral bone segment from parent joint (head) to child/tip (tail)."""
    bone_id: int
    bone_name: str
    head_pos: glm.vec3
    tail_pos: glm.vec3
    is_selected: bool = False


class BoneNode:
    """Single node in the armature/skeleton hierarchy."""

    def __init__(
        self,
        bone_id: int,
        name: str,
        bind_local_matrix: glm.mat4,
        offset_matrix: Optional[glm.mat4] = None,
        parent: Optional["BoneNode"] = None,
    ):
        self.id: int = bone_id
        self.name: str = name
        self.parent: Optional["BoneNode"] = parent
        self.children: List["BoneNode"] = []

        # Original rest/bind local transformation relative to parent
        self.bind_local_matrix: glm.mat4 = glm.mat4(bind_local_matrix)
        self.bind_pos, self.bind_rot, self.bind_scale = decompose_trs(self.bind_local_matrix)

        # Inverse bind matrix (mesh-space to bone-local space in bind pose)
        self.offset_matrix: glm.mat4 = (
            glm.mat4(offset_matrix) if offset_matrix is not None else glm.mat4(1.0)
        )
        self.has_skin_weights: bool = offset_matrix is not None

        # Current interactive local TRS state
        self.local_pos: glm.vec3 = glm.vec3(self.bind_pos)
        self.local_rot: glm.quat = glm.quat(self.bind_rot)
        self.local_scale: glm.vec3 = glm.vec3(self.bind_scale)

        # Interactive slider Euler offset in degrees (X, Y, Z) relative to bind pose
        self.slider_euler_deg: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)

        # Computed matrices
        self.local_matrix: glm.mat4 = glm.mat4(self.bind_local_matrix)
        self.world_matrix: glm.mat4 = glm.mat4(1.0)
        self.bind_world_matrix: glm.mat4 = glm.mat4(1.0)

        # Optional explicit leaf tail position in bone-local space (from Assimp/Blender `<bone>_end` nodes)
        self.end_local_pos: Optional[glm.vec3] = None

    def set_slider_rotation(self, rx_deg: float, ry_deg: float, rz_deg: float) -> None:
        """Apply interactive rotation offset (in degrees) on top of the bone's bind rotation."""
        self.slider_euler_deg = glm.vec3(float(rx_deg), float(ry_deg), float(rz_deg))
        delta_q = euler_deg_to_quat(self.slider_euler_deg)
        self.local_rot = glm.normalize(self.bind_rot * delta_q)
        self.local_matrix = compose_trs(self.local_pos, self.local_rot, self.local_scale)

    def reset_to_bind_pose(self) -> None:
        """Restore this bone's local transform to its original bind pose."""
        self.local_pos = glm.vec3(self.bind_pos)
        self.local_rot = glm.quat(self.bind_rot)
        self.local_scale = glm.vec3(self.bind_scale)
        self.slider_euler_deg = glm.vec3(0.0, 0.0, 0.0)
        self.local_matrix = glm.mat4(self.bind_local_matrix)

    def get_transform_data(self, world_space: bool = False) -> BoneTransformData:
        """Return Position, Euler XYZ (deg), Quaternion (WXYZ), and Scale in Local or World space."""
        if not world_space:
            euler = quat_to_euler_deg(self.local_rot)
            return BoneTransformData(
                position=glm.vec3(self.local_pos),
                euler_deg=euler,
                quaternion=glm.quat(self.local_rot),
                scale=glm.vec3(self.local_scale),
            )
        else:
            w_pos, w_rot, w_scale = decompose_trs(self.world_matrix)
            w_euler = quat_to_euler_deg(w_rot)
            return BoneTransformData(
                position=w_pos,
                euler_deg=w_euler,
                quaternion=w_rot,
                scale=w_scale,
            )

    @property
    def world_position(self) -> glm.vec3:
        return glm.vec3(
            self.world_matrix[3][0],
            self.world_matrix[3][1],
            self.world_matrix[3][2],
        )


class Skeleton:
    """Manages the complete bone tree, Forward Kinematics updates, and GPU skinning buffers."""

    def __init__(self):
        self.bones: List[BoneNode] = []
        self.bone_by_name: Dict[str, BoneNode] = {}
        self.bone_by_id: Dict[int, BoneNode] = {}
        self.root_bones: List[BoneNode] = []
        self.selected_bone_id: int = -1

        # Preallocated contiguous float32 buffer of 128 mat4s for GPU uniform upload
        self.skinning_matrices: np.ndarray = np.tile(
            np.eye(4, dtype=np.float32), (MAX_BONES, 1, 1)
        )
        self.is_dirty: bool = True

    @property
    def has_bones(self) -> bool:
        return len(self.bones) > 0

    def add_bone(
        self,
        name: str,
        bind_local_matrix: glm.mat4,
        offset_matrix: Optional[glm.mat4] = None,
        parent_name: Optional[str] = None,
    ) -> BoneNode:
        """Add or update a bone in the hierarchy."""
        if name in self.bone_by_name:
            node = self.bone_by_name[name]
            if offset_matrix is not None:
                node.offset_matrix = glm.mat4(offset_matrix)
                node.has_skin_weights = True
            return node

        bone_id = len(self.bones)
        parent_node = self.bone_by_name.get(parent_name) if parent_name else None
        node = BoneNode(
            bone_id=bone_id,
            name=name,
            bind_local_matrix=bind_local_matrix,
            offset_matrix=offset_matrix,
            parent=parent_node,
        )
        self.bones.append(node)
        self.bone_by_name[name] = node
        self.bone_by_id[bone_id] = node

        if parent_node is not None:
            parent_node.children.append(node)
        else:
            self.root_bones.append(node)

        return node

    def finalize_bind_pose(self) -> None:
        """
        Compute initial bind_world_matrix for all bones and synthesize inverse bind
        (offset_matrix) for any structural bones that didn't have explicit mesh weights.
        """
        def _traverse_bind(node: BoneNode, parent_world: glm.mat4):
            node.bind_world_matrix = parent_world * node.bind_local_matrix
            node.world_matrix = glm.mat4(node.bind_world_matrix)
            if not node.has_skin_weights:
                node.offset_matrix = glm.inverse(node.bind_world_matrix)
            for child in node.children:
                _traverse_bind(child, node.bind_world_matrix)

        for root in self.root_bones:
            _traverse_bind(root, glm.mat4(1.0))

        if self.bones and self.selected_bone_id < 0:
            self.selected_bone_id = self.bones[0].id

        self.update_forward_kinematics()

    def update_forward_kinematics(self) -> None:
        """Recompute world matrices down the hierarchy and update GPU skinning matrix buffer."""
        def _traverse_fk(node: BoneNode, parent_world: glm.mat4):
            node.world_matrix = parent_world * node.local_matrix
            if node.id < MAX_BONES:
                skin_mat = node.world_matrix * node.offset_matrix
                # Store column-major layout for direct OpenGL uniform upload
                for col in range(4):
                    for row in range(4):
                        self.skinning_matrices[node.id, col, row] = skin_mat[col][row]

            for child in node.children:
                _traverse_fk(child, node.world_matrix)

        for root in self.root_bones:
            _traverse_fk(root, glm.mat4(1.0))

        self.is_dirty = True

    def reset_bind_pose(self) -> None:
        """Reset all bones in the skeleton to their original bind pose."""
        for bone in self.bones:
            bone.reset_to_bind_pose()
        self.update_forward_kinematics()

    def get_bone(self, bone_id: int) -> Optional[BoneNode]:
        return self.bone_by_id.get(bone_id)

    def get_selected_bone(self) -> Optional[BoneNode]:
        return self.bone_by_id.get(self.selected_bone_id)

    def get_bone_focus_aabb(self, bone_id: int, scene_radius: float = 1.0) -> AABB:
        """
        Compute an AABB bounding box surrounding the given bone
        for smooth camera focus / framing (shortcut 'F').
        """
        bone = self.get_bone(bone_id)
        if bone is None:
            return AABB.empty()

        head = bone.world_position
        tail = None

        if bone.end_local_pos is not None and glm.length(bone.end_local_pos) > 1e-4:
            end_w = bone.world_matrix * glm.vec4(bone.end_local_pos, 1.0)
            tail = glm.vec3(end_w.x, end_w.y, end_w.z)
        elif bone.children:
            tail = bone.children[0].world_position

        if tail is not None and glm.length(tail - head) > 1e-4:
            min_p = glm.min(head, tail)
            max_p = glm.max(head, tail)
            seg_len = float(glm.length(tail - head))
            margin = max(seg_len * 0.45, float(scene_radius) * 0.04, 0.05)
        else:
            min_p = head
            max_p = head
            margin = max(float(scene_radius) * 0.06, 0.1)

        pad = glm.vec3(margin)
        return AABB(min_p - pad, max_p + pad)

    def extract_visual_primitives(
        self,
        model_scale_hint: float = 1.0,
        camera_distance: Optional[float] = None,
    ) -> Tuple[List[BoneSegment], List[Tuple[int, glm.vec3, float, bool]]]:
        """
        Generate 3D octahedral bone segments (1 anatomical segment per bone) and
        proportionally scaled joint spheres for the viewport overlay.
        - Uses explicit `<bone>_end` local coordinates when present (Blender/FBX leaf bones).
        - For multi-child parent bones (e.g., Rigify `DEF-spine` with IK hands/feet/thighs),
          connects only to axial continuation children (`local +Y`) instead of radiating
          giant wedges to disconnected offset branches.
        Returns:
          (segments: List[BoneSegment], joints: List[(bone_id, world_pos, joint_radius, is_selected)])
        """
        import math

        segments: List[BoneSegment] = []
        joints: List[Tuple[int, glm.vec3, float, bool]] = []

        scale_ref = max(float(model_scale_hint), 0.1)
        default_leaf_len = max(scale_ref * 0.05, 0.03)

        # Mild zoom adaptation so joints stay crisp when zooming in close to fingers/face
        zoom_factor = 1.0
        if camera_distance is not None and camera_distance > 0.0:
            zoom_factor = float(glm.clamp(camera_distance / (scale_ref * 2.4), 0.55, 1.20))

        for bone in self.bones:
            head = bone.world_position
            is_sel = bone.id == self.selected_bone_id

            # Determine bone's world +Y direction and scale
            y_col = glm.vec3(
                bone.world_matrix[1][0],
                bone.world_matrix[1][1],
                bone.world_matrix[1][2],
            )
            y_scale = glm.length(y_col)
            y_dir = (y_col / y_scale) if y_scale > 1e-5 else glm.vec3(0.0, 1.0, 0.0)

            tail: Optional[glm.vec3] = None

            # 1. Explicit `<bone>_end` node from FBX/Assimp hierarchy
            if bone.end_local_pos is not None and glm.length(bone.end_local_pos) > 1e-4:
                end_w = bone.world_matrix * glm.vec4(bone.end_local_pos, 1.0)
                tail = glm.vec3(end_w.x, end_w.y, end_w.z)

            # 2. Check children for connected axial continuation
            elif len(bone.children) == 1:
                child = bone.children[0]
                cand = child.world_position
                if glm.length(cand - head) > 1e-4:
                    tail = cand

            elif len(bone.children) > 1:
                # Look for children aligned with this bone's local +Y axis (e.g. spine -> spine.001, palm.01 -> Index1)
                axial_children: List[BoneNode] = []
                forward_y_vals: List[float] = []
                for c in bone.children:
                    by = float(c.bind_pos.y)
                    bxz = math.hypot(float(c.bind_pos.x), float(c.bind_pos.z))
                    if by > 1e-4:
                        if bxz <= 0.35 * by:
                            axial_children.append(c)
                        elif bxz <= 1.25 * by:
                            forward_y_vals.append(by)

                if axial_children:
                    # Pick the closest axial continuation child
                    best_c = min(axial_children, key=lambda c: float(c.bind_pos.y))
                    tail = best_c.world_position
                elif forward_y_vals:
                    # E.g. DEF-hand.L with 4 palm children all forward along +Y: draw 1 clean hand bone along +Y
                    avg_local_y = sum(forward_y_vals) / len(forward_y_vals)
                    tail = head + y_dir * (avg_local_y * y_scale * 0.85)

            # 3. Check if this is a leaf bone (terminal joint without child continuation)
            is_leaf = False
            if tail is None or glm.length(tail - head) <= 1e-4:
                is_leaf = True
                tail = None

            # Only append bone segment if there is a real child/continuation joint
            if tail is not None:
                seg_len = float(glm.length(tail - head))
                if seg_len > 1e-4:
                    segments.append(
                        BoneSegment(
                            bone_id=bone.id,
                            bone_name=bone.name,
                            head_pos=head,
                            tail_pos=tail,
                            is_selected=is_sel,
                        )
                    )
            else:
                seg_len = default_leaf_len

            # Joint marker radius proportional to bone scale & camera zoom
            min_j_rad = scale_ref * 0.0028
            max_j_rad = scale_ref * 0.0120
            if is_leaf and bone.parent is not None:
                p_dist = float(glm.length(head - bone.parent.world_position))
                base_j_rad = float(glm.clamp(p_dist * 0.12, min_j_rad * 1.2, max_j_rad * 1.5)) * zoom_factor
            else:
                base_j_rad = float(glm.clamp(seg_len * 0.075, min_j_rad, max_j_rad)) * zoom_factor

            if is_sel:
                base_j_rad *= 1.25

            joints.append((bone.id, head, base_j_rad, is_sel, is_leaf))

        return segments, joints
