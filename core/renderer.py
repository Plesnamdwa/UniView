"""
core/renderer.py
Hardware-accelerated OpenGL 3.3+ rendering engine using ModernGL.
Features:
- Static VBO and VAO caching: geometry is uploaded ONCE upon scene load (zero per-frame allocation)
- Dynamic Material & Texture management with live relinking
- Rendering Modes:
  - 0: Textured Mode (diffuse texture with Phong highlights, fallback to base color)
  - 1: Base Color / No Material (clean CAD/Clay style Phong shading with material override colors)
  - 2: Vertex Weight Heatmap (smooth Blue -> Cyan -> Green -> Yellow -> Red based on active bone weight)
  - 3: Unlit Flat
- 3D Octahedral Skeleton & Joint Overlay with optional X-Ray depth test bypass
- Infinite / Anti-Aliased Reference Ground Grid with dynamic distance fade
- 3D Corner Orientation XYZ Gizmo
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import glm
import moderngl

from core.math_utils import (
    glm_to_bytes,
    generate_octahedral_bone_mesh,
    generate_bone_align_matrix,
    generate_joint_sphere_mesh,
    generate_gimbal_rings_mesh,
)
from core.scene_loader import SceneData, SubMeshData
from core.camera import ArcballCamera
from core.bounding_box import compute_scene_bounds, generate_bbox_cage_lines, BoundingBoxInfo
from core.gimbal import generate_single_ring_mesh, get_bone_gimbal_matrix




@dataclass
class RenderOptions:
    """Configurable rendering parameters for the viewport."""
    mode: int = 0                   # 0: Textured, 1: Base Color, 2: Weight Heatmap, 3: Unlit
    show_grid: bool = True
    show_skeleton: bool = True
    xray_skeleton: bool = True
    show_gizmo: bool = True
    active_bone_id: int = 0
    bg_color: Tuple[float, float, float] = (0.085, 0.082, 0.115)
    grid_color: Tuple[float, float, float] = (0.34, 0.33, 0.43)
    show_wireframe: bool = False
    wireframe_color: Tuple[float, float, float, float] = (0.42, 0.68, 0.95, 0.70)
    show_bbox: bool = False
    active_gimbal_axis: Optional[str] = None
    show_fk_gimbal: bool = True





@dataclass
class GPUSubMesh:
    """Holds ModernGL GPU buffer handles for a single sub-mesh."""
    submesh: SubMeshData
    vao: moderngl.VertexArray
    vbos: List[moderngl.Buffer]
    ibo: Optional[moderngl.Buffer]

    def release(self) -> None:
        if self.vao:
            self.vao.release()
        for b in self.vbos:
            b.release()
        if self.ibo:
            self.ibo.release()


class GLRenderer:
    """Complete ModernGL rendering pipeline for UniView."""

    def __init__(self, ctx: moderngl.Context, shaders_dir: Optional[str] = None):
        self.ctx: moderngl.Context = ctx
        if shaders_dir is None:
            import sys
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            shaders_dir = os.path.join(base_dir, "shaders")
            if not os.path.isdir(shaders_dir):
                shaders_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders")
        self.shaders_dir: str = shaders_dir

        # Shader Programs
        self.mesh_prog: moderngl.Program = self._load_program("mesh.vert", "mesh.frag")
        self.grid_prog: moderngl.Program = self._load_program("grid.vert", "grid.frag")
        self.bone_prog: moderngl.Program = self._load_program("bone.vert", "bone.frag")
        self.gizmo_prog: moderngl.Program = self._load_program("gizmo.vert", "gizmo.frag")

        # GPU Meshes for current scene
        self.gpu_submeshes: List[GPUSubMesh] = []
        self._current_scene: Optional[SceneData] = None

        # Static primitive geometry
        self.grid_vao: moderngl.VertexArray = self._create_grid_geometry()
        self.bone_vao, self.bone_vbo = self._create_octahedral_bone_geometry()
        self.joint_vao, self.joint_vbo = self._create_joint_sphere_geometry()
        self.ring_vao, self.ring_vbo = self._create_gimbal_ring_geometry()
        self.gizmo_vao, self.gizmo_vbo = self._create_gizmo_geometry()

        # Dedicated 3-Axis Interactive FK Rotation Rings
        self.gimbal_x_vao, self.gimbal_x_vbo = self._create_single_ring_geometry("X")
        self.gimbal_y_vao, self.gimbal_y_vbo = self._create_single_ring_geometry("Y")
        self.gimbal_z_vao, self.gimbal_z_vbo = self._create_single_ring_geometry("Z")


        # Dynamic 3D measurement line VAO & VBO
        self.measure_line_vbo: moderngl.Buffer = self.ctx.buffer(reserve=2 * 6 * 4)
        self.measure_line_vao: moderngl.VertexArray = self.ctx.vertex_array(
            self.gizmo_prog,
            [(self.measure_line_vbo, "3f 3f", "in_position", "in_color")],
        )

        # Dynamic 3D bounding box cage line VAO & VBO (12 edges + 24 corner bracket lines)
        self.bbox_vbo: moderngl.Buffer = self.ctx.buffer(reserve=128 * 6 * 4)
        self.bbox_vao: moderngl.VertexArray = self.ctx.vertex_array(
            self.gizmo_prog,
            [(self.bbox_vbo, "3f 3f", "in_position", "in_color")],
        )


        # Dummy 1x1 white texture for untextured meshes
        self.dummy_texture: moderngl.Texture = self.ctx.texture((1, 1), 4, b"\xff\xff\xff\xff")
        self.dummy_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _read_shader_file(self, filename: str) -> str:
        path = os.path.join(self.shaders_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Shader file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_program(self, vert_name: str, frag_name: str) -> moderngl.Program:
        vert_src = self._read_shader_file(vert_name)
        frag_src = self._read_shader_file(frag_name)
        return self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

    def _create_grid_geometry(self) -> moderngl.VertexArray:
        """Create a unit horizontal ground quad on the XZ plane (+Y normal, scaled dynamically via u_model)."""
        size = 1.0
        verts = np.array([
            -size, 0.0, -size,
            -size, 0.0,  size,
             size, 0.0,  size,
            -size, 0.0, -size,
             size, 0.0,  size,
             size, 0.0, -size,
        ], dtype=np.float32)

        vbo = self.ctx.buffer(verts.tobytes())
        return self.ctx.vertex_array(self.grid_prog, [(vbo, "3f", "in_position")])

    def _create_octahedral_bone_geometry(self) -> Tuple[moderngl.VertexArray, moderngl.Buffer]:
        """Upload sleek unit octahedral bone geometry (24 vertices with normals)."""
        bone_data = generate_octahedral_bone_mesh(waist_ratio=0.18, radius=0.055)
        vbo = self.ctx.buffer(bone_data.tobytes())
        vao = self.ctx.vertex_array(self.bone_prog, [(vbo, "3f 3f", "in_position", "in_normal")])
        return vao, vbo

    def _create_joint_sphere_geometry(self) -> Tuple[moderngl.VertexArray, moderngl.Buffer]:
        """Upload unit joint sphere geometry."""
        sphere_data = generate_joint_sphere_mesh(radius=1.0, rings=8, sectors=12)
        vbo = self.ctx.buffer(sphere_data.tobytes())
        vao = self.ctx.vertex_array(self.bone_prog, [(vbo, "3f 3f", "in_position", "in_normal")])
        return vao, vbo

    def _create_gimbal_ring_geometry(self) -> Tuple[moderngl.VertexArray, moderngl.Buffer]:
        """Upload Unreal Engine style 3D gimbal ring geometry for terminal & selected joints."""
        ring_data = generate_gimbal_rings_mesh(radius=1.0, tube_radius=0.075, num_segments=24)
        vbo = self.ctx.buffer(ring_data.tobytes())
        vao = self.ctx.vertex_array(self.bone_prog, [(vbo, "3f 3f", "in_position", "in_normal")])
        return vao, vbo

    def _create_single_ring_geometry(self, axis: str) -> Tuple[moderngl.VertexArray, moderngl.Buffer]:
        """Upload individual single-axis 3D rotation ring geometry (X, Y, or Z)."""
        ring_data = generate_single_ring_mesh(axis=axis, radius=1.0, tube_radius=0.045, num_segments=48)
        vbo = self.ctx.buffer(ring_data.tobytes())
        vao = self.ctx.vertex_array(self.bone_prog, [(vbo, "3f 3f", "in_position", "in_normal")])
        return vao, vbo


    def _create_gizmo_geometry(self) -> Tuple[moderngl.VertexArray, moderngl.Buffer]:
        """Upload 3D coordinate axis lines for all 6 directions (+X/-X, +Y/-Y, +Z/-Z)."""
        axis_len = 1.0
        neg_len = 0.55
        # Line pairs: (px, py, pz, r, g, b)
        data = np.array([
            # +X Axis (Bright Coral Red)
            0.0, 0.0, 0.0,  1.0, 0.28, 0.28,
            axis_len, 0.0, 0.0,  1.0, 0.28, 0.28,
            # -X Axis (Subtle Muted Red)
            0.0, 0.0, 0.0,  0.55, 0.22, 0.22,
            -neg_len, 0.0, 0.0,  0.55, 0.22, 0.22,

            # +Y Axis (Bright Emerald Green)
            0.0, 0.0, 0.0,  0.25, 0.95, 0.35,
            0.0, axis_len, 0.0,  0.25, 0.95, 0.35,
            # -Y Axis (Subtle Muted Green)
            0.0, 0.0, 0.0,  0.20, 0.50, 0.25,
            0.0, -neg_len, 0.0,  0.20, 0.50, 0.25,

            # +Z Axis (Bright Sky Blue)
            0.0, 0.0, 0.0,  0.25, 0.65, 1.0,
            0.0, 0.0, axis_len,  0.25, 0.65, 1.0,
            # -Z Axis (Subtle Muted Blue)
            0.0, 0.0, 0.0,  0.22, 0.35, 0.60,
            0.0, 0.0, -neg_len,  0.22, 0.35, 0.60,
        ], dtype=np.float32)

        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(self.gizmo_prog, [(vbo, "3f 3f", "in_position", "in_color")])
        return vao, vbo

    def upload_scene(self, scene: SceneData) -> None:
        """
        Upload all sub-meshes of `scene` to contiguous GPU buffers ONCE.
        Old buffers are cleanly released from GPU memory.
        """
        self.clear_scene()
        self._current_scene = scene

        for sub in scene.submeshes:
            if sub.vertex_count == 0:
                continue

            vbo_pos = self.ctx.buffer(sub.positions.tobytes())
            vbo_norm = self.ctx.buffer(sub.normals.tobytes())
            vbo_uv = self.ctx.buffer(sub.uvs.tobytes())
            vbo_bones = self.ctx.buffer(sub.bone_ids.tobytes())
            vbo_weights = self.ctx.buffer(sub.bone_weights.tobytes())

            ibo = None
            if len(sub.indices) > 0:
                ibo = self.ctx.buffer(sub.indices.tobytes())

            vao = self.ctx.vertex_array(
                self.mesh_prog,
                [
                    (vbo_pos, "3f", "in_position"),
                    (vbo_norm, "3f", "in_normal"),
                    (vbo_uv, "2f", "in_uv"),
                    (vbo_bones, "4i", "in_bone_ids"),
                    (vbo_weights, "4f", "in_bone_weights"),
                ],
                index_buffer=ibo,
                index_element_size=4,  # uint32
            )

            gpu_sub = GPUSubMesh(
                submesh=sub,
                vao=vao,
                vbos=[vbo_pos, vbo_norm, vbo_uv, vbo_bones, vbo_weights],
                ibo=ibo,
            )
            self.gpu_submeshes.append(gpu_sub)

        # Preload any available material textures
        self._sync_material_textures(scene)

    def _sync_material_textures(self, scene: SceneData) -> None:
        """Upload or update ModernGL texture handles for any materials with pending image data."""
        for mat in scene.materials:
            if mat.gpu_texture_dirty or (mat.has_diffuse_texture and mat.gpu_texture is None):
                if mat.gpu_texture is not None:
                    try:
                        mat.gpu_texture.release()
                    except Exception:
                        pass
                    mat.gpu_texture = None

                if mat.diffuse_rgba_bytes and mat.diffuse_size:
                    try:
                        tex = self.ctx.texture(mat.diffuse_size, 4, mat.diffuse_rgba_bytes)
                        tex.build_mipmaps()
                        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                        mat.gpu_texture = tex
                    except Exception:
                        mat.gpu_texture = None
                mat.gpu_texture_dirty = False

    def clear_scene(self) -> None:
        """Release all allocated GPU mesh buffers for the current scene."""
        for gpu_sub in self.gpu_submeshes:
            gpu_sub.release()
        self.gpu_submeshes.clear()

        if self._current_scene:
            for mat in self._current_scene.materials:
                if mat.gpu_texture is not None:
                    try:
                        mat.gpu_texture.release()
                    except Exception:
                        pass
                    mat.gpu_texture = None
            self._current_scene = None

    def render_frame(
        self,
        viewport_w: int,
        viewport_h: int,
        camera: ArcballCamera,
        scene: Optional[SceneData],
        options: RenderOptions,
        model_matrix: Optional[glm.mat4] = None,
        measure_points: Optional[Tuple[Optional[glm.vec3], Optional[glm.vec3]]] = None,
    ) -> None:
        """Render complete viewport frame: background, grid, meshes, skeleton, measurement line, and gizmo."""
        if viewport_w <= 0 or viewport_h <= 0:
            return

        if model_matrix is None:
            model_matrix = glm.mat4(1.0)

        self.ctx.viewport = (0, 0, viewport_w, viewport_h)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        # Preserve destination alpha = 1.0 so QOpenGLWidget FBO never gets alpha holes during blending
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        # Theme-adaptive studio background color
        bg_r, bg_g, bg_b = options.bg_color
        self.ctx.clear(bg_r, bg_g, bg_b, 1.0, 1.0)

        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix()
        cam_pos = camera.position
        view_bytes = glm_to_bytes(view_mat)
        proj_bytes = glm_to_bytes(proj_mat)

        # 1. Render Ground Grid (stays fixed as the horizontal studio floor)
        if options.show_grid:
            self._render_ground_grid(camera, view_bytes, proj_bytes, scene, options)

        # 2. Render 3D Sub-Meshes (oriented by model_matrix for smooth Up-Vector transition)
        if scene and self.gpu_submeshes:
            self._sync_material_textures(scene)
            self._render_submeshes(camera, view_bytes, proj_bytes, scene, options, model_matrix=model_matrix)

        # 3. Render Skeletal Bone & Joint Overlay (oriented by model_matrix)
        if options.show_skeleton and scene and scene.skeleton.has_bones:
            self._render_skeleton_overlay(camera, view_mat, proj_mat, scene, options, model_matrix=model_matrix)

        # 3.5 Render 3D Bounding Box Dimensions Cage (if enabled)
        if options.show_bbox and scene:
            self._render_bounding_box_cage(camera, view_mat, proj_mat, model_matrix, scene)

        # 4. Render 3D Measurement Overlay (Point markers & connecting 3D line)
        if measure_points and (measure_points[0] is not None or measure_points[1] is not None):
            self._render_measurement_overlay(
                camera, view_mat, proj_mat, measure_points[0], measure_points[1], scene
            )

        # 5. Render Corner Orientation XYZ Gizmo
        if options.show_gizmo:
            self._render_corner_gizmo(viewport_w, viewport_h, camera)

    def _render_ground_grid(
        self,
        camera: ArcballCamera,
        view_bytes: bytes,
        proj_bytes: bytes,
        scene: Optional[SceneData],
        options: RenderOptions,
    ) -> None:
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        # Disable backface culling so the grid is visible from both above (+Y) and below (-Y)
        self.ctx.disable(moderngl.CULL_FACE)

        # Adapt grid scale and quad extent to scene size
        radius = scene.scene_aabb.radius if scene and scene.scene_aabb.is_valid else 2.0
        if radius < 15.0:
            grid_size = 1.0
        elif radius < 250.0:
            grid_size = 10.0
        else:
            grid_size = 50.0
        fade_dist = max(radius * 12.0, 40.0)
        grid_model = glm.scale(glm.mat4(1.0), glm.vec3(fade_dist, 1.0, fade_dist))

        prog = self.grid_prog
        prog["u_view"].write(view_bytes)
        prog["u_proj"].write(proj_bytes)
        prog["u_model"].write(glm_to_bytes(grid_model))
        prog["u_cam_pos"].value = (camera.position.x, camera.position.y, camera.position.z)
        prog["u_grid_size"].value = float(grid_size)
        prog["u_fade_dist"].value = float(fade_dist)
        if "u_grid_color" in prog:
            prog["u_grid_color"].value = options.grid_color

        self.grid_vao.render(moderngl.TRIANGLES, vertices=6)

    def _render_submeshes(
        self,
        camera: ArcballCamera,
        view_bytes: bytes,
        proj_bytes: bytes,
        scene: SceneData,
        options: RenderOptions,
        model_matrix: Optional[glm.mat4] = None,
    ) -> None:
        if model_matrix is None:
            model_matrix = glm.mat4(1.0)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        # Render two-sided so single-sided cloth, cape, pocket, and hair surfaces are always visible
        self.ctx.disable(moderngl.CULL_FACE)

        prog = self.mesh_prog
        prog["u_view"].write(view_bytes)
        prog["u_proj"].write(proj_bytes)
        prog["u_cam_pos"].value = (camera.position.x, camera.position.y, camera.position.z)
        prog["u_mode"].value = int(options.mode)
        prog["u_active_bone_id"].value = int(options.active_bone_id)

        # Upload skeletal skinning matrices if model is rigged
        if scene.skeleton.has_bones:
            prog["u_bones"].write(scene.skeleton.skinning_matrices.tobytes())

        # Texture unit 0 for diffuse
        prog["u_diffuse_texture"].value = 0

        for gpu_sub in self.gpu_submeshes:
            sub = gpu_sub.submesh
            if not sub.visible or sub.vertex_count == 0:
                continue

            prog["u_is_skinned"].value = 1 if (sub.has_bones and scene.skeleton.has_bones) else 0
            prog["u_model"].write(glm_to_bytes(model_matrix * sub.node_transform))

            # Retrieve assigned MaterialSlot
            mat_idx = sub.material_index
            mat = scene.materials[mat_idx] if 0 <= mat_idx < len(scene.materials) else scene.materials[0]

            prog["u_base_color"].value = (mat.base_color.x, mat.base_color.y, mat.base_color.z)
            prog["u_ambient"].value = (mat.ambient_color.x, mat.ambient_color.y, mat.ambient_color.z)
            prog["u_specular"].value = (mat.specular_color.x, mat.specular_color.y, mat.specular_color.z)
            prog["u_shininess"].value = float(mat.shininess)
            prog["u_opacity"].value = float(mat.opacity)
            prog["u_shading_type"].value = 1 if mat.shading_type == "Unlit" else 0

            # Bind texture or fallback dummy
            if options.mode == 0 and mat.gpu_texture is not None:
                mat.gpu_texture.use(location=0)
                prog["u_has_texture"].value = 1
            else:
                self.dummy_texture.use(location=0)
                prog["u_has_texture"].value = 0

            if gpu_sub.ibo is not None:
                gpu_sub.vao.render(moderngl.TRIANGLES, vertices=len(sub.indices))
            else:
                gpu_sub.vao.render(moderngl.TRIANGLES, vertices=sub.vertex_count)

        # Wireframe Overlay Pass (if enabled)
        if options.show_wireframe:
            self.ctx.wireframe = True
            # Push wireframe lines forward in z-buffer to completely eliminate z-fighting
            self.ctx.polygon_offset = (-1.5, -2.0)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

            prog["u_mode"].value = 3  # Unlit flat shading
            prog["u_base_color"].value = options.wireframe_color[:3]
            prog["u_opacity"].value = float(options.wireframe_color[3])
            prog["u_has_texture"].value = 0
            self.dummy_texture.use(location=0)

            for gpu_sub in self.gpu_submeshes:
                sub = gpu_sub.submesh
                if not sub.visible or sub.vertex_count == 0:
                    continue

                prog["u_is_skinned"].value = 1 if (sub.has_bones and scene.skeleton.has_bones) else 0
                prog["u_model"].write(glm_to_bytes(model_matrix * sub.node_transform))

                if gpu_sub.ibo is not None:
                    gpu_sub.vao.render(moderngl.TRIANGLES, vertices=len(sub.indices))
                else:
                    gpu_sub.vao.render(moderngl.TRIANGLES, vertices=sub.vertex_count)

            # Restore default solid polygon rasterization mode
            self.ctx.wireframe = False
            self.ctx.polygon_offset = (0.0, 0.0)


    def _render_skeleton_overlay(
        self,
        camera: ArcballCamera,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
        scene: SceneData,
        options: RenderOptions,
        model_matrix: Optional[glm.mat4] = None,
    ) -> None:
        """Render 3D octahedral bones and joint points with X-Ray bypass."""
        if model_matrix is None:
            model_matrix = glm.mat4(1.0)

        if options.xray_skeleton:
            # Bypass depth test so bones shine cleanly through the mesh geometry
            self.ctx.disable(moderngl.DEPTH_TEST)
        else:
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        view_bytes = glm_to_bytes(view_mat)
        proj_bytes = glm.value_ptr(proj_mat)
        proj_bytes = glm_to_bytes(proj_mat)

        prog = self.bone_prog
        prog["u_view"].write(view_bytes)
        prog["u_proj"].write(proj_bytes)
        prog["u_cam_pos"].value = (camera.position.x, camera.position.y, camera.position.z)

        # Scale hints based on scene size and camera distance
        radius = scene.scene_aabb.radius if scene.scene_aabb.is_valid else 1.0
        cam_dist = max(camera.distance, 0.1)
        segments, joints = scene.skeleton.extract_visual_primitives(
            model_scale_hint=radius,
            camera_distance=cam_dist,
        )

        # 1. Render Bone Segments
        col_regular_bone = (0.0, 0.78, 0.96, 0.68)   # Vivid Cyan (68% opacity)
        col_selected_bone = (1.0, 0.62, 0.05, 0.96)  # Bright Orange (96% opacity)

        for seg in segments:
            align_m = generate_bone_align_matrix(
                seg.head_pos,
                seg.tail_pos,
                max_thickness_len=radius * 0.14,
            )
            prog["u_model"].write(glm_to_bytes(model_matrix * align_m))

            if seg.is_selected:
                prog["u_bone_color"].value = col_selected_bone
                prog["u_is_selected"].value = 1
            else:
                prog["u_bone_color"].value = col_regular_bone
                prog["u_is_selected"].value = 0

            self.bone_vao.render(moderngl.TRIANGLES, vertices=24)

        # 2. Render Joint Spheres (individually scaled per bone length & camera distance)
        col_regular_joint = (0.25, 0.88, 1.0, 0.85)
        col_selected_joint = (1.0, 0.88, 0.15, 1.0)
        col_leaf_ring = (0.35, 0.95, 0.88, 0.92)  # Mint/electric cyan for Unreal style terminal rings

        for item in joints:
            if len(item) == 5:
                b_id, j_pos, j_rad, is_sel, is_leaf = item
            elif len(item) == 4:
                b_id, j_pos, j_rad, is_sel = item
                is_leaf = False
            else:
                b_id, j_pos, is_sel = item
                j_rad = max(radius * 0.008, 0.01)
                is_leaf = False

            trans_m = glm.translate(glm.mat4(1.0), j_pos) * glm.scale(glm.mat4(1.0), glm.vec3(j_rad))
            prog["u_model"].write(glm_to_bytes(model_matrix * trans_m))

            if is_sel:
                prog["u_bone_color"].value = col_selected_joint
                prog["u_is_selected"].value = 1
                # Center joint sphere
                self.joint_vao.render(moderngl.TRIANGLES)

                if options.show_fk_gimbal:
                    # 3D Viewport Interactive Rotation Gimbal (X-Red, Y-Green, Z-Blue)
                    # Compact scaling: snugly wraps around the selected joint sphere
                    gimbal_scale = min(max(j_rad * 2.2, 0.05), radius * 0.08)
                    bone_node = scene.skeleton.get_bone(b_id)
                    if bone_node:
                        _, _, ring_m = get_bone_gimbal_matrix(
                            bone=bone_node,
                            model_matrix=model_matrix,
                            gimbal_radius=gimbal_scale,
                        )
                        prog["u_model"].write(glm_to_bytes(ring_m))
                    else:
                        ring_m = glm.translate(glm.mat4(1.0), j_pos) * glm.scale(glm.mat4(1.0), glm.vec3(gimbal_scale))
                        prog["u_model"].write(glm_to_bytes(model_matrix * ring_m))

                    # Ring X (Pitch): Red or Active Highlight
                    is_x = (options.active_gimbal_axis == "X")
                    prog["u_bone_color"].value = (1.0, 0.92, 0.15, 1.0) if is_x else (1.0, 0.28, 0.28, 0.92)
                    prog["u_is_selected"].value = 1 if is_x else 0
                    self.gimbal_x_vao.render(moderngl.TRIANGLES)

                    # Ring Y (Yaw): Green or Active Highlight
                    is_y = (options.active_gimbal_axis == "Y")
                    prog["u_bone_color"].value = (1.0, 0.92, 0.15, 1.0) if is_y else (0.28, 0.95, 0.38, 0.92)
                    prog["u_is_selected"].value = 1 if is_y else 0
                    self.gimbal_y_vao.render(moderngl.TRIANGLES)

                    # Ring Z (Roll): Blue or Active Highlight
                    is_z = (options.active_gimbal_axis == "Z")
                    prog["u_bone_color"].value = (1.0, 0.92, 0.15, 1.0) if is_z else (0.28, 0.68, 1.0, 0.92)
                    prog["u_is_selected"].value = 1 if is_z else 0
                    self.gimbal_z_vao.render(moderngl.TRIANGLES)
                else:
                    ring_m = glm.translate(glm.mat4(1.0), j_pos) * glm.scale(glm.mat4(1.0), glm.vec3(j_rad * 1.5))
                    prog["u_model"].write(glm_to_bytes(model_matrix * ring_m))
                    self.ring_vao.render(moderngl.TRIANGLES)

            elif is_leaf:
                # Terminal leaf joint: render Unreal Engine style circular gimbal ring
                prog["u_bone_color"].value = col_leaf_ring
                prog["u_is_selected"].value = 0
                ring_m = glm.translate(glm.mat4(1.0), j_pos) * glm.scale(glm.mat4(1.0), glm.vec3(j_rad * 1.4))
                prog["u_model"].write(glm_to_bytes(model_matrix * ring_m))
                self.ring_vao.render(moderngl.TRIANGLES)
                # Plus a tiny center joint sphere
                center_m = glm.translate(glm.mat4(1.0), j_pos) * glm.scale(glm.mat4(1.0), glm.vec3(j_rad * 0.45))
                prog["u_model"].write(glm_to_bytes(model_matrix * center_m))
                self.joint_vao.render(moderngl.TRIANGLES)
            else:
                prog["u_bone_color"].value = col_regular_joint
                prog["u_is_selected"].value = 0
                self.joint_vao.render(moderngl.TRIANGLES)

        # Restore depth test state
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_corner_gizmo(
        self,
        viewport_w: int,
        viewport_h: int,
        camera: ArcballCamera,
    ) -> None:
        """Render a subtle 3D orientation axis gizmo in the bottom-left corner."""
        gizmo_size = 80
        margin = 15
        self.ctx.viewport = (margin, margin, gizmo_size, gizmo_size)

        # Clear depth for corner overlay
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Orthographic projection centered at origin
        ortho_proj = glm.ortho(-1.4, 1.4, -1.4, 1.4, -2.0, 2.0)
        # Orientation matrix (pure rotation of camera)
        rot_view = camera.get_orientation_matrix()

        prog = self.gizmo_prog
        prog["u_proj"].write(glm_to_bytes(ortho_proj))
        prog["u_view"].write(glm_to_bytes(rot_view))
        prog["u_model"].write(glm_to_bytes(glm.mat4(1.0)))

        self.gizmo_vao.render(moderngl.LINES, vertices=12)

        # Reset viewport
        self.ctx.viewport = (0, 0, viewport_w, viewport_h)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_measurement_overlay(
        self,
        camera: ArcballCamera,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
        p1: Optional[glm.vec3],
        p2: Optional[glm.vec3],
        scene: Optional[SceneData],
    ) -> None:
        """Render 3D measurement points and connecting line/rod in the viewport."""
        if p1 is None and p2 is None:
            return

        view_bytes = glm_to_bytes(view_mat)
        proj_bytes = glm_to_bytes(proj_mat)

        # X-Ray style overlay: disable depth test so measurements are never hidden inside geometry
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        radius = scene.scene_aabb.radius if (scene and scene.scene_aabb.is_valid) else 2.0
        marker_rad = max(radius * 0.016, 0.03)

        prog = self.bone_prog
        prog["u_view"].write(view_bytes)
        prog["u_proj"].write(proj_bytes)
        prog["u_cam_pos"].value = (camera.position.x, camera.position.y, camera.position.z)

        # 1. Render Point 1 marker (Neon Coral / Orange)
        if p1 is not None:
            prog["u_bone_color"].value = (1.0, 0.40, 0.12, 1.0)
            prog["u_is_selected"].value = 1
            m1 = glm.translate(glm.mat4(1.0), p1) * glm.scale(glm.mat4(1.0), glm.vec3(marker_rad))
            prog["u_model"].write(glm_to_bytes(m1))
            self.joint_vao.render(moderngl.TRIANGLES)

            # Outer ring marker
            prog["u_bone_color"].value = (0.2, 0.95, 0.85, 0.95)
            prog["u_is_selected"].value = 0
            m1_ring = glm.translate(glm.mat4(1.0), p1) * glm.scale(glm.mat4(1.0), glm.vec3(marker_rad * 1.5))
            prog["u_model"].write(glm_to_bytes(m1_ring))
            self.ring_vao.render(moderngl.TRIANGLES)

        # 2. Render Point 2 marker (Vivid Mint / Cyan)
        if p2 is not None:
            prog["u_bone_color"].value = (0.12, 0.95, 0.82, 1.0)
            prog["u_is_selected"].value = 1
            m2 = glm.translate(glm.mat4(1.0), p2) * glm.scale(glm.mat4(1.0), glm.vec3(marker_rad))
            prog["u_model"].write(glm_to_bytes(m2))
            self.joint_vao.render(moderngl.TRIANGLES)

            # Outer ring marker
            prog["u_bone_color"].value = (1.0, 0.85, 0.2, 0.95)
            prog["u_is_selected"].value = 0
            m2_ring = glm.translate(glm.mat4(1.0), p2) * glm.scale(glm.mat4(1.0), glm.vec3(marker_rad * 1.5))
            prog["u_model"].write(glm_to_bytes(m2_ring))
            self.ring_vao.render(moderngl.TRIANGLES)

        # 3. Render 3D connecting rod & line segment
        if p1 is not None and p2 is not None:
            dist = glm.length(p2 - p1)
            if dist > 1e-4:
                align_m = generate_bone_align_matrix(p1, p2, max_thickness_len=marker_rad * 0.35)
                prog["u_model"].write(glm_to_bytes(align_m))
                prog["u_bone_color"].value = (1.0, 0.90, 0.30, 0.95)  # Vivid Golden Yellow connector
                prog["u_is_selected"].value = 1
                self.bone_vao.render(moderngl.TRIANGLES, vertices=24)

            # High-visibility line segment
            line_data = np.array([
                p1.x, p1.y, p1.z,  1.0, 0.45, 0.15,
                p2.x, p2.y, p2.z,  0.15, 0.95, 0.82,
            ], dtype=np.float32)
            self.measure_line_vbo.write(line_data.tobytes())

            gprog = self.gizmo_prog
            gprog["u_view"].write(view_bytes)
            gprog["u_proj"].write(proj_bytes)
            gprog["u_model"].write(glm_to_bytes(glm.mat4(1.0)))
            self.measure_line_vao.render(moderngl.LINES, vertices=2)

        # Restore standard depth test
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_bounding_box_cage(
        self,
        camera: ArcballCamera,
        view_mat: glm.mat4,
        proj_mat: glm.mat4,
        model_matrix: Optional[glm.mat4],
        scene: SceneData,
    ) -> None:
        """Render 3D architectural bounding box cage with corner brackets around the model."""
        bbox = compute_scene_bounds(scene, model_matrix)
        if not bbox.is_valid:
            return

        lines = generate_bbox_cage_lines(bbox)
        if len(lines) == 0:
            return

        data_bytes = lines.tobytes()
        if len(data_bytes) > self.bbox_vbo.size:
            self.bbox_vbo.orphan(len(data_bytes))
        self.bbox_vbo.write(data_bytes)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        prog = self.gizmo_prog
        prog["u_view"].write(glm_to_bytes(view_mat))
        prog["u_proj"].write(glm_to_bytes(proj_mat))
        prog["u_model"].write(glm_to_bytes(glm.mat4(1.0)))

        self.bbox_vao.render(moderngl.LINES, vertices=len(lines))

