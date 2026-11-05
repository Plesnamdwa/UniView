"""
ui/gl_viewport.py
PyQt6 QOpenGLWidget hosting ModernGL context and hardware-accelerated 3D viewport.
Features:
- Native QOpenGLWidget + ModernGL FBO integration with High-DPI support
- Arcball Camera controls:
  - Left Mouse Button (LMB) Drag: Orbit camera
  - Right Mouse Button (RMB) / Middle Mouse Button Drag: Pan camera target
  - Mouse Scroll Wheel: Smooth Zoom In / Zoom Out
- Drag-and-Drop 3D asset importer directly onto viewport
- 60 FPS smooth camera interpolation when isolating sub-meshes or framing scene
- Real-time FPS counter & Camera stats signals
"""

from __future__ import annotations
import os
import time
import math
from typing import Optional, Tuple
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QDragEnterEvent, QDropEvent, QKeyEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import moderngl
import glm

from core.scene_loader import SceneData, SUPPORTED_EXTENSIONS
from core.camera import ArcballCamera
from core.renderer import GLRenderer, RenderOptions
from core.math_utils import AABB
from core.raycast import raycast_scene, pick_bone_at_screen
from core.gizmo import is_pos_in_gizmo_region, hit_test_gizmo_axis
from core.bounding_box import compute_scene_bounds, BoundingBoxInfo
from core.gimbal import hit_test_gimbal_rings, compute_gimbal_drag_angle
from ui.custom_widgets import (
    SegmentedControl,
    ToastNotification,
    ViewportDropOverlay,
    ViewportLoadingOverlay,
    MeasurementHUDCard,
    BoundingBoxHUDCard,
)



class GLViewport(QOpenGLWidget):
    """Interactive 3D Viewport widget powered by PyQt6 and ModernGL."""

    file_dropped = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    camera_updated = pyqtSignal(float, float, float, float)  # target_x, target_y, target_z, distance
    focus_requested = pyqtSignal()
    render_mode_requested = pyqtSignal(int)
    up_vector_changed = pyqtSignal(str, bool)  # mode ('Y' or 'Z'), is_flipped
    measure_mode_toggled = pyqtSignal(bool)
    bone_picked = pyqtSignal(int)
    wireframe_toggled = pyqtSignal(bool)
    bbox_toggled = pyqtSignal(bool)
    bone_fk_rotated = pyqtSignal(int, float, float, float)  # bone_id, rx, ry, rz
    gimbal_toggled = pyqtSignal(bool)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Rendering and Camera State
        self.ctx: Optional[moderngl.Context] = None
        self.renderer: Optional[GLRenderer] = None
        self.camera: ArcballCamera = ArcballCamera()
        self.render_options: RenderOptions = RenderOptions()

        self.scene_data: Optional[SceneData] = None
        self._pending_scene: Optional[SceneData] = None

        # Model Orientation & Up-Vector Animation State (Smooth Slerp Interpolation)
        self._cur_model_quat: glm.quat = glm.quat(1.0, 0.0, 0.0, 0.0)
        self._start_model_quat: glm.quat = glm.quat(1.0, 0.0, 0.0, 0.0)
        self._target_model_quat: glm.quat = glm.quat(1.0, 0.0, 0.0, 0.0)
        self._model_animating: bool = False
        self._model_anim_elapsed: float = 0.0
        self._model_anim_duration: float = 0.38
        self._up_mode: str = "Y"
        self._is_up_flipped: bool = False

        # Mouse interaction tracking
        self._last_mouse_pos: QPoint = QPoint()
        self._is_orbiting: bool = False
        self._is_panning: bool = False

        # Performance & FPS calculation
        self._last_frame_time: float = time.perf_counter()
        self._frame_count: int = 0
        self._fps_accumulator: float = 0.0
        self._last_cam_emit_time: float = 0.0
        self._last_cam_state: tuple = ()

        # Session 3: Floating Viewport Shading HUD (top-left pill selector)
        self.hud_shading = SegmentedControl(["Textured", "Base", "Heatmap", "Unlit"], self)
        self.hud_shading.setFixedSize(276, 32)
        self.hud_shading.move(14, 14)
        self.hud_shading.setToolTip("Shading mode (Shortcuts: 1 Textured · 2 Base · 3 Heatmap · 4 Unlit)")
        self.hud_shading.currentChanged.connect(self.render_mode_requested.emit)

        # Session 2 & 3: Floating Toast Notification, Drag-and-Drop Overlay & Loading Overlay
        self.toast = ToastNotification(self)
        self.drop_overlay = ViewportDropOverlay(self)
        self.loading_overlay = ViewportLoadingOverlay(self)

        # Session 3: 3D Measurement Tool State & Floating HUD Card
        self._is_measure_mode: bool = False
        self._measure_p1: Optional[glm.vec3] = None
        self._measure_p2: Optional[glm.vec3] = None
        self._measure_distance: Optional[float] = None
        self._measure_angle: Optional[float] = None
        self._press_pos: QPoint = QPoint()
        self._is_potential_click: bool = False

        self.hud_measure = MeasurementHUDCard(self)
        self.hud_measure.clear_requested.connect(self.clear_measurements)
        self.hud_measure.close_requested.connect(lambda: self.set_measure_mode(False))

        self.hud_bbox = BoundingBoxHUDCard(self)
        self.hud_bbox.close_requested.connect(lambda: self.set_show_bbox(False))

        # 3D FK Rotation Gimbal Drag State
        self._is_gimbal_dragging: bool = False
        self._gimbal_drag_axis: Optional[str] = None
        self._gimbal_drag_start_pos: Tuple[float, float] = (0.0, 0.0)
        self._gimbal_start_euler: glm.vec3 = glm.vec3(0.0)



    def initializeGL(self) -> None:
        """Called once upon OpenGL context creation."""
        # Attach ModernGL to the current active Qt OpenGL context
        self.ctx = moderngl.create_context()
        self.renderer = GLRenderer(self.ctx)

        # Upload scene if set prior to widget initialization
        if self._pending_scene is not None:
            self.scene_data = self._pending_scene
            self.renderer.upload_scene(self.scene_data)
            self._pending_scene = None

    def paintGL(self) -> None:
        """Render loop called whenever the viewport needs updating."""
        if not self.ctx or not self.renderer:
            return

        now = time.perf_counter()
        dt = max(now - self._last_frame_time, 0.001)
        self._last_frame_time = now

        # Update FPS counter
        self._frame_count += 1
        self._fps_accumulator += dt
        if self._fps_accumulator >= 0.5:
            current_fps = self._frame_count / self._fps_accumulator
            self.fps_updated.emit(round(current_fps, 1))
            self._frame_count = 0
            self._fps_accumulator = 0.0

        # Step camera animation (smooth lerp during mesh focus / framing)
        camera_moved = self.camera.update_animation(dt)
        if camera_moved:
            self.update()  # Request continuous redraw during animation

        # Step model orientation slerp animation (smooth Up-Vector lerp)
        if self._model_animating:
            self._model_anim_elapsed += dt
            t = min(self._model_anim_elapsed / self._model_anim_duration, 1.0)
            # Out-Cubic juicy easing
            ease_t = 1.0 - (1.0 - t) ** 3
            self._cur_model_quat = glm.slerp(self._start_model_quat, self._target_model_quat, ease_t)
            if t >= 1.0:
                self._cur_model_quat = self._target_model_quat
                self._model_animating = False
            self.update()

        model_m = glm.mat4_cast(self._cur_model_quat)

        # Handle High-DPI displays (Windows display scaling e.g., 125%, 150%)
        dpr = self.devicePixelRatio()
        viewport_w = int(self.width() * dpr)
        viewport_h = int(self.height() * dpr)
        self.camera.set_aspect_ratio(viewport_w, viewport_h)

        # Detect and bind Qt's internal QOpenGLWidget framebuffer
        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        # Render complete frame
        measure_pts = (
            (self._measure_p1, self._measure_p2)
            if (self._is_measure_mode or self._measure_p1 is not None)
            else None
        )
        self.renderer.render_frame(
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            camera=self.camera,
            scene=self.scene_data,
            options=self.render_options,
            model_matrix=model_m,
            measure_points=measure_pts,
        )

        # Update Bounding Box HUD if enabled
        if self.render_options.show_bbox and self.scene_data:
            bbox = compute_scene_bounds(self.scene_data, model_m)
            self.hud_bbox.set_bbox_info(bbox)

        # Emit updated camera coordinates for UI status bar (throttled to ~12 Hz for smooth 60 FPS)
        if now - self._last_cam_emit_time >= 0.08:
            tx = float(self.camera.target.x)
            ty = float(self.camera.target.y)
            tz = float(self.camera.target.z)
            dist = float(self.camera.distance)
            cam_state = (round(tx, 2), round(ty, 2), round(tz, 2), round(dist, 2))
            if cam_state != self._last_cam_state:
                self._last_cam_state = cam_state
                self._last_cam_emit_time = now
                self.camera_updated.emit(tx, ty, tz, dist)

    def resizeGL(self, w: int, h: int) -> None:
        """Handle viewport resize."""
        dpr = self.devicePixelRatio()
        self.camera.set_aspect_ratio(int(w * dpr), int(h * dpr))
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.toast._reposition()
        self.drop_overlay.sync_geometry()
        self.hud_measure._update_size_and_position()
        self.hud_bbox._update_size_and_position()
        if self.loading_overlay.isVisible():
            self.loading_overlay.setGeometry(0, 0, self.width(), self.height())

    def show_toast(self, text: str, kind: str = "info", duration_ms: int = 1850) -> None:
        """Display a juicy floating pill notification at the bottom-right of the viewport."""
        self.toast.show_toast(text, kind=kind, duration_ms=duration_ms)

    def set_loading(self, active: bool, filename: str = "") -> None:
        """Show or hide the centered loading overlay card."""
        if active:
            self.loading_overlay.show_loading(filename)
        else:
            self.loading_overlay.hide()

    # -------------------------------------------------------------------------
    # Public Scene and Viewport Control Methods
    # -------------------------------------------------------------------------

    def set_scene(self, scene: SceneData, frame_camera: bool = True) -> None:
        """Set the active 3D asset scene and upload geometry to GPU."""
        self.scene_data = scene
        if self.renderer:
            self.makeCurrent()
            self.renderer.upload_scene(scene)
            self.doneCurrent()
        else:
            self._pending_scene = scene

        if frame_camera and scene and scene.scene_aabb.is_valid:
            target_m = glm.mat4_cast(self._cur_model_quat)
            target_aabb = scene.scene_aabb.transformed(target_m)
            self.camera.frame_aabb(target_aabb, animate=False, save_as_home=True)

        self.update()

    def set_render_mode(self, mode: int) -> None:
        """0: Textured, 1: Base Color / No Material, 2: Weight Heatmap, 3: Unlit Flat."""
        self.render_options.mode = mode
        if hasattr(self, "hud_shading") and self.hud_shading.current_index() != mode:
            self.hud_shading.set_current_index(mode, animate=True)
        self.update()

    def set_show_grid(self, enabled: bool) -> None:
        self.render_options.show_grid = enabled
        self.update()

    def set_show_skeleton(self, enabled: bool) -> None:
        self.render_options.show_skeleton = enabled
        self.update()

    def set_xray_skeleton(self, enabled: bool) -> None:
        self.render_options.xray_skeleton = enabled
        self.update()

    def set_show_gizmo(self, enabled: bool) -> None:
        self.render_options.show_gizmo = enabled
        self.update()

    def set_active_bone_id(self, bone_id: int) -> None:
        self.render_options.active_bone_id = bone_id
        if self.scene_data and self.scene_data.skeleton.has_bones:
            self.scene_data.skeleton.selected_bone_id = bone_id
        self.update()

    def reset_camera(self) -> None:
        """Animate camera back to home framing position."""
        self.camera.reset_to_home(animate=True)
        self.update()

    def frame_scene(self) -> None:
        """Animate camera to frame the full scene AABB accounting for model orientation."""
        if self.scene_data and self.scene_data.scene_aabb.is_valid:
            target_m = glm.mat4_cast(self._target_model_quat)
            target_aabb = self.scene_data.scene_aabb.transformed(target_m)
            self.camera.frame_aabb(target_aabb, animate=True)
            self.update()

    def frame_aabb(self, aabb: AABB, animate: bool = True) -> None:
        """Animate camera to frame a specific bounding box (transformed to match model orientation)."""
        if aabb and aabb.is_valid:
            target_m = glm.mat4_cast(self._target_model_quat)
            target_aabb = aabb.transformed(target_m)
            self.camera.frame_aabb(target_aabb, animate=animate)
            self.update()

    def get_model_matrix(self) -> glm.mat4:
        """Get current 4x4 world transformation matrix representing avatar orientation."""
        return glm.mat4_cast(self._cur_model_quat)

    def set_model_up_vector(self, mode: str) -> None:
        """
        Smoothly transition model orientation to align with desired Up Vector.
        Mode:
          'Y'    - Set Y-axis as Up (Standard OpenGL / Maya / glTF coordinate system)
          'Z'    - Set Z-axis as Up (Unreal Engine / 3ds Max / Blender coordinate system)
          'FLIP' - Invert/Flip the current Up Vector vertically
        """
        if mode == "Y":
            self._up_mode = "Y"
            self._is_up_flipped = False
            base_q = glm.quat(1.0, 0.0, 0.0, 0.0)
            toast_text = "📐 Up Vector: Y-Axis Up"
        elif mode == "Z":
            self._up_mode = "Z"
            self._is_up_flipped = False
            # Rotate -90 degrees around X axis to map +Z into +Y
            base_q = glm.angleAxis(glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0))
            toast_text = "📐 Up Vector: Z-Axis Up"
        elif mode == "FLIP":
            self._is_up_flipped = not self._is_up_flipped
            if self._up_mode == "Z":
                base_q = glm.angleAxis(glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0))
            else:
                base_q = glm.quat(1.0, 0.0, 0.0, 0.0)
            state_str = "Flipped" if self._is_up_flipped else "Normal"
            toast_text = f"📐 Up Vector: {state_str} ({self._up_mode}-Axis)"
        else:
            return

        if self._is_up_flipped:
            # Rotate 180 degrees around Z axis to flip vertical axis
            flip_q = glm.angleAxis(glm.radians(180.0), glm.vec3(0.0, 0.0, 1.0))
            target_q = flip_q * base_q
        else:
            target_q = base_q

        # Begin smooth quaternion slerp transition
        self._start_model_quat = glm.quat(self._cur_model_quat)
        self._target_model_quat = target_q
        self._model_anim_elapsed = 0.0
        self._model_animating = True

        # Smoothly re-frame camera to encapsulate the new avatar posture
        if self.scene_data and self.scene_data.scene_aabb.is_valid:
            target_m = glm.mat4_cast(target_q)
            target_aabb = self.scene_data.scene_aabb.transformed(target_m)
            self.camera.frame_aabb(target_aabb, animate=True)

        self.up_vector_changed.emit(self._up_mode, self._is_up_flipped)
        self.show_toast(toast_text, kind="info")
        self.update()

    def reset_up_vector(self, animate: bool = False) -> None:
        """Reset model orientation to standard Y-Up."""
        self._up_mode = "Y"
        self._is_up_flipped = False
        target_q = glm.quat(1.0, 0.0, 0.0, 0.0)
        if animate:
            self._start_model_quat = glm.quat(self._cur_model_quat)
            self._target_model_quat = target_q
            self._model_anim_elapsed = 0.0
            self._model_animating = True
        else:
            self._cur_model_quat = target_q
            self._target_model_quat = target_q
            self._model_animating = False

        self.up_vector_changed.emit(self._up_mode, self._is_up_flipped)
        self.update()

    def refresh_materials(self) -> None:
        """Trigger viewport repaint when material colors, opacity, or textures change."""
        self.update()

    def set_theme_colors(self, bg_rgb: tuple[float, float, float], grid_rgb: tuple[float, float, float]) -> None:
        """Synchronize 3D viewport studio background and grid colors with Dark / Light UI theme."""
        self.render_options.bg_color = bg_rgb
        self.render_options.grid_color = grid_rgb
        self.hud_shading.update()
        self.toast.update()
        self.drop_overlay.update()
        self.loading_overlay.update()
        self.hud_measure.update()
        self.update()

    def set_wireframe_mode(self, enabled: bool) -> None:
        """Toggle wireframe mesh overlay on top of shaded geometry."""
        self.render_options.show_wireframe = bool(enabled)
        self.wireframe_toggled.emit(self.render_options.show_wireframe)
        status_txt = "ON" if self.render_options.show_wireframe else "OFF"
        self.show_toast(f"🕸️ Wireframe Overlay: {status_txt}", kind="info" if self.render_options.show_wireframe else "neutral")
        self.update()

    def toggle_wireframe(self) -> None:
        """Toggle wireframe overlay between active and inactive."""
        self.set_wireframe_mode(not self.render_options.show_wireframe)

    def set_show_bbox(self, enabled: bool) -> None:
        """Toggle 3D bounding box cage and real-time dimension HUD card."""
        self.render_options.show_bbox = bool(enabled)
        if enabled:
            model_m = glm.mat4_cast(self._cur_model_quat)
            bbox = compute_scene_bounds(self.scene_data, model_m)
            self.hud_bbox.set_bbox_info(bbox)
            self.hud_bbox.show_card()
            self.show_toast("📦 Bounding Box: ON", kind="info")
        else:
            self.hud_bbox.hide_card()
            self.show_toast("📦 Bounding Box: OFF", kind="neutral")
        self.bbox_toggled.emit(self.render_options.show_bbox)
        self.update()

    def toggle_show_bbox(self) -> None:
        """Toggle bounding box display."""
        self.set_show_bbox(not self.render_options.show_bbox)

    def set_show_fk_gimbal(self, enabled: bool) -> None:
        """Toggle 3D FK rotation gimbal visibility."""
        self.render_options.show_fk_gimbal = bool(enabled)
        self.gimbal_toggled.emit(self.render_options.show_fk_gimbal)
        self.show_toast(f"🔄 3D FK Gimbal: {'ON' if enabled else 'OFF'}", kind="info" if enabled else "neutral")
        self.update()

    def toggle_show_fk_gimbal(self) -> None:
        """Toggle 3D FK rotation gimbal between active and inactive."""
        self.set_show_fk_gimbal(not self.render_options.show_fk_gimbal)

    def _get_selected_bone_pivot_and_radius(self) -> Tuple[Optional[glm.vec3], float, Optional[any]]:
        """Return (world_pivot, gimbal_radius, bone_node) for the currently selected bone."""
        if (
            not self.scene_data
            or not self.scene_data.skeleton.has_bones
            or self.render_options.active_bone_id < 0
        ):
            return None, 1.0, None

        bone = self.scene_data.skeleton.find_bone_by_id(self.render_options.active_bone_id)
        if bone is None:
            return None, 1.0, None

        model_m = glm.mat4_cast(self._cur_model_quat)
        local_pivot = glm.vec3(bone.world_matrix[3].x, bone.world_matrix[3].y, bone.world_matrix[3].z)
        world_pivot = glm.vec3(model_m * glm.vec4(local_pivot, 1.0))

        cam_dist = float(glm.length(self.camera.position - world_pivot))
        radius = self.scene_data.scene_aabb.radius if self.scene_data.scene_aabb.is_valid else 1.0
        gimbal_rad = max(cam_dist * 0.08, radius * 0.04, 0.08)
        return world_pivot, gimbal_rad, bone



    # -------------------------------------------------------------------------
    # 3D Measurement Tool Control Methods
    # -------------------------------------------------------------------------

    def set_measure_mode(self, enabled: bool) -> None:
        """Toggle 3D interactive measurement tool."""
        self._is_measure_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.hud_measure.show_card()
            self.show_toast("📏 Measure Mode: Click model surface to place points", kind="info")
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.hud_measure.hide_card()
            self.clear_measurements()

        self.measure_mode_toggled.emit(enabled)
        self.update()

    def clear_measurements(self) -> None:
        """Clear all 3D measurement points and line."""
        self._measure_p1 = None
        self._measure_p2 = None
        self._measure_distance = None
        self._measure_angle = None
        self.hud_measure.set_measuring_state(point_count=0)
        self.update()

    def _on_measure_point_clicked(self, hit: glm.vec3, is_snapped: bool = False) -> None:
        """Handle 3D world hit point placement with optional vertex snapping feedback."""
        snap_badge = " (Vertex Snap)" if is_snapped else ""
        if self._measure_p1 is None:
            self._measure_p1 = hit
            self._measure_p2 = None
            self._measure_distance = None
            self._measure_angle = None
            self.hud_measure.set_measuring_state(point_count=1)
            self.show_toast(f"📍 Point 1{snap_badge}: ({hit.x:.2f}, {hit.y:.2f}, {hit.z:.2f})", kind="info")
        elif self._measure_p2 is None:
            self._measure_p2 = hit
            diff = self._measure_p2 - self._measure_p1
            dist = float(glm.length(diff))
            horiz_len = math.hypot(float(diff.x), float(diff.z))
            angle_deg = float(math.degrees(math.atan2(abs(float(diff.y)), horiz_len))) if dist > 1e-4 else 0.0
            self._measure_distance = dist
            self._measure_angle = angle_deg
            self.hud_measure.set_measuring_state(point_count=2, distance=dist, angle=angle_deg)
            self.show_toast(f"📏 Distance: {dist:.3f} · Angle: {angle_deg:.1f}°{snap_badge}", kind="success")
        else:
            # 3rd click starts fresh measurement
            self._measure_p1 = hit
            self._measure_p2 = None
            self._measure_distance = None
            self._measure_angle = None
            self.hud_measure.set_measuring_state(point_count=1)
            self.show_toast(f"📍 Point 1{snap_badge}: ({hit.x:.2f}, {hit.y:.2f}, {hit.z:.2f})", kind="info")

        self.update()

    # -------------------------------------------------------------------------
    # Mouse Interaction Events (with Click vs Drag Discrimination)
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.pos()
            self._is_potential_click = True

            # 0. Check if clicking on an active 3D FK Rotation Gimbal ring
            if self.render_options.show_fk_gimbal and self.render_options.active_gimbal_axis:
                pivot, rad, bone = self._get_selected_bone_pivot_and_radius()
                if bone is not None:
                    self._is_gimbal_dragging = True
                    self._gimbal_drag_axis = self.render_options.active_gimbal_axis
                    self._gimbal_drag_start_pos = (float(event.pos().x()), float(event.pos().y()))
                    self._gimbal_start_euler = glm.vec3(bone.slider_euler_deg)
                    self._is_orbiting = False
                    self._is_potential_click = False
                    event.accept()
                    return

            # If clicking in the gizmo corner, don't start orbit
            if self.render_options.show_gizmo and is_pos_in_gizmo_region(
                float(event.pos().x()), float(event.pos().y()), float(self.height())
            ):
                self._is_orbiting = False
                event.accept()
                return
            # Do NOT orbit immediately so single clicks don't cause jitter
            self._is_orbiting = False
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._is_panning = True
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # 1. Active 3D FK Gimbal Ring Dragging
        if self._is_gimbal_dragging:
            pivot, rad, bone = self._get_selected_bone_pivot_and_radius()
            if bone is not None and pivot is not None:
                delta_deg = compute_gimbal_drag_angle(
                    start_mouse=self._gimbal_drag_start_pos,
                    current_mouse=(float(event.pos().x()), float(event.pos().y())),
                    gimbal_center=pivot,
                    axis=self._gimbal_drag_axis,
                    camera=self.camera,
                    viewport_w=float(self.width()),
                    viewport_h=float(self.height()),
                )

                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    delta_deg *= 0.25

                new_rx = self._gimbal_start_euler.x
                new_ry = self._gimbal_start_euler.y
                new_rz = self._gimbal_start_euler.z

                if self._gimbal_drag_axis == "X":
                    new_rx = self._gimbal_start_euler.x + delta_deg
                    if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        new_rx = round(new_rx / 5.0) * 5.0
                elif self._gimbal_drag_axis == "Y":
                    new_ry = self._gimbal_start_euler.y + delta_deg
                    if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        new_ry = round(new_ry / 5.0) * 5.0
                elif self._gimbal_drag_axis == "Z":
                    new_rz = self._gimbal_start_euler.z + delta_deg
                    if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        new_rz = round(new_rz / 5.0) * 5.0

                new_rx = ((new_rx + 180.0) % 360.0) - 180.0
                new_ry = ((new_ry + 180.0) % 360.0) - 180.0
                new_rz = ((new_rz + 180.0) % 360.0) - 180.0

                bone.set_slider_rotation(new_rx, new_ry, new_rz)
                self.scene_data.skeleton.update_forward_kinematics()
                self.bone_fk_rotated.emit(bone.id, new_rx, new_ry, new_rz)

                axis_label = {"X": "Pitch (X)", "Y": "Yaw (Y)", "Z": "Roll (Z)"}.get(self._gimbal_drag_axis, "Rot")
                active_val = {"X": new_rx, "Y": new_ry, "Z": new_rz}.get(self._gimbal_drag_axis, 0.0)
                snap_badge = " [Snap 5°]" if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) else ""
                fine_badge = " [Fine]" if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else ""
                self.show_toast(f"🔄 {axis_label}: {active_val:+.1f}°{snap_badge}{fine_badge}", kind="info")
                self.update()
                event.accept()
                return

        # 2. Hover cursor feedback (pointing hand over gimbal ring, gizmo, or measure crosshair)
        if not event.buttons():
            gimbal_hit = None
            if self.render_options.show_fk_gimbal and self.render_options.show_skeleton:
                pivot, rad, bone = self._get_selected_bone_pivot_and_radius()
                if pivot is not None:
                    gimbal_hit = hit_test_gimbal_rings(
                        mouse_x=float(event.pos().x()),
                        mouse_y=float(event.pos().y()),
                        viewport_w=float(self.width()),
                        viewport_h=float(self.height()),
                        camera=self.camera,
                        gimbal_center=pivot,
                        gimbal_radius=rad,
                        hit_tolerance_px=16.0,
                    )

            if gimbal_hit is not None:
                if self.render_options.active_gimbal_axis != gimbal_hit:
                    self.render_options.active_gimbal_axis = gimbal_hit
                    self.update()
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                if self.render_options.active_gimbal_axis is not None:
                    self.render_options.active_gimbal_axis = None
                    self.update()

                if self.render_options.show_gizmo and is_pos_in_gizmo_region(
                    float(event.pos().x()), float(event.pos().y()), float(self.height())
                ):
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                elif self._is_measure_mode:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

        if event.buttons() & Qt.MouseButton.LeftButton:
            dist = (event.pos() - self._press_pos).manhattanLength()
            if dist > 4:
                self._is_potential_click = False
                self._is_orbiting = True

        dx = float(event.pos().x() - self._last_mouse_pos.x())
        dy = float(event.pos().y() - self._last_mouse_pos.y())
        self._last_mouse_pos = event.pos()

        if self._is_orbiting:
            self.camera.orbit(dx, dy)
            self.update()
        elif self._is_panning:
            self.camera.pan(dx, dy)
            self.update()

        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._is_gimbal_dragging:
            self._is_gimbal_dragging = False
            self.update()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:

            is_click = self._is_potential_click and ((event.pos() - self._press_pos).manhattanLength() <= 4)
            if is_click:
                # 1. Interactive Corner Gizmo: 1-click snap to standard Ortho/Axis views!
                if self.render_options.show_gizmo:
                    hit_gizmo = hit_test_gizmo_axis(
                        mouse_x=float(event.pos().x()),
                        mouse_y=float(event.pos().y()),
                        viewport_h=float(self.height()),
                        camera=self.camera,
                    )
                    if hit_gizmo is not None:
                        axis_label, target_yaw, target_pitch, view_name = hit_gizmo
                        self.camera.set_view_angles(target_yaw, target_pitch, animate=True)
                        self.show_toast(f"🧭 {view_name}", kind="info")
                        self.update()
                        self._is_potential_click = False
                        self._is_orbiting = False
                        event.accept()
                        return

                # 2. 3D Measurement Point Picking
                if self._is_measure_mode:
                    dpr = self.devicePixelRatio()
                    w = int(self.width() * dpr)
                    h = int(self.height() * dpr)
                    sx = float(event.pos().x() * dpr)
                    sy = float(event.pos().y() * dpr)
                    # Snap active if Shift/Ctrl is held, OR if HUD snap is enabled
                    snap_active = bool(
                        (event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
                        or self.hud_measure.is_snap_enabled()
                    )
                    hit = raycast_scene(
                        screen_x=sx,
                        screen_y=sy,
                        viewport_w=w,
                        viewport_h=h,
                        camera=self.camera,
                        scene=self.scene_data,
                        model_matrix=self.get_model_matrix(),
                        snap_to_vertex=snap_active,
                    )
                    if hit is not None:
                        self._on_measure_point_clicked(hit.point, hit.is_vertex_snapped)
                    else:
                        self.show_toast("Raycast: missed model surface", kind="warning", duration_ms=1200)

                # 3. Direct 3D Bone Picking from viewport!
                elif (
                    self.render_options.show_skeleton
                    and self.scene_data
                    and self.scene_data.skeleton
                    and self.scene_data.skeleton.has_bones
                ):
                    dpr = self.devicePixelRatio()
                    w = int(self.width() * dpr)
                    h = int(self.height() * dpr)
                    sx = float(event.pos().x() * dpr)
                    sy = float(event.pos().y() * dpr)
                    model_scale = self.scene_data.scene_aabb.radius if self.scene_data.scene_aabb.is_valid else 1.0
                    bone_id = pick_bone_at_screen(
                        screen_x=sx,
                        screen_y=sy,
                        viewport_w=w,
                        viewport_h=h,
                        camera=self.camera,
                        skeleton=self.scene_data.skeleton,
                        model_matrix=self.get_model_matrix(),
                        model_scale_hint=model_scale,
                    )
                    if bone_id is not None:
                        bone = self.scene_data.skeleton.get_bone(bone_id)
                        bone_name = bone.name if bone else f"Bone #{bone_id}"
                        self.show_toast(f"🦴 Selected: {bone_name}", kind="info")
                        self.bone_picked.emit(bone_id)
                        self.update()

            self._is_potential_click = False
            self._is_orbiting = False
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._is_panning = False
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Standard mouse wheel step is usually +/- 120
        delta = event.angleDelta().y()
        if delta != 0:
            steps = float(delta) / 120.0
            self.camera.zoom(steps)
            self.update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle 3D artist standard navigation hotkeys directly on the viewport."""
        if event.key() == Qt.Key.Key_M:
            self.set_measure_mode(not self._is_measure_mode)
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Escape and self._is_measure_mode:
            self.set_measure_mode(False)
            event.accept()
            return
        elif event.key() == Qt.Key.Key_F:
            self.focus_requested.emit()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_A:
            self.frame_scene()
            event.accept()
            return
        super().keyPressEvent(event)
        super().keyPressEvent(event)

    # -------------------------------------------------------------------------
    # Drag-and-Drop 3D Asset Handling
    # -------------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drop action if dragging a supported 3D model file."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                ext = os.path.splitext(filepath)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    self.drop_overlay.sync_geometry()
                    self.drop_overlay.show()
                    self.drop_overlay.raise_()
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        """Hide drop overlay when file drag leaves the viewport."""
        self.drop_overlay.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Emit file_dropped signal with the first supported 3D file."""
        self.drop_overlay.hide()
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                ext = os.path.splitext(filepath)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and os.path.isfile(filepath):
                    self.file_dropped.emit(filepath)
                    event.acceptProposedAction()
                    return
        event.ignore()
