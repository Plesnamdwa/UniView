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
from typing import Optional
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QDragEnterEvent, QDropEvent, QKeyEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import moderngl

from core.scene_loader import SceneData, SUPPORTED_EXTENSIONS
from core.camera import ArcballCamera
from core.renderer import GLRenderer, RenderOptions
from core.math_utils import AABB
from ui.custom_widgets import (
    SegmentedControl,
    ToastNotification,
    ViewportDropOverlay,
    ViewportLoadingOverlay,
)


class GLViewport(QOpenGLWidget):
    """Interactive 3D Viewport widget powered by PyQt6 and ModernGL."""

    file_dropped = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    camera_updated = pyqtSignal(float, float, float, float)  # target_x, target_y, target_z, distance
    focus_requested = pyqtSignal()
    render_mode_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Rendering and Camera State
        self.ctx: Optional[moderngl.Context] = None
        self.renderer: Optional[GLRenderer] = None
        self.camera: ArcballCamera = ArcballCamera()
        self.render_options: RenderOptions = RenderOptions()

        self.scene_data: Optional[SceneData] = None
        self._pending_scene: Optional[SceneData] = None

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

        # Handle High-DPI displays (Windows display scaling e.g., 125%, 150%)
        dpr = self.devicePixelRatio()
        viewport_w = int(self.width() * dpr)
        viewport_h = int(self.height() * dpr)
        self.camera.set_aspect_ratio(viewport_w, viewport_h)

        # Detect and bind Qt's internal QOpenGLWidget framebuffer
        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        # Render complete frame
        self.renderer.render_frame(
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            camera=self.camera,
            scene=self.scene_data,
            options=self.render_options,
        )

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
            self.camera.frame_aabb(scene.scene_aabb, animate=False, save_as_home=True)

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
        """Animate camera to frame the full scene AABB."""
        if self.scene_data and self.scene_data.scene_aabb.is_valid:
            self.camera.frame_aabb(self.scene_data.scene_aabb, animate=True)
            self.update()

    def frame_aabb(self, aabb: AABB, animate: bool = True) -> None:
        """Animate camera to frame a specific bounding box (used by sub-mesh Focus/Isolate)."""
        if aabb.is_valid:
            self.camera.frame_aabb(aabb, animate=animate)
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
        self.update()

    # -------------------------------------------------------------------------
    # Mouse Interaction Events
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_orbiting = True
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._is_panning = True
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
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
        if event.button() == Qt.MouseButton.LeftButton:
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
        if event.key() == Qt.Key.Key_F:
            self.focus_requested.emit()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_A:
            self.frame_scene()
            event.accept()
            return
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
