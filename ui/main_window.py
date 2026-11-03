"""
ui/main_window.py
Primary application window for UniView 2.0 - Minimalist 3D Asset & Rig Visualizer.
Features:
- Minimalist ("Juicy Simple") UI with runtime Dark Mode & Light Mode switching
- Sleek Top Header Bar with Brand Badge, Active File Pill, Quick Viewport Toggles, and Theme Switcher
- Central ModernGL 3D Viewport with High-DPI & Theme-Adaptive Studio Background/Grid
- Right Inspector Panel featuring the Expanding Label Stadium Tab Bar:
  - Files (Dependency Analyzer with drag-and-drop texture relinking)
  - Materials (Color & Opacity overrides with live QColorDialog)
  - Meshes (Sub-mesh inspector with Search filter, Focus/Isolate & Visibility eye)
  - Skeleton (Armature tree with Search filter, X-Ray overlay, Transform inspector, FK sliders & Heatmap)
"""

from __future__ import annotations
import os
import sys
from typing import Optional, List
from PyQt6.QtCore import Qt, QSize, QSettings
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QIcon, QDragEnterEvent, QDropEvent, QCloseEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton,
    QFrame,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
)

from core.scene_loader import SceneLoader, SceneData, SUPPORTED_EXTENSIONS, get_file_dialog_filter
from core.camera import ArcballCamera
from core.math_utils import AABB
from samples.generate_samples import generate_sample_assets

from ui.style import ThemeManager
from ui.custom_widgets import ExpandingTabWidget
from ui.gl_viewport import GLViewport
from ui.files_panel import FilesPanel
from ui.materials_panel import MaterialsPanel
from ui.meshes_panel import MeshesPanel
from ui.skeleton_panel import SkeletonPanel


class MainWindow(QMainWindow):
    """Main Application Window for UniView 2.0."""

    def __init__(self, initial_filepath: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("UniView - 3D Asset & Rig Visualizer")
        self.resize(1400, 900)
        self.setMinimumSize(800, 500)
        self.setAcceptDrops(True)

        base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        icon_path = os.path.join(base_dir, "docs", "images", "logo.ico")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.current_scene: Optional[SceneData] = None
        self.settings = QSettings("UniView", "UniView")
        self.recent_files: List[str] = self._load_recent_files()

        # ---------------------------------------------------------------------
        # 1. Central Workspace (Minimalist Header Bar + 3D OpenGL Viewport)
        # ---------------------------------------------------------------------
        central_container = QWidget(self)
        central_layout = QVBoxLayout(central_container)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self.header_bar = self._create_top_header_bar()
        central_layout.addWidget(self.header_bar)

        self.viewport = GLViewport(self)
        central_layout.addWidget(self.viewport, stretch=1)

        self.setCentralWidget(central_container)

        # ---------------------------------------------------------------------
        # 2. Right Dock Widget with Expanding Label Tab Bar
        # ---------------------------------------------------------------------
        self.dock = QDockWidget("Inspector", self)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        # Minimalist empty title bar widget so the Expanding Label Stadium sits cleanly at the top
        self.dock.setTitleBarWidget(QWidget(self.dock))

        self.tabs = ExpandingTabWidget(self)

        self.files_panel = FilesPanel(self)
        self.materials_panel = MaterialsPanel(self)
        self.meshes_panel = MeshesPanel(self)
        self.skeleton_panel = SkeletonPanel(self)

        self.tabs.addTab(self.files_panel, "Files", icon_kind="files")
        self.tabs.addTab(self.materials_panel, "Materials", icon_kind="materials")
        self.tabs.addTab(self.meshes_panel, "Meshes", icon_kind="meshes")
        self.tabs.addTab(self.skeleton_panel, "Skeleton", icon_kind="skeleton")

        self.dock.setWidget(self.tabs)
        self.dock.setMinimumWidth(375)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

        # ---------------------------------------------------------------------
        # 3. Minimalist Status Bar
        # ---------------------------------------------------------------------
        self.status_bar = self.statusBar()
        self._init_status_bar()

        # ---------------------------------------------------------------------
        # 4. Menu Bar & Actions
        # ---------------------------------------------------------------------
        self._create_menus_and_actions()

        # ---------------------------------------------------------------------
        # 5. Connect UI Signals & Sync Initial Theme (restored from QSettings)
        # ---------------------------------------------------------------------
        self._connect_signals()
        saved_theme = str(self.settings.value("theme/mode", ThemeManager.current_mode()))
        self.set_theme(saved_theme if saved_theme in ("dark", "light") else "dark")

        saved_geom = self.settings.value("window/geometry")
        if saved_geom is not None:
            try:
                self.restoreGeometry(saved_geom)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 6. Load Initial Scene or Default Base Character
        # ---------------------------------------------------------------------
        target_file = initial_filepath
        if not target_file or not os.path.isfile(target_file):
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            samples_dir = os.path.join(base_dir, "samples")
            if not os.path.isdir(samples_dir):
                samples_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "samples",
                )
            base_char = os.path.join(samples_dir, "UniView_BaseCharacter.fbx")
            if os.path.isfile(base_char):
                target_file = base_char
            else:
                sample_gltf = os.path.join(samples_dir, "UniView_CyberKnight.gltf")
                if not os.path.isfile(sample_gltf):
                    sample_gltf = generate_sample_assets()
                target_file = sample_gltf

        if target_file and os.path.isfile(target_file):
            self.open_file(target_file)

    def _create_top_header_bar(self) -> QFrame:
        """Create the sleek Minimalist Top Header & Quick Action Bar."""
        bar = QFrame(self)
        bar.setObjectName("TopHeaderBar")
        bar.setFixedHeight(46)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(8)

        # Brand badge
        self.lbl_brand = QLabel("UniView")
        self.lbl_brand.setObjectName("BrandBadge")
        layout.addWidget(self.lbl_brand)

        # Quick Open button
        self.btn_header_open = QPushButton("Open Asset")
        self.btn_header_open.setProperty("pill", True)
        self.btn_header_open.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_header_open.setToolTip("Open a 3D asset file (Ctrl+O)")
        self.btn_header_open.clicked.connect(self._on_menu_open_file)
        layout.addWidget(self.btn_header_open)

        # Active file pill
        self.lbl_header_file = QLabel("No Model Loaded")
        self.lbl_header_file.setObjectName("ActiveFilePill")
        layout.addWidget(self.lbl_header_file)

        layout.addStretch()

        # Quick Viewport Pill Toggles
        self.btn_pill_grid = QPushButton("Grid")
        self.btn_pill_grid.setProperty("pill", True)
        self.btn_pill_grid.setCheckable(True)
        self.btn_pill_grid.setChecked(True)
        self.btn_pill_grid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pill_grid.setToolTip("Toggle Ground Grid (G)")
        layout.addWidget(self.btn_pill_grid)

        self.btn_pill_rig = QPushButton("Rig")
        self.btn_pill_rig.setProperty("pill", True)
        self.btn_pill_rig.setCheckable(True)
        self.btn_pill_rig.setChecked(True)
        self.btn_pill_rig.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pill_rig.setToolTip("Toggle Skeleton Overlay (B)")
        layout.addWidget(self.btn_pill_rig)

        self.btn_pill_xray = QPushButton("X-Ray")
        self.btn_pill_xray.setProperty("pill", True)
        self.btn_pill_xray.setCheckable(True)
        self.btn_pill_xray.setChecked(True)
        self.btn_pill_xray.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pill_xray.setToolTip("Toggle Skeleton X-Ray Mode (X)")
        layout.addWidget(self.btn_pill_xray)

        self.btn_pill_focus = QPushButton("Focus (F)")
        self.btn_pill_focus.setProperty("pill", True)
        self.btn_pill_focus.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pill_focus.setToolTip("Focus camera on selected Bone, Mesh, or Material (F)")
        layout.addWidget(self.btn_pill_focus)

        # Theme Switcher Pill (Dark / Light mode)
        self.btn_theme_toggle = QPushButton("🌙 Dark")
        self.btn_theme_toggle.setProperty("pill", True)
        self.btn_theme_toggle.setProperty("primary", True)
        self.btn_theme_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme_toggle.setToolTip("Switch between Dark Mode and Light Mode (Ctrl+T)")
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        layout.addWidget(self.btn_theme_toggle)

        # Shortcuts Help Pill (?)
        self.btn_help_shortcuts = QPushButton("?")
        self.btn_help_shortcuts.setProperty("pill", True)
        self.btn_help_shortcuts.setFixedWidth(30)
        self.btn_help_shortcuts.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_help_shortcuts.setToolTip("Keyboard Shortcuts & Navigation Cheat-Sheet (F1 / ?)")
        self.btn_help_shortcuts.clicked.connect(self._show_shortcuts_dialog)
        layout.addWidget(self.btn_help_shortcuts)

        return bar

    def _init_status_bar(self) -> None:
        """Create clean minimalist status bar indicators."""
        self.lbl_status_file = QLabel("Ready")
        self.lbl_status_file.setStyleSheet("font-weight: 600; padding-left: 6px;")

        self.lbl_status_stats = QLabel("No model loaded")
        self.lbl_status_stats.setStyleSheet("padding-left: 12px;")

        self.lbl_status_fps = QLabel("FPS: --")
        self.lbl_status_fps.setStyleSheet("font-weight: 700; padding-right: 12px;")

        self.lbl_status_camera = QLabel("Target: (0.00, 0.00, 0.00) | Dist: 0.00")
        self.lbl_status_camera.setStyleSheet("padding-right: 12px;")

        self.status_bar.addWidget(self.lbl_status_file)
        self.status_bar.addWidget(self.lbl_status_stats)
        self.status_bar.addPermanentWidget(self.lbl_status_camera)
        self.status_bar.addPermanentWidget(self.lbl_status_fps)

    def _create_menus_and_actions(self) -> None:
        """Construct top menu bar with keyboard shortcuts and Theme Settings."""
        menubar = self.menuBar()

        # ------------------ File Menu ------------------
        menu_file = menubar.addMenu("&File")

        act_open = QAction("&Open 3D Asset...", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_menu_open_file)
        menu_file.addAction(act_open)

        self.menu_recent = menu_file.addMenu("Open &Recent")
        self._rebuild_recent_files_menu()

        act_load_sample = QAction("Load Default Base Character", self)
        act_load_sample.triggered.connect(self._on_load_sample_clicked)
        menu_file.addAction(act_load_sample)

        menu_file.addSeparator()

        act_relink_folder = QAction("Scan Folder for Missing Textures...", self)
        act_relink_folder.triggered.connect(self.files_panel._scan_folder_for_missing)
        menu_file.addAction(act_relink_folder)

        menu_file.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # ------------------ View Menu ------------------
        menu_view = menubar.addMenu("&View")

        act_focus_selected = QAction("&Focus Selected", self)
        act_focus_selected.setShortcut(QKeySequence(Qt.Key.Key_F))
        act_focus_selected.setStatusTip("Focus camera on the selected Bone, Mesh, or Material (F)")
        act_focus_selected.triggered.connect(self.focus_selection)
        menu_view.addAction(act_focus_selected)

        act_frame_scene = QAction("Frame &All / Model", self)
        act_frame_scene.setShortcuts([QKeySequence(Qt.Key.Key_A), QKeySequence("Shift+F")])
        act_frame_scene.setStatusTip("Frame the entire 3D model in the viewport (A / Shift+F)")
        act_frame_scene.triggered.connect(self.viewport.frame_scene)
        menu_view.addAction(act_frame_scene)

        act_reset_cam = QAction("&Reset Camera", self)
        act_reset_cam.setShortcut(QKeySequence(Qt.Key.Key_Home))
        act_reset_cam.setStatusTip("Reset camera to initial home framing (Home)")
        act_reset_cam.triggered.connect(self.viewport.reset_camera)
        menu_view.addAction(act_reset_cam)

        menu_view.addSeparator()

        self.act_grid = QAction("Ground &Grid", self, checkable=True)
        self.act_grid.setChecked(True)
        self.act_grid.setShortcut(QKeySequence(Qt.Key.Key_G))
        self.act_grid.toggled.connect(self._on_grid_toggled)
        menu_view.addAction(self.act_grid)

        self.act_gizmo = QAction("3D &Gizmo Axes", self, checkable=True)
        self.act_gizmo.setChecked(True)
        self.act_gizmo.toggled.connect(self.viewport.set_show_gizmo)
        menu_view.addAction(self.act_gizmo)

        menu_view.addSeparator()

        # Theme Submenu (Dark Mode / Light Mode)
        menu_theme = menu_view.addMenu("🎨 UI &Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)

        self.act_theme_dark = QAction("🌙 Dark Mode (Obsidian Indigo)", self, checkable=True)
        self.act_theme_dark.setChecked(True)
        self.act_theme_dark.triggered.connect(lambda: self.set_theme("dark"))
        theme_group.addAction(self.act_theme_dark)
        menu_theme.addAction(self.act_theme_dark)

        self.act_theme_light = QAction("☀️ Light Mode (Minimalist Alabaster)", self, checkable=True)
        self.act_theme_light.triggered.connect(lambda: self.set_theme("light"))
        theme_group.addAction(self.act_theme_light)
        menu_theme.addAction(self.act_theme_light)

        menu_theme.addSeparator()
        act_toggle_theme = QAction("Toggle Dark / Light Mode", self)
        act_toggle_theme.setShortcut(QKeySequence("Ctrl+T"))
        act_toggle_theme.triggered.connect(self.toggle_theme)
        menu_theme.addAction(act_toggle_theme)

        menu_view.addSeparator()

        act_fullscreen = QAction("&Toggle Fullscreen", self)
        act_fullscreen.setShortcut(QKeySequence(Qt.Key.Key_F11))
        act_fullscreen.triggered.connect(self._toggle_fullscreen)
        menu_view.addAction(act_fullscreen)

        # ------------------ Shading Menu ------------------
        menu_shading = menubar.addMenu("&Shading")

        act_mode_textured = QAction("&1: Textured Mode", self)
        act_mode_textured.setShortcut(QKeySequence(Qt.Key.Key_1))
        act_mode_textured.triggered.connect(lambda: self._set_shading_mode(0))
        menu_shading.addAction(act_mode_textured)

        act_mode_base = QAction("&2: Base Color / No Material", self)
        act_mode_base.setShortcut(QKeySequence(Qt.Key.Key_2))
        act_mode_base.triggered.connect(lambda: self._set_shading_mode(1))
        menu_shading.addAction(act_mode_base)

        act_mode_heatmap = QAction("&3: Vertex Weight Heatmap", self)
        act_mode_heatmap.setShortcut(QKeySequence(Qt.Key.Key_3))
        act_mode_heatmap.triggered.connect(lambda: self._set_shading_mode(2))
        menu_shading.addAction(act_mode_heatmap)

        act_mode_unlit = QAction("&4: Unlit Flat", self)
        act_mode_unlit.setShortcut(QKeySequence(Qt.Key.Key_4))
        act_mode_unlit.triggered.connect(lambda: self._set_shading_mode(3))
        menu_shading.addAction(act_mode_unlit)

        # ------------------ Skeleton Menu ------------------
        menu_skel = menubar.addMenu("&Skeleton")

        self.act_bones = QAction("&Show Bones Overlay", self, checkable=True)
        self.act_bones.setChecked(True)
        self.act_bones.setShortcut(QKeySequence(Qt.Key.Key_B))
        self.act_bones.toggled.connect(self._on_menu_bones_toggled)
        menu_skel.addAction(self.act_bones)

        self.act_xray = QAction("&X-Ray Mode", self, checkable=True)
        self.act_xray.setChecked(True)
        self.act_xray.setShortcut(QKeySequence(Qt.Key.Key_X))
        self.act_xray.toggled.connect(self._on_menu_xray_toggled)
        menu_skel.addAction(self.act_xray)

        menu_skel.addSeparator()

        act_reset_bind = QAction("&Reset Bind Pose", self)
        act_reset_bind.setShortcut(QKeySequence("Ctrl+R"))
        act_reset_bind.triggered.connect(self.skeleton_panel._on_reset_bind_pose_clicked)
        menu_skel.addAction(act_reset_bind)

        # ------------------ Help Menu ------------------
        menu_help = menubar.addMenu("&Help")

        act_shortcuts = QAction("⌨️ &Keyboard Shortcuts", self)
        act_shortcuts.setShortcuts([QKeySequence(Qt.Key.Key_F1), QKeySequence("?")])
        act_shortcuts.triggered.connect(self._show_shortcuts_dialog)
        menu_help.addAction(act_shortcuts)

        act_formats = QAction("&Supported Formats", self)
        act_formats.triggered.connect(self._show_supported_formats_dialog)
        menu_help.addAction(act_formats)

        act_about = QAction("&About UniView", self)
        act_about.triggered.connect(self._show_about_dialog)
        menu_help.addAction(act_about)

    def _connect_signals(self) -> None:
        """Wire all signals between header bar, viewport, and dock inspector panels."""
        # Top Header Bar signals
        self.btn_pill_grid.toggled.connect(self.act_grid.setChecked)
        self.btn_pill_rig.toggled.connect(self.act_bones.setChecked)
        self.btn_pill_xray.toggled.connect(self.act_xray.setChecked)
        self.btn_pill_focus.clicked.connect(self.focus_selection)

        # Viewport signals
        self.viewport.file_dropped.connect(self.open_file)
        self.viewport.fps_updated.connect(lambda fps: self.lbl_status_fps.setText(f"FPS: {fps:.1f}"))
        self.viewport.camera_updated.connect(self._on_camera_updated)
        self.viewport.focus_requested.connect(self.focus_selection)
        self.viewport.render_mode_requested.connect(self._set_shading_mode)

        # Files Panel signals
        self.files_panel.dependency_relinked.connect(self._on_dependency_relinked)
        self.files_panel.refresh_viewport_requested.connect(self.viewport.refresh_materials)

        # Materials Panel signals
        self.materials_panel.render_mode_changed.connect(self._set_shading_mode)
        self.materials_panel.material_modified.connect(self.viewport.refresh_materials)
        self.materials_panel.focus_requested.connect(self.focus_selection)

        # Meshes Panel signals
        self.meshes_panel.isolate_toggled.connect(self._on_mesh_isolated)
        self.meshes_panel.visibility_toggled.connect(lambda idx, vis: self.viewport.update())
        self.meshes_panel.refresh_viewport_requested.connect(self.viewport.update)
        self.meshes_panel.focus_requested.connect(self.focus_selection)

        # Skeleton Panel signals
        self.skeleton_panel.active_bone_changed.connect(self.viewport.set_active_bone_id)
        self.skeleton_panel.overlay_options_changed.connect(self._on_skeleton_overlays_changed)
        self.skeleton_panel.fk_rotation_changed.connect(self.viewport.update)
        self.skeleton_panel.bind_pose_reset.connect(lambda: self.viewport.show_toast("Bind pose restored", kind="info"))
        self.skeleton_panel.refresh_viewport_requested.connect(self.viewport.update)
        self.skeleton_panel.focus_requested.connect(self.focus_selection)

    # -------------------------------------------------------------------------
    # Theme Engine Control (Dark Mode / Light Mode)
    # -------------------------------------------------------------------------

    def set_theme(self, mode: str) -> None:
        """Switch the entire application and 3D viewport between Dark and Light mode."""
        palette = ThemeManager.set_mode(mode)
        qss = ThemeManager.stylesheet(palette.name)

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qss)
        else:
            self.setStyleSheet(qss)

        # Update 3D OpenGL Viewport background and ground grid colors
        self.viewport.set_theme_colors(palette.viewport_bg_rgb, palette.grid_line_rgb)

        # Sync Header Toggle Button & Menu Actions
        if palette.is_dark:
            self.btn_theme_toggle.setText("🌙 Dark")
            self.act_theme_dark.setChecked(True)
        else:
            self.btn_theme_toggle.setText("☀️ Light")
            self.act_theme_light.setChecked(True)

        # Refresh custom-painted widgets
        self.tabs.refresh_theme()
        self.files_panel.refresh_theme()
        self.materials_panel.refresh_theme()
        self.meshes_panel.refresh_theme()
        self.skeleton_panel.refresh_theme()
        self.settings.setValue("theme/mode", palette.name)

    def toggle_theme(self) -> None:
        """Toggle between Dark Mode and Light Mode."""
        next_mode = "light" if ThemeManager.is_dark() else "dark"
        self.set_theme(next_mode)
        mode_label = "Dark Mode" if ThemeManager.is_dark() else "Light Mode"
        self.viewport.show_toast(f"Theme: {mode_label}", kind="info")

    # -------------------------------------------------------------------------
    # Scene & Focus Operations
    # -------------------------------------------------------------------------

    def focus_selection(self) -> None:
        """
        Focus camera on the currently selected item (Bone, Mesh, or Material)
        WITHOUT altering mesh visibility or hiding anything.
        Falls back to framing the full model if nothing specific is selected.
        """
        if not self.current_scene:
            return

        current_tab = self.tabs.currentWidget()

        # 1. If currently on Skeleton tab
        if current_tab == self.skeleton_panel:
            bone = self.skeleton_panel.get_selected_bone()
            if bone is not None:
                scene_rad = self.current_scene.scene_aabb.radius if self.current_scene.scene_aabb.is_valid else 1.0
                bone_aabb = self.skeleton_panel.get_bone_focus_aabb(bone, scene_rad)
                if bone_aabb.is_valid:
                    self.viewport.frame_aabb(bone_aabb, animate=True)
                    return

        # 2. If currently on Meshes tab
        elif current_tab == self.meshes_panel:
            sub = self.meshes_panel.get_selected_submesh()
            if sub is not None and sub.aabb.is_valid:
                self.viewport.frame_aabb(sub.aabb, animate=True)
                return

        # 3. If currently on Materials tab
        elif current_tab == self.materials_panel:
            mat_idx = self.materials_panel.selected_material_index
            mat_aabb = AABB.empty()
            for sub in self.current_scene.submeshes:
                if sub.material_index == mat_idx and sub.aabb.is_valid:
                    mat_aabb = mat_aabb.merge(sub.aabb)
            if mat_aabb.is_valid:
                self.viewport.frame_aabb(mat_aabb, animate=True)
                return

        # 4. Contextual fallback if on Viewport or Files tab:
        if self.current_scene.skeleton.has_bones and self.current_scene.skeleton.selected_bone_id >= 0:
            bone = self.skeleton_panel.get_selected_bone()
            if bone is not None:
                scene_rad = self.current_scene.scene_aabb.radius if self.current_scene.scene_aabb.is_valid else 1.0
                bone_aabb = self.skeleton_panel.get_bone_focus_aabb(bone, scene_rad)
                if bone_aabb.is_valid:
                    self.viewport.frame_aabb(bone_aabb, animate=True)
                    return

        sub = self.meshes_panel.get_selected_submesh()
        if sub is not None and sub.aabb.is_valid:
            self.viewport.frame_aabb(sub.aabb, animate=True)
            return

        # 5. Default fallback: frame entire model
        self.viewport.frame_scene()

    def open_file(self, filepath: str) -> bool:
        """Load and display a 3D asset file."""
        if not os.path.isfile(filepath):
            QMessageBox.critical(self, "File Not Found", f"Cannot open file:\n{filepath}")
            return False

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            if self.isVisible():
                self.viewport.set_loading(True, os.path.basename(filepath))
                QApplication.processEvents()
            scene = SceneLoader.load(filepath)
            self.current_scene = scene

            # Pass scene to viewport and all dock panels
            self.viewport.set_scene(scene, frame_camera=True)
            self.files_panel.set_scene(scene)
            self.materials_panel.set_scene(scene)
            self.meshes_panel.set_scene(scene)
            self.skeleton_panel.set_scene(scene)

            # Update header & status bar
            self.lbl_header_file.setText(f"📄 {scene.filename}")
            self.lbl_status_file.setText(f"📄 {scene.filename}")
            self.lbl_status_stats.setText(
                f"Meshes: {len(scene.submeshes)}  ·  Polys: {scene.total_faces:,}  ·  Verts: {scene.total_vertices:,}"
            )
            self.setWindowTitle(f"UniView - {scene.filename}")

            # Switch to Skeleton tab if model is rigged, else Meshes tab
            if scene.skeleton.has_bones:
                self.tabs.setCurrentWidget(self.skeleton_panel)
            else:
                self.tabs.setCurrentWidget(self.meshes_panel)

            self._add_recent_file(filepath)
            self.viewport.show_toast(f"Loaded {scene.filename}", kind="success")
            return True
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error Loading 3D Model",
                f"Failed to open 3D asset:\n{filepath}\n\nError details:\n{exc}",
            )
            return False
        finally:
            self.viewport.set_loading(False)
            QApplication.restoreOverrideCursor()

    def _on_menu_open_file(self) -> None:
        """Open QFileDialog to browse for a 3D asset file."""
        initial_dir = ""
        if self.current_scene:
            initial_dir = os.path.dirname(self.current_scene.filepath)

        file_filter = get_file_dialog_filter()
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Asset File",
            initial_dir,
            file_filter,
        )
        if filepath:
            self.open_file(filepath)

    def _on_load_sample_clicked(self) -> None:
        """Load default base character asset."""
        base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        samples_dir = os.path.join(base_dir, "samples")
        if not os.path.isdir(samples_dir):
            samples_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "samples",
            )
        base_char = os.path.join(samples_dir, "UniView_BaseCharacter.fbx")
        if os.path.isfile(base_char):
            self.open_file(base_char)
        else:
            sample_path = generate_sample_assets()
            self.open_file(sample_path)

    def _on_dependency_relinked(self, dep_index: int, image_path: str) -> None:
        """Sync relinked textures with Materials panel and Viewport."""
        if self.current_scene:
            self.materials_panel.rebuild_slots_list()
            self.viewport.refresh_materials()
            self.viewport.show_toast(f"Texture relinked: {os.path.basename(image_path)}", kind="success")

    def _on_mesh_isolated(self, submesh_idx: int, is_isolated: bool, aabb: AABB) -> None:
        """Smoothly frame camera onto isolated sub-mesh AABB."""
        self.viewport.frame_aabb(aabb, animate=True)

    def _on_skeleton_overlays_changed(self, show_bones: bool, xray_mode: bool, heatmap_mode: bool) -> None:
        self.viewport.set_show_skeleton(show_bones)
        self.viewport.set_xray_skeleton(xray_mode)
        self.btn_pill_rig.blockSignals(True)
        self.btn_pill_rig.setChecked(show_bones)
        self.btn_pill_rig.blockSignals(False)
        self.btn_pill_xray.blockSignals(True)
        self.btn_pill_xray.setChecked(xray_mode)
        self.btn_pill_xray.blockSignals(False)
        if heatmap_mode:
            self.viewport.set_render_mode(2)
        else:
            current_mode = self.materials_panel.mode_segment.current_index()
            self.viewport.set_render_mode(current_mode)

    def _on_grid_toggled(self, checked: bool) -> None:
        self.viewport.set_show_grid(checked)
        self.btn_pill_grid.blockSignals(True)
        self.btn_pill_grid.setChecked(checked)
        self.btn_pill_grid.blockSignals(False)

    def _on_menu_bones_toggled(self, checked: bool) -> None:
        self.skeleton_panel.chk_show_bones.setChecked(checked)

    def _on_menu_xray_toggled(self, checked: bool) -> None:
        self.skeleton_panel.chk_xray.setChecked(checked)

    def _set_shading_mode(self, mode: int) -> None:
        if mode == 0:
            self.materials_panel.mode_segment.set_current_index(0)
            self.skeleton_panel.chk_heatmap.blockSignals(True)
            self.skeleton_panel.chk_heatmap.setChecked(False)
            self.skeleton_panel.chk_heatmap.blockSignals(False)
        elif mode == 1:
            self.materials_panel.mode_segment.set_current_index(1)
            self.skeleton_panel.chk_heatmap.blockSignals(True)
            self.skeleton_panel.chk_heatmap.setChecked(False)
            self.skeleton_panel.chk_heatmap.blockSignals(False)
        elif mode == 2:
            self.skeleton_panel.chk_heatmap.blockSignals(True)
            self.skeleton_panel.chk_heatmap.setChecked(True)
            self.skeleton_panel.chk_heatmap.blockSignals(False)
        elif mode == 3:
            self.skeleton_panel.chk_heatmap.blockSignals(True)
            self.skeleton_panel.chk_heatmap.setChecked(False)
            self.skeleton_panel.chk_heatmap.blockSignals(False)

        self.viewport.set_render_mode(mode)

    def _on_camera_updated(self, tx: float, ty: float, tz: float, dist: float) -> None:
        self.lbl_status_camera.setText(f"Target: ({tx:+.2f}, {ty:+.2f}, {tz:+.2f}) | Dist: {dist:.2f}")

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _show_supported_formats_dialog(self) -> None:
        formats_text = (
            "<h3>Supported 3D Asset Formats</h3>"
            "<p><b>Tier-1 Rigged & Static Mesh Formats:</b><br>"
            "• <code>.fbx</code> (Autodesk FBX - Blender, Mixamo, Unreal Engine 5)<br>"
            "• <code>.gltf</code>, <code>.glb</code> (glTF 2.0 with PBR materials and skeletal skins)<br>"
            "• <code>.obj</code> (Wavefront OBJ with MTL materials)</p>"
            "<p><b>Extended DCC & CAD Formats:</b><br>"
            "• <code>.dae</code> (Collada), <code>.stl</code> (Stereolithography), <code>.ply</code> (Stanford)<br>"
            "• <code>.3ds</code> (3D Studio), <code>.wrl</code> (VRML), <code>.off</code> (Object File Format)<br>"
            "• <code>.3mf</code>, <code>.amf</code> (Additive Manufacturing)<br>"
            "• <code>.step</code>, <code>.iges</code>, <code>.ifc</code>, <code>.brep</code>, <code>.fcstd</code>, <code>.bim</code>, <code>.3dm</code> (CAD/BIM via extended loaders)</p>"
        )
        QMessageBox.information(self, "Supported Formats", formats_text)

    def _show_about_dialog(self) -> None:
        about_text = (
            "<h2>UniView 2.0</h2>"
            "<p><b>Minimalist 3D Asset & Rig Visualizer</b></p>"
            "<p>Engineered with Python, PyQt6, ModernGL, PyGLM, and PyAssimp.</p>"
            "<p>Features hardware-accelerated OpenGL 3.3+ rendering, Expanding Label Navigation, "
            "Dark & Light Mode studio themes, interactive Forward Kinematics, and vertex weight heatmaps.</p>"
        )
        QMessageBox.about(self, "About UniView", about_text)

    def _show_shortcuts_dialog(self) -> None:
        shortcuts_html = (
            "<h3>⌨️ Keyboard Shortcuts & Navigation</h3>"
            "<table cellspacing='6' cellpadding='2'>"
            "<tr><td><b>Left Mouse Drag</b></td><td>Orbit Camera around Target</td></tr>"
            "<tr><td><b>Right / Middle Drag</b></td><td>Pan Camera Target</td></tr>"
            "<tr><td><b>Mouse Scroll Wheel</b></td><td>Smooth Zoom In / Out</td></tr>"
            "<tr><td><b>F / Double-Click</b></td><td>Focus Camera on Selected Bone, Mesh, or Material</td></tr>"
            "<tr><td><b>A / Shift+F</b></td><td>Frame Entire 3D Model</td></tr>"
            "<tr><td><b>Home</b></td><td>Reset Camera to Home Position</td></tr>"
            "<tr><td><b>1 · 2 · 3 · 4</b></td><td>Shading: Textured · Base · Weight Heatmap · Unlit</td></tr>"
            "<tr><td><b>G</b></td><td>Toggle Ground Grid</td></tr>"
            "<tr><td><b>B</b></td><td>Toggle Skeleton Bones Overlay</td></tr>"
            "<tr><td><b>X</b></td><td>Toggle Skeleton X-Ray Mode</td></tr>"
            "<tr><td><b>Ctrl+R</b></td><td>Reset Skeleton Bind Pose</td></tr>"
            "<tr><td><b>Ctrl+T</b></td><td>Switch Dark Mode / Light Mode</td></tr>"
            "<tr><td><b>Ctrl+O</b></td><td>Open 3D Asset File</td></tr>"
            "<tr><td><b>F11</b></td><td>Toggle Fullscreen</td></tr>"
            "</table>"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_html)

    def _load_recent_files(self) -> List[str]:
        raw = self.settings.value("files/recent", [])
        if isinstance(raw, str):
            raw = [raw] if raw else []
        elif not isinstance(raw, list):
            raw = []
        return [p for p in raw if isinstance(p, str) and os.path.isfile(p)][:8]

    def _add_recent_file(self, filepath: str) -> None:
        norm = os.path.abspath(filepath)
        self.recent_files = [p for p in self.recent_files if os.path.abspath(p) != norm]
        self.recent_files.insert(0, norm)
        self.recent_files = self.recent_files[:8]
        self.settings.setValue("files/recent", self.recent_files)
        self._rebuild_recent_files_menu()

    def _rebuild_recent_files_menu(self) -> None:
        if not hasattr(self, "menu_recent"):
            return
        self.menu_recent.clear()
        if not self.recent_files:
            empty_act = QAction("No Recent Files", self)
            empty_act.setEnabled(False)
            self.menu_recent.addAction(empty_act)
            return
        for path in self.recent_files:
            act = QAction(f"{os.path.basename(path)}  —  {path}", self)
            act.triggered.connect(lambda checked=False, p=path: self.open_file(p))
            self.menu_recent.addAction(act)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.settings.setValue("theme/mode", ThemeManager.current_mode())
        self.settings.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)

    # -------------------------------------------------------------------------
    # Drag-and-Drop 3D File directly onto Main Window
    # -------------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                ext = os.path.splitext(filepath)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and os.path.isfile(filepath):
                    self.open_file(filepath)
                    event.acceptProposedAction()
                    return
        event.ignore()
