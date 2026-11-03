"""
ui/meshes_panel.py
Detailed Sub-Mesh Inspector ("Meshes" Tab) for UniView.
Features:
- Lists every individual sub-mesh node inside the 3D file with poly & vertex counts
- Each row contains two right-aligned action icons:
  1. Focus / Isolate Icon (4-corner expand/target '⛶'):
     Hides all other sub-meshes except the clicked one, and smoothly interpolates (zooms and centers)
     the camera to frame this specific sub-mesh's AABB. Clicking it again restores visibility of all meshes.
  2. Visibility Eye Icon ('👁' / '🚫'):
     Toggles `visible = True/False` for that specific sub-mesh in the OpenGL draw loop.
- Batch "Show All" and "Hide All" toolbar shortcuts
"""

from __future__ import annotations
from typing import Optional, Dict
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
)

from core.scene_loader import SceneData, SubMeshData
from core.math_utils import AABB
from ui.custom_widgets import IconButton as ActionIconButton


class MeshesPanel(QWidget):
    """Sub-Mesh Inspector dock panel."""

    isolate_toggled = pyqtSignal(int, bool, object)    # (submesh_idx, is_isolated, AABB)
    visibility_toggled = pyqtSignal(int, bool)        # (submesh_idx, is_visible)
    submesh_selected = pyqtSignal(int)                # (submesh_idx)
    focus_requested = pyqtSignal()                   # Emitted to zoom/center camera on selected mesh (F)
    refresh_viewport_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_data: Optional[SceneData] = None
        self._selected_submesh_index: Optional[int] = None

        # Track button references for live UI state syncing
        self._isolate_buttons: Dict[int, ActionIconButton] = {}
        self._eye_buttons: Dict[int, ActionIconButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header bar with summary & batch shortcuts
        top_bar = QHBoxLayout()
        self.summary_label = QLabel("No meshes loaded")
        self.summary_label.setStyleSheet("font-weight: 600;")
        top_bar.addWidget(self.summary_label)
        top_bar.addStretch()

        self.btn_focus = QPushButton("Focus (F)")
        self.btn_focus.setProperty("pill", True)
        self.btn_focus.setProperty("primary", True)
        self.btn_focus.setToolTip("Focus camera on the selected sub-mesh (F)")
        self.btn_focus.clicked.connect(self.focus_requested.emit)
        self.btn_focus.setCursor(Qt.CursorShape.PointingHandCursor)
        top_bar.addWidget(self.btn_focus)

        self.btn_show_all = QPushButton("Show All")
        self.btn_show_all.setProperty("pill", True)
        self.btn_show_all.setToolTip("Make all sub-meshes visible")
        self.btn_show_all.clicked.connect(self._show_all_meshes)
        self.btn_show_all.setCursor(Qt.CursorShape.PointingHandCursor)
        top_bar.addWidget(self.btn_show_all)

        self.btn_hide_all = QPushButton("Hide All")
        self.btn_hide_all.setProperty("pill", True)
        self.btn_hide_all.setToolTip("Hide all sub-meshes")
        self.btn_hide_all.clicked.connect(self._hide_all_meshes)
        self.btn_hide_all.setCursor(Qt.CursorShape.PointingHandCursor)
        top_bar.addWidget(self.btn_hide_all)

        layout.addLayout(top_bar)

        # Minimalist Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search sub-meshes...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._filter_meshes)
        layout.addWidget(self.search_input)

        # Sub-Meshes Tree Widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Sub-Mesh Node", "Material", "Actions"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tree.header().resizeSection(2, 64)
        self.tree.setRootIsDecorated(False)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree.itemDoubleClicked.connect(lambda item, col: self.focus_requested.emit())
        layout.addWidget(self.tree)

        # Usage hint banner
        hint_label = QLabel("Double-click or F to focus  ·  corners = isolate  ·  eye = show / hide")
        hint_label.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        layout.addWidget(hint_label)

    def set_scene(self, scene: Optional[SceneData]) -> None:
        """Update panel with sub-meshes from the loaded scene."""
        self.scene_data = scene
        self.rebuild_mesh_list()

    def rebuild_mesh_list(self) -> None:
        """Populate tree with sub-mesh rows and action buttons."""
        self.tree.clear()
        self._isolate_buttons.clear()
        self._eye_buttons.clear()

        if not self.scene_data:
            self.summary_label.setText("No meshes loaded")
            self.btn_show_all.setEnabled(False)
            self.btn_hide_all.setEnabled(False)
            return

        self.btn_show_all.setEnabled(True)
        self.btn_hide_all.setEnabled(True)

        num_sub = len(self.scene_data.submeshes)
        total_poly = self.scene_data.total_faces
        total_vert = self.scene_data.total_vertices
        self.summary_label.setText(f"Meshes: {num_sub} | Polys: {total_poly:,} | Verts: {total_vert:,}")

        for sub in self.scene_data.submeshes:
            item = QTreeWidgetItem(self.tree)
            item.setData(0, Qt.ItemDataRole.UserRole, sub.index)
            item.setSizeHint(0, QSize(0, 30))
            item.setSizeHint(2, QSize(60, 30))

            # Column 0: Node Name with Poly & Vertex info
            item.setText(0, sub.name)
            item.setToolTip(0, f"Vertices: {sub.vertex_count:,}\nFaces: {sub.face_count:,}\nSkinned: {sub.has_bones}")

            # Column 1: Material name tag
            mat_name = "Default"
            if 0 <= sub.material_index < len(self.scene_data.materials):
                mat_name = self.scene_data.materials[sub.material_index].name
            item.setText(1, f"[{sub.material_index}] {mat_name}")
            item.setForeground(1, Qt.GlobalColor.gray)

            # Column 2: Two right-aligned action buttons (Focus/Isolate + Visibility Eye)
            actions_widget = QWidget()
            actions_widget.setStyleSheet("background: transparent;")
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(4)
            actions_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # 1. Focus / Isolate Icon (4-corner expand/target '⛶')
            is_isolated = (self.scene_data.isolated_submesh_index == sub.index)
            isolate_btn = ActionIconButton(
                "isolate",
                tooltip="Focus & Isolate this sub-mesh (hides others, frames AABB)",
                active=is_isolated,
            )
            isolate_btn.clicked.connect(lambda checked, idx=sub.index: self._on_isolate_clicked(idx))
            self._isolate_buttons[sub.index] = isolate_btn
            actions_layout.addWidget(isolate_btn)

            # 2. Visibility Eye Icon ('👁' / '🚫')
            eye_btn = ActionIconButton(
                "eye" if sub.visible else "eye_off",
                tooltip="Toggle mesh visibility",
                active=sub.visible,
            )
            eye_btn.clicked.connect(lambda checked, idx=sub.index: self._on_visibility_clicked(idx))
            self._eye_buttons[sub.index] = eye_btn
            actions_layout.addWidget(eye_btn)

            self.tree.setItemWidget(item, 2, actions_widget)

    def get_selected_submesh(self) -> Optional[SubMeshData]:
        """Return the currently selected SubMeshData, or the first mesh if none selected."""
        if not self.scene_data or not self.scene_data.submeshes:
            return None
        items = self.tree.selectedItems()
        if items:
            idx = items[0].data(0, Qt.ItemDataRole.UserRole)
            if idx is not None and 0 <= idx < len(self.scene_data.submeshes):
                return self.scene_data.submeshes[idx]
        if self._selected_submesh_index is not None and 0 <= self._selected_submesh_index < len(self.scene_data.submeshes):
            return self.scene_data.submeshes[self._selected_submesh_index]
        return self.scene_data.submeshes[0]

    def _on_tree_selection_changed(self) -> None:
        """Track user-selected sub-mesh row."""
        items = self.tree.selectedItems()
        if items:
            idx = items[0].data(0, Qt.ItemDataRole.UserRole)
            if idx is not None and isinstance(idx, int):
                self._selected_submesh_index = idx
                self.submesh_selected.emit(idx)

    def _on_isolate_clicked(self, submesh_idx: int) -> None:
        """Handle Focus/Isolate button click on sub-mesh."""
        if not self.scene_data:
            return

        was_isolated, target_aabb = self.scene_data.toggle_isolate_submesh(submesh_idx)

        # Update button visual states
        for idx, btn in self._isolate_buttons.items():
            btn.set_active(idx == self.scene_data.isolated_submesh_index)

        # Update eye button states to reflect current mesh visibility
        for sub in self.scene_data.submeshes:
            eye_btn = self._eye_buttons.get(sub.index)
            if eye_btn:
                eye_btn.set_kind("eye" if sub.visible else "eye_off")
                eye_btn.set_active(sub.visible)

        # Notify viewport to smoothly interpolate camera to AABB and redraw
        self.isolate_toggled.emit(submesh_idx, was_isolated, target_aabb)
        self.refresh_viewport_requested.emit()

    def _on_visibility_clicked(self, submesh_idx: int) -> None:
        """Handle Visibility Eye icon click on sub-mesh."""
        if not self.scene_data or submesh_idx < 0 or submesh_idx >= len(self.scene_data.submeshes):
            return

        sub = self.scene_data.submeshes[submesh_idx]
        sub.visible = not sub.visible

        eye_btn = self._eye_buttons.get(submesh_idx)
        if eye_btn:
            eye_btn.set_kind("eye" if sub.visible else "eye_off")
            eye_btn.set_active(sub.visible)

        self.visibility_toggled.emit(submesh_idx, sub.visible)
        self.refresh_viewport_requested.emit()

    def _show_all_meshes(self) -> None:
        """Set all sub-meshes visible."""
        if not self.scene_data:
            return
        self.scene_data.isolated_submesh_index = None
        for sub in self.scene_data.submeshes:
            sub.visible = True

        for idx, btn in self._isolate_buttons.items():
            btn.set_active(False)
        for idx, btn in self._eye_buttons.items():
            btn.set_kind("eye")
            btn.set_active(True)

        self.refresh_viewport_requested.emit()

    def _hide_all_meshes(self) -> None:
        """Set all sub-meshes hidden."""
        if not self.scene_data:
            return
        self.scene_data.isolated_submesh_index = None
        for sub in self.scene_data.submeshes:
            sub.visible = False

        for idx, btn in self._isolate_buttons.items():
            btn.set_active(False)
        for idx, btn in self._eye_buttons.items():
            btn.set_kind("eye_off")
            btn.set_active(False)

        self.refresh_viewport_requested.emit()

    def _filter_meshes(self, query: str) -> None:
        """Fast in-place filter of sub-mesh rows without rebuilding the tree."""
        q = query.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            match = (not q) or (q in item.text(0).lower()) or (q in item.text(1).lower())
            item.setHidden(not match)

    def refresh_theme(self) -> None:
        for btn in self._isolate_buttons.values():
            btn._apply_style()
        for btn in self._eye_buttons.values():
            btn._apply_style()
