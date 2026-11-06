"""
ui/skeleton_panel.py
Skeletal & Rigging Inspector Panel ("Skeleton" Tab) for UniView.
Features:
- Bone Hierarchy Tree (Parent-Child tree structure from root bones to finger/tip bones)
  - Selecting a bone highlights it in the 3D viewport (bright orange)
- Overlay Controls:
  - Checkbox: "Show Bones Overlay"
  - Checkbox: "X-Ray Mode (See through mesh geometry)"
  - Checkbox: "Vertex Weight Heatmap" (Blue -> Cyan -> Green -> Yellow -> Red gradient)
- Transform Inspector:
  - Toggle between [ Local Space ] and [ World Space ]
  - Position: X, Y, Z
  - Rotation: Euler XYZ in degrees + Quaternion (W, X, Y, Z)
  - Scale: X, Y, Z
- Interactive Forward Kinematics (FK) Rotation Sliders:
  - Sliders for X, Y, Z rotation (-180° to +180°) with live Forward Kinematics deformation
  - "↺ Reset Bind Pose" button to restore rest pose across all bones
"""

from __future__ import annotations
from typing import Optional, Dict
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QGroupBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QSlider,
    QFormLayout,
    QHeaderView,
)
import glm

from core.math_utils import AABB
from core.scene_loader import SceneData
from core.skeleton import Skeleton, BoneNode
from ui.custom_widgets import ToggleSwitch, SegmentedControl


class SkeletonPanel(QWidget):
    """Skeletal & Rigging Inspector dock panel."""

    active_bone_changed = pyqtSignal(int)             # Selected bone ID
    overlay_options_changed = pyqtSignal(bool, bool, bool)  # (show_bones, xray_mode, heatmap_mode)
    fk_rotation_changed = pyqtSignal()               # Emitted when a bone is rotated via sliders
    bind_pose_reset = pyqtSignal()                   # Emitted when bind pose is restored
    focus_requested = pyqtSignal()                   # Emitted to zoom/center camera on selected bone (F)
    refresh_viewport_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_data: Optional[SceneData] = None
        self._selected_bone: Optional[BoneNode] = None
        self._world_space: bool = False
        self._updating_ui: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Empty-state banner when a loaded model is a static mesh (no bones)
        self.empty_banner = QLabel("Static Mesh — No Skeleton / Armature in this asset")
        self.empty_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_banner.setStyleSheet(
            "padding: 10px 12px; border-radius: 10px; font-weight: 600; background: rgba(124, 92, 252, 0.12);"
        )
        self.empty_banner.hide()
        layout.addWidget(self.empty_banner)

        # ---------------------------------------------------------------------
        # 1. Overlay Display Controls
        # ---------------------------------------------------------------------
        overlay_box = QGroupBox("Armature && Rig Overlays")
        overlay_layout = QVBoxLayout(overlay_box)
        overlay_layout.setContentsMargins(12, 14, 12, 10)
        overlay_layout.setSpacing(8)

        row_toggles = QHBoxLayout()
        self.chk_show_bones = ToggleSwitch("Show Bones")
        self.chk_show_bones.setChecked(True)
        self.chk_show_bones.toggled.connect(self._on_overlay_toggled)
        row_toggles.addWidget(self.chk_show_bones)

        self.chk_xray = ToggleSwitch("X-Ray")
        self.chk_xray.setToolTip("Render bones visible through mesh geometry")
        self.chk_xray.setChecked(True)
        self.chk_xray.toggled.connect(self._on_overlay_toggled)
        row_toggles.addWidget(self.chk_xray)
        row_toggles.addStretch()
        overlay_layout.addLayout(row_toggles)

        self.chk_heatmap = ToggleSwitch("Weight Heatmap")
        self.chk_heatmap.setToolTip("Colorize mesh vertices based on active bone weight (Blue = 0.0, Red = 1.0)")
        self.chk_heatmap.setChecked(False)
        self.chk_heatmap.toggled.connect(self._on_heatmap_toggled)
        overlay_layout.addWidget(self.chk_heatmap)

        layout.addWidget(overlay_box)

        # ---------------------------------------------------------------------
        # 2. Bone Hierarchy Tree
        # ---------------------------------------------------------------------
        tree_box = QGroupBox("Bone Hierarchy")
        tree_layout = QVBoxLayout(tree_box)
        tree_layout.setContentsMargins(8, 12, 8, 8)
        tree_layout.setSpacing(6)

        self.search_bone_input = QLineEdit()
        self.search_bone_input.setPlaceholderText("🔍 Search bone hierarchy...")
        self.search_bone_input.setClearButtonEnabled(True)
        self.search_bone_input.textChanged.connect(self._filter_bones)
        tree_layout.addWidget(self.search_bone_input)

        self.bone_tree = QTreeWidget()
        self.bone_tree.setColumnCount(1)
        self.bone_tree.setHeaderHidden(True)
        self.bone_tree.setIndentation(12)
        self.bone_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.bone_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.bone_tree.itemDoubleClicked.connect(lambda item, col: self.focus_requested.emit())
        tree_layout.addWidget(self.bone_tree)

        layout.addWidget(tree_box, stretch=1)

        # ---------------------------------------------------------------------
        # 3. Transform Inspector (Local vs World Space)
        # ---------------------------------------------------------------------
        transform_box = QGroupBox("Transform Inspector")
        trans_outer_layout = QVBoxLayout(transform_box)
        trans_outer_layout.setContentsMargins(8, 12, 8, 8)
        trans_outer_layout.setSpacing(6)

        # Space Selector Toggle: [ Local Space ] vs [ World Space ] + Focus Button
        space_bar = QHBoxLayout()
        self.space_segment = SegmentedControl(["Local", "World"])
        self.space_segment.setToolTip("Coordinate space for the readouts below")
        self.space_segment.setMaximumWidth(170)
        self.space_segment.currentChanged.connect(self._on_space_radio_clicked)
        space_bar.addWidget(self.space_segment)
        space_bar.addStretch()

        self.btn_focus_bone = QPushButton("Focus (F)")
        self.btn_focus_bone.setProperty("pill", True)
        self.btn_focus_bone.setProperty("primary", True)
        self.btn_focus_bone.setToolTip("Focus camera on the selected bone (F)")
        self.btn_focus_bone.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_focus_bone.clicked.connect(self.focus_requested.emit)
        space_bar.addWidget(self.btn_focus_bone)

        trans_outer_layout.addLayout(space_bar)

        # Readouts Form
        readout_layout = QFormLayout()
        readout_layout.setContentsMargins(4, 4, 4, 4)
        readout_layout.setSpacing(4)
        readout_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.lbl_pos = QLabel("X: 0.00  Y: 0.00  Z: 0.00")
        self.lbl_pos.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: 600;")
        readout_layout.addRow("Position:", self.lbl_pos)

        self.lbl_euler = QLabel("X: 0.0°  Y: 0.0°  Z: 0.0°")
        self.lbl_euler.setStyleSheet("color: #7C5CFC; font-family: 'Consolas', monospace; font-size: 11px; font-weight: 600;")
        readout_layout.addRow("Euler XYZ:", self.lbl_euler)

        self.lbl_quat = QLabel("W: 1.000  X: 0.000  Y: 0.000  Z: 0.000")
        self.lbl_quat.setStyleSheet("font-family: 'Consolas', monospace; font-size: 10px;")
        readout_layout.addRow("Quaternion:", self.lbl_quat)

        self.lbl_scale = QLabel("X: 1.00  Y: 1.00  Z: 1.00")
        self.lbl_scale.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        readout_layout.addRow("Scale:", self.lbl_scale)

        trans_outer_layout.addLayout(readout_layout)
        layout.addWidget(transform_box, stretch=0)

        # ---------------------------------------------------------------------
        # 4. Interactive Forward Kinematics (FK) Rotation & Reset Bind Pose
        # ---------------------------------------------------------------------
        fk_box = QGroupBox("Interactive FK Rotation (Local)")
        fk_layout = QVBoxLayout(fk_box)
        fk_layout.setContentsMargins(8, 12, 8, 8)
        fk_layout.setSpacing(6)

        def make_slider_row(axis_name: str, color_hex: str):
            row = QHBoxLayout()
            lbl_tag = QLabel(axis_name)
            lbl_tag.setStyleSheet(f"color: {color_hex}; font-weight: bold; width: 14px;")
            row.addWidget(lbl_tag)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-180, 180)
            slider.setValue(0)
            row.addWidget(slider)

            lbl_val = QLabel("0°")
            lbl_val.setFixedWidth(40)
            lbl_val.setStyleSheet("font-family: 'Consolas', monospace; font-weight: 600;")
            row.addWidget(lbl_val)
            return row, slider, lbl_val

        # X Slider (Pitch / Red)
        rx_row, self.slider_rx, self.lbl_rx_val = make_slider_row("X:", "#ff5555")
        self.slider_rx.valueChanged.connect(self._on_fk_slider_changed)
        fk_layout.addLayout(rx_row)

        # Y Slider (Yaw / Green)
        ry_row, self.slider_ry, self.lbl_ry_val = make_slider_row("Y:", "#55ff55")
        self.slider_ry.valueChanged.connect(self._on_fk_slider_changed)
        fk_layout.addLayout(ry_row)

        # Z Slider (Roll / Blue)
        rz_row, self.slider_rz, self.lbl_rz_val = make_slider_row("Z:", "#5599ff")
        self.slider_rz.valueChanged.connect(self._on_fk_slider_changed)
        fk_layout.addLayout(rz_row)

        # Reset Bind Pose Button
        btn_row = QHBoxLayout()
        self.btn_reset_bind = QPushButton("↺ Reset Bind Pose")
        self.btn_reset_bind.setToolTip("Restore all bones in the armature to their original bind pose")
        self.btn_reset_bind.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_bind.clicked.connect(self._on_reset_bind_pose_clicked)
        btn_row.addWidget(self.btn_reset_bind)
        fk_layout.addLayout(btn_row)

        layout.addWidget(fk_box, stretch=0)

    def set_scene(self, scene: Optional[SceneData]) -> None:
        """Populate bone hierarchy tree from the loaded scene."""
        self.scene_data = scene
        self.rebuild_bone_tree()

    def rebuild_bone_tree(self) -> None:
        """Construct hierarchical bone tree in QTreeWidget."""
        self.bone_tree.clear()
        self._selected_bone = None

        if not self.scene_data or not self.scene_data.skeleton.has_bones:
            self.empty_banner.show()
            self.setEnabled(False)
            self._update_readout_empty()
            return

        self.empty_banner.hide()
        self.setEnabled(True)
        skeleton = self.scene_data.skeleton

        # Recursively add bone nodes
        def add_bone_item(parent_widget, bone: BoneNode):
            item = QTreeWidgetItem(parent_widget)
            item.setText(0, f"🦴 {bone.name}")
            skin_str = "Skinned" if bone.has_skin_weights else "Structural"
            item.setToolTip(0, f"Bone: {bone.name} (ID: {bone.id}, {skin_str})")
            item.setData(0, Qt.ItemDataRole.UserRole, bone.id)

            for child in bone.children:
                add_bone_item(item, child)
            return item

        for root in skeleton.root_bones:
            add_bone_item(self.bone_tree, root)

        self.bone_tree.expandAll()

        # Select first bone by default
        if skeleton.bones:
            first_bone = skeleton.bones[0]
            self.select_bone_by_id(first_bone.id)

    def get_selected_bone(self) -> Optional[BoneNode]:
        """Return the currently selected BoneNode, or root bone if none explicitly selected."""
        if not self.scene_data or not self.scene_data.skeleton.has_bones:
            return None
        if self._selected_bone is not None:
            return self._selected_bone
        if self.scene_data.skeleton.selected_bone_id >= 0:
            return self.scene_data.skeleton.get_bone(self.scene_data.skeleton.selected_bone_id)
        if self.scene_data.skeleton.bones:
            return self.scene_data.skeleton.bones[0]
        return None

    def get_bone_focus_aabb(self, bone: Optional[BoneNode] = None, scene_radius: float = 1.0) -> AABB:
        """Return the framing AABB for the given bone (or selected bone)."""
        target_bone = bone or self.get_selected_bone()
        if target_bone is None or not self.scene_data:
            return AABB.empty()
        return self.scene_data.skeleton.get_bone_focus_aabb(target_bone.id, scene_radius)

    def select_bone_by_id(self, bone_id: int) -> None:
        """Programmatically select a bone in the tree by its ID, or deselect if bone_id < 0."""
        if bone_id < 0:
            self.bone_tree.clearSelection()
            self._selected_bone = None
            if self.scene_data and self.scene_data.skeleton:
                self.scene_data.skeleton.selected_bone_id = -1
            self._update_readout_empty()
            self._update_fk_slider_values()
            return

        def search_item(item: QTreeWidgetItem) -> Optional[QTreeWidgetItem]:
            if item.data(0, Qt.ItemDataRole.UserRole) == bone_id:
                return item
            for i in range(item.childCount()):
                res = search_item(item.child(i))
                if res is not None:
                    return res
            return None

        for i in range(self.bone_tree.topLevelItemCount()):
            found = search_item(self.bone_tree.topLevelItem(i))
            if found:
                self.bone_tree.setCurrentItem(found)
                return

    def _on_tree_selection_changed(self) -> None:
        """Handle bone selected in the hierarchy tree."""
        items = self.bone_tree.selectedItems()
        if not items or not self.scene_data:
            return

        bone_id = items[0].data(0, Qt.ItemDataRole.UserRole)
        bone = self.scene_data.skeleton.get_bone(bone_id)
        if bone:
            self._selected_bone = bone
            self.scene_data.skeleton.selected_bone_id = bone.id
            self.active_bone_changed.emit(bone.id)
            self._update_transform_readouts()
            self._update_fk_slider_values()
            self.refresh_viewport_requested.emit()

    def _update_transform_readouts(self) -> None:
        """Update position, Euler XYZ, Quaternion, and scale readout labels."""
        if not self._selected_bone:
            self._update_readout_empty()
            return

        t = self._selected_bone.get_transform_data(world_space=self._world_space)

        self.lbl_pos.setText(f"X: {t.position.x:+.2f}  Y: {t.position.y:+.2f}  Z: {t.position.z:+.2f}")
        self.lbl_euler.setText(f"X: {t.euler_deg.x:+.1f}°  Y: {t.euler_deg.y:+.1f}°  Z: {t.euler_deg.z:+.1f}°")
        self.lbl_quat.setText(f"W: {t.quaternion.w:+.3f}  X: {t.quaternion.x:+.3f}  Y: {t.quaternion.y:+.3f}  Z: {t.quaternion.z:+.3f}")
        self.lbl_scale.setText(f"X: {t.scale.x:.2f}  Y: {t.scale.y:.2f}  Z: {t.scale.z:.2f}")

    def _update_readout_empty(self) -> None:
        self.lbl_pos.setText("X: 0.00  Y: 0.00  Z: 0.00")
        self.lbl_euler.setText("X: 0.0°  Y: 0.0°  Z: 0.0°")
        self.lbl_quat.setText("W: 1.000  X: 0.000  Y: 0.000  Z: 0.000")
        self.lbl_scale.setText("X: 1.00  Y: 1.00  Z: 1.00")

    def _update_fk_slider_values(self) -> None:
        """Sync FK sliders to the active bone's interactive rotation offset."""
        if not self._selected_bone:
            self._updating_ui = True
            try:
                self.slider_rx.setValue(0)
                self.slider_ry.setValue(0)
                self.slider_rz.setValue(0)
                self.lbl_rx_val.setText("0°")
                self.lbl_ry_val.setText("0°")
                self.lbl_rz_val.setText("0°")
            finally:
                self._updating_ui = False
            return
        self._updating_ui = True
        try:
            rx = int(round(self._selected_bone.slider_euler_deg.x))
            ry = int(round(self._selected_bone.slider_euler_deg.y))
            rz = int(round(self._selected_bone.slider_euler_deg.z))

            self.slider_rx.setValue(rx)
            self.slider_ry.setValue(ry)
            self.slider_rz.setValue(rz)

            self.lbl_rx_val.setText(f"{rx}°")
            self.lbl_ry_val.setText(f"{ry}°")
            self.lbl_rz_val.setText(f"{rz}°")
        finally:
            self._updating_ui = False

    def _on_fk_slider_changed(self) -> None:
        """Handle interactive FK rotation slider movement."""
        if self._updating_ui or not self._selected_bone or not self.scene_data:
            return

        rx = float(self.slider_rx.value())
        ry = float(self.slider_ry.value())
        rz = float(self.slider_rz.value())

        self.lbl_rx_val.setText(f"{int(rx)}°")
        self.lbl_ry_val.setText(f"{int(ry)}°")
        self.lbl_rz_val.setText(f"{int(rz)}°")

        # Apply rotation offset and propagate FK down the bone hierarchy
        self._selected_bone.set_slider_rotation(rx, ry, rz)
        self.scene_data.skeleton.update_forward_kinematics()

        self._update_transform_readouts()
        self.fk_rotation_changed.emit()
        self.refresh_viewport_requested.emit()

    def _on_reset_bind_pose_clicked(self) -> None:
        """Reset all bones across the skeleton to bind pose."""
        if not self.scene_data or not self.scene_data.skeleton.has_bones:
            return

        self.scene_data.skeleton.reset_bind_pose()
        self._update_fk_slider_values()
        self._update_transform_readouts()
        self.bind_pose_reset.emit()
        self.refresh_viewport_requested.emit()

    def _on_space_radio_clicked(self, space_id: int) -> None:
        self._world_space = (space_id == 1)
        self._update_transform_readouts()

    def _on_overlay_toggled(self) -> None:
        show_bones = self.chk_show_bones.isChecked()
        xray_mode = self.chk_xray.isChecked()
        heatmap_mode = self.chk_heatmap.isChecked()
        self.overlay_options_changed.emit(show_bones, xray_mode, heatmap_mode)
        self.refresh_viewport_requested.emit()

    def _on_heatmap_toggled(self, checked: bool) -> None:
        self._on_overlay_toggled()

    def _filter_bones(self, query: str) -> None:
        """Fast recursive filter of bone tree items without rebuilding."""
        q = query.strip().lower()

        def filter_item(item: QTreeWidgetItem) -> bool:
            self_match = (not q) or (q in item.text(0).lower())
            child_match = False
            for i in range(item.childCount()):
                if filter_item(item.child(i)):
                    child_match = True
            visible = self_match or child_match
            item.setHidden(not visible)
            if visible and q:
                item.setExpanded(True)
            return visible

        for i in range(self.bone_tree.topLevelItemCount()):
            filter_item(self.bone_tree.topLevelItem(i))

    def refresh_theme(self) -> None:
        self.update()
