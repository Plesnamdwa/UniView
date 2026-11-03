"""
ui/materials_panel.py
Material & Base Color Override Panel ("Materials" Tab) for UniView.
Features:
- Global viewport toggle for "No Material / Base Color Mode" vs "Textured Mode"
- Detects and lists all distinct Material slots assigned to the model
- Selected Material Details Inspector:
  - Source: Model / Custom Override (with a single-click "Reset to Model" button)
  - Type: Phong / Unlit
  - Color (Base/Diffuse): Interactive Color Swatch button + Hex Code label with live QColorDialog
  - Ambient: Interactive Color Swatch + Hex Code
  - Specular: Interactive Color Swatch + Hex Code
  - Opacity: Percentage slider (0% - 100%) and percentage label
"""

from __future__ import annotations
from typing import Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QSlider,
    QFormLayout,
)
import glm

from core.scene_loader import SceneData
from core.material import MaterialSlot, vec3_to_hex
from ui.custom_widgets import ColorSwatchButton, SegmentedControl, make_color_dot_icon
from ui.style import ThemeManager


class MaterialsPanel(QWidget):
    """Materials and color override inspector panel."""

    render_mode_changed = pyqtSignal(int)      # 0: Textured, 1: Base Color
    material_modified = pyqtSignal()          # Emitted to request viewport repaint
    focus_requested = pyqtSignal()            # Emitted to zoom/center camera on selected material meshes (F)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_data: Optional[SceneData] = None
        self._selected_mat_index: int = 0
        self._updating_ui: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ---------------------------------------------------------------------
        # 1. Global Viewport Mode Selector
        # ---------------------------------------------------------------------
        mode_box = QGroupBox("Viewport Display")
        mode_layout = QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(12, 14, 12, 10)

        self.mode_segment = SegmentedControl(["Textured", "Base Color"])
        self.mode_segment.setToolTip("Textured shows texture maps · Base Color shows flat material colors (1 / 2)")
        self.mode_segment.currentChanged.connect(self._on_mode_radio_clicked)
        mode_layout.addWidget(self.mode_segment)
        layout.addWidget(mode_box)

        # ---------------------------------------------------------------------
        # 2. Material Slots List
        # ---------------------------------------------------------------------
        slots_box = QGroupBox("Material Slots")
        slots_layout = QVBoxLayout(slots_box)
        slots_layout.setContentsMargins(8, 8, 8, 8)

        self.slot_list = QListWidget()
        self.slot_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.slot_list.currentRowChanged.connect(self._on_slot_selection_changed)
        self.slot_list.itemDoubleClicked.connect(lambda item: self.focus_requested.emit())
        slots_layout.addWidget(self.slot_list)
        layout.addWidget(slots_box, stretch=1)

        # ---------------------------------------------------------------------
        # 3. Selected Material Details Inspector
        # ---------------------------------------------------------------------
        self.details_box = QGroupBox("Details Inspector")
        details_layout = QFormLayout(self.details_box)
        details_layout.setContentsMargins(10, 10, 10, 10)
        details_layout.setSpacing(8)
        details_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Slot Name & Index
        self.lbl_mat_name = QLabel("None")
        self.lbl_mat_name.setStyleSheet("font-weight: 700;")
        details_layout.addRow("Material:", self.lbl_mat_name)

        # Source & Reset Button
        source_row = QHBoxLayout()
        self.lbl_source = QLabel("Model")
        self.lbl_source.setStyleSheet("color: #10B981; font-weight: 700;")
        source_row.addWidget(self.lbl_source)
        source_row.addSpacing(10)

        self.btn_reset_model = QPushButton("↺ Reset to Model")
        self.btn_reset_model.setMinimumSize(134, 24)
        self.btn_reset_model.setToolTip("Restore original model colors and properties")
        self.btn_reset_model.clicked.connect(self._reset_current_material)
        self.btn_reset_model.setEnabled(False)
        source_row.addWidget(self.btn_reset_model)
        source_row.addStretch()
        details_layout.addRow("Source:", source_row)

        # Type (Phong / Unlit)
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Phong", "Unlit"])
        self.combo_type.currentTextChanged.connect(self._on_shading_type_changed)
        details_layout.addRow("Type:", self.combo_type)

        # Base / Diffuse Color
        self.swatch_base = ColorSwatchButton(glm.vec3(0.75, 0.75, 0.78))
        self.swatch_base.color_changed.connect(self._on_base_color_changed)
        details_layout.addRow("Color (Base):", self.swatch_base)

        # Ambient Color
        self.swatch_ambient = ColorSwatchButton(glm.vec3(0.08, 0.08, 0.10))
        self.swatch_ambient.color_changed.connect(self._on_ambient_color_changed)
        details_layout.addRow("Ambient:", self.swatch_ambient)

        # Specular Color
        self.swatch_specular = ColorSwatchButton(glm.vec3(0.25, 0.25, 0.25))
        self.swatch_specular.color_changed.connect(self._on_specular_color_changed)
        details_layout.addRow("Specular:", self.swatch_specular)

        # Opacity Slider & Label
        opacity_row = QHBoxLayout()
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider_changed)
        opacity_row.addWidget(self.slider_opacity)

        self.lbl_opacity = QLabel("100%")
        self.lbl_opacity.setFixedWidth(40)
        self.lbl_opacity.setStyleSheet("font-family: 'Consolas', monospace; font-weight: 600;")
        opacity_row.addWidget(self.lbl_opacity)
        details_layout.addRow("Opacity:", opacity_row)

        layout.addWidget(self.details_box, stretch=0)

    def refresh_theme(self) -> None:
        self.swatch_base._update_swatch_style()
        self.swatch_ambient._update_swatch_style()
        self.swatch_specular._update_swatch_style()
        if self.scene_data and 0 <= self._selected_mat_index < len(self.scene_data.materials):
            self._update_source_badge(self.scene_data.materials[self._selected_mat_index])

    @property
    def selected_material_index(self) -> int:
        return self._selected_mat_index

    def set_scene(self, scene: Optional[SceneData]) -> None:
        """Populate material slots from the loaded scene."""
        self.scene_data = scene
        self.rebuild_slots_list()

    def rebuild_slots_list(self) -> None:
        """Re-render material slot items in QListWidget."""
        self.slot_list.clear()
        if not self.scene_data or not self.scene_data.materials:
            self.details_box.setEnabled(False)
            self.lbl_mat_name.setText("None")
            return

        self.details_box.setEnabled(True)

        for i, mat in enumerate(self.scene_data.materials):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.slot_list.addItem(item)
            self._refresh_slot_list_item(i)

        # Select first slot by default
        self.slot_list.setCurrentRow(0)

    def _on_slot_selection_changed(self, row: int) -> None:
        """Update the details inspector when a different material slot is selected."""
        if not self.scene_data or row < 0 or row >= len(self.scene_data.materials):
            return

        self._selected_mat_index = row
        mat = self.scene_data.materials[row]

        self._updating_ui = True
        try:
            self.lbl_mat_name.setText(f"[{mat.index}] {mat.name}")
            self._update_source_badge(mat)

            # Shading Type
            idx = self.combo_type.findText(mat.shading_type)
            if idx >= 0:
                self.combo_type.setCurrentIndex(idx)

            # Colors
            self.swatch_base.set_color(mat.base_color)
            self.swatch_ambient.set_color(mat.ambient_color)
            self.swatch_specular.set_color(mat.specular_color)

            # Opacity
            pct = mat.opacity_percent
            self.slider_opacity.setValue(pct)
            self.lbl_opacity.setText(f"{pct}%")
        finally:
            self._updating_ui = False

    def _update_source_badge(self, mat: MaterialSlot) -> None:
        p = ThemeManager.palette()
        if mat.source == "Custom Override":
            self.lbl_source.setText("Custom Override")
            self.lbl_source.setStyleSheet(f"color: {p.warning}; font-weight: 700;")
            self.btn_reset_model.setEnabled(True)
        else:
            self.lbl_source.setText("Model")
            self.lbl_source.setStyleSheet(f"color: {p.success}; font-weight: 700;")
            self.btn_reset_model.setEnabled(False)

    def _on_mode_radio_clicked(self, mode_id: int) -> None:
        self.render_mode_changed.emit(mode_id)

    def _on_base_color_changed(self, color: glm.vec3) -> None:
        if self._updating_ui or not self.scene_data:
            return
        mat = self.scene_data.materials[self._selected_mat_index]
        mat.set_base_color(color)
        self._update_source_badge(mat)
        self._refresh_slot_list_item(self._selected_mat_index)
        self.material_modified.emit()

    def _on_ambient_color_changed(self, color: glm.vec3) -> None:
        if self._updating_ui or not self.scene_data:
            return
        mat = self.scene_data.materials[self._selected_mat_index]
        mat.set_ambient_color(color)
        self._update_source_badge(mat)
        self.material_modified.emit()

    def _on_specular_color_changed(self, color: glm.vec3) -> None:
        if self._updating_ui or not self.scene_data:
            return
        mat = self.scene_data.materials[self._selected_mat_index]
        mat.set_specular_color(color)
        self._update_source_badge(mat)
        self.material_modified.emit()

    def _on_shading_type_changed(self, text: str) -> None:
        if self._updating_ui or not self.scene_data:
            return
        mat = self.scene_data.materials[self._selected_mat_index]
        mat.set_shading_type(text)
        self._update_source_badge(mat)
        self.material_modified.emit()

    def _on_opacity_slider_changed(self, value: int) -> None:
        self.lbl_opacity.setText(f"{value}%")
        if self._updating_ui or not self.scene_data:
            return
        mat = self.scene_data.materials[self._selected_mat_index]
        mat.set_opacity_percent(value)
        self._update_source_badge(mat)
        self.material_modified.emit()

    def _reset_current_material(self) -> None:
        """Reset currently selected material to original model defaults."""
        if not self.scene_data:
            return
        mat = self.scene_data.materials[self._selected_mat_index]
        mat.reset_to_model()
        self._on_slot_selection_changed(self._selected_mat_index)
        self._refresh_slot_list_item(self._selected_mat_index)
        self.material_modified.emit()

    def _refresh_slot_list_item(self, row: int) -> None:
        item = self.slot_list.item(row)
        if item and self.scene_data and 0 <= row < len(self.scene_data.materials):
            mat = self.scene_data.materials[row]
            tag = "   · edited" if mat.source == "Custom Override" else ""
            item.setText(f"{mat.name}{tag}")
            item.setIcon(make_color_dot_icon(mat.base_hex))
            tex_info = "has texture" if mat.has_diffuse_texture else "no texture"
            item.setToolTip(f"Slot [{mat.index}] · {mat.base_hex} · {tex_info}\nDouble-click to focus camera")
