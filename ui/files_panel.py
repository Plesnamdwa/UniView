"""
ui/files_panel.py
File & Dependency Analyzer Panel ("Files" Tab) for UniView.
Features:
- Two collapsible tree categories:
  1. Missing Files: Lists unresolved textures with Browse/Relink button AND drag-and-drop support
     directly onto rows to dynamically bind textures in OpenGL memory.
  2. Available Files: Lists loaded main 3D asset file and resolved textures with resolution & size info.
- Batch Folder Scanner: Allows user to point to a folder to auto-resolve all missing textures in one click.
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
)

from core.scene_loader import SceneData
from core.material import DependencyFile
from ui.custom_widgets import DropTargetTreeWidget, IconButton, SUPPORTED_IMAGE_EXTENSIONS


class FilesPanel(QWidget):
    """File & Dependency Analyzer dock panel."""

    dependency_relinked = pyqtSignal(int, str)  # (dep_index, new_image_path)
    refresh_viewport_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_data: Optional[SceneData] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header bar with summary & batch folder scanner
        top_bar = QHBoxLayout()
        self.summary_label = QLabel("No 3D asset loaded")
        self.summary_label.setStyleSheet("font-weight: 600;")
        top_bar.addWidget(self.summary_label)
        top_bar.addStretch()

        self.scan_folder_btn = QPushButton("Scan Folder")
        self.scan_folder_btn.setProperty("pill", True)
        self.scan_folder_btn.setProperty("primary", True)
        self.scan_folder_btn.setToolTip("Select a folder to automatically batch-resolve all missing textures")
        self.scan_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scan_folder_btn.clicked.connect(self._scan_folder_for_missing)
        self.scan_folder_btn.setEnabled(False)
        top_bar.addWidget(self.scan_folder_btn)

        layout.addLayout(top_bar)

        # Drop-Target Tree Widget (clean 2-column layout so filenames are never truncated)
        self.tree = DropTargetTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.header().resizeSection(1, 36)
        self.tree.image_dropped_on_item.connect(self._on_image_dropped_on_item)
        layout.addWidget(self.tree)

        # Drag hint banner
        self.hint_label = QLabel("Tip: Click the folder button or drag & drop an image onto a missing row to relink.")
        self.hint_label.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        layout.addWidget(self.hint_label)

    def set_scene(self, scene: Optional[SceneData]) -> None:
        """Update panel with the dependencies of the loaded scene."""
        self.scene_data = scene
        self.rebuild_tree()

    def rebuild_tree(self) -> None:
        """Repopulate the tree with Missing Files and Available Files."""
        self.tree.clear()
        if not self.scene_data:
            self.summary_label.setText("No 3D asset loaded")
            self.scan_folder_btn.setEnabled(False)
            return

        missing = self.scene_data.missing_files
        available = self.scene_data.available_files
        self.scan_folder_btn.setEnabled(len(missing) > 0)

        # Update summary text
        self.summary_label.setText(
            f"Missing: {len(missing)}  |  Available: {len(available)}"
        )

        # ---------------------------------------------------------------------
        # 1. Missing Files Category
        # ---------------------------------------------------------------------
        missing_root = QTreeWidgetItem(self.tree)
        missing_root.setText(0, f"Missing Files ({len(missing)})" if missing else "Missing Files")
        missing_root.setFirstColumnSpanned(True)
        missing_root.setExpanded(True)
        font_bold = missing_root.font(0)
        font_bold.setBold(True)
        missing_root.setFont(0, font_bold)

        for dep in missing:
            dep_idx = self.scene_data.dependencies.index(dep)
            item = QTreeWidgetItem(missing_root)
            item.setText(0, dep.filename)
            item.setSizeHint(0, QSize(0, 30))
            item.setSizeHint(1, QSize(34, 30))

            mat_names = []
            for m_i in dep.material_indices:
                if 0 <= m_i < len(self.scene_data.materials):
                    mat_names.append(self.scene_data.materials[m_i].name)
            mat_str = ", ".join(mat_names) if mat_names else "Model"
            item.setToolTip(
                0,
                f"File: {dep.filename}\nType: {dep.texture_semantic} ({mat_str})\nReferenced path: {dep.raw_path}",
            )
            item.setData(0, Qt.ItemDataRole.UserRole, dep_idx)

            # Action column: sleek folder-relink icon button matching Online 3D Viewer
            relink_wrap = QWidget()
            relink_wrap.setStyleSheet("background: transparent;")
            relink_layout = QHBoxLayout(relink_wrap)
            relink_layout.setContentsMargins(2, 0, 2, 0)
            relink_layout.setSpacing(0)
            relink_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            relink_btn = IconButton("relink", tooltip=f"Browse or drop image to relink '{dep.filename}'")
            relink_btn.clicked.connect(lambda checked, idx=dep_idx: self._browse_relink(idx))
            relink_layout.addWidget(relink_btn)
            self.tree.setItemWidget(item, 1, relink_wrap)

        # ---------------------------------------------------------------------
        # 2. Available Files Category
        # ---------------------------------------------------------------------
        available_root = QTreeWidgetItem(self.tree)
        available_root.setText(0, f"Available Files ({len(available)})" if available else "Available Files")
        available_root.setFirstColumnSpanned(True)
        available_root.setExpanded(True)
        available_root.setFont(0, font_bold)

        for dep in available:
            dep_idx = self.scene_data.dependencies.index(dep)
            item = QTreeWidgetItem(available_root)
            item.setText(0, dep.filename)
            item.setSizeHint(0, QSize(0, 26))

            if dep.file_type == "Model":
                ext = os.path.splitext(dep.filename)[1].upper()
                file_size_kb = 0.0
                if dep.resolved_path and os.path.isfile(dep.resolved_path):
                    file_size_kb = os.path.getsize(dep.resolved_path) / 1024.0
                info_str = f"3D Asset ({ext}, {file_size_kb:.1f} KB)"
            else:
                res_str = f"{dep.image_size[0]}x{dep.image_size[1]}" if dep.image_size else "Loaded"
                info_str = f"{dep.texture_semantic} ({res_str})"

            item.setToolTip(0, f"File: {dep.filename}\nInfo: {info_str}\nPath: {dep.resolved_path or 'Embedded'}")
            item.setData(0, Qt.ItemDataRole.UserRole, dep_idx)

    def _browse_relink(self, dep_index: int) -> None:
        """Open QFileDialog to select a replacement image for dependency."""
        if not self.scene_data:
            return
        dep = self.scene_data.dependencies[dep_index]
        start_dir = os.path.dirname(self.scene_data.filepath)
        # If the referenced raw_path has a subfolder (e.g. .fbm) that exists next to the model, start there
        raw_sub = os.path.dirname(dep.raw_path.replace("\\", os.sep).replace("/", os.sep))
        if raw_sub:
            candidate_dir = os.path.join(start_dir, raw_sub)
            if os.path.isdir(candidate_dir):
                start_dir = candidate_dir

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Relink Texture for '{dep.filename}'",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.tga *.bmp *.dds *.webp);;All Files (*.*)",
        )
        if file_path:
            self._apply_relink(dep_index, file_path)

    def _on_image_dropped_on_item(self, item: QTreeWidgetItem, image_path: str) -> None:
        """Handle image dropped directly onto a tree row."""
        dep_idx = item.data(0, Qt.ItemDataRole.UserRole)
        if dep_idx is not None and isinstance(dep_idx, int):
            self._apply_relink(dep_idx, image_path)

    def _apply_relink(self, dep_index: int, image_path: str) -> None:
        """Execute dependency relink, reload texture bytes, and rebuild UI tree."""
        if not self.scene_data:
            return
        success = self.scene_data.relink_dependency(dep_index, image_path)
        if success:
            self.dependency_relinked.emit(dep_index, image_path)
            self.refresh_viewport_requested.emit()
            self.rebuild_tree()
        else:
            QMessageBox.warning(
                self,
                "Relink Failed",
                f"Could not load image file:\n{image_path}\nMake sure it is a valid supported image format.",
            )

    def _scan_folder_for_missing(self) -> None:
        """Scan a user-selected folder to automatically resolve matching missing texture filenames."""
        if not self.scene_data:
            return

        missing = self.scene_data.missing_files
        if not missing:
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Textures Folder to Auto-Resolve",
            os.path.dirname(self.scene_data.filepath),
        )
        if not folder or not os.path.isdir(folder):
            return

        # Index files in selected folder
        folder_files: Dict[str, str] = {}
        for root, _, files in os.walk(folder):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    folder_files[f.lower()] = os.path.join(root, f)

        resolved_count = 0
        last_relinked = (-1, "")
        for dep in missing:
            dep_idx = self.scene_data.dependencies.index(dep)
            key = dep.filename.lower()
            if key in folder_files:
                matched_path = folder_files[key]
                if self.scene_data.relink_dependency(dep_idx, matched_path):
                    resolved_count += 1
                    last_relinked = (dep_idx, matched_path)

        if resolved_count > 0:
            if last_relinked[0] >= 0:
                self.dependency_relinked.emit(last_relinked[0], last_relinked[1])
            self.refresh_viewport_requested.emit()
            self.rebuild_tree()
            QMessageBox.information(
                self,
                "Batch Relink Complete",
                f"Successfully resolved {resolved_count} missing texture(s) from '{os.path.basename(folder)}'!",
            )
        else:
            QMessageBox.information(
                self,
                "Batch Relink",
                "No matching filenames were found in the selected folder.",
            )

    def refresh_theme(self) -> None:
        self.rebuild_tree()
