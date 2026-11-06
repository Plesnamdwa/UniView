"""
ui/custom_widgets.py
Custom Minimalist ("Juicy Simple") PyQt6 Widgets for UniView 2.0:
- ExpandingTabBar & ExpandingTabWidget:
  Implements the "Expanding Label" navigation style ("The active tab grows into a pill
  and its label slides out. Others shrink.") with custom monoline vector icons and 60 FPS
  hardware-smooth QPainter rendering.
- ColorSwatchButton: Interactive pill swatch with hex readout and live QColorDialog
- ActionIconButton: Minimalist vector-capable icon button for Isolate, Eye, and Relink
- DropTargetTreeWidget: Drag-and-drop tree widget for texture relinking
"""

from __future__ import annotations
import os
import math
from typing import Optional, List, Dict
from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
    QSize,
    QRect,
    QRectF,
    QPointF,
    QTimer,
    QVariantAnimation,
    QEasingCurve,
)
from PyQt6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QBrush,
    QPen,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
)
from PyQt6.QtWidgets import (
    QPushButton,
    QCheckBox,
    QColorDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QWidget,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QSizePolicy,
)
import glm
from core.bounding_box import BoundingBoxInfo

from core.material import vec3_to_hex, hex_to_vec3
from ui.style import ThemeManager, ThemePalette

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds", ".webp"}


# =============================================================================
# 1. Monoline Vector Icon Renderer (QPainterPath)
# =============================================================================

def draw_monoline_icon(painter: QPainter, icon_kind: str, center: QPointF, size: float, color: QColor) -> None:
    """
    Draw a crisp, minimalist monoline vector icon centered at `center` within `size x size`.
    Matches the clean stroke aesthetic of the Expanding Label reference design.
    """
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    pen = QPen(color, 1.75, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    r = size * 0.45
    cx, cy = center.x(), center.y()

    if icon_kind == "files":
        # Clean Minimalist Folder / File Tab Icon
        path = QPainterPath()
        x0, y0 = cx - r, cy - r * 0.78
        w, h = r * 2.0, r * 1.56
        # Folder tab on top-left and rounded body
        path.moveTo(x0, y0 + h * 0.22)
        path.lineTo(x0, y0 + h - 3.0)
        path.quadTo(x0, y0 + h, x0 + 3.0, y0 + h)
        path.lineTo(x0 + w - 3.0, y0 + h)
        path.quadTo(x0 + w, y0 + h, x0 + w, y0 + h - 3.0)
        path.lineTo(x0 + w, y0 + h * 0.26 + 2.5)
        path.quadTo(x0 + w, y0 + h * 0.26, x0 + w - 2.5, y0 + h * 0.26)
        path.lineTo(x0 + w * 0.48, y0 + h * 0.26)
        path.lineTo(x0 + w * 0.36, y0)
        path.lineTo(x0 + 2.5, y0)
        path.quadTo(x0, y0, x0, y0 + 2.5)
        path.closeSubpath()
        painter.drawPath(path)

    elif icon_kind == "materials":
        # Clean Minimalist Material Sphere / Swatch Rings
        painter.drawEllipse(QPointF(cx, cy), r * 0.92, r * 0.92)
        arc_rect = QRectF(cx - r * 0.58, cy - r * 0.58, r * 1.16, r * 1.16)
        painter.drawArc(arc_rect, 35 * 16, 125 * 16)
        # Specular highlight dot
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx + r * 0.30, cy - r * 0.30), 1.6, 1.6)

    elif icon_kind == "meshes":
        # Isometric 3D Wireframe Cube
        path = QPainterPath()
        pts = []
        for i in range(6):
            angle = math.radians(i * 60.0 - 90.0)
            pts.append(QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle)))
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()
        # Inner Y edges of cube
        path.moveTo(pts[5])
        path.lineTo(QPointF(cx, cy))
        path.lineTo(pts[1])
        path.moveTo(QPointF(cx, cy))
        path.lineTo(pts[3])
        painter.drawPath(path)

    elif icon_kind == "skeleton":
        # Clean Connected Joint / Bone Node Icon
        p1 = QPointF(cx - r * 0.58, cy + r * 0.58)
        p2 = QPointF(cx + r * 0.58, cy - r * 0.58)
        joint_r = r * 0.34
        # Connecting shaft between joints
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        dist = math.hypot(dx, dy)
        if dist > 1e-4:
            ux, uy = dx / dist, dy / dist
            painter.drawLine(
                QPointF(p1.x() + ux * joint_r, p1.y() + uy * joint_r),
                QPointF(p2.x() - ux * joint_r, p2.y() - uy * joint_r),
            )
        painter.drawEllipse(p1, joint_r, joint_r)
        painter.drawEllipse(p2, joint_r, joint_r)

    painter.restore()


# =============================================================================
# 2. Expanding Label Tab Bar & Container Widget
# =============================================================================

class ExpandingTabBar(QWidget):
    """
    Minimalist Expanding Label Tab Bar matching the "5 Expanding Label" design:
    - Outer rounded stadium capsule container
    - Active tab grows into an inner pill and its label slides out smoothly
    - Other tabs shrink down to crisp monoline vector icons
    """

    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Tab items: [{"label": str, "icon": str, "badge": str}]
        self._tabs: List[Dict[str, str]] = []
        self._current_index: int = 0
        self._hover_index: int = -1

        # Expansion progress [0.0 .. 1.0] per tab for 60 FPS smooth transitions
        self._progress: List[float] = []
        self._start_progress: List[float] = []
        self._target_progress: List[float] = []
        self._tab_rects: List[QRectF] = []

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(210)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)

    def add_tab(self, label: str, icon_kind: str = "files") -> int:
        idx = len(self._tabs)
        clean_label = (
            label.replace("📁", "")
            .replace("🎨", "")
            .replace("🧊", "")
            .replace("🦴", "")
            .strip()
        )
        if not icon_kind:
            low = clean_label.lower()
            if "file" in low:
                icon_kind = "files"
            elif "mat" in low:
                icon_kind = "materials"
            elif "mesh" in low:
                icon_kind = "meshes"
            else:
                icon_kind = "skeleton"

        self._tabs.append({"label": clean_label, "icon": icon_kind, "badge": ""})
        is_first = (idx == 0)
        self._progress.append(1.0 if is_first else 0.0)
        self._start_progress.append(1.0 if is_first else 0.0)
        self._target_progress.append(1.0 if is_first else 0.0)
        self.update()
        return idx

    def set_tab_badge(self, index: int, badge_text: str) -> None:
        if 0 <= index < len(self._tabs):
            self._tabs[index]["badge"] = badge_text
            self.update()

    def current_index(self) -> int:
        return self._current_index

    def set_current_index(self, index: int, animate: bool = True) -> None:
        if index < 0 or index >= len(self._tabs):
            return
        changed = (index != self._current_index)
        self._current_index = index

        if animate and self.isVisible():
            self._anim.stop()
            self._start_progress = list(self._progress)
            self._target_progress = [1.0 if i == index else 0.0 for i in range(len(self._tabs))]
            self._anim.start()
        else:
            self._anim.stop()
            self._progress = [1.0 if i == index else 0.0 for i in range(len(self._tabs))]
            self._start_progress = list(self._progress)
            self._target_progress = list(self._progress)
            self.update()

        if changed:
            self.currentChanged.emit(index)

    def _on_anim_step(self, value) -> None:
        t = float(value)
        for i in range(len(self._progress)):
            self._progress[i] = self._start_progress[i] + (self._target_progress[i] - self._start_progress[i]) * t
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        new_hover = -1
        for i, rect in enumerate(self._tab_rects):
            if rect.contains(pos):
                new_hover = i
                break
        if new_hover != self._hover_index:
            self._hover_index = new_hover
            if 0 <= new_hover < len(self._tabs):
                self.setToolTip(self._tabs[new_hover]["label"])
            else:
                self.setToolTip("")
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hover_index != -1:
            self._hover_index = -1
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            for i, rect in enumerate(self._tab_rects):
                if rect.contains(pos):
                    self.set_current_index(i, animate=True)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        if not self._tabs:
            return

        palette: ThemePalette = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = float(self.width())
        h = float(self.height())

        # 1. Draw Outer Stadium Container
        outer_margin_x = 4.0
        outer_margin_y = 3.0
        outer_rect = QRectF(
            outer_margin_x,
            outer_margin_y,
            w - outer_margin_x * 2.0,
            h - outer_margin_y * 2.0,
        )
        outer_radius = outer_rect.height() * 0.5

        painter.setPen(QPen(QColor(palette.tab_outer_border), 1.5))
        painter.setBrush(QBrush(QColor(palette.tab_outer_bg)))
        painter.drawRoundedRect(outer_rect, outer_radius, outer_radius)

        # 2. Compute Dynamic Expanding / Shrinking Tab Rects inside the Stadium
        inner_pad = 5.0
        avail_x = outer_rect.left() + inner_pad
        avail_y = outer_rect.top() + inner_pad
        avail_w = outer_rect.width() - inner_pad * 2.0
        avail_h = outer_rect.height() - inner_pad * 2.0

        font = QFont(self.font())
        font.setPixelSize(13)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        n = len(self._tabs)
        raw_weights: List[float] = []
        label_widths: List[float] = []

        for i, tab in enumerate(self._tabs):
            p = self._progress[i]
            display_str = tab["label"]
            if tab["badge"]:
                display_str = f"{display_str} · {tab['badge']}"
            lw = float(fm.horizontalAdvance(display_str))
            label_widths.append(lw)
            # Base icon slot weight = 44, expanded adds label width + padding
            raw_weights.append(44.0 + p * (lw + 28.0))

        total_weight = max(sum(raw_weights), 1.0)
        self._tab_rects.clear()

        curr_x = avail_x
        for i in range(n):
            slot_w = (raw_weights[i] / total_weight) * avail_w
            rect = QRectF(curr_x, avail_y, slot_w, avail_h)
            self._tab_rects.append(rect)
            curr_x += slot_w

        # 3. Draw Each Tab (Inner Pill + Monoline Vector Icon + Sliding Label)
        inactive_col = QColor(palette.tab_inactive_fg)
        hover_col = QColor(palette.tab_hover_fg)
        active_col = QColor(palette.tab_active_fg)

        for i, tab in enumerate(self._tabs):
            rect = self._tab_rects[i]
            p = self._progress[i]
            is_hovered = (i == self._hover_index)

            # A. Draw growing Inner Pill when active / animating
            if p > 0.01:
                painter.save()
                painter.setOpacity(min(max(p, 0.0), 1.0))
                pill_radius = rect.height() * 0.5
                pill_fill = QColor(palette.tab_pill_bg)
                pill_border = QColor(palette.tab_pill_border)
                painter.setPen(QPen(pill_border, 1.4))
                painter.setBrush(QBrush(pill_fill))
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), pill_radius, pill_radius)
                painter.restore()
            elif is_hovered:
                # Subtle hover circle/pill for inactive icon
                painter.save()
                painter.setOpacity(0.35)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(palette.hover_bg)))
                pill_radius = rect.height() * 0.5
                painter.drawRoundedRect(rect.adjusted(3.0, 2.0, -3.0, -2.0), pill_radius, pill_radius)
                painter.restore()

            # Interpolate foreground color between inactive/hover and active
            base_fg = hover_col if is_hovered else inactive_col
            fg_r = int(base_fg.red() + (active_col.red() - base_fg.red()) * p)
            fg_g = int(base_fg.green() + (active_col.green() - base_fg.green()) * p)
            fg_b = int(base_fg.blue() + (active_col.blue() - base_fg.blue()) * p)
            fg_color = QColor(fg_r, fg_g, fg_b)

            display_str = tab["label"]
            if tab["badge"]:
                display_str = f"{display_str} · {tab['badge']}"
            lw = label_widths[i]

            icon_size = 17.0
            gap = 8.0 * p
            content_w = icon_size + gap + (lw * p)
            start_x = rect.center().x() - content_w * 0.5
            icon_center = QPointF(start_x + icon_size * 0.5, rect.center().y())

            # B. Draw Monoline Vector Icon
            draw_monoline_icon(painter, tab["icon"], icon_center, icon_size, fg_color)

            # C. Draw Sliding Out Label (Clipped smoothly to progress p)
            if p > 0.03:
                painter.save()
                painter.setOpacity(min(max((p - 0.05) / 0.95, 0.0), 1.0))
                text_left = start_x + icon_size + 8.0
                clip_w = max(0.0, (rect.right() - 8.0) - text_left) * p
                clip_rect = QRectF(text_left, rect.top(), clip_w + 4.0, rect.height())
                painter.setClipRect(clip_rect)

                # Slight horizontal slide-out offset as pill expands
                slide_dx = (1.0 - p) * -6.0
                text_rect = QRectF(text_left + slide_dx, rect.top(), lw + 10.0, rect.height())
                painter.setPen(fg_color)
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    display_str,
                )
                painter.restore()

        painter.end()


class ExpandingTabWidget(QWidget):
    """
    Drop-in replacement for QTabWidget using ExpandingTabBar at the top
    and QStackedWidget for the inspector panels below.
    """

    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.tab_bar = ExpandingTabBar(self)
        self.stack = QStackedWidget(self)

        layout.addWidget(self.tab_bar)
        layout.addWidget(self.stack, stretch=1)

        self.tab_bar.currentChanged.connect(self._on_bar_index_changed)

    def addTab(self, widget: QWidget, label: str, icon_kind: str = "") -> int:
        idx = self.stack.addWidget(widget)
        self.tab_bar.add_tab(label, icon_kind=icon_kind)
        return idx

    def setDocumentMode(self, enabled: bool) -> None:
        pass

    def currentIndex(self) -> int:
        return self.stack.currentIndex()

    def currentWidget(self) -> Optional[QWidget]:
        return self.stack.currentWidget()

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)
            self.tab_bar.set_current_index(index, animate=True)

    def setCurrentWidget(self, widget: QWidget) -> None:
        idx = self.stack.indexOf(widget)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def setTabBadge(self, index: int, badge_text: str) -> None:
        self.tab_bar.set_tab_badge(index, badge_text)

    def _on_bar_index_changed(self, index: int) -> None:
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)
            self.currentChanged.emit(index)

    def refresh_theme(self) -> None:
        self.tab_bar.update()


# =============================================================================
# 3. ColorSwatchButton, ActionIconButton & DropTargetTreeWidget
# =============================================================================

class ColorSwatchButton(QWidget):
    """
    Minimalist Interactive Color Swatch Pill.
    Displays a rounded color capsule alongside its Hex Code badge.
    Clicking opens QColorDialog for live real-time color overrides.
    """

    color_changed = pyqtSignal(object)  # Emits glm.vec3(r, g, b)

    def __init__(self, initial_color: glm.vec3 = glm.vec3(0.8, 0.8, 0.8), parent=None):
        super().__init__(parent)
        self._color: glm.vec3 = glm.vec3(initial_color)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.swatch_btn = QPushButton()
        self.swatch_btn.setFixedSize(40, 22)
        self.swatch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swatch_btn.setToolTip("Click to choose color override")
        self.swatch_btn.clicked.connect(self._open_color_dialog)
        layout.addWidget(self.swatch_btn)

        self.hex_label = QLabel(vec3_to_hex(self._color))
        self.hex_label.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; font-weight: 600;")
        layout.addWidget(self.hex_label)
        layout.addStretch()

        self._update_swatch_style()

    def get_color(self) -> glm.vec3:
        return glm.vec3(self._color)

    def set_color(self, color: glm.vec3, emit_signal: bool = False) -> None:
        self._color = glm.vec3(color)
        self.hex_label.setText(vec3_to_hex(self._color))
        self._update_swatch_style()
        if emit_signal:
            self.color_changed.emit(self.get_color())

    def _update_swatch_style(self) -> None:
        hex_str = vec3_to_hex(self._color)
        p = ThemeManager.palette()
        self.swatch_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {hex_str};
                border: 1.5px solid {p.card_border};
                border-radius: 11px;
            }}
            QPushButton:hover {{
                border: 2px solid {p.accent};
            }}
            """
        )

    def _open_color_dialog(self) -> None:
        init_qcolor = QColor(
            int(glm.clamp(self._color.x, 0.0, 1.0) * 255),
            int(glm.clamp(self._color.y, 0.0, 1.0) * 255),
            int(glm.clamp(self._color.z, 0.0, 1.0) * 255),
        )

        dialog = QColorDialog(init_qcolor, self)
        dialog.setWindowTitle("Select Material Color")
        dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)

        def on_current_color_changed(qc: QColor):
            if qc.isValid():
                new_vec = glm.vec3(qc.redF(), qc.greenF(), qc.blueF())
                self.set_color(new_vec, emit_signal=True)

        dialog.currentColorChanged.connect(on_current_color_changed)

        if dialog.exec():
            final_c = dialog.selectedColor()
            if final_c.isValid():
                new_vec = glm.vec3(final_c.redF(), final_c.greenF(), final_c.blueF())
                self.set_color(new_vec, emit_signal=True)
        else:
            self.set_color(self._color, emit_signal=True)


class ActionIconButton(QPushButton):
    """
    Minimalist rounded icon action button for sub-mesh Isolate/Focus, Visibility Eye, and Relink.
    """

    def __init__(self, text: str, tooltip: str = "", active: bool = False, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(26, 24)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._is_active = active
        self._apply_style()

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self._apply_style()

    def is_active(self) -> bool:
        return self._is_active

    def _apply_style(self) -> None:
        p = ThemeManager.palette()
        if self._is_active:
            bg = p.selected_bg
            border = p.tab_pill_border
            color = p.selected_text
        else:
            bg = p.input_bg
            border = p.card_border
            color = p.text_secondary

        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 7px;
                color: {color};
                font-size: 12px;
                font-weight: 600;
                padding: 1px;
            }}
            QPushButton:hover {{
                background-color: {p.hover_bg};
                border-color: {p.accent};
                color: {p.text_primary};
            }}
            QPushButton:pressed {{
                background-color: {p.accent};
                color: #FFFFFF;
            }}
            """
        )


# =============================================================================
# 4. Juicy Controls: ToggleSwitch, SegmentedControl, IconButton (Session 1)
# =============================================================================

def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    t = min(max(t, 0.0), 1.0)
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


class ToggleSwitch(QCheckBox):
    """
    Animated iOS-style toggle switch with a label.
    Subclasses QCheckBox so the API (setChecked / isChecked / toggled) stays identical.
    """

    TRACK_W = 34
    TRACK_H = 18

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos: float = 0.0
        self._hover: bool = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.toggled.connect(self._start_anim)

    def setChecked(self, checked: bool) -> None:  # noqa: N802 (Qt naming)
        super().setChecked(checked)
        if not self.isVisible():
            self._anim.stop()
            self._pos = 1.0 if checked else 0.0
            self.update()

    def _start_anim(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _on_anim(self, v) -> None:
        self._pos = float(v)
        self.update()

    def showEvent(self, event) -> None:
        self._pos = 1.0 if self.isChecked() else 0.0
        super().showEvent(event)

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text()) + 10 if self.text() else 0
        return QSize(self.TRACK_W + 4 + text_w, max(self.TRACK_H + 6, fm.height() + 6))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(1.0 if self.isEnabled() else 0.45)

        y0 = (self.height() - self.TRACK_H) / 2.0
        track = QRectF(2.0, y0, self.TRACK_W, self.TRACK_H)
        off_col = QColor(p.card_border)
        on_col = QColor(p.accent_hover if self._hover else p.accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_lerp_color(off_col, on_col, self._pos))
        painter.drawRoundedRect(track, self.TRACK_H / 2.0, self.TRACK_H / 2.0)

        knob_d = self.TRACK_H - 4.0
        knob_x = track.left() + 2.0 + (self.TRACK_W - knob_d - 4.0) * self._pos
        # Soft shadow + white knob (slightly bigger while hovered = "juicy")
        grow = 0.8 if self._hover else 0.0
        painter.setBrush(QColor(0, 0, 0, 40))
        painter.drawEllipse(QRectF(knob_x - grow, y0 + 2.6 - grow, knob_d + 2 * grow, knob_d + 2 * grow))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(QRectF(knob_x - grow, y0 + 2.0 - grow, knob_d + 2 * grow, knob_d + 2 * grow))

        if self.text():
            painter.setPen(QColor(p.text_primary))
            text_rect = QRectF(track.right() + 10.0, 0, self.width() - track.right() - 10.0, self.height())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        painter.end()


class SegmentedControl(QWidget):
    """
    Pill-shaped segmented control with an animated sliding highlight.
    Emits currentChanged(int) when the user picks a segment.
    """

    currentChanged = pyqtSignal(int)

    def __init__(self, labels: List[str], parent=None):
        super().__init__(parent)
        self._labels: List[str] = list(labels)
        self._index: int = 0
        self._hover: int = -1
        self._pos: float = 0.0  # animated index position (float)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)

    def current_index(self) -> int:
        return self._index

    def set_current_index(self, index: int, emit_signal: bool = False, animate: bool = True) -> None:
        if not (0 <= index < len(self._labels)):
            return
        changed = index != self._index
        self._index = index
        self._anim.stop()
        if animate and self.isVisible():
            self._anim.setStartValue(self._pos)
            self._anim.setEndValue(float(index))
            self._anim.start()
        else:
            self._pos = float(index)
            self.update()
        if changed and emit_signal:
            self.currentChanged.emit(index)

    def _on_anim(self, v) -> None:
        self._pos = float(v)
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        w = sum(fm.horizontalAdvance(t) + 28 for t in self._labels) + 8
        return QSize(w, 32)

    def _seg_at(self, x: float) -> int:
        n = max(len(self._labels), 1)
        seg_w = (self.width() - 6.0) / n
        return int(min(max((x - 3.0) // seg_w, 0), n - 1))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        h = self._seg_at(event.position().x())
        if h != self._hover:
            self._hover = h
            self.update()

    def leaveEvent(self, event) -> None:
        self._hover = -1
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.set_current_index(self._seg_at(event.position().x()), emit_signal=True)

    def paintEvent(self, event) -> None:
        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(1.0 if self.isEnabled() else 0.45)

        outer = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QPen(QColor(p.card_border), 1.0))
        painter.setBrush(QColor(p.input_bg))
        painter.drawRoundedRect(outer, outer.height() / 2, outer.height() / 2)

        n = max(len(self._labels), 1)
        seg_w = (self.width() - 6.0) / n
        pill = QRectF(3.0 + seg_w * self._pos, 3.0, seg_w, self.height() - 6.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(p.accent))
        painter.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)

        font = QFont(self.font())
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        for i, label in enumerate(self._labels):
            r = QRectF(3.0 + seg_w * i, 0, seg_w, self.height())
            # How much the sliding pill covers this segment -> text turns white
            cover = max(0.0, 1.0 - abs(self._pos - i))
            base = QColor(p.text_primary if i == self._hover else p.text_secondary)
            painter.setPen(_lerp_color(base, QColor("#FFFFFF"), cover))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
        painter.end()


class IconButton(QPushButton):
    """
    Small rounded button that paints a crisp vector icon (no emoji).
    Kinds: 'eye', 'eye_off', 'isolate', 'relink', 'focus'.
    `active=True` fills it with the soft accent color.
    """

    def __init__(self, kind: str, tooltip: str = "", active: bool = False, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._is_active = active
        self._hover = False
        self.setFixedSize(24, 22)
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setStyleSheet(
            "QPushButton { background: transparent; border: none; padding: 0px; margin: 0px; min-height: 0px; min-width: 0px; }"
        )

    # Backwards-compatible API with ActionIconButton
    def set_active(self, active: bool) -> None:
        self._is_active = active
        self.update()

    def is_active(self) -> bool:
        return self._is_active

    def set_kind(self, kind: str) -> None:
        self._kind = kind
        self.update()

    def _apply_style(self) -> None:
        self.update()

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(1.0, 1.0, self.width() - 2.0, self.height() - 2.0)

        if self.isDown():
            bg, fg = QColor(p.accent), QColor("#FFFFFF")
        elif self._is_active:
            bg, fg = QColor(p.accent_soft_bg), QColor(p.accent_soft_text)
        elif self._hover:
            bg, fg = QColor(p.hover_bg), QColor(p.text_primary)
        else:
            bg, fg = QColor(0, 0, 0, 0), QColor(p.text_muted)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(r, 8.0, 8.0)

        pen = QPen(fg, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy = r.center().x(), r.center().y()
        k = self._kind

        if k in ("eye", "eye_off"):
            path = QPainterPath()
            path.moveTo(cx - 7.0, cy)
            path.quadTo(cx, cy - 7.5, cx + 7.0, cy)
            path.quadTo(cx, cy + 7.5, cx - 7.0, cy)
            painter.drawPath(path)
            painter.drawEllipse(QPointF(cx, cy), 2.2, 2.2)
            if k == "eye_off":
                painter.drawLine(QPointF(cx - 6.0, cy + 6.0), QPointF(cx + 6.0, cy - 6.0))
        elif k in ("isolate", "focus"):
            s, c = 6.0, 3.0
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                x, y = cx + sx * s, cy + sy * s
                painter.drawLine(QPointF(x, y), QPointF(x - sx * c, y))
                painter.drawLine(QPointF(x, y), QPointF(x, y - sy * c))
            if k == "focus":
                painter.drawEllipse(QPointF(cx, cy), 1.6, 1.6)
            else:
                painter.drawRect(QRectF(cx - 2.2, cy - 2.2, 4.4, 4.4))
        elif k == "relink":
            # Folder with an up-arrow
            path = QPainterPath()
            path.moveTo(cx - 7.0, cy - 4.0)
            path.lineTo(cx - 3.0, cy - 4.0)
            path.lineTo(cx - 1.5, cy - 2.5)
            path.lineTo(cx + 7.0, cy - 2.5)
            path.lineTo(cx + 7.0, cy + 5.0)
            path.lineTo(cx - 7.0, cy + 5.0)
            path.closeSubpath()
            painter.drawPath(path)
            painter.drawLine(QPointF(cx, cy + 3.0), QPointF(cx, cy - 0.5))
            painter.drawLine(QPointF(cx - 2.0, cy + 1.2), QPointF(cx, cy - 0.8))
            painter.drawLine(QPointF(cx + 2.0, cy + 1.2), QPointF(cx, cy - 0.8))
        painter.end()


def make_color_dot_icon(hex_color: str, size: int = 14) -> "QIcon":
    """Create a round color-dot icon (used for material slots)."""
    from PyQt6.QtGui import QPixmap, QIcon
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor(0, 0, 0, 60), 1.0))
    painter.setBrush(QColor(hex_color))
    painter.drawEllipse(QRectF(1.0, 1.0, size - 2.0, size - 2.0))
    painter.end()
    return QIcon(pm)


class DropTargetTreeWidget(QTreeWidget):
    """
    QTreeWidget supporting drag-and-drop of external image files directly onto a tree row.
    Highlights the hovered target row with an accent border/glow while dragging.
    """

    image_dropped_on_item = pyqtSignal(object, str)  # (QTreeWidgetItem, file_path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.DropOnly)
        self._drag_hover_rect: Optional[QRect] = None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        item = self.itemAt(event.position().toPoint())
        if item is not None and event.mimeData().hasUrls():
            self._drag_hover_rect = self.visualItemRect(item)
            self.viewport().update()
            event.acceptProposedAction()
        else:
            if self._drag_hover_rect is not None:
                self._drag_hover_rect = None
                self.viewport().update()
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._drag_hover_rect = None
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self._drag_hover_rect = None
        self.viewport().update()
        item = self.itemAt(event.position().toPoint())
        if item is not None and event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                filepath = url.toLocalFile()
                ext = os.path.splitext(filepath)[1].lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS and os.path.isfile(filepath):
                    self.image_dropped_on_item.emit(item, filepath)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._drag_hover_rect is not None:
            p = ThemeManager.palette()
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            r = QRectF(self._drag_hover_rect).adjusted(2.0, 1.0, -2.0, -1.0)
            fill = QColor(p.accent)
            fill.setAlpha(40)
            painter.setBrush(fill)
            pen = QPen(QColor(p.accent), 1.8, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRoundedRect(r, 8.0, 8.0)
            painter.end()


# =============================================================================
# 5. Session 2 & 3: Toast Notification, Drop Overlay & Floating Viewport HUD
# =============================================================================

class ToastNotification(QWidget):
    """
    Lightweight floating pill notification at the bottom-right of the viewport.
    Automatically slides up, stays for ~1.8s, and fades out cleanly.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.hide()

        self._text: str = ""
        self._kind: str = "info"  # "info", "success", "warning"
        self._opacity: float = 0.0

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)
        self._anim.finished.connect(self._on_anim_finished)

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._fade_out)
        self._fading_out: bool = False

    def show_toast(self, text: str, kind: str = "info", duration_ms: int = 1850) -> None:
        self._text = text
        self._kind = kind
        self._fading_out = False

        font = QFont(self.font())
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.DemiBold)
        fm = QFontMetrics(font)
        w = fm.horizontalAdvance(text) + 44
        h = 36
        self.resize(w, h)
        self._reposition()
        self.show()
        self.raise_()

        self._hold_timer.stop()
        self._anim.stop()
        self._anim.setStartValue(self._opacity)
        self._anim.setEndValue(1.0)
        self._anim.start()
        self._hold_timer.start(duration_ms)

    def _fade_out(self) -> None:
        self._fading_out = True
        self._anim.stop()
        self._anim.setStartValue(self._opacity)
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_anim_step(self, val) -> None:
        self._opacity = float(val)
        self._reposition()
        self.update()

    def _on_anim_finished(self) -> None:
        if self._fading_out and self._opacity <= 0.02:
            self.hide()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        margin_x = 18
        base_y = parent.height() - self.height() - 18
        slide_offset = int((1.0 - self._opacity) * 10.0)
        self.move(max(margin_x, parent.width() - self.width() - margin_x), base_y + slide_offset)

    def paintEvent(self, event) -> None:
        if self._opacity <= 0.01:
            return
        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(self._opacity)

        r = QRectF(1.0, 1.0, self.width() - 2.0, self.height() - 2.0)
        bg = QColor(p.card_bg)
        bg.setAlpha(242)
        border_col = QColor(
            p.success if self._kind == "success" else (p.warning if self._kind == "warning" else p.accent)
        )
        painter.setPen(QPen(border_col, 1.5))
        painter.setBrush(bg)
        painter.drawRoundedRect(r, r.height() * 0.5, r.height() * 0.5)

        # Status dot on the left
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(border_col)
        painter.drawEllipse(QPointF(16.0, r.center().y()), 4.5, 4.5)

        font = QFont(self.font())
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor(p.text_primary))
        text_rect = QRectF(28.0, 0.0, self.width() - 40.0, self.height())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text)
        painter.end()


class ViewportDropOverlay(QWidget):
    """
    Translucent dashed card overlay shown on top of the 3D viewport when
    the user drags a supported 3D file over the viewport.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.hide()

    def sync_geometry(self) -> None:
        parent = self.parentWidget()
        if parent:
            self.setGeometry(0, 0, parent.width(), parent.height())

    def paintEvent(self, event) -> None:
        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Subtle backdrop dim
        backdrop = QColor(p.window_bg)
        backdrop.setAlpha(150 if p.is_dark else 170)
        painter.fillRect(self.rect(), backdrop)

        # Centered card
        card_w = min(340.0, self.width() - 48.0)
        card_h = 150.0
        cx, cy = self.width() * 0.5, self.height() * 0.5
        card_rect = QRectF(cx - card_w * 0.5, cy - card_h * 0.5, card_w, card_h)

        card_bg = QColor(p.card_bg)
        card_bg.setAlpha(240)
        painter.setBrush(card_bg)
        pen = QPen(QColor(p.accent), 2.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(card_rect, 18.0, 18.0)

        # Draw 3D cube icon inside soft accent circle
        badge_r = 24.0
        badge_center = QPointF(cx, card_rect.top() + 46.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(p.accent_soft_bg))
        painter.drawEllipse(badge_center, badge_r, badge_r)

        draw_monoline_icon(painter, "meshes", badge_center, 22.0, QColor(p.accent_soft_text))

        # Title text
        font_title = QFont(self.font())
        font_title.setPixelSize(14)
        font_title.setWeight(QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(p.text_primary))
        title_rect = QRectF(card_rect.left(), card_rect.top() + 80.0, card_w, 24.0)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "Drop 3D Asset to Open")

        # Subtitle text
        font_sub = QFont(self.font())
        font_sub.setPixelSize(11)
        painter.setFont(font_sub)
        painter.setPen(QColor(p.text_muted))
        sub_rect = QRectF(card_rect.left(), card_rect.top() + 104.0, card_w, 20.0)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, "FBX · glTF / GLB · OBJ · DAE · STL · PLY")
        painter.end()


class ViewportLoadingOverlay(QWidget):
    """
    Sleek centered status card displayed on top of the 3D viewport while a heavy 3D asset is loading.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._filename: str = ""
        self.hide()

    def show_loading(self, filename: str) -> None:
        self._filename = filename
        parent = self.parentWidget()
        if parent:
            self.setGeometry(0, 0, parent.width(), parent.height())
        self.show()
        self.raise_()
        self.repaint()

    def paintEvent(self, event) -> None:
        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        backdrop = QColor(p.window_bg)
        backdrop.setAlpha(135 if p.is_dark else 155)
        painter.fillRect(self.rect(), backdrop)

        card_w = min(320.0, self.width() - 48.0)
        card_h = 96.0
        cx, cy = self.width() * 0.5, self.height() * 0.5
        card_rect = QRectF(cx - card_w * 0.5, cy - card_h * 0.5, card_w, card_h)

        card_bg = QColor(p.card_bg)
        card_bg.setAlpha(245)
        painter.setBrush(card_bg)
        painter.setPen(QPen(QColor(p.accent), 1.5))
        painter.drawRoundedRect(card_rect, 16.0, 16.0)

        font_title = QFont(self.font())
        font_title.setPixelSize(13)
        font_title.setWeight(QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QColor(p.text_primary))
        title_rect = QRectF(card_rect.left() + 16.0, card_rect.top() + 24.0, card_w - 32.0, 22.0)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "Loading 3D Asset…")

        font_sub = QFont(self.font())
        font_sub.setPixelSize(11)
        painter.setFont(font_sub)
        painter.setPen(QColor(p.accent_soft_text))
        sub_rect = QRectF(card_rect.left() + 16.0, card_rect.top() + 50.0, card_w - 32.0, 20.0)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, self._filename)
        painter.end()


class MeasurementHUDCard(QWidget):
    """
    Juicy floating HUD pill card displayed during 3D Measure mode.
    Displays:
    - Interactive live guidance ("Click surface for Point 1", "Point 1 placed")
    - Precise 3D distance (<o---o> 39.755) and inclination angle (📐 13.2°)
    - Clear and dismiss buttons
    """
    clear_requested = pyqtSignal()
    close_requested = pyqtSignal()
    snap_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.hide()

        self._distance: Optional[float] = None
        self._angle: Optional[float] = None
        self._state_text: str = "Click surface to place Point 1"
        self._has_points: bool = False
        self._opacity: float = 0.0

        self._snap_enabled: bool = True
        self._btn_snap_rect = QRectF()
        self._btn_clear_rect = QRectF()
        self._btn_close_rect = QRectF()
        self._is_snap_hovered: bool = False
        self._is_clear_hovered: bool = False
        self._is_close_hovered: bool = False

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)
        self._anim.finished.connect(self._on_anim_finished)

    def is_snap_enabled(self) -> bool:
        """Return True if vertex snapping is active for measurements."""
        return self._snap_enabled

    def set_snap_enabled(self, enabled: bool) -> None:
        self._snap_enabled = enabled
        self.update()

    def set_measuring_state(
        self,
        point_count: int,
        distance: Optional[float] = None,
        angle: Optional[float] = None,
    ) -> None:
        self._distance = distance
        self._angle = angle
        self._has_points = (point_count > 0)

        if point_count == 0:
            self._state_text = "Click surface to place Point 1"
        elif point_count == 1:
            self._state_text = "Point 1 placed · Click surface for Point 2"
        else:
            self._state_text = ""

        self._update_size_and_position()
        self.update()

    def show_card(self) -> None:
        self.show()
        self.raise_()
        self._anim.stop()
        self._anim.setStartValue(self._opacity)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def hide_card(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._opacity)
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_anim_step(self, val) -> None:
        self._opacity = float(val)
        self._update_size_and_position()
        self.update()

    def _on_anim_finished(self) -> None:
        if self._opacity <= 0.02:
            self.hide()

    def _update_size_and_position(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return

        w = 385 if self._distance is not None else 330
        h = 38
        self.resize(w, h)

        base_y = parent.height() - h - 22
        slide_offset = int((1.0 - self._opacity) * 12.0)
        self.move((parent.width() - w) // 2, base_y + slide_offset)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if self._btn_snap_rect.contains(pos):
                self._snap_enabled = not self._snap_enabled
                self.snap_toggled.emit(self._snap_enabled)
                self.update()
                event.accept()
                return
            if self._btn_clear_rect.contains(pos):
                self.clear_requested.emit()
                event.accept()
                return
            if self._btn_close_rect.contains(pos):
                self.close_requested.emit()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        was_snap = self._is_snap_hovered
        was_clear = self._is_clear_hovered
        was_close = self._is_close_hovered
        self._is_snap_hovered = self._btn_snap_rect.contains(pos)
        self._is_clear_hovered = self._btn_clear_rect.contains(pos)
        self._is_close_hovered = self._btn_close_rect.contains(pos)
        if (
            was_snap != self._is_snap_hovered
            or was_clear != self._is_clear_hovered
            or was_close != self._is_close_hovered
        ):
            self.update()
        if self._is_snap_hovered or self._is_clear_hovered or self._is_close_hovered:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        if self._opacity <= 0.01:
            return

        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(self._opacity)

        w = float(self.width())
        h = float(self.height())
        r = QRectF(1.0, 1.0, w - 2.0, h - 2.0)

        # Background capsule
        bg = QColor(p.card_bg)
        bg.setAlpha(245)
        painter.setBrush(bg)
        painter.setPen(QPen(QColor(p.accent), 1.5))
        painter.drawRoundedRect(r, h * 0.5, h * 0.5)

        if self._distance is not None and self._angle is not None:
            # Mode with 2 points measured: show Distance and Angle side-by-side
            font_val = QFont(self.font())
            font_val.setPixelSize(12)
            font_val.setWeight(QFont.Weight.Bold)
            painter.setFont(font_val)

            # Left section: Distance readout
            dist_str = f"Dist: {self._distance:.3f}"
            painter.setPen(QColor(p.accent))
            dist_rect = QRectF(16.0, 0.0, 105.0, h)
            painter.drawText(dist_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, dist_str)

            # Divider line
            painter.setPen(QPen(QColor(p.card_border), 1.0))
            painter.drawLine(int(125), int(8), int(125), int(h - 8))

            # Middle section: Angle readout
            angle_str = f"Ang: {self._angle:.1f}°"
            painter.setPen(QColor(p.success if hasattr(p, 'success') else p.accent))
            angle_rect = QRectF(135.0, 0.0, 90.0, h)
            painter.drawText(angle_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, angle_str)

            # Snap toggle button
            btn_snap_x = w - 124.0
            self._btn_snap_rect = QRectF(btn_snap_x, 7.0, 56.0, 24.0)
            if self._snap_enabled:
                painter.setBrush(QColor(p.accent_soft_bg))
                painter.setPen(QPen(QColor(p.accent), 1.2))
            else:
                painter.setBrush(QColor(p.card_bg))
                painter.setPen(QPen(QColor(p.card_border), 1.0))
            if self._is_snap_hovered:
                painter.setPen(QPen(QColor(p.accent), 1.8))
            painter.drawRoundedRect(self._btn_snap_rect, 12.0, 12.0)

            font_snap = QFont(self.font())
            font_snap.setPixelSize(10)
            font_snap.setWeight(QFont.Weight.Bold)
            painter.setFont(font_snap)
            painter.setPen(QColor(p.accent if self._snap_enabled else p.text_secondary))
            painter.drawText(self._btn_snap_rect, Qt.AlignmentFlag.AlignCenter, "Snap")

            # Clear button
            btn_clear_x = w - 62.0
            self._btn_clear_rect = QRectF(btn_clear_x, 6.0, 24.0, 26.0)
            if self._is_clear_hovered:
                painter.setBrush(QColor(p.accent_soft_bg))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(self._btn_clear_rect, 6.0, 6.0)
            painter.setPen(QColor(p.text_secondary))
            font_btn = QFont(self.font())
            font_btn.setPixelSize(12)
            font_btn.setWeight(QFont.Weight.Bold)
            painter.setFont(font_btn)
            painter.drawText(self._btn_clear_rect, Qt.AlignmentFlag.AlignCenter, "↺")

            # Close / Dismiss button
            btn_close_x = w - 34.0
            self._btn_close_rect = QRectF(btn_close_x, 6.0, 24.0, 26.0)
            if self._is_close_hovered:
                painter.setBrush(QColor(p.danger if hasattr(p, 'danger') else '#FF4A6E'))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(self._btn_close_rect, 6.0, 6.0)
                painter.setPen(QColor('#FFFFFF'))
            else:
                painter.setPen(QColor(p.text_secondary))
            painter.drawText(self._btn_close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        else:
            # Snap toggle button
            btn_snap_x = w - 96.0
            self._btn_snap_rect = QRectF(btn_snap_x, 7.0, 56.0, 24.0)
            if self._snap_enabled:
                painter.setBrush(QColor(p.accent_soft_bg))
                painter.setPen(QPen(QColor(p.accent), 1.2))
            else:
                painter.setBrush(QColor(p.card_bg))
                painter.setPen(QPen(QColor(p.card_border), 1.0))
            if self._is_snap_hovered:
                painter.setPen(QPen(QColor(p.accent), 1.8))
            painter.drawRoundedRect(self._btn_snap_rect, 12.0, 12.0)

            font_snap = QFont(self.font())
            font_snap.setPixelSize(10)
            font_snap.setWeight(QFont.Weight.Bold)
            painter.setFont(font_snap)
            painter.setPen(QColor(p.accent if self._snap_enabled else p.text_secondary))
            painter.drawText(self._btn_snap_rect, Qt.AlignmentFlag.AlignCenter, "Snap")

            # Guidance text
            font_txt = QFont(self.font())
            font_txt.setPixelSize(11)
            font_txt.setWeight(QFont.Weight.Medium)
            painter.setFont(font_txt)
            painter.setPen(QColor(p.text_primary))
            text_rect = QRectF(16.0, 0.0, btn_snap_x - 22.0, h)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._state_text)

            # Close button
            btn_close_x = w - 34.0
            self._btn_close_rect = QRectF(btn_close_x, 6.0, 24.0, 26.0)
            self._btn_clear_rect = QRectF()
            if self._is_close_hovered:
                painter.setBrush(QColor(p.danger if hasattr(p, 'danger') else '#FF4A6E'))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(self._btn_close_rect, 6.0, 6.0)
                painter.setPen(QColor('#FFFFFF'))
            else:
                painter.setPen(QColor(p.text_secondary))
            font_btn = QFont(self.font())
            font_btn.setPixelSize(11)
            font_btn.setWeight(QFont.Weight.Bold)
            painter.setFont(font_btn)
            painter.drawText(self._btn_close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        painter.end()





class BoundingBoxHUDCard(QWidget):
    """
    Sleek floating HUD card displaying real-time 3D Bounding Box dimensions (Width, Height, Depth, Volume).
    """
    close_requested = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.hide()

        self._bbox_info: Optional[BoundingBoxInfo] = None
        self._opacity: float = 0.0
        self._btn_close_rect = QRectF()
        self._is_close_hovered: bool = False

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)
        self._anim.finished.connect(self._on_anim_finished)

    def set_bbox_info(self, info: Optional[BoundingBoxInfo]) -> None:
        self._bbox_info = info
        self._update_size_and_position()
        self.update()

    def show_card(self) -> None:
        self.show()
        self._anim.stop()
        self._anim.setStartValue(self._opacity)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def hide_card(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._opacity)
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_anim_step(self, value) -> None:
        self._opacity = float(value)
        self._update_size_and_position()
        self.update()

    def _on_anim_finished(self) -> None:
        if self._opacity <= 0.01:
            self.hide()

    def _update_size_and_position(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return

        font_dim = QFont(self.font())
        font_dim.setPixelSize(11)
        font_dim.setWeight(QFont.Weight.Bold)
        fm = QFontMetrics(font_dim)

        font_vol = QFont(self.font())
        font_vol.setPixelSize(11)
        font_vol.setWeight(QFont.Weight.Bold)
        fm_vol = QFontMetrics(font_vol)

        sx = self._bbox_info.width if (self._bbox_info and self._bbox_info.is_valid) else 0.0
        sy = self._bbox_info.height if (self._bbox_info and self._bbox_info.is_valid) else 0.0
        sz = self._bbox_info.depth if (self._bbox_info and self._bbox_info.is_valid) else 0.0
        vol = self._bbox_info.volume if (self._bbox_info and self._bbox_info.is_valid) else 0.0

        w_str = f"W {sx:.2f}m"
        h_str = f"H {sy:.2f}m"
        d_str = f"D {sz:.2f}m"
        vol_str = f"{vol:.2f}m³"

        # Dynamically measure content to prevent text clipping
        content_w = (
            20.0  # left margin
            + fm.horizontalAdvance(w_str)
            + 24.0  # separator 1
            + fm.horizontalAdvance(h_str)
            + 24.0  # separator 2
            + fm.horizontalAdvance(d_str)
            + 24.0  # separator 3
            + fm_vol.horizontalAdvance(vol_str)
            + 32.0  # spacing before close button
            + 20.0  # close button
            + 16.0  # right margin
        )
        w = max(390, int(content_w))
        h = 38
        self.resize(w, h)
        base_y = parent.height() - h - 22
        # If measurement HUD is active, stack neatly above it
        if hasattr(parent, "hud_measure") and parent.hud_measure.isVisible():
            base_y -= (h + 8)
        slide_offset = int((1.0 - self._opacity) * 12.0)
        self.move((parent.width() - w) // 2, base_y + slide_offset)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if self._btn_close_rect.contains(pos):
                self.close_requested.emit()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        was_close = self._is_close_hovered
        self._is_close_hovered = self._btn_close_rect.contains(pos)
        if was_close != self._is_close_hovered:
            self.update()
        if self._is_close_hovered:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        if self._opacity <= 0.01:
            return

        p = ThemeManager.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(self._opacity)

        w = float(self.width())
        h = float(self.height())
        r = QRectF(1.0, 1.0, w - 2.0, h - 2.0)

        # Background stadium capsule matching MeasurementHUDCard
        bg = QColor(p.card_bg)
        bg.setAlpha(245)
        painter.setBrush(bg)
        painter.setPen(QPen(QColor(p.card_border), 1.2))
        painter.drawRoundedRect(r, h * 0.5, h * 0.5)

        # Close button (far right)
        btn_size = 20.0
        self._btn_close_rect = QRectF(w - 28.0, (h - btn_size) * 0.5, btn_size, btn_size)
        if self._is_close_hovered:
            painter.setBrush(QColor('#FF4A6E'))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self._btn_close_rect, btn_size * 0.5, btn_size * 0.5)
            painter.setPen(QColor('#FFFFFF'))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QColor(p.text_secondary))
        font_btn = QFont(self.font())
        font_btn.setPixelSize(10)
        font_btn.setWeight(QFont.Weight.Bold)
        painter.setFont(font_btn)
        painter.drawText(self._btn_close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        # Dimensions row: W (X), H (Y), D (Z) and Volume (m3)
        if self._bbox_info and self._bbox_info.is_valid:
            sx = self._bbox_info.width
            sy = self._bbox_info.height
            sz = self._bbox_info.depth

            font_dim = QFont(self.font())
            font_dim.setPixelSize(11)
            font_dim.setWeight(QFont.Weight.Bold)
            fm = QFontMetrics(font_dim)

            font_sep = QFont(self.font())
            font_sep.setPixelSize(10)

            font_vol = QFont(self.font())
            font_vol.setPixelSize(11)
            font_vol.setWeight(QFont.Weight.Bold)
            fm_vol = QFontMetrics(font_vol)

            w_str = f"W {sx:.2f}m"
            h_str = f"H {sy:.2f}m"
            d_str = f"D {sz:.2f}m"
            vol_str = f"{self._bbox_info.volume:.2f}m³"

            curr_x = 20.0

            # W: Coral/Red
            painter.setFont(font_dim)
            painter.setPen(QColor('#FF5C5C'))
            w_w = fm.horizontalAdvance(w_str)
            painter.drawText(QRectF(curr_x, 0.0, w_w + 2.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, w_str)
            curr_x += w_w + 10.0

            # Separator 1
            painter.setFont(font_sep)
            painter.setPen(QColor(p.card_border))
            painter.drawText(QRectF(curr_x, 0.0, 8.0, h), Qt.AlignmentFlag.AlignCenter, "│")
            curr_x += 8.0 + 10.0

            # H: Emerald Green
            painter.setFont(font_dim)
            painter.setPen(QColor('#4ADE80'))
            h_w = fm.horizontalAdvance(h_str)
            painter.drawText(QRectF(curr_x, 0.0, h_w + 2.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, h_str)
            curr_x += h_w + 10.0

            # Separator 2
            painter.setFont(font_sep)
            painter.setPen(QColor(p.card_border))
            painter.drawText(QRectF(curr_x, 0.0, 8.0, h), Qt.AlignmentFlag.AlignCenter, "│")
            curr_x += 8.0 + 10.0

            # D: Sky Blue
            painter.setFont(font_dim)
            painter.setPen(QColor('#38BDF8'))
            d_w = fm.horizontalAdvance(d_str)
            painter.drawText(QRectF(curr_x, 0.0, d_w + 2.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, d_str)
            curr_x += d_w + 10.0

            # Separator 3
            painter.setFont(font_sep)
            painter.setPen(QColor(p.card_border))
            painter.drawText(QRectF(curr_x, 0.0, 8.0, h), Qt.AlignmentFlag.AlignCenter, "│")
            curr_x += 8.0 + 10.0

            # Volume: Soft Violet/Purple
            painter.setFont(font_vol)
            painter.setPen(QColor('#C084FC'))
            vol_w = fm_vol.horizontalAdvance(vol_str)
            painter.drawText(QRectF(curr_x, 0.0, vol_w + 6.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, vol_str)
        else:
            font_dim = QFont(self.font())
            font_dim.setPixelSize(11)
            painter.setFont(font_dim)
            painter.setPen(QColor(p.text_secondary))
            painter.drawText(QRectF(20.0, 0.0, 280.0, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "No active model bounds")

        painter.end()


class BlenderOrientationGizmo(QWidget):
    """
    Blender-style 3D Viewport Orientation Navigation Widget.
    Features:
    - 6 axes: +X (Red), +Y (Green), +Z (Blue) solid circles with text 'X', 'Y', 'Z';
      -X, -Y, -Z translucent circular rings.
    - True 3D depth-sorting based on camera orientation matrix.
    - 1-click view snapping (+X Right, -X Left, +Y Top, -Y Bottom, +Z Front, -Z Back).
    - Mouse drag to orbit the camera smoothly in real time.
    - Interactive hover feedback with outer highlight halo.
    """
    view_snap_requested = pyqtSignal(float, float, str)
    orbit_requested = pyqtSignal(float, float)

    AXIS_PRESETS = [
        ("+X", glm.vec3(1.0, 0.0, 0.0), 90.0, 0.0, "Right View (+X)", "#EF4444", "X", True),
        ("-X", glm.vec3(-1.0, 0.0, 0.0), -90.0, 0.0, "Left View (-X)", "#EF4444", "", False),
        ("+Y", glm.vec3(0.0, 1.0, 0.0), 0.0, 89.0, "Top View (+Y)", "#65A30D", "Y", True),
        ("-Y", glm.vec3(0.0, -1.0, 0.0), 0.0, -89.0, "Bottom View (-Y)", "#65A30D", "", False),
        ("+Z", glm.vec3(0.0, 0.0, 1.0), 0.0, 0.0, "Front View (+Z)", "#2563EB", "Z", True),
        ("-Z", glm.vec3(0.0, 0.0, -1.0), 180.0, 0.0, "Back View (-Z)", "#2563EB", "", False),
    ]

    def __init__(self, parent: QWidget, camera: any):
        super().__init__(parent)
        self.camera = camera
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setFixedSize(96, 96)

        self._hovered_axis: Optional[str] = None
        self._is_dragging: bool = False
        self._press_pos: QPointF = QPointF()
        self._last_mouse_pos: QPointF = QPointF()

    def _get_projected_axes(self) -> List[Tuple[str, float, float, float, str, str, bool, float, float, str]]:
        """
        Projects all 6 axis endpoints into widget 2D coordinates.
        Returns list of (label, sx, sy, sz, color_hex, letter, is_pos, yaw, pitch, name).
        """
        cx = float(self.width()) * 0.5
        cy = float(self.height()) * 0.5
        arm_radius = 34.0

        rot_view = self.camera.get_orientation_matrix()
        projected = []

        for label, world_dir, yaw, pitch, name, color_hex, letter, is_pos in self.AXIS_PRESETS:
            v_view = rot_view * glm.vec4(world_dir, 0.0)
            sx = cx + float(v_view.x) * arm_radius
            sy = cy - float(v_view.y) * arm_radius
            sz = float(v_view.z)
            projected.append((label, sx, sy, sz, color_hex, letter, is_pos, yaw, pitch, name))

        return projected

    def _hit_test(self, mx: float, my: float) -> Optional[Tuple[str, float, float, str]]:
        """Hit test mouse coordinate against all 6 projected axis circles."""
        projected = self._get_projected_axes()
        best_match = None
        best_depth = -float("inf")
        best_dist = float("inf")

        for label, sx, sy, sz, _, _, is_pos, yaw, pitch, name in projected:
            hit_rad = 12.0 if is_pos else 9.0
            dist = math.hypot(mx - sx, my - sy)
            if dist <= hit_rad:
                if sz > best_depth or (abs(sz - best_depth) < 0.15 and dist < best_dist):
                    best_depth = sz
                    best_dist = dist
                    best_match = (label, yaw, pitch, name)

        return best_match

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position()
            self._last_mouse_pos = event.position()
            self._is_dragging = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._is_dragging:
            dx = float(pos.x() - self._last_mouse_pos.x())
            dy = float(pos.y() - self._last_mouse_pos.y())
            self._last_mouse_pos = pos
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self.orbit_requested.emit(dx, dy)
            self.update()
            event.accept()
            return

        hit = self._hit_test(pos.x(), pos.y())
        hover_label = hit[0] if hit else None
        if hover_label != self._hovered_axis:
            self._hovered_axis = hover_label
            self.update()

        if self._hovered_axis is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            drag_dist = (event.position() - self._press_pos).manhattanLength()
            self._is_dragging = False

            if drag_dist <= 4.0:
                hit = self._hit_test(event.position().x(), event.position().y())
                if hit is not None:
                    _, yaw, pitch, name = hit
                    self.view_snap_requested.emit(yaw, pitch, name)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cx = float(self.width()) * 0.5
        cy = float(self.height()) * 0.5

        # Background circular plate (subtle dark translucent, Blender style)
        plate_color = QColor(24, 26, 32)
        plate_color.setAlpha(125)
        painter.setBrush(plate_color)
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1.0))
        painter.drawEllipse(QPointF(cx, cy), 43.0, 43.0)

        # Get projected 3D axes and sort by view depth (sz ascending: back to front)
        projected = self._get_projected_axes()
        projected.sort(key=lambda item: item[3])

        # 1. Draw positive stems (lines from center to positive endpoints)
        for label, sx, sy, sz, color_hex, _, is_pos, _, _, _ in projected:
            if is_pos:
                pen = QPen(QColor(color_hex), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(cx, cy), QPointF(sx, sy))

        # 2. Draw center origin dot
        painter.setBrush(QColor(160, 175, 195, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 2.8, 2.8)

        # 3. Draw depth-sorted circles
        font_axis = QFont(self.font())
        font_axis.setPixelSize(10)
        font_axis.setWeight(QFont.Weight.Bold)

        for label, sx, sy, sz, color_hex, letter, is_pos, _, _, _ in projected:
            is_hover = (self._hovered_axis == label)

            if is_pos:
                # Solid circle (+X Red, +Y Green, +Z Blue)
                rad = 11.5 if is_hover else 10.0
                col = QColor(color_hex)
                if is_hover:
                    col = col.lighter(120)

                painter.setBrush(col)
                if is_hover:
                    painter.setPen(QPen(QColor(255, 255, 255, 230), 1.8))
                else:
                    painter.setPen(QPen(QColor(255, 255, 255, 40), 0.8))
                painter.drawEllipse(QPointF(sx, sy), rad, rad)

                # Axis letter (X, Y, Z) in crisp bold text
                painter.setFont(font_axis)
                painter.setPen(QColor('#FFFFFF'))
                rect_text = QRectF(sx - rad, sy - rad, rad * 2.0, rad * 2.0)
                painter.drawText(rect_text, Qt.AlignmentFlag.AlignCenter, letter)
            else:
                # Translucent ring (-X, -Y, -Z)
                rad = 8.5 if is_hover else 7.5
                col_border = QColor(color_hex)
                col_border.setAlpha(255 if is_hover else 160)

                col_fill = QColor(color_hex)
                col_fill.setAlpha(100 if is_hover else 40)

                painter.setBrush(col_fill)
                painter.setPen(QPen(col_border, 1.8 if is_hover else 1.4))
                painter.drawEllipse(QPointF(sx, sy), rad, rad)

        painter.end()
