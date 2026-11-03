"""
ui/style.py
Minimalist ("Juicy Simple") Theme Engine & Stylesheets for UniView 2.0.
Supports instant runtime switching between:
- Dark Mode (Obsidian & Indigo Glow — matching the Expanding Label reference)
- Light Mode (Alabaster & Soft Violet — matching the Minimalist Workspace reference)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ThemePalette:
    name: str
    is_dark: bool

    # Window & Surface Colors
    window_bg: str
    panel_bg: str
    card_bg: str
    card_border: str
    input_bg: str
    hover_bg: str
    selected_bg: str
    selected_text: str

    # Typography
    text_primary: str
    text_secondary: str
    text_muted: str

    # Accents
    accent: str
    accent_hover: str
    accent_soft_bg: str
    accent_soft_text: str

    # Expanding Label Tab Bar Colors (Matching Reference Image)
    tab_outer_bg: str
    tab_outer_border: str
    tab_pill_bg: str
    tab_pill_border: str
    tab_active_fg: str
    tab_inactive_fg: str
    tab_hover_fg: str

    # Semantic Badges
    success: str
    warning: str
    danger: str

    # 3D OpenGL Viewport & Grid Colors (RGB floats 0..1)
    viewport_bg_rgb: Tuple[float, float, float]
    grid_line_rgb: Tuple[float, float, float]


DARK_PALETTE = ThemePalette(
    name="dark",
    is_dark=True,
    window_bg="#111018",
    panel_bg="#161520",
    card_bg="#1D1B29",
    card_border="#2B283D",
    input_bg="#14131D",
    hover_bg="#282538",
    selected_bg="#393464",
    selected_text="#FFFFFF",
    text_primary="#F4F3FA",
    text_secondary="#C4C1D8",
    text_muted="#85819E",
    accent="#7C5CFC",
    accent_hover="#9175FF",
    accent_soft_bg="#2B254A",
    accent_soft_text="#D5C8FF",
    # Exact match to the "5 Expanding Label" dark reference image
    tab_outer_bg="#191724",
    tab_outer_border="#2C293E",
    tab_pill_bg="#393464",
    tab_pill_border="#6E66A8",
    tab_active_fg="#FFFFFF",
    tab_inactive_fg="#8E8CA3",
    tab_hover_fg="#D8D6E8",
    success="#34D399",
    warning="#FBBF24",
    danger="#F87171",
    viewport_bg_rgb=(0.085, 0.082, 0.115),
    grid_line_rgb=(0.34, 0.33, 0.43),
)

LIGHT_PALETTE = ThemePalette(
    name="light",
    is_dark=False,
    window_bg="#F3F4F9",
    panel_bg="#F8F9FC",
    card_bg="#FFFFFF",
    card_border="#E5E6F0",
    input_bg="#F2F3F8",
    hover_bg="#EFEAFF",
    selected_bg="#EDE8FF",
    selected_text="#4E31CC",
    text_primary="#1D1E2C",
    text_secondary="#4A4D63",
    text_muted="#8589A0",
    accent="#7C5CFC",
    accent_hover="#6846F5",
    accent_soft_bg="#EDE8FF",
    accent_soft_text="#5B37EB",
    # Clean Minimalist Light counterpart for Expanding Label bar
    tab_outer_bg="#ECEAF6",
    tab_outer_border="#DCD8EE",
    tab_pill_bg="#FFFFFF",
    tab_pill_border="#7C5CFC",
    tab_active_fg="#4E31CC",
    tab_inactive_fg="#78758F",
    tab_hover_fg="#3C3760",
    success="#10B981",
    warning="#D97706",
    danger="#EF4444",
    viewport_bg_rgb=(0.925, 0.930, 0.955),
    grid_line_rgb=(0.70, 0.72, 0.78),
)


def build_stylesheet(p: ThemePalette) -> str:
    """Compile a complete, high-performance Qt Stylesheet for the given palette."""
    return f"""
/* =========================================================================
   Global Minimalist Typography & Surface
   ========================================================================= */
QWidget {{
    background-color: {p.panel_bg};
    color: {p.text_primary};
    font-family: "Segoe UI Variable Display", "Segoe UI", "Inter", "SF Pro Display", Arial, sans-serif;
    font-size: 12px;
    selection-background-color: {p.accent};
    selection-color: #FFFFFF;
}}

QMainWindow {{
    background-color: {p.window_bg};
}}

QMainWindow::separator {{
    background-color: {p.card_border};
    width: 1px;
    height: 1px;
}}

QMainWindow::separator:hover {{
    background-color: {p.accent};
}}

/* =========================================================================
   Minimalist Menu Bar & Dropdown Menus
   ========================================================================= */
QMenuBar {{
    background-color: {p.window_bg};
    color: {p.text_secondary};
    border-bottom: 1px solid {p.card_border};
    padding: 3px 8px;
    font-weight: 500;
}}

QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 6px;
}}

QMenuBar::item:selected {{
    background-color: {p.hover_bg};
    color: {p.text_primary};
}}

QMenuBar::item:pressed {{
    background-color: {p.accent};
    color: #FFFFFF;
}}

QMenu {{
    background-color: {p.card_bg};
    color: {p.text_primary};
    border: 1px solid {p.card_border};
    padding: 6px;
    border-radius: 10px;
}}

QMenu::item {{
    padding: 7px 26px 7px 14px;
    border-radius: 6px;
    background: transparent;
}}

QMenu::item:selected {{
    background-color: {p.selected_bg};
    color: {p.selected_text};
    font-weight: 600;
}}

QMenu::separator {{
    height: 1px;
    background-color: {p.card_border};
    margin: 5px 8px;
}}

/* =========================================================================
   Top Header Bar & Status Bar
   ========================================================================= */
QFrame#TopHeaderBar {{
    background-color: {p.card_bg};
    border-bottom: 1px solid {p.card_border};
}}

QLabel#BrandBadge {{
    background-color: {p.accent};
    color: #FFFFFF;
    font-weight: 700;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 8px;
}}

QLabel#ActiveFilePill {{
    background-color: {p.input_bg};
    color: {p.text_secondary};
    border: 1px solid {p.card_border};
    font-weight: 600;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 11px;
}}

QStatusBar {{
    background-color: {p.card_bg};
    color: {p.text_secondary};
    border-top: 1px solid {p.card_border};
    font-size: 11px;
    padding: 2px 10px;
}}

QStatusBar::item {{
    border: none;
}}

QStatusBar QLabel {{
    background: transparent;
    color: {p.text_secondary};
    padding: 0 6px;
}}

/* =========================================================================
   Dock Widget & Inspector Container
   ========================================================================= */
QDockWidget {{
    color: {p.text_primary};
    font-weight: 600;
    titlebar-close-icon: url(none);
    titlebar-normal-icon: url(none);
}}

QDockWidget::title {{
    background-color: {p.panel_bg};
    padding: 4px 12px;
    border-bottom: none;
}}

/* =========================================================================
   Minimalist Cards (QGroupBox & Card Frames)
   ========================================================================= */
QGroupBox {{
    background-color: {p.card_bg};
    border: 1px solid {p.card_border};
    border-radius: 12px;
    margin-top: 8px;
    padding: 16px 10px 10px 10px;
    font-weight: 600;
    color: {p.text_primary};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 0px;
    padding: 2px 8px;
    border-radius: 6px;
    background-color: {p.accent_soft_bg};
    color: {p.accent_soft_text};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.4px;
}}

QGroupBox QLabel, QGroupBox QCheckBox, QGroupBox QRadioButton {{
    background-color: transparent;
}}

QFrame[card="true"] {{
    background-color: {p.card_bg};
    border: 1px solid {p.card_border};
    border-radius: 12px;
}}

/* =========================================================================
   Tree & List Views (Clean Rounded Cards with Generous Row Padding)
   ========================================================================= */
QTreeWidget, QListView, QListWidget {{
    background-color: {p.card_bg};
    color: {p.text_primary};
    border: 1px solid {p.card_border};
    border-radius: 12px;
    outline: none;
    padding: 6px;
}}

QTreeWidget::item, QListWidget::item {{
    padding: 6px 8px;
    border-radius: 8px;
    margin: 2px 2px;
}}

QTreeWidget::item:hover, QListWidget::item:hover {{
    background-color: {p.hover_bg};
    color: {p.text_primary};
}}

QTreeWidget::item:selected, QListWidget::item:selected {{
    background-color: {p.selected_bg};
    color: {p.selected_text};
    font-weight: 600;
}}

QHeaderView::section {{
    background-color: {p.card_bg};
    color: {p.text_muted};
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid {p.card_border};
    font-weight: 600;
    font-size: 11px;
}}

/* =========================================================================
   Buttons & Pill Controls
   ========================================================================= */
QPushButton {{
    background-color: {p.card_bg};
    color: {p.text_primary};
    border: 1px solid {p.card_border};
    border-radius: 8px;
    padding: 6px 13px;
    font-weight: 600;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {p.hover_bg};
    border-color: {p.accent};
}}

QPushButton:pressed {{
    background-color: {p.accent};
    border-color: {p.accent};
    color: #FFFFFF;
}}

QPushButton:checked {{
    background-color: {p.selected_bg};
    border-color: {p.tab_pill_border};
    color: {p.selected_text};
}}

QPushButton:disabled {{
    background-color: {p.input_bg};
    color: {p.text_muted};
    border-color: {p.card_border};
}}

QPushButton[primary="true"] {{
    background-color: {p.accent};
    border-color: {p.accent};
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 9px;
}}

QPushButton[primary="true"]:hover {{
    background-color: {p.accent_hover};
    border-color: {p.accent_hover};
}}

QPushButton[pill="true"] {{
    border-radius: 13px;
    padding: 4px 12px;
    font-size: 11px;
}}

/* =========================================================================
   Search LineEdit & ComboBox
   ========================================================================= */
QLineEdit {{
    background-color: {p.card_bg};
    color: {p.text_primary};
    border: 1px solid {p.card_border};
    border-radius: 10px;
    padding: 6px 12px;
    font-size: 12px;
}}

QLineEdit:focus {{
    border: 1.5px solid {p.accent};
}}

QComboBox {{
    background-color: {p.input_bg};
    color: {p.text_primary};
    border: 1px solid {p.card_border};
    border-radius: 8px;
    padding: 5px 10px;
    font-weight: 500;
}}

QComboBox:hover {{
    border-color: {p.accent};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {p.card_bg};
    color: {p.text_primary};
    border: 1px solid {p.card_border};
    border-radius: 8px;
    selection-background-color: {p.selected_bg};
    selection-color: {p.selected_text};
    padding: 4px;
}}

/* =========================================================================
   Juicy Sliders, Checkboxes & Radio Buttons
   ========================================================================= */
QSlider {{
    background: transparent;
}}

QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {p.input_bg};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {p.accent};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 2px solid {p.accent};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {p.accent};
    border: 2px solid #FFFFFF;
}}

QCheckBox, QRadioButton {{
    spacing: 8px;
    color: {p.text_primary};
    font-weight: 500;
    background: transparent;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 5px;
    border: 1.5px solid {p.text_muted};
    background-color: {p.input_bg};
}}

QCheckBox::indicator:hover {{
    border-color: {p.accent};
}}

QCheckBox::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 9px;
    border: 1.5px solid {p.text_muted};
    background-color: {p.input_bg};
}}

QRadioButton::indicator:hover {{
    border-color: {p.accent};
}}

QRadioButton::indicator:checked {{
    background-color: {p.accent};
    border: 4px solid {p.card_bg};
}}

/* =========================================================================
   Minimalist Scrollbars
   ========================================================================= */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px 1px;
}}

QScrollBar::handle:vertical {{
    background: {p.card_border};
    min-height: 28px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {p.text_muted};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 1px 2px;
}}

QScrollBar::handle:horizontal {{
    background: {p.card_border};
    min-width: 28px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {p.text_muted};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QToolTip {{
    background-color: {p.card_bg};
    color: {p.text_primary};
    border: 1px solid {p.accent};
    padding: 5px 8px;
    border-radius: 6px;
}}
"""


DARK_THEME_QSS = build_stylesheet(DARK_PALETTE)
LIGHT_THEME_QSS = build_stylesheet(LIGHT_PALETTE)


class ThemeManager:
    """Centralized singleton manager for Dark / Light minimalist theme state."""

    _current_mode: str = "dark"

    @classmethod
    def current_mode(cls) -> str:
        return cls._current_mode

    @classmethod
    def is_dark(cls) -> bool:
        return cls._current_mode == "dark"

    @classmethod
    def palette(cls, mode: str | None = None) -> ThemePalette:
        target = (mode or cls._current_mode).lower()
        return LIGHT_PALETTE if target == "light" else DARK_PALETTE

    @classmethod
    def stylesheet(cls, mode: str | None = None) -> str:
        target = (mode or cls._current_mode).lower()
        return LIGHT_THEME_QSS if target == "light" else DARK_THEME_QSS

    @classmethod
    def set_mode(cls, mode: str) -> ThemePalette:
        cls._current_mode = "light" if mode.lower() == "light" else "dark"
        return cls.palette()

    @classmethod
    def toggle_mode(cls) -> ThemePalette:
        cls._current_mode = "light" if cls._current_mode == "dark" else "dark"
        return cls.palette()
