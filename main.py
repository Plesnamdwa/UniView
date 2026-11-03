"""
main.py
Application Entry Point for UniView - 3D Asset & Rig Visualizer.
Usage:
    python main.py                     # Launches UniView with auto-generated sample rigged asset
    python main.py path/to/model.fbx   # Opens specified 3D model directly
"""

from __future__ import annotations
import os
import sys

# Ensure Windows DLL directories (Conda Library/bin or PyInstaller bundle) are registered for assimp.dll resolution
if os.name == "nt":
    candidate_bins = [
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.dirname(os.path.abspath(sys.executable)),
    ]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidate_bins.insert(0, getattr(sys, "_MEIPASS"))
    for bin_dir in candidate_bins:
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QSurfaceFormat

from ui.style import DARK_THEME_QSS
from ui.main_window import MainWindow


def main() -> None:
    # Configure high-quality OpenGL 3.3 Core profile surface format for QOpenGLWidget
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)
    gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    gl_format.setDepthBufferSize(24)
    gl_format.setStencilBufferSize(8)
    gl_format.setSamples(4)  # 4x MSAA antialiasing
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    app.setApplicationName("UniView")
    app.setOrganizationName("UniView")
    app.setStyleSheet(DARK_THEME_QSS)

    initial_file = sys.argv[1] if len(sys.argv) > 1 else None

    # Instantiate Main Window and maximize by default
    window = MainWindow(initial_filepath=initial_file)
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
