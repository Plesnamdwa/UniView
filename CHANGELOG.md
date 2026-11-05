# Changelog

All notable changes to **UniView** will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] - 2026-11-06

### 🌟 Added
- **Wireframe Overlay Mode (`W`)**:
  - Polygon-offset wireframe pass (`core/renderer.py`) rendered directly over shaded, textured, heatmap, or unlit models with zero z-fighting.
  - Theme-aware wireframe colors: sleek cyber blue `(0.42, 0.68, 0.95, 0.70)` in dark mode and crisp blueprint slate `(0.18, 0.24, 0.38, 0.80)` in light mode.
  - Top header quick toggle pill (`Wire`) and View menu shortcut `W`.
- **3D Bounding Box Dimensions & Architectural Cage Display (`Shift+B` / `O`)**:
  - Procedural 3D bounding box cage (`core/bounding_box.py`) with 12 edge lines and corner accent brackets.
  - Accounts dynamically for model matrices, up-vector rotations, and node hierarchies.
  - Floating `BoundingBoxHUDCard` widget in viewport displaying real-time dimensions ($W$ Width, $H$ Height, $D$ Depth), bounding volume in $\text{m}^3$, and center coordinates.
  - View menu shortcut `Shift+B` / `O` and top header pill (`BBox`).
- **Interactive 3D Viewport Rotation Gimbal for FK Posing (`R`)**:
  - 3 dedicated, color-coded orthogonal rotation rings rendered at the active joint pivot: Pitch ($X$-axis, Red), Yaw ($Y$-axis, Green), and Roll ($Z$-axis, Blue) (`core/gimbal.py`).
  - Screen-space hit-testing with interactive hover highlight (active ring turns into glowing amber).
  - Natural mouse drag rotation with angular delta calculation around projected gimbal center.
  - Modifiers: `Ctrl` enables 5° angle snapping; `Shift` enables 0.25x ultra-fine precision mode.
  - Live bidirectional synchronization: dragging in the 3D viewport immediately recalculates skeletal forward kinematics, deforms the skinned mesh in real time at 60 FPS, updates FK rotation sliders in the side panel, and refreshes the Transform Inspector.

---

## [2.0.0] - 2026-11-05

### 🌟 Added
- **Unreal Engine Style Skeleton & Direct 3D Bone Picking**:
  - Direct 3D Bone Picking from Viewport: clicking any joint sphere or octahedral bone segment directly in 3D selects the bone in hierarchy, updates the transform inspector, FK sliders, and weight heatmap.
  - Re-architected anatomical octahedral bone connectivity (`core/skeleton.py`) so bone cones strictly connect from parent to real child joints.
  - Eliminated artificial spike projections on leaf bones (fingertips, top of skull, toes).
  - Added 3D orthogonal circular gimbal rings (`core/math_utils.py` & `core/renderer.py`) for all terminal joints and highlighted active bones.
- **Up-Vector Orientation Lerp & Interactive Corner Gizmo**:
  - Interactive Corner Orientation Gizmo (`core/gizmo.py`): 1-click snapping to standard Orthographic views (Front +Z, Back -Z, Right +X, Left -X, Top +Y, Bottom -Y) with shortest-path camera slerp interpolation and hover cursor feedback.
  - 1-click smooth transition between OpenGL / glTF standard (`Y-Up`), Unreal Engine / 3ds Max / Blender standard (`Z-Up`), and vertical inversion (`Flip`).
  - Implemented 60 FPS quaternion slerp easing (`OutCubic` curve $\approx 0.38\text{s}$) for silky smooth model rotation.
  - Studio floor ground grid stays locked at $Y=0$, while the avatar rotates in world space.
  - Camera automatically calculates new bounding box (`AABB.transformed`) and re-frames the avatar smoothly.
  - Top header pill dropdown (`📐 Y-Up`) and `View -> Up Vector Orientation` menu.
- **3D Interactive Measurement Tool with Vertex Snapping**:
  - Added vectorized *Möller-Trumbore* 3D raycasting (`core/raycast.py`) achieving $< 5\text{ms}$ intersection checks over tens of thousands of triangles.
  - **🧲 Vertex Snapping Mode**: Locks measurement hit points to the exact coordinates of the closest mesh polygon vertex for CAD precision (HUD toggle button & `Shift`/`Ctrl` modifier).
  - **Intelligent Click vs Drag Discrimination**: Distinguishes between mouse drag ($> 4\text{px}$) to orbit camera normally and single click ($\le 4\text{px}$) to accurately drop measurement points on polygon surfaces.
  - Visual 3D markers with coral/mint spheres, gimbal rings, and golden connecting 3D connector rod.
  - Floating pill HUD card displaying live Euclidean distance, inclination angle, `🧲 Snap` toggle, with 1-click clear (`↺`) and dismiss (`✕`).
  - Keyboard shortcut `M` and top header pill button (`📏 Measure`).

### 🐛 Fixed
- Fixed runtime `NameError` crash (`QMenu` and `QPoint` missing from imports) when opening Up-Vector popup dropdown from the header bar.
- Fixed camera focusing (`F`) when the model is rotated in Z-Up or Flipped orientation by transforming submesh and bone bounding boxes accordingly.

---

## [1.0.0] - 2026-11-04

### 🌟 Added
- **Minimalist "Juicy Simple" UI Redesign**:
  - Dual studio themes: **Obsidian Dark** (`#111018`) and **Alabaster Light** (`#F3F4F9`) with `Ctrl+T` runtime switching.
  - Custom `ExpandingTabBar` with 60 FPS animated sliding pill capsule and monoline vector icons.
  - Custom vector widgets: `ToggleSwitch`, `SegmentedControl`, and `IconButton` rendered with `QPainter`.
  - Floating viewport HUD for shading modes (`Textured`, `Base Color`, `Heatmap`, `Unlit`).
  - Bottom-right floating `ToastNotification` pills for micro-feedback.
  - Drag-and-drop viewport overlay card (`Drop 3D Asset to Open`).
  - Status bar camera throttling (~12 Hz) to prevent Qt relayout overhead during 60 FPS orbit.
  - `QSettings` persistence for window geometry, theme preference, and Recent Files history.
- **Standalone Windows Executable**:
  - Bundled standalone zero-dependency executable (`UniView.exe`) via PyInstaller with embedded C++ Assimp runtime.
