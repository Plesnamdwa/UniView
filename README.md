# UniView 2.0 🌌

**UniView** is a lightweight, responsive, and minimalist desktop 3D asset & rig visualizer built with **Python 3.10+**, **PyQt6**, **ModernGL (OpenGL 3.3+)**, **PyAssimp**, and **PyGLM**.

Designed with a juicy, distraction-free modern UI featuring Dark/Light studio modes, an animated Expanding Label Inspector, instant texture relinking, and interactive Forward Kinematics (FK) rig inspection.

---

## ✨ Features

- **Minimalist "Juicy Simple" UI**:
  - Seamless runtime switching between **Obsidian Dark** and **Alabaster Light** modes (`Ctrl+T`).
  - Expanding Label stadium tab bar with custom monoline vector icons.
  - Floating translucent Shading HUD and responsive micro-toast notifications.
  - Drag-and-drop 3D asset importer directly onto the viewport with overlay card.
- **Hardware-Accelerated 3D Viewport**:
  - Powered by ModernGL and Arcball Camera (smooth 60 FPS Orbit, Pan, and Zoom).
  - High-DPI support (Windows display scaling 125%, 150%, 200%).
  - Ground grid with dynamic contrast and separate RGB/Alpha blending.
- **Comprehensive Inspector Panels**:
  - **📁 Files**: Dependency analyzer for detecting missing textures with 1-click batch folder scanner and drag-and-drop texture relinking onto rows.
  - **🎨 Materials**: Interactive color & opacity overrides with live color picker and shading modes (*Textured*, *Base Color*, *Weight Heatmap*, *Unlit*).
  - **📦 Meshes**: Sub-mesh inspector with real-time search, poly/vertex count stats, isolate focus (`⛶`), and individual visibility toggles (`👁`).
  - **🦴 Skeleton**: Armature tree hierarchy, bone search filter, X-Ray overlay mode, vertex weight heatmap, local/world transform inspector, and interactive Forward Kinematics (FK) rotation sliders with bind pose reset (`Ctrl+R`).
- **3D Artist Focus Workflow**:
  - Press **`F`** or double-click any sub-mesh/bone to smoothly focus and center the camera on it without altering visibility.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| **LMB Drag** | Orbit Camera around target |
| **RMB / MMB Drag** | Pan Camera target |
| **Mouse Scroll** | Smooth Zoom In / Zoom Out |
| **`F` / Double Click** | Focus camera on selected bone, sub-mesh, or material |
| **`A` / `Shift+F`** | Frame entire 3D model |
| **`Home`** | Reset camera to initial framing position |
| **`1` / `2` / `3` / `4`** | Shading modes: *Textured*, *Base Color*, *Weight Heatmap*, *Unlit* |
| **`G`** | Toggle Ground Grid |
| **`B`** | Toggle Skeleton Bones overlay |
| **`X`** | Toggle Skeleton X-Ray mode |
| **`Ctrl + R`** | Reset Skeleton bind pose |
| **`Ctrl + T`** | Toggle Dark / Light theme |
| **`Ctrl + O`** | Open 3D asset file |
| **`F1` / `?`** | Open Keyboard Shortcuts cheat-sheet |
| **`F11`** | Toggle Fullscreen |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher (Conda recommended).
- C++ Assimp library runtime (required by `pyassimp`).

Using Conda:
```bash
conda create -n uniview python=3.10 -y
conda activate uniview
conda install -c conda-forge assimp=5.4.3 -y
```

### 2. Installation
Clone the repository and install Python dependencies:
```bash
git clone https://github.com/your-username/UniView.git
cd UniView
pip install -r requirements.txt
```

### 3. Running the Application
```bash
python main.py
```
Or open a specific 3D model directly:
```bash
python main.py "path/to/your/model.fbx"
```

On Windows, you can also double-click `run_uniview.bat` or run `create_desktop_shortcut.bat` to create a desktop shortcut.

---

## 📂 Supported Formats

- **Autodesk FBX** (`.fbx` - 7.x binary & ASCII, skeletal rigs, meshes)
- **glTF 2.0** (`.gltf`, `.glb` - PBR materials & skeletal skins)
- **Wavefront OBJ** (`.obj` with MTL material definitions)
- **Collada** (`.dae`)
- **Polygon File Format** (`.ply`), **Stereolithography** (`.stl`), and other Assimp-supported formats.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
