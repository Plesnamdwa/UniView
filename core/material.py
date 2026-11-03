"""
core/material.py
Material & File Dependency management for UniView.
Supports:
- Tracking Available vs Missing dependency files (for the 'Files' tab)
- Dynamic texture relinking via Browse or Drag-and-Drop
- Material Slot state for the 'Materials' tab:
  - Source: 'Model' vs 'Custom Override'
  - Type: 'Phong' vs 'Unlit'
  - Base/Diffuse Color (RGB + Hex string e.g. #cc001c)
  - Ambient Color (RGB + Hex string e.g. #000000)
  - Specular Color (RGB + Hex string e.g. #330007)
  - Opacity (0.0 - 1.0 / 0% - 100%)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import os
import glm
from PIL import Image


def vec3_to_hex(color: glm.vec3) -> str:
    """Convert a normalized glm.vec3(r, g, b) in [0, 1] to a lowercase hex code (e.g., '#cc001c')."""
    r = int(round(float(glm.clamp(color.x, 0.0, 1.0)) * 255.0))
    g = int(round(float(glm.clamp(color.y, 0.0, 1.0)) * 255.0))
    b = int(round(float(glm.clamp(color.z, 0.0, 1.0)) * 255.0))
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_vec3(hex_code: str) -> glm.vec3:
    """Convert a hex string (e.g., '#cc001c') to a normalized glm.vec3(r, g, b) in [0, 1]."""
    clean = hex_code.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        return glm.vec3(0.8, 0.8, 0.8)
    try:
        r = int(clean[0:2], 16) / 255.0
        g = int(clean[2:4], 16) / 255.0
        b = int(clean[4:6], 16) / 255.0
        return glm.vec3(r, g, b)
    except ValueError:
        return glm.vec3(0.8, 0.8, 0.8)


@dataclass
class DependencyFile:
    """Represents an external or embedded file dependency (3D model or texture image)."""
    filename: str
    raw_path: str
    resolved_path: Optional[str] = None
    is_available: bool = False
    file_type: str = "Texture"  # "Model" or "Texture"
    texture_semantic: str = "Diffuse"  # "Diffuse", "Normal", "Specular", "Other"
    material_indices: List[int] = field(default_factory=list)
    image_size: Optional[Tuple[int, int]] = None


@dataclass
class MaterialSlot:
    """Represents a distinct material slot assigned to one or more sub-meshes."""
    index: int
    name: str
    source: str = "Model"          # "Model" or "Custom Override"
    shading_type: str = "Phong"    # "Phong" or "Unlit"

    base_color: glm.vec3 = field(default_factory=lambda: glm.vec3(0.75, 0.75, 0.78))
    ambient_color: glm.vec3 = field(default_factory=lambda: glm.vec3(0.08, 0.08, 0.10))
    specular_color: glm.vec3 = field(default_factory=lambda: glm.vec3(0.25, 0.25, 0.25))
    shininess: float = 32.0
    opacity: float = 1.0           # 0.0 to 1.0 (0% to 100%)

    # Original model values for Reset
    orig_base_color: glm.vec3 = field(default_factory=lambda: glm.vec3(0.75, 0.75, 0.78))
    orig_ambient_color: glm.vec3 = field(default_factory=lambda: glm.vec3(0.08, 0.08, 0.10))
    orig_specular_color: glm.vec3 = field(default_factory=lambda: glm.vec3(0.25, 0.25, 0.25))
    orig_opacity: float = 1.0
    orig_shading_type: str = "Phong"

    # Texture paths & GPU handle
    diffuse_texture_path: Optional[str] = None
    normal_texture_path: Optional[str] = None
    resolved_diffuse_path: Optional[str] = None
    resolved_normal_path: Optional[str] = None

    # Cached RGBA bytes and dimensions for instant GPU upload
    diffuse_rgba_bytes: Optional[bytes] = None
    diffuse_size: Optional[Tuple[int, int]] = None
    gpu_texture: Any = None        # moderngl.Texture bound at runtime
    gpu_texture_dirty: bool = False

    def snapshot_defaults(self) -> None:
        """Save current properties as the original model defaults."""
        self.orig_base_color = glm.vec3(self.base_color)
        self.orig_ambient_color = glm.vec3(self.ambient_color)
        self.orig_specular_color = glm.vec3(self.specular_color)
        self.orig_opacity = float(self.opacity)
        self.orig_shading_type = self.shading_type
        self.source = "Model"

    def set_base_color(self, color: glm.vec3) -> None:
        self.base_color = glm.vec3(color)
        self.source = "Custom Override"

    def set_ambient_color(self, color: glm.vec3) -> None:
        self.ambient_color = glm.vec3(color)
        self.source = "Custom Override"

    def set_specular_color(self, color: glm.vec3) -> None:
        self.specular_color = glm.vec3(color)
        self.source = "Custom Override"

    def set_opacity_percent(self, pct: float) -> None:
        self.opacity = float(glm.clamp(pct / 100.0, 0.0, 1.0))
        self.source = "Custom Override"

    def set_shading_type(self, shading_type: str) -> None:
        if shading_type in ("Phong", "Unlit"):
            self.shading_type = shading_type
            self.source = "Custom Override"

    def reset_to_model(self) -> None:
        """Restore original model colors, shading type, and opacity."""
        self.base_color = glm.vec3(self.orig_base_color)
        self.ambient_color = glm.vec3(self.orig_ambient_color)
        self.specular_color = glm.vec3(self.orig_specular_color)
        self.opacity = float(self.orig_opacity)
        self.shading_type = self.orig_shading_type
        self.source = "Model"

    @property
    def base_hex(self) -> str:
        return vec3_to_hex(self.base_color)

    @property
    def ambient_hex(self) -> str:
        return vec3_to_hex(self.ambient_color)

    @property
    def specular_hex(self) -> str:
        return vec3_to_hex(self.specular_color)

    @property
    def opacity_percent(self) -> int:
        return int(round(self.opacity * 100.0))

    @property
    def has_diffuse_texture(self) -> bool:
        return self.diffuse_rgba_bytes is not None and self.diffuse_size is not None

    def load_diffuse_image(self, image_path: str) -> bool:
        """Load an image file from disk using Pillow into RGBA float/byte buffer for OpenGL."""
        if not image_path or not os.path.isfile(image_path):
            return False
        try:
            with Image.open(image_path) as img:
                # Flip top-to-bottom for OpenGL UV convention if not already flipped,
                # note: PyAssimp with aiProcess_FlipUVs expects standard top-to-bottom image bytes,
                # or standard OpenGL image loading. Using standard RGBA bytes:
                rgba = img.convert("RGBA")
                self.diffuse_size = rgba.size
                self.diffuse_rgba_bytes = rgba.tobytes()
                self.resolved_diffuse_path = os.path.abspath(image_path)
                self.gpu_texture_dirty = True
                return True
        except Exception:
            return False
