"""
core/scene_loader.py
Multi-format 3D Asset, Material Dependency, Sub-Mesh, and Skeletal Rig Loader using PyAssimp.
Optimized for:
- Tier-1 Formats: .fbx (Blender / Maya / UE5), .gltf / .glb, .obj
- Extended Formats: .dae, .stl, .ply, .3ds, .wrl, .off, .3mf, .amf, .step, .iges, .ifc, .brep, .fcstd, .bim, .3dm
- Contiguous NumPy float32 / int32 / uint32 buffers for one-time ModernGL VBO/VAO upload
- File & Dependency Analyzer (Available vs Missing textures + dynamic relinking)
- Full Hierarchical Bone Tree & Vertex Skinning Weight extraction
"""

from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import glm

# Ensure Conda's Library/bin is registered on Windows BEFORE importing pyassimp
if os.name == "nt":
    _conda_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_conda_bin):
        try:
            os.add_dll_directory(_conda_bin)
        except Exception:
            pass
        if _conda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

import pyassimp
import pyassimp.postprocess as pp

from core.math_utils import AABB, assimp_mat_to_glm, glm_to_numpy, decompose_trs
from core.material import MaterialSlot, DependencyFile
from core.skeleton import Skeleton


SUPPORTED_EXTENSIONS: List[str] = [
    ".fbx", ".gltf", ".glb", ".obj",
    ".dae", ".stl", ".ply", ".3ds", ".wrl", ".off", ".3mf", ".amf",
    ".step", ".stp", ".iges", ".igs", ".ifc", ".brep", ".fcstd", ".bim", ".3dm",
]


def get_file_dialog_filter() -> str:
    """Return a clean QFileDialog filter string prioritizing FBX, GLTF/GLB, and OBJ."""
    all_Wildcards = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
    return (
        f"All Supported 3D Assets ({all_Wildcards});;"
        "Tier-1 Game & Rig Assets (*.fbx *.gltf *.glb *.obj);;"
        "Autodesk FBX (*.fbx);;"
        "glTF 2.0 (*.gltf *.glb);;"
        "Wavefront OBJ (*.obj);;"
        "Collada / Mesh Formats (*.dae *.stl *.ply *.3ds *.wrl *.off *.3mf *.amf);;"
        "CAD & BIM Formats (*.step *.stp *.iges *.igs *.ifc *.brep *.fcstd *.bim *.3dm);;"
        "All Files (*.*)"
    )


@dataclass
class SubMeshData:
    """Contiguous CPU/GPU-ready geometry and metadata for an individual sub-mesh."""
    index: int
    name: str
    material_index: int
    visible: bool = True

    # Contiguous NumPy arrays (uploaded once to ModernGL VBO)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    uvs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    bone_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    bone_weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint32))

    node_transform: glm.mat4 = field(default_factory=lambda: glm.mat4(1.0))
    has_bones: bool = False
    aabb: AABB = field(default_factory=AABB.empty)

    @property
    def vertex_count(self) -> int:
        return int(len(self.positions))

    @property
    def face_count(self) -> int:
        return int(len(self.indices) // 3)


@dataclass
class SceneData:
    """Complete loaded 3D scene representation."""
    filepath: str
    filename: str
    submeshes: List[SubMeshData] = field(default_factory=list)
    materials: List[MaterialSlot] = field(default_factory=list)
    dependencies: List[DependencyFile] = field(default_factory=list)
    skeleton: Skeleton = field(default_factory=Skeleton)
    scene_aabb: AABB = field(default_factory=AABB.empty)

    # Isolation state tracking for the "Meshes" tab Focus/Isolate button
    isolated_submesh_index: Optional[int] = None
    _saved_visibility: Dict[int, bool] = field(default_factory=dict)

    @property
    def total_vertices(self) -> int:
        return sum(m.vertex_count for m in self.submeshes)

    @property
    def total_faces(self) -> int:
        return sum(m.face_count for m in self.submeshes)

    @property
    def missing_files(self) -> List[DependencyFile]:
        return [d for d in self.dependencies if not d.is_available]

    @property
    def available_files(self) -> List[DependencyFile]:
        return [d for d in self.dependencies if d.is_available]

    def toggle_isolate_submesh(self, submesh_idx: int) -> Tuple[bool, AABB]:
        """
        Toggle isolation on `submesh_idx`.
        - If isolating: saves current visibility of all sub-meshes, hides all except `submesh_idx`,
          and returns `(True, submesh.aabb)`.
        - If un-isolating (clicking the already-isolated sub-mesh): restores previous visibility of all
          sub-meshes and returns `(False, self.scene_aabb)`.
        """
        if self.isolated_submesh_index == submesh_idx:
            # Restore visibility
            for m in self.submeshes:
                m.visible = self._saved_visibility.get(m.index, True)
            self.isolated_submesh_index = None
            self._saved_visibility.clear()
            return False, self.scene_aabb
        else:
            # Save current visibility only if we weren't already in isolated mode
            if self.isolated_submesh_index is None:
                self._saved_visibility = {m.index: m.visible for m in self.submeshes}

            self.isolated_submesh_index = submesh_idx
            target_aabb = self.scene_aabb
            for m in self.submeshes:
                if m.index == submesh_idx:
                    m.visible = True
                    target_aabb = m.aabb
                else:
                    m.visible = False
            return True, target_aabb

    def relink_dependency(self, dep_index: int, new_image_path: str) -> bool:
        """
        Dynamically relink a missing (or existing) texture dependency to `new_image_path`.
        Updates the DependencyFile state and reloads RGBA bytes into all associated MaterialSlots.
        """
        if dep_index < 0 or dep_index >= len(self.dependencies):
            return False
        if not os.path.isfile(new_image_path):
            return False

        dep = self.dependencies[dep_index]
        abs_path = os.path.abspath(new_image_path)

        success_any = False
        for mat_idx in dep.material_indices:
            if 0 <= mat_idx < len(self.materials):
                mat = self.materials[mat_idx]
                if dep.texture_semantic == "Normal":
                    mat.resolved_normal_path = abs_path
                    success_any = True
                elif dep.texture_semantic == "Diffuse":
                    if mat.load_diffuse_image(abs_path):
                        dep.image_size = mat.diffuse_size
                        success_any = True

        # Verify image and record dimensions with PIL if not already loaded as diffuse
        if dep.image_size is None or not success_any:
            try:
                from PIL import Image
                with Image.open(abs_path) as im:
                    dep.image_size = im.size
                success_any = True
            except Exception:
                if not success_any:
                    return False

        dep.resolved_path = abs_path
        dep.is_available = True
        return True


class SceneLoader:
    """Loads 3D assets using PyAssimp and builds a complete SceneData structure."""

    @staticmethod
    def _clean_path_str(val) -> str:
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        elif isinstance(val, list) and len(val) > 0:
            val = str(val[0])
        s = str(val).strip().strip("\x00").strip('"').strip("'")
        return s

    @staticmethod
    def _resolve_texture_path(model_dir: str, raw_tex_path: str, model_filepath: str = "") -> Optional[str]:
        """
        Attempt to resolve `raw_tex_path` relative to `model_dir`.
        Note: For standalone `.fbx` files that reference external `.fbm` folders or external
        unbundled files not matching the `.fbx` bundle, we do not silently pull from another
        model's `.fbm` folder—marking them as Missing Files (matching Online 3D Viewer) so
        the user can inspect and relink them cleanly in the Files tab.
        """
        if not raw_tex_path or raw_tex_path.startswith("*"):
            return None

        norm_raw = raw_tex_path.replace("\\", os.sep).replace("/", os.sep)
        lower_raw = norm_raw.lower()

        # If the path references an external .fbm folder (or standalone .fbx external texture),
        # treat it as unresolved/missing on initial single-file load so Missing Files detects it cleanly.
        if ".fbm" in lower_raw or model_filepath.lower().endswith(".fbx"):
            return None

        if os.path.isabs(norm_raw) and os.path.isfile(norm_raw):
            return os.path.abspath(norm_raw)

        rel_candidate = os.path.join(model_dir, norm_raw)
        if os.path.isfile(rel_candidate):
            return os.path.abspath(rel_candidate)

        base_candidate = os.path.join(model_dir, os.path.basename(norm_raw))
        if os.path.isfile(base_candidate):
            return os.path.abspath(base_candidate)

        return None

    @staticmethod
    def _extract_gltf_extra_textures(filepath: str) -> Dict[int, List[Tuple[str, str]]]:
        """
        For .gltf files, parse the JSON directly as well to ensure 100% of referenced
        baseColorTexture and normalTexture URIs are captured even if missing on disk.
        Returns {material_idx: [(uri, semantic), ...]}.
        """
        result: Dict[int, List[Tuple[str, str]]] = {}
        if not filepath.lower().endswith(".gltf"):
            return result
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                doc = json.load(f)
            images = doc.get("images", [])
            textures = doc.get("textures", [])
            materials = doc.get("materials", [])

            def get_uri(tex_info: Optional[dict]) -> Optional[str]:
                if not isinstance(tex_info, dict):
                    return None
                t_idx = tex_info.get("index")
                if t_idx is None or not (0 <= t_idx < len(textures)):
                    return None
                s_idx = textures[t_idx].get("source")
                if s_idx is None or not (0 <= s_idx < len(images)):
                    return None
                uri = images[s_idx].get("uri")
                if uri and not uri.startswith("data:"):
                    return uri
                return None

            for m_idx, mat in enumerate(materials):
                entries: List[Tuple[str, str]] = []
                pbr = mat.get("pbrMetallicRoughness", {})
                base_uri = get_uri(pbr.get("baseColorTexture"))
                if base_uri:
                    entries.append((base_uri, "Diffuse"))
                norm_uri = get_uri(mat.get("normalTexture"))
                if norm_uri:
                    entries.append((norm_uri, "Normal"))
                if entries:
                    result[m_idx] = entries
        except Exception:
            pass
        return result

    @classmethod
    def _check_and_upgrade_legacy_fbx(cls, abs_path: str) -> str:
        """
        Detect legacy FBX versions (< 7100, e.g. FBX 6.1 / 2006-2010 binary or ASCII)
        which Assimp does not support natively, and transparently upgrade to
        modern FBX 7.4+ binary using the installed Autodesk FBX SDK.
        """
        if not abs_path.lower().endswith(".fbx") or not os.path.isfile(abs_path):
            return abs_path

        needs_upgrade = False
        try:
            with open(abs_path, "rb") as f:
                hdr = f.read(27)
            if hdr.startswith(b"Kaydara FBX Binary"):
                import struct
                version = struct.unpack("<I", hdr[23:27])[0]
                if version < 7100:
                    needs_upgrade = True
            elif not hdr.startswith(b"Kaydara FBX Binary"):
                # ASCII or non-standard legacy FBX header
                needs_upgrade = True
        except Exception:
            pass

        if needs_upgrade:
            try:
                import fbx
                import tempfile
                mgr = fbx.FbxManager.Create()
                imp = fbx.FbxImporter.Create(mgr, "")
                if imp.Initialize(abs_path):
                    scene = fbx.FbxScene.Create(mgr, "scene")
                    if imp.Import(scene):
                        imp.Destroy()
                        temp_dir = tempfile.gettempdir()
                        base_name = os.path.splitext(os.path.basename(abs_path))[0]
                        upgraded_path = os.path.join(temp_dir, f"uniview_{base_name}_v7700.fbx")
                        exp = fbx.FbxExporter.Create(mgr, "")
                        if exp.Initialize(upgraded_path, -1, mgr.GetIOSettings()):
                            if exp.Export(scene):
                                exp.Destroy()
                                mgr.Destroy()
                                return upgraded_path
                        exp.Destroy()
                imp.Destroy()
                mgr.Destroy()
            except Exception:
                pass

        return abs_path

    @classmethod
    def load(cls, filepath: str) -> SceneData:
        """
        Load a 3D model file via PyAssimp and return a populated SceneData object.
        Raises ValueError or RuntimeError with descriptive message if unsupported or corrupted.
        """
        abs_path = os.path.abspath(filepath)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"3D asset file not found: {abs_path}")

        model_dir = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)

        # Transparently convert legacy FBX (< 7100, FBX 6.1) if needed
        actual_import_path = cls._check_and_upgrade_legacy_fbx(abs_path)

        scene_data = SceneData(filepath=abs_path, filename=filename)

        # Register the main 3D asset file itself under Available Files
        scene_data.dependencies.append(
            DependencyFile(
                filename=filename,
                raw_path=abs_path,
                resolved_path=abs_path,
                is_available=True,
                file_type="Model",
                texture_semantic="3D Asset",
            )
        )

        flags = (
            pp.aiProcess_Triangulate
            | pp.aiProcess_GenSmoothNormals
            | pp.aiProcess_CalcTangentSpace
            | pp.aiProcess_JoinIdenticalVertices
            | pp.aiProcess_SortByPType
            | pp.aiProcess_FlipUVs
        )

        try:
            with pyassimp.load(actual_import_path, processing=flags) as ai_scene:
                cls._parse_materials_and_dependencies(ai_scene, abs_path, model_dir, scene_data)
                cls._parse_nodes_meshes_and_skeleton(ai_scene, scene_data)
        except Exception as exc:
            err_msg = str(exc)
            ext = os.path.splitext(abs_path)[1].lower()
            if "unsupported, old format version" in err_msg or "old format version" in err_msg:
                raise RuntimeError(
                    f"File '{filename}' menggunakan format FBX versi lama (FBX 6.1 / 2010 atau lebih lama) yang tidak didukung langsung oleh Assimp.\n"
                    "Silakan simpan ulang atau ekspor kembali melalui Blender / Autodesk FBX Converter ke versi FBX 2014 atau lebih baru (7.4+)."
                ) from exc
            cad_exts = {".step", ".stp", ".iges", ".igs", ".ifc", ".brep", ".fcstd", ".bim", ".3dm"}
            if ext in cad_exts:
                raise RuntimeError(
                    f"Format CAD/BIM '{ext}' membutuhkan konverter CAD opsional atau gagal diparsing oleh Assimp:\n{exc}"
                ) from exc
            raise RuntimeError(f"Failed to load 3D asset '{filename}':\n{exc}") from exc

        return scene_data

    @classmethod
    def _parse_materials_and_dependencies(
        cls,
        ai_scene,
        abs_path: str,
        model_dir: str,
        scene_data: SceneData,
    ) -> None:
        gltf_tex_map = cls._extract_gltf_extra_textures(abs_path)
        dep_by_key: Dict[str, int] = {}
        used_mat_indices: Set[int] = {
            int(getattr(m, "materialindex", 0)) for m in ai_scene.meshes
        }

        def register_texture_dep(raw_tex: str, semantic: str, mat_idx: int) -> Optional[str]:
            raw_clean = cls._clean_path_str(raw_tex)
            if not raw_clean or raw_clean.startswith("*"):
                return None

            tex_filename = os.path.basename(raw_clean.replace("\\", "/"))
            resolved = cls._resolve_texture_path(model_dir, raw_clean, abs_path)

            key = tex_filename.lower()
            if key in dep_by_key:
                dep = scene_data.dependencies[dep_by_key[key]]
                if mat_idx not in dep.material_indices:
                    dep.material_indices.append(mat_idx)
                return dep.resolved_path

            dep_obj = DependencyFile(
                filename=tex_filename,
                raw_path=raw_clean,
                resolved_path=resolved,
                is_available=(resolved is not None),
                file_type="Texture",
                texture_semantic=semantic,
                material_indices=[mat_idx],
            )
            dep_by_key[key] = len(scene_data.dependencies)
            scene_data.dependencies.append(dep_obj)
            return resolved

        for m_idx, ai_mat in enumerate(ai_scene.materials):
            if used_mat_indices and (m_idx not in used_mat_indices) and m_idx == len(ai_scene.materials) - 1:
                # Skip Assimp's auto-appended unreferenced trailing DefaultMaterial
                continue
            props = {}
            for k, v in ai_mat.properties.items():
                props[str(k).lower()] = v

            raw_name = props.get("name", f"Material_{m_idx}")
            mat_name = cls._clean_path_str(raw_name) or f"Material_{m_idx}"

            # Extract Base / Diffuse color
            diffuse = props.get("diffuse", props.get("base", [0.75, 0.75, 0.78]))
            if hasattr(diffuse, "__len__") and len(diffuse) >= 3:
                base_col = glm.vec3(float(diffuse[0]), float(diffuse[1]), float(diffuse[2]))
            else:
                base_col = glm.vec3(0.75, 0.75, 0.78)

            # Extract Ambient color
            ambient = props.get("ambient", [0.05, 0.05, 0.07])
            if hasattr(ambient, "__len__") and len(ambient) >= 3:
                amb_col = glm.vec3(float(ambient[0]), float(ambient[1]), float(ambient[2]))
            else:
                amb_col = glm.vec3(0.05, 0.05, 0.07)

            # Extract Specular color
            specular = props.get("specular", [0.22, 0.22, 0.24])
            if hasattr(specular, "__len__") and len(specular) >= 3:
                spec_col = glm.vec3(float(specular[0]), float(specular[1]), float(specular[2]))
            else:
                spec_col = glm.vec3(0.22, 0.22, 0.24)

            opacity_val = props.get("opacity", 1.0)
            if hasattr(opacity_val, "__len__"):
                opacity_val = opacity_val[0]
            opacity = float(glm.clamp(float(opacity_val), 0.0, 1.0))

            slot = MaterialSlot(
                index=m_idx,
                name=mat_name,
                base_color=base_col,
                ambient_color=amb_col,
                specular_color=spec_col,
                opacity=opacity,
            )

            # Check Assimp material properties for texture files and classify semantics accurately
            tex_candidates: List[Tuple[str, str]] = []
            for k_raw, v_raw in ai_mat.properties.items():
                if isinstance(k_raw, tuple):
                    key_name, tex_type = str(k_raw[0]).lower(), int(k_raw[1])
                else:
                    key_name, tex_type = str(k_raw).lower(), 0

                if "file" in key_name:
                    path_str = cls._clean_path_str(v_raw)
                    if path_str:
                        p_low = path_str.lower()
                        if tex_type in (6, 5) or "normal" in key_name or "bump" in key_name or "norm" in p_low:
                            sem = "Normal"
                        elif (
                            "shininess" in key_name
                            or "rough" in key_name
                            or "spec" in key_name
                            or "metal" in key_name
                            or "rough" in p_low
                            or "metal" in p_low
                            or "spec" in p_low
                        ):
                            sem = "Roughness"
                        else:
                            sem = "Diffuse"
                        tex_candidates.append((path_str, sem))

            # Supplement with glTF JSON textures if any
            if m_idx in gltf_tex_map:
                existing_basenames = {os.path.basename(p.replace("\\", "/")).lower() for p, _ in tex_candidates}
                for uri, sem in gltf_tex_map[m_idx]:
                    bname = os.path.basename(uri.replace("\\", "/")).lower()
                    if bname not in existing_basenames:
                        tex_candidates.append((uri, sem))

            # Sort per material: Diffuse (Base Color) first, Normal second, Roughness/Other third
            sem_priority = {"Diffuse": 0, "Normal": 1, "Roughness": 2}
            tex_candidates.sort(key=lambda item: sem_priority.get(item[1], 3))

            for raw_tex, sem in tex_candidates:
                resolved_path = register_texture_dep(raw_tex, sem, m_idx)
                if sem == "Normal":
                    slot.normal_texture_path = raw_tex
                    slot.resolved_normal_path = resolved_path
                elif sem == "Diffuse":
                    if slot.diffuse_texture_path is None:
                        slot.diffuse_texture_path = raw_tex
                        if resolved_path and slot.load_diffuse_image(resolved_path):
                            # Store resolution back into DependencyFile
                            key = os.path.basename(raw_tex.replace("\\", "/")).lower()
                            if key in dep_by_key:
                                scene_data.dependencies[dep_by_key[key]].image_size = slot.diffuse_size

            slot.snapshot_defaults()
            scene_data.materials.append(slot)

        if not scene_data.materials:
            default_slot = MaterialSlot(index=0, name="Default_Material")
            default_slot.snapshot_defaults()
            scene_data.materials.append(default_slot)

    @classmethod
    def _parse_nodes_meshes_and_skeleton(cls, ai_scene, scene_data: SceneData) -> None:
        # 1. Traverse node hierarchy to record node transforms, parent links, and mesh node names
        node_local_transforms: Dict[str, glm.mat4] = {}
        node_world_transforms: Dict[str, glm.mat4] = {}
        node_parent_map: Dict[str, Optional[str]] = {}
        node_order: List[str] = []
        mesh_idx_to_node_info: Dict[int, Tuple[str, glm.mat4, int]] = {}

        def _walk_node(ai_node, parent_name: Optional[str], parent_world: glm.mat4):
            n_name = cls._clean_path_str(ai_node.name) or f"Node_{len(node_order)}"
            local_m = assimp_mat_to_glm(ai_node.transformation)
            world_m = parent_world * local_m

            if n_name not in node_local_transforms:
                node_local_transforms[n_name] = local_m
                node_world_transforms[n_name] = world_m
                node_parent_map[n_name] = parent_name
                node_order.append(n_name)

            # Record mesh associations for this node (unwrap ctypes LP_Node if needed)
            c_node = ai_node.contents if hasattr(ai_node, "contents") else ai_node
            num_node_meshes = int(getattr(c_node, "mNumMeshes", 0))
            if num_node_meshes > 0 and hasattr(c_node, "mMeshes") and c_node.mMeshes:
                for i in range(num_node_meshes):
                    m_idx_int = int(c_node.mMeshes[i])
                    if m_idx_int not in mesh_idx_to_node_info:
                        mesh_idx_to_node_info[m_idx_int] = (n_name, world_m, num_node_meshes)

            for child in ai_node.children:
                _walk_node(child, n_name, world_m)

        if ai_scene.rootnode is not None:
            _walk_node(ai_scene.rootnode, None, glm.mat4(1.0))

        # 2. Discover all bones referenced across ai_scene.meshes
        bone_offsets: Dict[str, glm.mat4] = {}
        for ai_mesh in ai_scene.meshes:
            if hasattr(ai_mesh, "bones") and ai_mesh.bones:
                for ai_bone in ai_mesh.bones:
                    b_name = cls._clean_path_str(ai_bone.name)
                    if b_name and b_name not in bone_offsets:
                        bone_offsets[b_name] = assimp_mat_to_glm(ai_bone.offsetmatrix)

        # If bones exist, include their ancestor nodes up to the armature root so the bone tree is connected
        relevant_bones: Set[str] = set(bone_offsets.keys())
        if relevant_bones:
            # Expand ancestors that connect bones together (excluding Assimp FBX helper pivot nodes)
            added_ancestors: Set[str] = set()
            for b_name in list(relevant_bones):
                curr = node_parent_map.get(b_name)
                chain = []
                while curr is not None:
                    if "_$AssimpFbx$_" not in curr:
                        chain.append(curr)
                    if curr in relevant_bones:
                        added_ancestors.update(chain)
                        break
                    curr = node_parent_map.get(curr)
            relevant_bones.update(added_ancestors)

            def _find_bone_ancestor(n_name: str) -> Optional[str]:
                curr = node_parent_map.get(n_name)
                while curr is not None:
                    if curr in relevant_bones:
                        return curr
                    curr = node_parent_map.get(curr)
                return None

            # Build Skeleton in topological order (parents before children)
            skeleton = scene_data.skeleton
            for n_name in node_order:
                if n_name in relevant_bones:
                    p_name = _find_bone_ancestor(n_name)
                    w_m = node_world_transforms.get(n_name, glm.mat4(1.0))
                    if p_name is None:
                        local_m = w_m
                    else:
                        p_w_m = node_world_transforms.get(p_name, glm.mat4(1.0))
                        local_m = glm.inverse(p_w_m) * w_m
                    skeleton.add_bone(
                        name=n_name,
                        bind_local_matrix=local_m,
                        offset_matrix=bone_offsets.get(n_name),
                        parent_name=p_name,
                    )

            # Any bones that weren't in the node tree (rare edge case)
            for b_name, off_m in bone_offsets.items():
                if b_name not in skeleton.bone_by_name:
                    inv_off = glm.inverse(off_m)
                    skeleton.add_bone(name=b_name, bind_local_matrix=inv_off, offset_matrix=off_m)

            # Attach explicit `<bone>_end` leaf tip positions from Blender/FBX node tree
            for bone in skeleton.bones:
                end_name = f"{bone.name}_end"
                if end_name in node_local_transforms:
                    end_pos, _, _ = decompose_trs(node_local_transforms[end_name])
                    bone.end_local_pos = end_pos

            skeleton.finalize_bind_pose()

        # 3. Parse every sub-mesh into contiguous float32/int32/uint32 arrays
        used_names: Dict[str, int] = {}
        overall_aabb = AABB.empty()

        for m_idx, ai_mesh in enumerate(ai_scene.meshes):
            raw_m_name = cls._clean_path_str(ai_mesh.name)
            node_name, node_world_mat, node_mesh_count = mesh_idx_to_node_info.get(
                m_idx, ("", glm.mat4(1.0), 0)
            )
            clean_node_name = node_name.split("_$AssimpFbx$_")[0] if node_name else ""

            # Prefer the scene object node name (e.g., Baju 2_Low, Pocket_Low, Rambut_Low)
            # over Blender's internal geometry datablock name (e.g., Plane.003, Sphere.002)
            # when the node represents a single sub-mesh object.
            if (
                node_mesh_count == 1
                and clean_node_name
                and clean_node_name.lower() not in ("rootnode", "root", "scene")
            ):
                base_name = clean_node_name
            else:
                base_name = raw_m_name or clean_node_name or f"SubMesh_{m_idx}"

            if base_name in used_names:
                used_names[base_name] += 1
                sub_name = f"{base_name}_{used_names[base_name]}"
            else:
                used_names[base_name] = 0
                sub_name = base_name

            verts = np.asarray(ai_mesh.vertices, dtype=np.float32)
            if verts.ndim == 1:
                verts = verts.reshape(-1, 3)
            num_verts = len(verts)
            if num_verts == 0:
                continue

            # Normals
            if hasattr(ai_mesh, "normals") and ai_mesh.normals is not None and len(ai_mesh.normals) == num_verts:
                norms = np.asarray(ai_mesh.normals, dtype=np.float32).reshape(-1, 3)
            else:
                norms = np.zeros((num_verts, 3), dtype=np.float32)
                norms[:, 1] = 1.0

            # UVs (channel 0)
            uvs = np.zeros((num_verts, 2), dtype=np.float32)
            if (
                hasattr(ai_mesh, "texturecoords")
                and ai_mesh.texturecoords is not None
                and len(ai_mesh.texturecoords) > 0
                and ai_mesh.texturecoords[0] is not None
                and len(ai_mesh.texturecoords[0]) == num_verts
            ):
                tc0 = np.asarray(ai_mesh.texturecoords[0], dtype=np.float32)
                if tc0.ndim == 2 and tc0.shape[1] >= 2:
                    uvs = np.ascontiguousarray(tc0[:, :2], dtype=np.float32)

            # Faces / Triangle Indices
            faces = np.asarray(ai_mesh.faces, dtype=np.uint32)
            indices = np.ascontiguousarray(faces.flatten(), dtype=np.uint32)

            # Vertex Bone IDs (4 per vertex) and Bone Weights (4 per vertex)
            bone_ids = np.zeros((num_verts, 4), dtype=np.int32)
            bone_weights = np.zeros((num_verts, 4), dtype=np.float32)
            vertex_slot_counts = np.zeros((num_verts,), dtype=np.int32)

            has_mesh_bones = hasattr(ai_mesh, "bones") and ai_mesh.bones is not None and len(ai_mesh.bones) > 0
            if has_mesh_bones and scene_data.skeleton.has_bones:
                for ai_bone in ai_mesh.bones:
                    b_name = cls._clean_path_str(ai_bone.name)
                    b_node = scene_data.skeleton.bone_by_name.get(b_name)
                    if b_node is None:
                        continue
                    b_id = b_node.id
                    for vw in ai_bone.weights:
                        v_id = int(vw.vertexid)
                        w_val = float(vw.weight)
                        if 0 <= v_id < num_verts and w_val > 1e-5:
                            slot_i = vertex_slot_counts[v_id]
                            if slot_i < 4:
                                bone_ids[v_id, slot_i] = b_id
                                bone_weights[v_id, slot_i] = w_val
                                vertex_slot_counts[v_id] = slot_i + 1
                            else:
                                # Replace smallest weight if this weight is larger
                                min_slot = int(np.argmin(bone_weights[v_id]))
                                if w_val > bone_weights[v_id, min_slot]:
                                    bone_ids[v_id, min_slot] = b_id
                                    bone_weights[v_id, min_slot] = w_val

                # Normalize weights per vertex where sum > 0
                w_sums = bone_weights.sum(axis=1, keepdims=True)
                valid_mask = (w_sums[:, 0] > 1e-6)
                bone_weights[valid_mask] /= w_sums[valid_mask]

            # If static mesh (not skinned, e.g. Pocket_Low), bake node_world_mat into positions & normals
            if not has_mesh_bones and node_world_mat != glm.mat4(1.0):
                mat_np = glm_to_numpy(node_world_mat)
                ones = np.ones((num_verts, 1), dtype=np.float32)
                verts = np.ascontiguousarray((np.hstack([verts, ones]) @ mat_np.T)[:, :3], dtype=np.float32)
                rot_np = mat_np[:3, :3]
                norms = np.ascontiguousarray(norms @ rot_np.T, dtype=np.float32)
                n_len = np.linalg.norm(norms, axis=1, keepdims=True)
                n_len = np.maximum(n_len, 1e-6)
                norms = np.ascontiguousarray(norms / n_len, dtype=np.float32)
                effective_node_mat = glm.mat4(1.0)
                sub_aabb = AABB.from_vertices(verts)
            else:
                verts = np.ascontiguousarray(verts, dtype=np.float32)
                norms = np.ascontiguousarray(norms, dtype=np.float32)
                effective_node_mat = glm.mat4(1.0)
                # Compute AABB in world-space using the mesh node's world transform
                sub_aabb = AABB.from_vertices(verts, transform=node_world_mat)

            overall_aabb = overall_aabb.merge(sub_aabb)

            mat_idx = int(getattr(ai_mesh, "materialindex", 0))
            if mat_idx < 0 or mat_idx >= len(scene_data.materials):
                mat_idx = 0

            submesh = SubMeshData(
                index=len(scene_data.submeshes),
                name=sub_name,
                material_index=mat_idx,
                visible=True,
                positions=verts,
                normals=norms,
                uvs=uvs,
                bone_ids=np.ascontiguousarray(bone_ids, dtype=np.int32),
                bone_weights=np.ascontiguousarray(bone_weights, dtype=np.float32),
                indices=indices,
                node_transform=effective_node_mat,
                has_bones=bool(has_mesh_bones and scene_data.skeleton.has_bones),
                aabb=sub_aabb,
            )
            scene_data.submeshes.append(submesh)

        if not overall_aabb.is_valid:
            overall_aabb = AABB(glm.vec3(-1.0, 0.0, -1.0), glm.vec3(1.0, 2.0, 1.0))
        scene_data.scene_aabb = overall_aabb
