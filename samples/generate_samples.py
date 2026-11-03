"""
samples/generate_samples.py
Generates a rich, standard-compliant rigged multi-mesh glTF 2.0 asset (`samples/UniView_CyberKnight.gltf` + `.bin`)
and companion texture files in `samples/textures/` for immediate out-of-the-box testing of UniView.

Features of the generated asset:
1. 4 Distinct Sub-Meshes:
   - `Body_Low` (Pelvis, Torso & Legs)
   - `Armor_Chest` (Layered chestplate & shoulder pauldrons)
   - `Helmet_Low` (Head & Visor crest)
   - `Gauntlets_Low` (Left & Right articulated arms)
2. 3 Distinct Material Slots:
   - `Body_Mat` (References missing `Body_BaseColor.png` & `Body_Normal_OpenGL.png`)
   - `Armor_Crimson_Mat` (References missing `Armor_BaseColor.png`, base color #cc001c)
   - `Helmet_Gold_Mat` (References available `textures/Helmet_Albedo.png`)
3. Ready-to-Relink Texture Files in `samples/textures/`:
   - `samples/textures/Helmet_Albedo.png` (Automatically resolved as Available)
   - `samples/textures/Armor_BaseColor.png` (Ready for drag-and-drop / browse relinking onto Missing row!)
   - `samples/textures/Body_BaseColor.png` (Ready for drag-and-drop / browse relinking onto Missing row!)
   - `samples/textures/Body_Normal_OpenGL.png` (Ready for drag-and-drop / browse relinking!)
4. Hierarchical 8-Bone Skeleton (`skin`) with smooth vertex weights (`JOINTS_0`, `WEIGHTS_0`):
   - `Root` (0, 0, 0)
     └── `Pelvis` (0, 0.85, 0)
           └── `Spine` (0, 1.20, 0)
                 └── `Chest` (0, 1.55, 0)
                       ├── `Head` (0, 1.95, 0)
                       ├── `L_Arm` (0.45, 1.65, 0)
                       │     └── `L_ForeArm` (0.85, 1.65, 0)
                       └── `R_Arm` (-0.45, 1.65, 0)
                             └── `R_ForeArm` (-0.85, 1.65, 0)
"""

from __future__ import annotations
import os
import json
import math
import struct
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw


def _create_procedural_textures(tex_dir: str) -> None:
    """Create visually distinct procedural textures for testing Available & Relinked states."""
    os.makedirs(tex_dir, exist_ok=True)

    # 1. Helmet_Albedo.png (Available right away - Gold & Cyan cyber visor grid)
    img_helmet = Image.new("RGBA", (256, 256), (230, 185, 45, 255))
    draw = ImageDraw.Draw(img_helmet)
    for i in range(0, 256, 32):
        draw.line([(i, 0), (i, 256)], fill=(180, 130, 20, 255), width=2)
        draw.line([(0, i), (256, i)], fill=(180, 130, 20, 255), width=2)
    draw.rectangle([24, 96, 232, 160], fill=(20, 210, 240, 255), outline=(10, 80, 110, 255), width=4)
    img_helmet.save(os.path.join(tex_dir, "Helmet_Albedo.png"))

    # 2. Armor_BaseColor.png (In textures/ folder so user can drag-and-drop relink `Armor_BaseColor.png`)
    img_armor = Image.new("RGBA", (256, 256), (204, 16, 36, 255))
    draw_a = ImageDraw.Draw(img_armor)
    for y in range(0, 256, 32):
        for x in range(0, 256, 32):
            if ((x // 32) + (y // 32)) % 2 == 0:
                draw_a.rectangle([x, y, x + 31, y + 31], fill=(165, 12, 28, 255))
    draw_a.rectangle([8, 8, 247, 247], outline=(245, 195, 60, 255), width=6)
    img_armor.save(os.path.join(tex_dir, "Armor_BaseColor.png"))

    # 3. Body_BaseColor.png (Dark carbon-fiber undersuit grid)
    img_body = Image.new("RGBA", (256, 256), (48, 54, 66, 255))
    draw_b = ImageDraw.Draw(img_body)
    for i in range(0, 256, 16):
        draw_b.line([(i, 0), (i, 256)], fill=(68, 76, 92, 255), width=2)
        draw_b.line([(0, i), (256, i)], fill=(34, 38, 46, 255), width=2)
    img_body.save(os.path.join(tex_dir, "Body_BaseColor.png"))

    # 4. Body_Normal_OpenGL.png (OpenGL tangent-space normal map)
    img_norm = Image.new("RGBA", (256, 256), (128, 128, 255, 255))
    draw_n = ImageDraw.Draw(img_norm)
    for i in range(16, 256, 32):
        draw_n.rectangle([i - 6, i - 6, i + 6, i + 6], outline=(160, 96, 255, 255), width=2)
    img_norm.save(os.path.join(tex_dir, "Body_Normal_OpenGL.png"))


def _make_box_geometry(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    subdiv_y: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a subdivided 3D box/limb geometry with smooth vertical subdivisions
    so skeletal skinning bends cleanly along its height/length.
    Returns (positions (N,3), normals (N,3), uvs (N,2), indices (M,3)).
    """
    cx, cy, cz = center
    sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5

    positions = []
    normals = []
    uvs = []
    indices = []

    def add_quad_grid(p00, p10, p11, p01, normal, steps_u=2, steps_v=subdiv_y):
        base_idx = len(positions)
        p00 = np.array(p00, dtype=np.float32)
        p10 = np.array(p10, dtype=np.float32)
        p11 = np.array(p11, dtype=np.float32)
        p01 = np.array(p01, dtype=np.float32)
        n = np.array(normal, dtype=np.float32)

        for iv in range(steps_v + 1):
            v = iv / steps_v
            left = p00 * (1.0 - v) + p01 * v
            right = p10 * (1.0 - v) + p11 * v
            for iu in range(steps_u + 1):
                u = iu / steps_u
                pt = left * (1.0 - u) + right * u
                positions.append(pt)
                normals.append(n)
                uvs.append([u, v])

        row_len = steps_u + 1
        for iv in range(steps_v):
            for iu in range(steps_u):
                i0 = base_idx + iv * row_len + iu
                i1 = i0 + 1
                i2 = i0 + row_len + 1
                i3 = i0 + row_len
                indices.append([i0, i1, i2])
                indices.append([i0, i2, i3])

    # Front (+Z)
    add_quad_grid([cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
                  [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz], [0, 0, 1])
    # Back (-Z)
    add_quad_grid([cx + sx, cy - sy, cz - sz], [cx - sx, cy - sy, cz - sz],
                  [cx - sx, cy + sy, cz - sz], [cx + sx, cy + sy, cz - sz], [0, 0, -1])
    # Right (+X)
    add_quad_grid([cx + sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz - sz],
                  [cx + sx, cy + sy, cz - sz], [cx + sx, cy + sy, cz + sz], [1, 0, 0])
    # Left (-X)
    add_quad_grid([cx - sx, cy - sy, cz - sz], [cx - sx, cy - sy, cz + sz],
                  [cx - sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz - sz], [-1, 0, 0])
    # Top (+Y)
    add_quad_grid([cx - sx, cy + sy, cz + sz], [cx + sx, cy + sy, cz + sz],
                  [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz], [0, 1, 0], 2, 2)
    # Bottom (-Y)
    add_quad_grid([cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
                  [cx + sx, cy - sy, cz + sz], [cx - sx, cy - sy, cz + sz], [0, -1, 0], 2, 2)

    return (
        np.array(positions, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(uvs, dtype=np.float32),
        np.array(indices, dtype=np.uint16),
    )


def _merge_parts(parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    all_pos, all_norm, all_uv, all_idx = [], [], [], []
    offset = 0
    for pos, norm, uv, idx in parts:
        all_pos.append(pos)
        all_norm.append(norm)
        all_uv.append(uv)
        all_idx.append(idx + offset)
        offset += len(pos)
    return (
        np.vstack(all_pos).astype(np.float32),
        np.vstack(all_norm).astype(np.float32),
        np.vstack(all_uv).astype(np.float32),
        np.vstack(all_idx).astype(np.uint16),
    )


def _compute_skin_weights(positions: np.ndarray, bone_world_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign smooth 4-bone skinning weights (`JOINTS_0`, `WEIGHTS_0`) to each vertex
    based on inverse-distance falloff to the 9 skeleton joints.
    """
    num_verts = len(positions)
    joints = np.zeros((num_verts, 4), dtype=np.uint16)
    weights = np.zeros((num_verts, 4), dtype=np.float32)

    for i, v in enumerate(positions):
        dists = np.linalg.norm(bone_world_positions - v, axis=1)
        # Sort 4 closest bones
        closest_idx = np.argsort(dists)[:4]
        closest_dists = np.maximum(dists[closest_idx], 0.02)

        # Sharp inverse-cubic weighting for crisp yet smooth limb bending
        raw_w = 1.0 / (closest_dists ** 3)
        total_w = np.sum(raw_w)
        norm_w = raw_w / total_w

        joints[i] = closest_idx.astype(np.uint16)
        weights[i] = norm_w.astype(np.float32)

    return joints, weights


def generate_sample_assets(output_dir: Optional[str] = None) -> str:
    """
    Generate `UniView_CyberKnight.gltf` + `UniView_CyberKnight.bin` and texture files.
    Returns the absolute path to the generated `.gltf` file.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    tex_dir = os.path.join(output_dir, "textures")
    _create_procedural_textures(tex_dir)

    # Define 9 hierarchical bones:
    # 0: Root (0, 0, 0)
    # 1: Pelvis (0, 0.85, 0) parent 0
    # 2: Spine (0, 1.20, 0) parent 1
    # 3: Chest (0, 1.55, 0) parent 2
    # 4: Head (0, 1.95, 0) parent 3
    # 5: L_Arm (0.42, 1.62, 0) parent 3
    # 6: L_ForeArm (0.85, 1.62, 0) parent 5
    # 7: R_Arm (-0.42, 1.62, 0) parent 3
    # 8: R_ForeArm (-0.85, 1.62, 0) parent 7
    bone_defs = [
        ("Root", None, [0.0, 0.0, 0.0]),
        ("Pelvis", 0, [0.0, 0.85, 0.0]),
        ("Spine", 1, [0.0, 1.20, 0.0]),
        ("Chest", 2, [0.0, 1.55, 0.0]),
        ("Head", 3, [0.0, 1.95, 0.0]),
        ("L_Arm", 3, [0.42, 1.62, 0.0]),
        ("L_ForeArm", 5, [0.85, 1.62, 0.0]),
        ("R_Arm", 3, [-0.42, 1.62, 0.0]),
        ("R_ForeArm", 7, [-0.85, 1.62, 0.0]),
    ]

    bone_world_pos = np.array([b[2] for b in bone_defs], dtype=np.float32)

    # Build 4 distinct sub-meshes:
    # 1. Body_Low (Legs, Pelvis, Spine core, Neck) -> Material 0 (Body_Mat)
    body_geom = _merge_parts([
        _make_box_geometry((-0.16, 0.42, 0.0), (0.18, 0.84, 0.20), subdiv_y=5),  # Right leg
        _make_box_geometry((0.16, 0.42, 0.0), (0.18, 0.84, 0.20), subdiv_y=5),   # Left leg
        _make_box_geometry((0.0, 0.92, 0.0), (0.50, 0.24, 0.26), subdiv_y=3),    # Pelvis block
        _make_box_geometry((0.0, 1.25, 0.0), (0.38, 0.42, 0.22), subdiv_y=4),    # Spine abdomen
        _make_box_geometry((0.0, 1.84, 0.0), (0.16, 0.16, 0.16), subdiv_y=2),    # Neck
    ])

    # 2. Armor_Chest (Breastplate, Pauldrons, Backplate) -> Material 1 (Armor_Crimson_Mat)
    armor_geom = _merge_parts([
        _make_box_geometry((0.0, 1.56, 0.02), (0.58, 0.44, 0.34), subdiv_y=4),   # Chest torso plate
        _make_box_geometry((0.40, 1.70, 0.0), (0.26, 0.22, 0.30), subdiv_y=2),   # Left pauldron
        _make_box_geometry((-0.40, 1.70, 0.0), (0.26, 0.22, 0.30), subdiv_y=2),  # Right pauldron
        _make_box_geometry((0.0, 0.82, 0.14), (0.22, 0.24, 0.06), subdiv_y=2),   # Belt tasset
    ])

    # 3. Helmet_Low (Head, Visor, Crest) -> Material 2 (Helmet_Gold_Mat)
    helmet_geom = _merge_parts([
        _make_box_geometry((0.0, 2.08, 0.0), (0.32, 0.34, 0.34), subdiv_y=3),    # Main helmet
        _make_box_geometry((0.0, 2.10, 0.16), (0.28, 0.10, 0.08), subdiv_y=2),   # Visor ridge
        _make_box_geometry((0.0, 2.30, -0.02), (0.08, 0.14, 0.36), subdiv_y=2),  # Top crest
    ])

    # 4. Gauntlets_Low (Left & Right Upper/Forearms) -> Material 1 (Armor_Crimson_Mat)
    gauntlets_geom = _merge_parts([
        _make_box_geometry((0.62, 1.62, 0.0), (0.36, 0.16, 0.16), subdiv_y=4),   # L Upper Arm
        _make_box_geometry((1.02, 1.62, 0.0), (0.38, 0.18, 0.18), subdiv_y=4),   # L Forearm Gauntlet
        _make_box_geometry((-0.62, 1.62, 0.0), (0.36, 0.16, 0.16), subdiv_y=4),  # R Upper Arm
        _make_box_geometry((-1.02, 1.62, 0.0), (0.38, 0.18, 0.18), subdiv_y=4),  # R Forearm Gauntlet
    ])

    submeshes = [
        ("Body_Low", body_geom, 0),
        ("Armor_Chest", armor_geom, 1),
        ("Helmet_Low", helmet_geom, 2),
        ("Gauntlets_Low", gauntlets_geom, 1),
    ]

    # Build binary buffer (.bin) for glTF 2.0
    bin_chunks = bytearray()
    buffer_views = []
    accessors = []

    def append_accessor(arr: np.ndarray, component_type: int, acc_type: str, target: Optional[int] = None) -> int:
        # Align to 4-byte boundary
        while len(bin_chunks) % 4 != 0:
            bin_chunks.append(0)
        byte_offset = len(bin_chunks)
        raw = arr.tobytes()
        bin_chunks.extend(raw)

        bv = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(raw),
        }
        if target is not None:
            bv["target"] = target
        bv_idx = len(buffer_views)
        buffer_views.append(bv)

        acc = {
            "bufferView": bv_idx,
            "byteOffset": 0,
            "componentType": component_type,
            "count": len(arr),
            "type": acc_type,
        }
        if acc_type == "VEC3" and arr.dtype == np.float32:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        elif acc_type == "SCALAR" and len(arr) > 0:
            acc["min"] = [int(arr.min())]
            acc["max"] = [int(arr.max())]

        acc_idx = len(accessors)
        accessors.append(acc)
        return acc_idx

    # Inverse bind matrices for the 9 bones (column-major 4x4 float32 each)
    ibm_list = []
    for _, _, wpos in bone_defs:
        # Column-major 4x4 translation(-wpos)
        mat_col_major = np.eye(4, dtype=np.float32)
        mat_col_major[3, 0] = -wpos[0]
        mat_col_major[3, 1] = -wpos[1]
        mat_col_major[3, 2] = -wpos[2]
        ibm_list.append(mat_col_major)
    ibm_array = np.array(ibm_list, dtype=np.float32)
    ibm_accessor_idx = append_accessor(ibm_array, 5126, "MAT4", target=None)

    gltf_meshes = []
    for name, (pos, norm, uv, idx), mat_idx in submeshes:
        j_arr, w_arr = _compute_skin_weights(pos, bone_world_pos)
        pos_acc = append_accessor(pos, 5126, "VEC3", target=34962)
        norm_acc = append_accessor(norm, 5126, "VEC3", target=34962)
        uv_acc = append_accessor(uv, 5126, "VEC2", target=34962)
        j_acc = append_accessor(j_arr, 5123, "VEC4", target=34962)
        w_acc = append_accessor(w_arr, 5126, "VEC4", target=34962)
        idx_acc = append_accessor(idx.flatten(), 5123, "SCALAR", target=34963)

        gltf_meshes.append({
            "name": name,
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": norm_acc,
                    "TEXCOORD_0": uv_acc,
                    "JOINTS_0": j_acc,
                    "WEIGHTS_0": w_acc,
                },
                "indices": idx_acc,
                "material": mat_idx,
                "mode": 4,
            }],
        })

    # Construct glTF nodes:
    # Nodes 0..8 are the 9 bones
    nodes = []
    for bone_idx, (b_name, parent_idx, wpos) in enumerate(bone_defs):
        if parent_idx is None:
            local_t = wpos
        else:
            p_pos = bone_defs[parent_idx][2]
            local_t = [wpos[0] - p_pos[0], wpos[1] - p_pos[1], wpos[2] - p_pos[2]]

        children = [i for i, b in enumerate(bone_defs) if b[1] == bone_idx]
        node_dict = {
            "name": b_name,
            "translation": local_t,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        }
        if children:
            node_dict["children"] = children
        nodes.append(node_dict)

    # Nodes 9..12 are the 4 sub-mesh nodes referencing skin 0
    mesh_node_indices = []
    for m_idx, (m_name, _, _) in enumerate(submeshes):
        n_idx = len(nodes)
        mesh_node_indices.append(n_idx)
        nodes.append({
            "name": m_name,
            "mesh": m_idx,
            "skin": 0,
        })

    bin_filename = "UniView_CyberKnight.bin"
    bin_path = os.path.join(output_dir, bin_filename)
    with open(bin_path, "wb") as f:
        f.write(bin_chunks)

    # Note on images:
    # Image 0 & 1 intentionally point to "Body_BaseColor.png" & "Body_Normal_OpenGL.png" (Missing in root, present in textures/ for relink test!)
    # Image 2 intentionally points to "Armor_BaseColor.png" (Missing in root, present in textures/ for relink test!)
    # Image 3 points to "textures/Helmet_Albedo.png" (Resolved & Available automatically!)
    gltf_doc = {
        "asset": {
            "version": "2.0",
            "generator": "UniView Procedural Sample Generator",
        },
        "scene": 0,
        "scenes": [{
            "name": "CyberKnight_Scene",
            "nodes": [0] + mesh_node_indices,
        }],
        "nodes": nodes,
        "skins": [{
            "name": "CyberKnight_Armature",
            "inverseBindMatrices": ibm_accessor_idx,
            "skeleton": 0,
            "joints": list(range(len(bone_defs))),
        }],
        "meshes": gltf_meshes,
        "materials": [
            {
                "name": "Body_Carbon_Mat",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.22, 0.25, 0.30, 1.0],
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.2,
                    "roughnessFactor": 0.7,
                },
                "normalTexture": {"index": 1},
            },
            {
                "name": "Armor_Crimson_Mat",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.80, 0.06, 0.14, 1.0],
                    "baseColorTexture": {"index": 2},
                    "metallicFactor": 0.6,
                    "roughnessFactor": 0.35,
                },
            },
            {
                "name": "Helmet_Gold_Mat",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.95, 0.78, 0.22, 1.0],
                    "baseColorTexture": {"index": 3},
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.25,
                },
            },
        ],
        "textures": [
            {"source": 0, "name": "Body_BaseColor_Tex"},
            {"source": 1, "name": "Body_Normal_Tex"},
            {"source": 2, "name": "Armor_BaseColor_Tex"},
            {"source": 3, "name": "Helmet_Albedo_Tex"},
        ],
        "images": [
            {"uri": "Body_BaseColor.png", "name": "Body_BaseColor"},
            {"uri": "Body_Normal_OpenGL.png", "name": "Body_Normal_OpenGL"},
            {"uri": "Armor_BaseColor.png", "name": "Armor_BaseColor"},
            {"uri": "textures/Helmet_Albedo.png", "name": "Helmet_Albedo"},
        ],
        "buffers": [{
            "uri": bin_filename,
            "byteLength": len(bin_chunks),
        }],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    gltf_path = os.path.join(output_dir, "UniView_CyberKnight.gltf")
    with open(gltf_path, "w", encoding="utf-8") as f:
        json.dump(gltf_doc, f, indent=2)

    return gltf_path


if __name__ == "__main__":
    out = generate_sample_assets()
    print(f"Generated sample rigged model at: {out}")
