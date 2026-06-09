"""
space/_glb.py
-------------
Builds a GLB mesh (sphere per point) from UMAP coords for gr.Model3D.
PointCloud primitives render at 1px in Three.js regardless of scale;
small spheres give controllable apparent size.
"""
from __future__ import annotations
import os
import json
import struct
import tempfile
from pathlib import Path
import numpy as np


def _gradio_tmp() -> str:
    # Gradio 5 only serves files inside its own temp dir; /tmp/tmpXXX.glb is
    # outside it and returns 403. Match the same path Gradio uses internally.
    d = os.environ.get("GRADIO_TEMP_DIR") or str(
        (Path(tempfile.gettempdir()) / "gradio").resolve()
    )
    os.makedirs(d, exist_ok=True)
    return d

_COLORS_RGB: list[tuple[int, int, int]] = [
    (230, 237, 243),  # student  — #e6edf3
    (124,  58, 237),  # teacher0 — #7c3aed
    (  6, 182, 212),  # teacher1 — #06b6d4
    (245, 158,  11),  # teacher2 — #f59e0b
    ( 52, 211, 153),  # teacher3 — #34d399
    (244, 114, 182),  # teacher4 — #f472b6
]
_PROBE_COLOR = (255, 255, 255)


def _hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _inject_material(glb: bytes) -> bytes:
    """Add a matte PBR material to every primitive in a GLB.

    trimesh exports vertex-colored meshes with POSITION + COLOR_0 but no
    material, so model-viewer (the PBR renderer behind gr.Model3D) falls back
    to the glTF default material (metallicFactor=1, roughnessFactor=1). Under
    model-viewer's neutral environment a fully-metallic surface renders dark,
    which is why the model looked gray. A matte (metallic=0) material lets the
    per-vertex COLOR_0 show through, lit correctly.
    """
    json_len = struct.unpack("<I", glb[12:16])[0]
    json_bytes = glb[20:20 + json_len]
    bin_chunk = glb[20 + json_len:]  # keeps its own 8-byte chunk header
    gltf = json.loads(json_bytes)
    gltf.setdefault("materials", []).append({
        "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.85,
        },
        "doubleSided": True,
    })
    midx = len(gltf["materials"]) - 1
    for mesh in gltf.get("meshes", []):
        for prim in mesh["primitives"]:
            prim["material"] = midx
    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    new_json += b" " * ((-len(new_json)) % 4)   # 4-byte align, pad with spaces
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(new_json) + len(bin_chunk))
    out += struct.pack("<II", len(new_json), 0x4E4F534A)  # JSON chunk header
    out += new_json
    out += bin_chunk
    return bytes(out)


def build_glb(
    viz: dict,
    coords3d: "np.ndarray | None",
    probe_points: list[dict],
) -> str | None:
    """Return path to a temporary .glb with one small sphere per embedding point."""
    if coords3d is None or len(coords3d) == 0 or not viz.get("model_names"):
        return None

    import trimesh

    model_names = viz["model_names"]
    labels      = np.array(viz["labels"])

    # Adaptive radius: 1.8 % of the data bounding-box diagonal
    span   = float(np.linalg.norm(coords3d.max(axis=0) - coords3d.min(axis=0)))
    radius = max(span * 0.018, 0.04)

    # Build template sphere once, scale per group
    tpl        = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    tpl_v      = tpl.vertices.astype(np.float64)   # (42, 3)
    tpl_f      = tpl.faces                          # (80, 3)
    n_v        = len(tpl_v)

    all_verts  : list[np.ndarray] = []
    all_faces  : list[np.ndarray] = []
    all_colors : list[np.ndarray] = []
    offset = 0

    def _add_group(pts: np.ndarray, rgb: tuple[int, int, int], r: float) -> None:
        nonlocal offset
        color = np.array([*rgb, 255], dtype=np.uint8)
        for pt in pts:
            all_verts.append(tpl_v * r + pt)
            all_faces.append(tpl_f + offset)
            all_colors.append(np.tile(color, (n_v, 1)))
            offset += n_v

    for i, name in enumerate(model_names):
        mask = labels == name
        if not mask.any():
            continue
        pts = coords3d[mask].astype(np.float64)
        r   = radius * (1.6 if name == "student" else 1.0)
        _add_group(pts, _COLORS_RGB[i % len(_COLORS_RGB)], r)

    if probe_points:
        probe_pts = np.array([[p["x"], p["y"], p["z"]] for p in probe_points],
                             dtype=np.float64)
        _add_group(probe_pts, _PROBE_COLOR, radius * 2.0)

    if not all_verts:
        return None

    vertices = np.concatenate(all_verts,  axis=0)
    faces    = np.concatenate(all_faces,  axis=0)
    colors   = np.concatenate(all_colors, axis=0)

    # vertex_colors in the constructor → COLOR_0 attribute on export.
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=colors, process=False
    )
    _ = mesh.vertex_normals  # force smooth normals so the PBR renderer can shade

    glb_bytes = _inject_material(mesh.export(file_type="glb", include_normals=True))

    tmp = tempfile.NamedTemporaryFile(
        suffix=".glb", dir=_gradio_tmp(), delete=False
    )
    tmp.write(glb_bytes)
    tmp.close()
    return tmp.name


def build_legend_html(viz: dict) -> str:
    """Colored dot legend matching the GLB sphere colors."""
    if not viz.get("model_names"):
        return ""
    items = []
    for i, name in enumerate(viz["model_names"]):
        r, g, b = _COLORS_RGB[i % len(_COLORS_RGB)]
        dot_color = _hex(r, g, b)
        is_student = name == "student"
        label = "student — Qwen2.5-0.5B" if is_student else f"{name} — teacher"
        size  = "10px" if is_student else "8px"
        items.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:{size};height:{size};border-radius:50%;'
            f'background:{dot_color};flex-shrink:0;"></div>'
            f'<span style="font-size:11px;color:#8b949e;font-family:monospace;">{label}</span>'
            f'</div>'
        )
    items.append(
        '<div style="display:flex;align-items:center;gap:6px;">'
        '<div style="width:8px;height:8px;border-radius:50%;background:#ffffff;flex-shrink:0;"></div>'
        '<span style="font-size:11px;color:#8b949e;font-family:monospace;">probe — your input</span>'
        '</div>'
    )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:10px 18px;padding:8px 2px;">'
        + "".join(items)
        + '</div>'
    )
