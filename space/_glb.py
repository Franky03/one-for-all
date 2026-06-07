"""
space/_glb.py
-------------
Builds a GLB mesh (sphere per point) from UMAP coords for gr.Model3D.
PointCloud primitives render at 1px in Three.js regardless of scale;
small spheres give controllable apparent size.
"""
from __future__ import annotations
import tempfile
import numpy as np

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

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = colors

    path = tempfile.mktemp(suffix=".glb")
    mesh.export(path)
    return path


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
        '<span style="font-size:11px;color:#8b949e;font-family:monospace;">● probe — your input</span>'
        '</div>'
    )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:10px 18px;padding:8px 2px;">'
        + "".join(items)
        + '</div>'
    )
