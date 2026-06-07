"""
space/_three.py
--------------
Generates a self-contained HTML+Three.js snippet for the 3D soul space.
Called both at startup (base model points) and on each probe (adds probe point).

The snippet is injected into a gr.HTML component. Camera position persists
across rebuilds via localStorage so the view doesn't reset on each probe.
"""
from __future__ import annotations

import json
import numpy as np

MODEL_COLORS = ["#e6edf3", "#7c3aed", "#06b6d4", "#f59e0b", "#34d399", "#f472b6"]

_EMPTY = """
<div style="width:100%;height:520px;background:#0d1117;border-radius:8px;
  display:flex;align-items:center;justify-content:center;
  color:#8b949e;font-family:monospace;font-size:13px;">
  no embedding data — run export_viz to populate
</div>
"""


def build_canvas_page(viz: dict, coords3d: "np.ndarray | None",
                      probe_points: list[dict]) -> str:
    """Full standalone HTML page served by the /ofa-canvas FastAPI route."""
    if coords3d is None or len(coords3d) == 0 or not viz.get("model_names"):
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
            f"<body style='margin:0;background:#0d1117'>{_EMPTY}</body></html>"
        )
    fragment = build_umap_html(viz, coords3d, probe_points)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>*{margin:0;padding:0;box-sizing:border-box}"
        "html,body{width:100%;height:100%;background:#0d1117}</style>"
        "</head><body>" + fragment + "</body></html>"
    )


def build_iframe_tag(ts: str = "0") -> str:
    """Iframe pointing to /ofa-canvas with cache-busting timestamp."""
    return (
        f'<iframe src="/ofa-canvas?v={ts}" '
        f'style="width:100%;height:520px;border:none;border-radius:8px;display:block;">'
        f'</iframe>'
    )


def build_umap_html(viz: dict, coords3d: "np.ndarray | None",
                    probe_points: list[dict]) -> str:
    """Return a self-contained HTML+Three.js soul space for the given data."""
    if coords3d is None or len(coords3d) == 0 or not viz.get("model_names"):
        return _EMPTY

    model_names = viz["model_names"]
    labels_arr  = np.array(viz["labels"])

    models_json: list[dict] = []
    for i, name in enumerate(model_names):
        mask = labels_arr == name
        if not mask.any():
            continue
        pts = coords3d[mask].tolist()
        models_json.append({
            "name":      name,
            "color":     MODEL_COLORS[i % len(MODEL_COLORS)],
            "points":    pts,
            "isStudent": name == "student",
        })

    payload = json.dumps({"models": models_json, "probes": probe_points})
    return _TEMPLATE.replace("__OFA_DATA__", payload)


# ── Three.js HTML template ─────────────────────────────────────────────────
_TEMPLATE = r"""
<style>
#ofa-root{position:relative;width:100%;height:520px;background:#0d1117;border-radius:8px;overflow:hidden;user-select:none}
#ofa-canvas{display:block;width:100%;height:100%}
#ofa-sub{position:absolute;top:12px;left:16px;font-size:11px;color:#8b949e;font-family:monospace;pointer-events:none}
#ofa-legend{position:absolute;bottom:12px;left:16px;display:flex;flex-wrap:wrap;gap:8px;pointer-events:none}
.ofa-li{display:flex;align-items:center;gap:5px;font-size:11px;color:#8b949e;font-family:monospace}
.ofa-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.ofa-lbl{position:absolute;font-size:11px;font-family:monospace;color:#e6edf3;
  background:rgba(13,17,23,0.88);border:1px solid #30363d;border-radius:4px;
  padding:2px 8px;pointer-events:none;transform:translate(-50%,-130%);white-space:nowrap}
.ofa-lbl.student{color:#fff;border-color:#7c3aed;background:rgba(124,58,237,0.18)}
</style>
<div id="ofa-root">
  <canvas id="ofa-canvas"></canvas>
  <div id="ofa-sub">soul space — UMAP 3D &nbsp;·&nbsp; drag to rotate &nbsp;·&nbsp; scroll to zoom</div>
  <div id="ofa-legend"></div>
</div>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const DATA   = __OFA_DATA__;
const root   = document.getElementById('ofa-root');
const canvas = document.getElementById('ofa-canvas');
const legend = document.getElementById('ofa-legend');

// ── renderer ──────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0d1117, 1);

// ── camera + controls ─────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 500);
camera.position.set(3, 2, 7);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping  = true;
controls.dampingFactor  = 0.07;
controls.rotateSpeed    = 0.65;
controls.zoomSpeed      = 1.2;
controls.enablePan      = true;
controls.minDistance    = 0.5;
controls.maxDistance    = 40;

// persist camera across HTML rebuilds (probe updates)
const CAM_KEY = 'ofa-soul-cam-v1';
try {
  const s = JSON.parse(localStorage.getItem(CAM_KEY));
  if (s) {
    camera.position.set(s.px, s.py, s.pz);
    controls.target.set(s.tx, s.ty, s.tz);
    controls.update();
  }
} catch(_) {}
controls.addEventListener('change', () => {
  try {
    localStorage.setItem(CAM_KEY, JSON.stringify({
      px: camera.position.x, py: camera.position.y, pz: camera.position.z,
      tx: controls.target.x,  ty: controls.target.y,  tz: controls.target.z,
    }));
  } catch(_) {}
});

// ── scene ─────────────────────────────────────────────────────────
const scene = new THREE.Scene();

// normalise all model points to [-2, 2] so the cloud fits any UMAP scale
const allPts = DATA.models.flatMap(m => m.points);
let normFn = p => p;
if (allPts.length > 0) {
  const xs = allPts.map(p => p[0]), ys = allPts.map(p => p[1]), zs = allPts.map(p => p[2]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const [z0, z1] = [Math.min(...zs), Math.max(...zs)];
  const cx = (x0+x1)/2, cy = (y0+y1)/2, cz = (z0+z1)/2;
  const span = Math.max(x1-x0, y1-y0, z1-z0, 1e-6);
  const k = 4 / span;
  normFn = ([x, y, z]) => [(x-cx)*k, (y-cy)*k, (z-cz)*k];
  // also normalise probe points that were stored in UMAP space
  if (DATA.probes) {
    DATA.probes = DATA.probes.map(p => {
      const [nx, ny, nz] = normFn([p.x, p.y, p.z]);
      return { ...p, x: nx, y: ny, z: nz };
    });
  }
}

// label DOM elements, updated each frame via project()
const labelEls = [];

// ── model point clouds ────────────────────────────────────────────
DATA.models.forEach((m, idx) => {
  const pts = m.points.map(normFn);
  if (!pts.length) return;

  const pos = new Float32Array(pts.length * 3);
  pts.forEach(([x,y,z], i) => { pos[i*3]=x; pos[i*3+1]=y; pos[i*3+2]=z; });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  const mat = new THREE.PointsMaterial({
    color:           new THREE.Color(m.color),
    size:            m.isStudent ? 0.20 : 0.09,
    sizeAttenuation: true,
    transparent:     true,
    opacity:         m.isStudent ? 1.0 : 0.80,
    depthWrite:      m.isStudent,
  });

  scene.add(new THREE.Points(geo, mat));

  // centroid for label anchor
  const cx = pts.reduce((s,p) => s+p[0], 0) / pts.length;
  const cy = pts.reduce((s,p) => s+p[1], 0) / pts.length;
  const cz = pts.reduce((s,p) => s+p[2], 0) / pts.length;

  const el = document.createElement('div');
  el.className = 'ofa-lbl' + (m.isStudent ? ' student' : '');
  el.textContent = m.isStudent ? 'student · Qwen2.5-0.5B' : m.name;
  root.appendChild(el);
  labelEls.push({ el, world: new THREE.Vector3(cx, cy, cz) });

  // legend
  const li = document.createElement('div'); li.className = 'ofa-li';
  const dot = document.createElement('div'); dot.className = 'ofa-dot';
  dot.style.background = m.color;
  if (m.isStudent) { dot.style.width = dot.style.height = '10px'; }
  li.appendChild(dot);
  li.appendChild(document.createTextNode(m.isStudent
    ? 'student — Qwen2.5-0.5B'
    : `${m.name} — teacher`));
  legend.appendChild(li);
});

// ── live probe points ─────────────────────────────────────────────
if (DATA.probes && DATA.probes.length) {
  const n   = DATA.probes.length;
  const pos = new Float32Array(n * 3);
  DATA.probes.forEach((p, i) => { pos[i*3]=p.x; pos[i*3+1]=p.y; pos[i*3+2]=p.z; });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  // inner bright point
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
    color: 0xffffff, size: 0.26, sizeAttenuation: true,
    transparent: true, opacity: 1.0, depthWrite: true,
  })));
  // outer glow (additive blending)
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
    color: 0xaaaaff, size: 0.55, sizeAttenuation: true,
    transparent: true, opacity: 0.35,
    blending: THREE.AdditiveBlending, depthWrite: false,
  })));
}

// ── resize ────────────────────────────────────────────────────────
function onResize() {
  const w = root.clientWidth, h = root.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
onResize();
new ResizeObserver(onResize).observe(root);

// ── render loop ───────────────────────────────────────────────────
let raf;
function tick() {
  raf = requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene, camera);

  // reproject HTML labels each frame
  const W = root.clientWidth, H = root.clientHeight;
  for (const { el, world } of labelEls) {
    const p = world.clone().project(camera);
    const behind = p.z > 1;
    el.style.display = behind ? 'none' : 'block';
    if (!behind) {
      el.style.left = ((p.x *  0.5 + 0.5) * W) + 'px';
      el.style.top  = ((p.y * -0.5 + 0.5) * H) + 'px';
    }
  }
}
tick();

// cleanup on removal
new MutationObserver(() => {
  if (!root.isConnected) { cancelAnimationFrame(raf); renderer.dispose(); }
}).observe(document.body, { childList: true, subtree: true });
</script>
"""
