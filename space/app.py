"""
space/app.py — One for All HuggingFace ZeroGPU Gradio Space.

Run locally (from space/ dir, with a local viz_data.json):
    cd space && VIZ_DATA_PATH=/path/to/viz_data.json python app.py
"""
from __future__ import annotations
import os
import html as _html_stdlib
import gradio as gr
import spaces

import _data
import _fig
import _glb
import _html
import _probe

# ── Startup: load data, fit UMAP, load student ───────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")

_local_viz = os.environ.get("VIZ_DATA_PATH")
try:
    if _local_viz:
        VIZ = _data.load_from_path(_local_viz)
        print(f"[ofa-space] loaded viz from {_local_viz}")
    else:
        VIZ = _data.load_and_parse(HF_TOKEN)
except Exception as e:
    print(f"[ofa-space] viz_data.json not available ({e}), using empty state")
    VIZ = _data.make_empty_viz()

try:
    if VIZ["stacked"].shape[0] > 3:
        REDUCER  = _data.fit_umap3d(VIZ["stacked"])
        COORDS3D = REDUCER.embedding_
        print(f"[ofa-space] UMAP done: {COORDS3D.shape}")
    else:
        print(f"[ofa-space] not enough points for UMAP: {VIZ['stacked'].shape[0]}")
        REDUCER  = None
        COORDS3D = None
except Exception as e:
    print(f"[ofa-space] UMAP failed ({e}), 3D disabled")
    REDUCER  = None
    COORDS3D = None

try:
    TOK, STUDENT, GATING = _probe.load_student(HF_TOKEN)
    _MODEL_READY = True
except Exception as e:
    print(f"[ofa-space] Student not available ({e}). Probe disabled.")
    TOK = STUDENT = GATING = None
    _MODEL_READY = False

_INIT_GLB = _glb.build_glb(VIZ, COORDS3D, [])
print(f"[ofa-space] GLB path: {_INIT_GLB}")
if _INIT_GLB:
    print(f"[ofa-space] GLB exists: {os.path.exists(_INIT_GLB)}, size: {os.path.getsize(_INIT_GLB)} bytes")

# Camera: pull back far enough to see all points at startup
if COORDS3D is not None:
    import numpy as _np
    _span = float(_np.linalg.norm(COORDS3D.max(axis=0) - COORDS3D.min(axis=0)))
    _CAM = (45, 30, _span * 1.8)
else:
    _CAM = (45, 30, 10)


def _response_html(text: str) -> str:
    safe = _html_stdlib.escape(text).replace("\n", "<br>")
    return (
        '<div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;'
        'padding:14px;margin-top:8px;">'
        '<div style="font-size:10px;color:#8b949e;font-family:monospace;'
        'margin-bottom:8px;letter-spacing:0.04em;">MODEL RESPONSE</div>'
        f'<div style="font-size:13px;color:#e6edf3;line-height:1.65;">{safe}</div>'
        "</div>"
    )


# ── ZeroGPU probe handler ─────────────────────────────────────────────────
@spaces.GPU
def probe_fn(text: str, probe_points: list) -> tuple:
    no_change = _glb.build_glb(VIZ, COORDS3D, probe_points), probe_points, "", "", ""
    if not text.strip():
        return no_change
    if not _MODEL_READY or REDUCER is None:
        msg = _html.gate_html([0.2] * 5, VIZ["teacher_names"] or ["—"] * 5)
        return _glb.build_glb(VIZ, COORDS3D, probe_points), probe_points, "", msg, ""
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    STUDENT.to(device)
    answer               = _probe.generate_response(text, STUDENT, TOK)
    new_pt, gate_weights = _probe.run_probe(text, STUDENT, TOK, GATING, REDUCER)
    updated              = probe_points + [new_pt]
    glb_path             = _glb.build_glb(VIZ, COORDS3D, updated)
    gate_h               = _html.gate_html(gate_weights, VIZ["teacher_names"])
    task_h               = _html.task_html(gate_weights, VIZ["teacher_names"])
    resp_h               = _response_html(answer)
    return glb_path, updated, resp_h, gate_h, task_h


# ── CSS ───────────────────────────────────────────────────────────────────
CSS = """
/* ── Variables ─────────────────────────────────────────── */
:root {
  --bg:           #080b10;
  --panel:        #0d1117;
  --panel2:       #111620;
  --border:       #1c2129;
  --border-hi:    #30363d;
  --indigo:       #7c3aed;
  --indigo-dim:   rgba(124,58,237,0.18);
  --cyan:         #06b6d4;
  --amber:        #f59e0b;
  --green:        #34d399;
  --pink:         #f472b6;
  --text:         #e6edf3;
  --text-dim:     #8b949e;
  --text-faint:   #484f58;
  --mono:         "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
  --radius:       10px;
}

/* ── Base ───────────────────────────────────────────────── */
* { box-sizing: border-box; }

.gradio-container {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 90% 60% at 10% -5%,  rgba(124,58,237,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 70% 50% at 90% 105%,  rgba(6,182,212,0.05)  0%, transparent 55%)
    !important;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  min-height: 100vh;
}

footer { display: none !important; }

/* ── All blocks: remove default boxy look ───────────────── */
.block {
  background: var(--panel) !important;
  border:     1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 0 0 0 transparent !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
  padding: 0 !important;
}
.block:hover {
  border-color: var(--border-hi) !important;
}

/* Remove the double-border that Gradio adds */
.block .block { border: none !important; background: transparent !important; }

/* ── Label text ─────────────────────────────────────────── */
label > span, .label-wrap span, .block-label span {
  font-family: var(--mono) !important;
  font-size:   10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  color: var(--text-faint) !important;
}

/* ── Textarea / input ───────────────────────────────────── */
textarea, input[type="text"], input[type="number"] {
  background:   #060a10 !important;
  border:       1px solid var(--border-hi) !important;
  color:        var(--text) !important;
  border-radius: 8px !important;
  font-size:    13px !important;
  line-height:  1.6 !important;
  transition:   border-color 0.2s, box-shadow 0.2s !important;
  resize: vertical !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--indigo) !important;
  box-shadow:   0 0 0 3px rgba(124,58,237,0.12),
                0 0 18px rgba(124,58,237,0.18) !important;
  outline: none !important;
}
textarea::placeholder { color: var(--text-faint) !important; }

/* ── Primary button ─────────────────────────────────────── */
button.primary, button[variant="primary"] {
  background:   linear-gradient(135deg, #6d28d9, var(--indigo)) !important;
  border:       1px solid rgba(124,58,237,0.45) !important;
  border-radius: 8px !important;
  color:        #fff !important;
  font-family:  var(--mono) !important;
  font-size:    12px !important;
  font-weight:  700 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  padding:      10px 22px !important;
  transition:   all 0.2s ease !important;
  box-shadow:   0 2px 10px rgba(124,58,237,0.22) !important;
  cursor: pointer !important;
}
button.primary:hover {
  background:   linear-gradient(135deg, var(--indigo), #8b5cf6) !important;
  box-shadow:   0 4px 20px rgba(124,58,237,0.45),
                0 0 0 1px rgba(124,58,237,0.35) !important;
  transform:    translateY(-1px) !important;
}
button.primary:active { transform: translateY(0) !important; }

/* Secondary buttons */
button.secondary {
  background:  var(--panel2) !important;
  border:      1px solid var(--border-hi) !important;
  color:       var(--text-dim) !important;
  border-radius: 8px !important;
  transition:  all 0.2s !important;
}
button.secondary:hover {
  border-color: var(--indigo) !important;
  color: var(--text) !important;
}

/* ── Tabs: underline style (Linear / GitHub / Vercel) ───── */
.tabs > .tab-nav,
div[role="tablist"] {
  background:   transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
  padding: 0 2px !important;
}
.tab-nav button, div[role="tab"] {
  background:    transparent !important;
  border:        none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color:         var(--text-dim) !important;
  font-family:   var(--mono) !important;
  font-size:     11px !important;
  font-weight:   600 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  padding:       10px 18px 9px !important;
  margin-bottom: -1px !important;
  transition:    color 0.18s, border-color 0.18s !important;
  box-shadow: none !important;
}
.tab-nav button:hover {
  color:         var(--text) !important;
  border-bottom-color: rgba(124,58,237,0.4) !important;
}
.tab-nav button.selected {
  color:        var(--indigo) !important;
  border-bottom: 2px solid var(--indigo) !important;
  background:   transparent !important;
  box-shadow:   none !important;
}

/* ── Plotly / charts: transparent background ────────────── */
.plot-container, .plot-container > div, .js-plotly-plot {
  background: transparent !important;
}

/* ── Model3D container ──────────────────────────────────── */
div[data-testid="model3d"], .model3D-component {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 0 40px rgba(124,58,237,0.08) inset !important;
}

/* ── Scrollbars ─────────────────────────────────────────── */
::-webkit-scrollbar              { width: 5px; height: 5px; }
::-webkit-scrollbar-track        { background: var(--bg); }
::-webkit-scrollbar-thumb        { background: var(--border-hi); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover  { background: var(--text-faint); }

/* ── Animated LIVE badge ─────────────────────────────────── */
@keyframes pulse-dot {
  0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--cyan); }
  50%       { opacity: 0.5; box-shadow: 0 0 2px var(--cyan); }
}
.live-dot { animation: pulse-dot 1.8s ease-in-out infinite; }

/* ── Fade-in on load ─────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.gradio-container > .main > .wrap { animation: fadeUp 0.45s ease; }

/* ── Row/col gaps ────────────────────────────────────────── */
.gap { gap: 14px !important; }
.row { gap: 14px !important; }
"""

# ── Layout ────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="One for All") as demo:

    gr.HTML(_html.header_html())
    probe_state = gr.State([])

    with gr.Tabs():

        # ── Tab 1: Almas ──────────────────────────────────────────────────
        with gr.TabItem("Souls"):
            with gr.Row():
                with gr.Column(scale=6):
                    umap_plot = gr.Model3D(
                        value=_INIT_GLB,
                        display_mode="solid",
                        clear_color=[0.031, 0.043, 0.063, 1.0],
                        height=500,
                        label=None,
                        camera_position=_CAM,
                    )
                    gr.HTML(_glb.build_legend_html(VIZ))
                with gr.Column(scale=4):
                    gr.HTML(
                        '<div style="display:flex;align-items:center;gap:8px;'
                        'font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:8px;">'
                        '<span style="color:#06b6d4;">⚡</span>Probe the student'
                        '<span style="font-family:monospace;font-size:10px;color:#06b6d4;'
                        'border:1px solid rgba(6,182,212,0.4);border-radius:4px;padding:2px 7px;">LIVE</span>'
                        '</div>'
                    )
                    prompt_box = gr.Textbox(
                        lines=4,
                        placeholder="Ask anything — code, math, language…",
                        label="",
                    )
                    run_btn = gr.Button("Run", variant="primary")
                    resp_out = gr.HTML()
                    gate_out = gr.HTML()
                    task_out = gr.HTML()
                    gr.HTML(
                        '<div style="font-size:11px;color:#8b949e;margin-top:8px;'
                        'font-family:monospace;">↑ new probe point will appear in soul space</div>'
                    )

        # ── Tab 2: Geometria ──────────────────────────────────────────────
        with gr.TabItem("Geometry"):
            with gr.Row():
                with gr.Column(scale=7):
                    gr.Plot(
                        value=_fig.build_cka_fig(VIZ["cka"]),
                        label="CKA geometry alignment",
                    )
                with gr.Column(scale=3):
                    cka_matrix = VIZ["cka"].get("matrix", [])
                    if cka_matrix:
                        import numpy as _np
                        mat = _np.array(cka_matrix)
                        n   = mat.shape[0]
                        mask = ~_np.eye(n, dtype=bool)
                        mean_off = float(mat[mask].mean())
                        masked = mat.copy()
                        _np.fill_diagonal(masked, 1.0)
                        min_idx  = _np.unravel_index(masked.argmin(), masked.shape)
                        hard_pair = (VIZ["cka"]["models"][min_idx[0]],
                                     VIZ["cka"]["models"][min_idx[1]])
                        hard_val = float(masked[min_idx])
                        gr.HTML(
                            f'<div style="background:#161b22;border:1px solid #30363d;'
                            f'border-radius:6px;padding:18px;margin-top:8px;">'
                            f'<div style="font-size:28px;font-family:monospace;color:#06b6d4;'
                            f'font-weight:700;">{mean_off:.3f}</div>'
                            f'<div style="font-size:11px;color:#8b949e;margin-top:4px;">'
                            f'mean off-diagonal CKA</div>'
                            f'<div style="margin-top:16px;font-size:11px;color:#8b949e;">hardest pair</div>'
                            f'<div style="font-family:monospace;font-size:12px;color:#f59e0b;margin-top:4px;">'
                            f'{hard_pair[0]} ↔ {hard_pair[1]}'
                            f'  <span style="color:#8b949e;">{hard_val:.2f}</span></div>'
                            f'</div>'
                        )

        # ── Tab 3: Treino ─────────────────────────────────────────────────
        with gr.TabItem("Training"):
            with gr.Row():
                gr.Plot(
                    value=_fig.build_curves_fig(VIZ["curves"]),
                    label="Loss curves",
                )
                gr.Plot(
                    value=_fig.build_gate_area_fig(VIZ["curves"]),
                    label="Gate evolution",
                )

    # ── Event wiring ──────────────────────────────────────────────────────
    run_btn.click(
        probe_fn,
        inputs=[prompt_box, probe_state],
        outputs=[umap_plot, probe_state, resp_out, gate_out, task_out],
    )


if __name__ == "__main__":
    demo.launch()
