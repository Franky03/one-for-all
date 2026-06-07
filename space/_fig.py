from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

DARK = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="JetBrains Mono, monospace, sans-serif"),
    margin=dict(t=48, r=16, b=48, l=56),
)
GRID_COLOR = "#21262d"

# index 0 = student, 1-5 = teachers
MODEL_COLORS = ["#e6edf3", "#7c3aed", "#06b6d4", "#f59e0b", "#34d399", "#f472b6"]


def build_base_traces(viz: dict, coords3d: np.ndarray) -> list:
    """One Scatter3d trace per model. Returns a plain list — never mutate it."""
    labels = np.array(viz["labels"])
    traces = []
    for i, name in enumerate(viz["model_names"]):
        c = coords3d[labels == name]
        traces.append(go.Scatter3d(
            x=c[:, 0].tolist(), y=c[:, 1].tolist(), z=c[:, 2].tolist(),
            mode="markers", name=name,
            marker=dict(
                color=MODEL_COLORS[i % len(MODEL_COLORS)],
                size=5 if name != "student" else 6,
                opacity=0.85,
            ),
        ))
    return traces


def rebuild_fig(base_traces: list, probe_points: list[dict]) -> go.Figure:
    """Fresh Figure from immutable base traces + accumulated probe points.

    probe_points: list of {"x": float, "y": float, "z": float, "label": str}
    Never mutates base_traces.
    """
    fig = go.Figure(data=list(base_traces))  # copy, not reference
    if probe_points:
        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in probe_points],
            y=[p["y"] for p in probe_points],
            z=[p["z"] for p in probe_points],
            mode="markers", name="live probe",
            marker=dict(color="#ffffff", size=9, symbol="diamond", opacity=1.0,
                        line=dict(color="#7c3aed", width=1)),
        ))
    fig.update_layout(
        **DARK,
        title=dict(text="Soul space — UMAP 3D", font=dict(size=13)),
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, showticklabels=False, title=""),
            yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, showticklabels=False, title=""),
            zaxis=dict(showgrid=True, gridcolor=GRID_COLOR, showticklabels=False, title=""),
        ),
        legend=dict(bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d", borderwidth=1,
                    font=dict(size=11)),
        uirevision="soul",  # keeps camera position between updates
    )
    return fig


def build_cka_fig(cka: dict) -> go.Figure:
    if not cka or "matrix" not in cka:
        return go.Figure(layout={**DARK, "title": "No CKA data"})
    fig = go.Figure(go.Heatmap(
        z=cka["matrix"], x=cka["models"], y=cka["models"],
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar=dict(title="CKA", thickness=14, tickfont=dict(color="#e6edf3", size=11)),
        text=[[f"{v:.2f}" for v in row] for row in cka["matrix"]],
        texttemplate="%{text}", textfont=dict(size=11),
    ))
    fig.update_layout(
        **DARK,
        title=dict(text="CKA geometry alignment — all pairs", font=dict(size=13)),
        xaxis=dict(side="bottom", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    return fig


def _ema(vals: list[float], alpha: float = 0.9) -> list[float]:
    out, s = [], vals[0]
    for v in vals:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def build_curves_fig(curves: dict) -> go.Figure:
    if not curves or not curves.get("steps"):
        return go.Figure(layout={**DARK, "title": "No training data"})
    steps = curves["steps"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=_ema(curves["task"]), name="task",
                             line=dict(color="#06b6d4", width=2)))
    if curves.get("kd"):
        fig.add_trace(go.Scatter(x=steps, y=_ema(curves["kd"]), name="kd (qwen)",
                                 line=dict(color="#34d399", width=2)))
    fig.add_trace(go.Scatter(x=steps, y=_ema(curves["geo"]), name="geo",
                             line=dict(color="#7c3aed", width=2)))
    fig.add_trace(go.Scatter(x=steps, y=_ema(curves["total"]), name="total",
                             line=dict(color="#f59e0b", width=1.5, dash="dot")))
    fig.update_layout(
        **DARK,
        title=dict(text="Loss curves (EMA α=0.9)", font=dict(size=13)),
        xaxis=dict(title="step", gridcolor=GRID_COLOR),
        yaxis=dict(title="loss", gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1),
    )
    return fig


def build_gate_area_fig(curves: dict) -> go.Figure:
    if not curves or not curves.get("gate"):
        return go.Figure(layout={**DARK, "title": "No gate data"})
    steps = curves["steps"]
    names = curves["teacher_names"]
    teacher_colors = MODEL_COLORS[1:]
    fig = go.Figure()
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=steps,
            y=[g[i] for g in curves["gate"]],
            name=name, mode="lines", stackgroup="g",
            line=dict(color=teacher_colors[i % len(teacher_colors)], width=0.5),
        ))
    fig.update_layout(
        **DARK,
        title=dict(text="Gate — teacher routing over steps", font=dict(size=13)),
        yaxis=dict(title="weight", range=[0, 1], gridcolor=GRID_COLOR),
        xaxis=dict(title="step", gridcolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1),
    )
    return fig
