"""
space/_boot.py — shared startup + backend dispatch for both Space entrypoints.

app.py (Gradio Blocks, legacy) and server_app.py (gr.Server + custom frontend)
load the exact same runtime: viz data, UMAP reducer, and the student in either
backend (torch / llamacpp). Keeping it here means the two UIs can't drift.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import _data
import _probe


@dataclass
class Runtime:
    hf_token: str | None = None
    backend: str = "torch"
    viz: dict = field(default_factory=dict)
    reducer: Any = None
    coords3d: Any = None
    tok: Any = None
    student: Any = None
    gating: Any = None
    lcs: Any = None
    model_ready: bool = False
    load_error: str | None = None


def load_runtime() -> Runtime:
    rt = Runtime(hf_token=os.environ.get("HF_TOKEN"), backend=_probe.BACKEND)

    local_viz = os.environ.get("VIZ_DATA_PATH")
    try:
        if local_viz:
            rt.viz = _data.load_from_path(local_viz)
            print(f"[ofa-space] loaded viz from {local_viz}")
        else:
            rt.viz = _data.load_and_parse(rt.hf_token)
    except Exception as e:
        print(f"[ofa-space] viz_data.json not available ({e}), using empty state")
        rt.viz = _data.make_empty_viz()

    try:
        if rt.viz["stacked"].shape[0] > 3:
            rt.reducer = _data.fit_umap3d(rt.viz["stacked"])
            rt.coords3d = rt.reducer.embedding_
            print(f"[ofa-space] UMAP done: {rt.coords3d.shape}")
        else:
            print(f"[ofa-space] not enough points for UMAP: {rt.viz['stacked'].shape[0]}")
    except Exception as e:
        print(f"[ofa-space] UMAP failed ({e}), 3D disabled")
        rt.reducer = rt.coords3d = None

    try:
        if rt.backend == "llamacpp":
            # llama.cpp is pure CPU C++ — safe to load at startup, no CUDA touch.
            rt.lcs = _probe.load_student_llamacpp(rt.hf_token)
            print(f"[ofa-space] llama.cpp backend ready ({_probe.GGUF_FILE})")
        else:
            # torch + ZeroGPU: load on CPU at startup so the weights live in the
            # main process and fork into each @spaces.GPU call (copy-on-write).
            # The `spaces` lib patches torch.cuda.is_available()→True at startup,
            # which makes peft/transformers attempt .cuda() outside a GPU context
            # and crash — so force it False just for the load, then restore.
            import torch
            _orig = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
            try:
                rt.tok, rt.student, rt.gating = _probe.load_student(rt.hf_token)
            finally:
                torch.cuda.is_available = _orig
            print("[ofa-space] torch student loaded on CPU (moves to GPU per call)")
        rt.model_ready = True
    except Exception as e:
        print(f"[ofa-space] Student not available ({e}). Probe disabled.")
        rt.load_error = str(e)
        rt.model_ready = False

    return rt


# ── Backend dispatch: same call sites drive torch and llama.cpp ────────────

def to_device(rt: Runtime) -> None:
    """Move the torch student AND the gating head onto the GPU. Called inside
    @spaces.GPU, where ZeroGPU has attached a real CUDA device. Both must share
    a device — gating runs on the student's hidden state."""
    if rt.backend == "llamacpp" or rt.student is None:
        return
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    rt.student.to(device)
    if rt.gating is not None:
        rt.gating.to(device)


def stream(rt: Runtime, text: str):
    if rt.backend == "llamacpp":
        return _probe.stream_generate_llamacpp(text, rt.lcs)
    return _probe.stream_generate(text, rt.student, rt.tok, rt.gating)


def stream_pair(rt: Runtime, text: str):
    if rt.backend == "llamacpp":
        return _probe.stream_pair_llamacpp(text, rt.lcs)
    return _probe.stream_pair(text, rt.student, rt.tok, rt.gating)


def final_probe(rt: Runtime, text: str):
    if rt.backend == "llamacpp":
        return _probe.run_probe_llamacpp(text, rt.lcs, rt.reducer)
    return _probe.run_probe(text, rt.student, rt.tok, rt.gating, rt.reducer)


# ── Payload for the custom frontend (server_app.py) ────────────────────────

# Frontend palette — student first, then teachers (qwen, smollm, phi, gemma,
# minicpm, nemotron). No purple by design; the legacy Blocks UI keeps its own
# palette in _fig.py.
MODEL_COLORS = ["#e6edf3", "#38bdf8", "#f59e0b", "#f43f5e", "#2dd4bf", "#f472b6", "#76b900"]

# Canonical 6-teacher lineup. viz_data.json exported from an older 5-teacher
# run is a prefix of this — pad it so the UI already shows nemotron (its meter
# stays at 0 until a 6-gate checkpoint is published).
CANONICAL_TEACHERS = ["qwen", "smollm", "phi", "gemma", "minicpm", "nemotron"]


def viz_payload(viz: dict, coords3d, backend: str = "torch",
                model_ready: bool = False) -> dict:
    """Everything the JS frontend needs in one JSON-safe dict."""
    models = []
    if coords3d is not None and viz.get("labels"):
        labels = viz["labels"]
        for i, name in enumerate(viz.get("model_names", [])):
            pts = [list(map(float, coords3d[j]))
                   for j, lab in enumerate(labels) if lab == name]
            models.append({
                "name": name,
                "color": MODEL_COLORS[i % len(MODEL_COLORS)],
                "points": pts,
            })
    teachers = viz.get("teacher_names", [])
    if teachers == CANONICAL_TEACHERS[:len(teachers)]:
        teachers = CANONICAL_TEACHERS
    return {
        "models": models,
        "teachers": teachers,
        "teacher_colors": [MODEL_COLORS[1 + i % (len(MODEL_COLORS) - 1)]
                           for i in range(len(teachers))],
        "cka": viz.get("cka", {}),
        "curves": viz.get("curves", {}),
        "backend": backend,
        "model_ready": model_ready,
    }
