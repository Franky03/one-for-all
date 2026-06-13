"""
space/server_app.py — One for All, custom frontend on gr.Server.

gr.Server = FastAPI with Gradio's API engine: @app.api functions become
queued/streaming endpoints (SSE), and we serve our own HTML/JS at "/" —
the browser talks to the endpoints via @gradio/client (required for ZeroGPU).

Run locally:
    cd space && VIZ_DATA_PATH=/path/to/viz_data.json python server_app.py

The legacy Blocks UI (app.py) keeps working — both share _boot.py.
"""
from __future__ import annotations

from pathlib import Path

import spaces
from fastapi.responses import FileResponse, HTMLResponse
from gradio import Server

import _boot

RT = _boot.load_runtime()
FRONTEND = Path(__file__).parent / "frontend"

app = Server()


# ── API: static viz payload (soul space, CKA, curves, meta) ────────────────
@app.api(name="viz")
def viz() -> dict:
    return _boot.viz_payload(RT.viz, RT.coords3d, RT.backend, RT.model_ready)


# ── API: streaming probe — text + live gates, final soul-space point ───────
@app.api(name="probe", stream_every=0.1)
@spaces.GPU
def probe(text: str) -> dict:  # annotation = type of each streamed chunk
    if not text.strip():
        yield {"error": "empty prompt", "done": True}
        return
    _boot.to_device(RT)   # lazy-loads the student here (real CUDA inside @spaces.GPU)
    if not RT.model_ready:
        yield {"error": RT.load_error or "model not ready", "done": True}
        return
    partial, gates = "", []
    for partial, gates in _boot.stream(RT, text):
        yield {"text": partial, "gates": gates, "done": False}
    point = None
    if RT.reducer is not None:
        point, gates = _boot.final_probe(RT, text)
    yield {"text": partial, "gates": gates, "point": point, "done": True}


# ── API: streaming arena — base (LoRA off) vs deku, interleaved ────────────
@app.api(name="arena", stream_every=0.1)
@spaces.GPU
def arena(text: str) -> dict:  # annotation = type of each streamed chunk
    if not text.strip():
        yield {"error": "empty prompt", "done": True}
        return
    _boot.to_device(RT)   # lazy-loads the student here (real CUDA inside @spaces.GPU)
    if not RT.model_ready:
        yield {"error": RT.load_error or "model not ready", "done": True}
        return
    base = deku = ""
    gates: list[float] = []
    for base, deku, gates in _boot.stream_pair(RT, text):
        yield {"base": base, "deku": deku, "gates": gates, "done": False}
    yield {"base": base, "deku": deku, "gates": gates, "done": True}


# ── Custom frontend (overrides Gradio's default UI at "/") ────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    return (FRONTEND / "index.html").read_text()


@app.get("/static/{name}")
async def static(name: str):
    target = (FRONTEND / name).resolve()
    if target.parent != FRONTEND.resolve() or not target.exists():
        return HTMLResponse("not found", status_code=404)
    return FileResponse(target)


if __name__ == "__main__":
    app.launch()
