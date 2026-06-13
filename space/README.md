---
title: One for All — Soul Transfer
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.18.0"
app_file: server_app.py
pinned: false
license: mit
---

![One for All demo](one_for_all.gif)

**One for All** distills 6 heterogeneous teacher LLMs (Qwen2.5-1.5B, SmolLM2-1.7B, Phi-3.5-mini, gemma-2-2b-it, MiniCPM-2B, Nemotron-Mini-4B) into a single Qwen2.5-0.5B student via gated CKA geometry distillation (Path B — geometry-only, tokenizer-agnostic).

### Tabs

- **∀ Souls** — 3D UMAP soul space of all models over 24 probe texts. Type a prompt and watch the student answer **token by token** while the gate bars show, live, which teacher's geometry is dominant. The pooled prompt lands as a new point in soul space.
- **⚔ Arena** — one prompt, two models: the untouched Qwen2.5-0.5B base races the distilled student side by side (same weights, LoRA off vs on).
- **⬡ Geometry** — CKA alignment heatmap across all model pairs.
- **↗ Training** — Loss curves and gate evolution over training steps.

### UI

`server_app.py` is a **custom frontend on `gr.Server`** 🎨 — our own HTML/JS/Three.js
(`frontend/`) talking to Gradio API endpoints via `@gradio/client`. The legacy
Blocks UI still works: set `app_file: app.py` (it also runs on sdk 6.18).

### Backends

- default — torch + ZeroGPU (per-token gates).
- `OFA_BACKEND=llamacpp` — the student runs as a **GGUF through llama.cpp** 🦙 (CPU-friendly; gates computed on the prompt embedding via numpy, no torch in the hot path).
