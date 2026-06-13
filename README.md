# One for All — Multi-Teacher Soul Transfer

![One for All demo](static/one_for_all.gif)

Six teacher SLMs (1.5–4B — Qwen2.5, SmolLM2, Phi-3.5, gemma-2, MiniCPM,
Nemotron-Mini) distill into one smaller student (0.5B) that learns to absorb
each teacher's representation geometry. A gating network conditioned on the
student's own hidden state decides, online, how much of each teacher to absorb.

**Loss:** `L = λ1·L_task + λ2·L_KL(qwen) + λ3·L_geo(gated CKA)`

**Path B — geometry-only, tokenizer-agnostic.** KL distillation uses only the
Qwen teacher (same tokenizer as student → no vocab mismatch). Geometry distillation
uses all 6 teachers via masked-mean pooling (sequence-level, tokenizer-agnostic).

---

## Setup

```bash
cd /home/kai/ofa
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft accelerate bitsandbytes datasets \
            umap-learn plotly huggingface_hub lm_eval
```

---

## Verify (no GPU, no downloads)

```bash
# pure layer — config, schedule, fingerprint
python -m ofa.smoke_test_pure

# core math — CKA, gating, composite loss + backward (torch only)
python -m ofa.smoke_test_core

# Space modules — _data, _fig, _html, _probe (24 tests)
.venv/bin/python -m pytest tests/space/test_space.py -q
```

---

## Benchmark (Modal — GPU obrigatório)

Roda no Modal (A10G). Compara baseline vs deku e imprime tabela de delta.

```bash
cd /home/kai

# baseline + deku — gsm8k, arc_challenge, hellaswag (padrão)
modal run ofa/modal_app.py::benchmark

# tarefas customizadas
modal run ofa/modal_app.py::benchmark --tasks gsm8k,mmlu --batch-size 4
```

Resultados salvos no volume em `ofa_student/benchmarks/`. Para baixar:

```bash
modal volume get ofa-student ofa_student/benchmarks ./benchmarks
```

---

## Train (Modal — A100 80GB)

```bash
cd /home/kai

# full run
modal run ofa/modal_app.py --steps 5000

# cheap smoke (exercises teacher forwards + geometry after warmup)
modal run ofa/modal_app.py --steps 6 --warmup-steps 2 --batch-size 4
```

Checkpoints saved to Modal volume `ofa-student` every 500 steps.

---

## Export GGUF (llama.cpp — após o treino)

Merge do LoRA + conversão para GGUF (f16 + q8_0) + `gating.npz` (gate em numpy).
Roda em CPU no Modal:

```bash
cd /home/kai

# salva no volume em ofa_student/gguf/
modal run ofa/modal_app.py::export_gguf

# salva e publica em build-small-hackathon/deku-gguf
modal run ofa/modal_app.py::export_gguf --push
```

Para o Space servir via llama.cpp (CPU, sem ZeroGPU): setar `OFA_BACKEND=llamacpp`
em **Settings → Variables and secrets** do Space.

---

## Export viz data (after training)

Runs student + 5 teachers on 24 probe texts, writes `viz_data.json` to volume:

```bash
modal run ofa/modal_app.py::export_viz
```

---

## Pull artifacts from Modal volume

```bash
# viz data for the Space
modal volume get ofa-student ofa_student/viz_data.json ./viz_data.json

# trained model (LoRA adapter + gating + projections + tokenizer)
mkdir -p ofa_student_local
modal volume get ofa-student ofa_student/final ofa_student_local/
```

---

## Push to HuggingFace

```bash
source /home/kai/echo/.venv/bin/activate

# 1. model → build-small-hackathon/deku
cp /home/kai/ofa/docs/deku_model_card.md ofa_student_local/final/README.md
cd /home/kai/ofa
python push_to_hub.py \
  --local-dir ../ofa_student_local/final \
  --repo build-small-hackathon/deku \
  --message "Deku v2 — KL + GSM8K/ARC dataset mix"

# 2. viz data → build-small-hackathon/ofa-viz-data (dataset)
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("build-small-hackathon/ofa-viz-data", repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj="/home/kai/viz_data.json",
    path_in_repo="viz_data.json",
    repo_id="build-small-hackathon/ofa-viz-data",
    repo_type="dataset",
    commit_message="regen viz_data",
)
EOF

# 3. Space → build-small-hackathon/one-for-all
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("build-small-hackathon/one-for-all", repo_type="space",
                space_sdk="gradio", exist_ok=True, private=False)
api.upload_folder(
    folder_path="/home/kai/ofa/space",
    repo_id="build-small-hackathon/one-for-all",
    repo_type="space",
    commit_message="update Space",
)
EOF
```

After pushing the Space: set `HF_TOKEN` in **Settings → Variables and secrets**,
then **Factory reboot** so it picks up the new viz data and model.

---

## Module map

### Pure layer (no torch)
| File | Role |
|------|------|
| `config.py` | `OFAConfig` — one dataclass fully describes a run. `dataset_mix`, `LossSchedule` (3-phase λ). |
| `metrics/fingerprint.py` | `SoulFingerprint` — per-teacher contribution α_i, task-conditional. |

### Core math (needs torch)
| File | Role |
|------|------|
| `metrics/cka.py` | Differentiable linear CKA + `cka_loss`. |
| `distill/gating.py` | `GatingNetwork` (self-referential routing) + `TeacherProjections` (Π_i). |
| `distill/losses.py` | `composite_loss` = task + `qwen_kl_loss` + gated CKA geometry. |
| `distill/trainer.py` | `OFATrainer`: forward → gate → loss → backward. Emits `TrainState` per step. |

### Models & data
| File | Role |
|------|------|
| `models/loader.py` | `ModelBundle`: 5 teachers in 4-bit (frozen), student with LoRA. |
| `data/dataloader.py` | Weighted mix of OpenHermes + GSM8K + ARC via `interleave_datasets`. |

### Space (`space/`)
| File | Role |
|------|------|
| `_data.py` | Load `viz_data.json` from HF dataset, fit 3D UMAP at startup. |
| `_fig.py` | Plotly figures: UMAP 3D, CKA heatmap, loss curves, gate area. |
| `_html.py` | Gate bars, task badge, header. |
| `_probe.py` | Student loading + probe + token streaming, torch and llama.cpp backends. |
| `_boot.py` | Shared runtime (viz + UMAP + student) and backend dispatch for both UIs. |
| `server_app.py` | **Main entrypoint** — `gr.Server` API + custom frontend (`frontend/`). |
| `frontend/` | Custom UI: Three.js soul space, streaming console, quirk meters. |
| `app.py` | Legacy Gradio Blocks UI (4 tabs incl. Arena) — kept as fallback. |
