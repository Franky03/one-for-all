"""
ofa/modal_app.py
----------------
Modal entrypoint. Training is Modal-first: this file defines the GPU image,
mounts a volume for the student weights, and runs the One for All loop there.

The student is saved to the Modal volume (cfg.output_dir). Pushing to the
HuggingFace Hub is a SEPARATE, manual step (see push_to_hub.py) so weights are
never auto-published.

Run:
    modal run ofa/modal_app.py --steps 5000
"""

from __future__ import annotations

import modal

# ----------------------------------------------------------------- image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        # Pin <5: gemma-2/Phi-3.5/MiniCPM community remote-code is written for the
        # 4.x API (e.g. MiniCPM imports transformers.utils.is_torch_fx_available,
        # removed in 5.0). 4.44+ has gemma-2/Phi-3.5/Qwen2.5 support.
        "transformers>=4.44,<5",
        "peft>=0.11",
        "accelerate>=0.30",
        "bitsandbytes>=0.43",
        "datasets>=2.19",
        "umap-learn>=0.5",
        "numpy",
        "lm_eval>=0.4",
    )
    # Ship the local `ofa` package into the remote container. Without this the
    # GPU job cannot `import ofa.*` (automount is not guaranteed in modal>=0.64).
    .add_local_python_source("ofa")
)

app = modal.App("one-for-all", image=image)

# Persistent volume for student weights + telemetry (survives between runs).
volume = modal.Volume.from_name("ofa-student", create_if_missing=True)
VOL_PATH = "/vol"


@app.function(
    gpu="A100-80GB",
    volumes={VOL_PATH: volume},
    timeout=60 * 60 * 6,
    # HF token for gated (gemma-2) / download auth. Create once with:
    #   modal secret create huggingface HF_TOKEN=hf_xxx   (or HUGGING_FACE_HUB_TOKEN)
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_ofa(steps: int = 5000, batch_size: int = 8, max_seq_len: int = 512,
              warmup_steps: int = 1000, geo_steps: int = 500):
    """
    Run the full One for All training loop on GPU and save to the volume.

    Defaults are memory-conservative (batch 8, seq 512) for a safe first run on a
    single A100. For a CHEAP end-to-end smoke that actually exercises the teacher
    forwards + geometry loss (which only start after warmup), pass a tiny warmup:
        modal run ofa/modal_app.py --steps 6 --warmup-steps 2 --batch-size 4
    """
    import json
    import os
    import torch
    import transformers

    # Confirm the image actually picked up the <5 pin (the is_torch_fx_available
    # ImportError from MiniCPM only happens on transformers 5.x).
    print(f"[ofa] transformers=={transformers.__version__} torch=={torch.__version__}")

    from ofa.config import OFAConfig, LossSchedule
    from ofa.models.loader import ModelBundle
    from ofa.distill.gating import GatingNetwork, TeacherProjections
    from ofa.distill.trainer import OFATrainer
    from ofa.data.dataloader import make_dataloader  # built separately

    cfg = OFAConfig(
        train_steps=steps,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        schedule=LossSchedule(warmup_steps=warmup_steps, geo_steps=geo_steps),
        output_dir=f"{VOL_PATH}/ofa_student",
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. load models
    bundle = ModelBundle(cfg)
    bundle.load_teachers()
    bundle.load_student()

    # 2. trainable modules
    gating = GatingNetwork(cfg.student.hidden_dim, cfg.n_teachers()).cuda()
    projections = TeacherProjections(cfg).cuda()

    # 3. trainer + data
    trainer = OFATrainer(cfg, bundle, gating, projections)
    loader = make_dataloader(cfg, bundle.student_tokenizer)

    # 4. train with periodic checkpoint + telemetry flush
    def log_fn(state):
        if state.step % 50 == 0:
            print(f"[step {state.step}] {state.components}")
        if state.step % 500 == 0 and state.step > 0:
            _save(cfg, bundle, gating, projections, trainer, tag=f"step{state.step}")
            volume.commit()

    history = trainer.train(loader, log_fn=log_fn)

    # 5. final save
    _save(cfg, bundle, gating, projections, trainer, tag="final")
    with open(f"{cfg.output_dir}/history.json", "w") as f:
        json.dump(history.states, f)
    with open(f"{cfg.output_dir}/config.json", "w") as f:
        f.write(cfg.to_json())
    volume.commit()
    print(f"Training complete. Student saved to {cfg.output_dir}")


def _save(cfg, bundle, gating, projections, trainer, tag: str):
    """Persist student LoRA + gating + projections to the volume."""
    import os
    import torch
    out = os.path.join(cfg.output_dir, tag)
    os.makedirs(out, exist_ok=True)
    bundle.student.save_pretrained(out)                       # LoRA adapter
    bundle.student_tokenizer.save_pretrained(out)
    torch.save(gating.state_dict(), os.path.join(out, "gating.pt"))
    torch.save(projections.state_dict(), os.path.join(out, "projections.pt"))


# ----------------------------------------------------------------- viz export
def _probe_texts() -> list[str]:
    """Fixed, diverse probe set so the CKA/UMAP picture is reproducible."""
    return [
        "Write a Python function that returns the nth Fibonacci number.",
        "Explain how a hash map achieves average O(1) lookups.",
        "Debug: why does `for i in range(len(a)): a.pop(i)` skip elements?",
        "Implement binary search over a sorted list of integers.",
        "Prove that the square root of 2 is irrational.",
        "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
        "Differentiate f(x) = x^3 * ln(x) with respect to x.",
        "Solve for x: 3x + 7 = 2x - 5.",
        "Explain step by step why the sky appears blue.",
        "What are the trade-offs between TCP and UDP?",
        "Summarize the causes of the French Revolution.",
        "Who wrote 'One Hundred Years of Solitude' and in what year?",
        "What is the capital of Australia?",
        "Describe photosynthesis in two sentences.",
        "Translate to Portuguese: 'The weather is nice today.'",
        "Traduce al español: 'Knowledge is power.'",
        "Traduisez en français: 'Where is the train station?'",
        "Write a haiku about the ocean at dawn.",
        "Give three tips for writing clear commit messages.",
        "Explain the difference between supervised and unsupervised learning.",
        "What is recursion and when should you avoid it?",
        "Outline a balanced weekly workout plan for a beginner.",
        "Compare the iterative and recursive forms of factorial.",
        "Why is premature optimization considered harmful?",
    ]


@app.function(
    gpu="A10G",
    volumes={VOL_PATH: volume},
    timeout=60 * 90,   # 6 teachers + Nemotron download + 256k-vocab forward > 30min
    secrets=[modal.Secret.from_name("huggingface")],
)
def export_viz(ckpt: str = "final", probe_size: int = 24):
    """
    Run student + 5 teachers on a probe set, masked-mean pool their last hidden
    states, and write viz_data.json to the volume:
      * embeddings: raw (P, d_student) float32 arrays — student in its own space,
                    each teacher projected into student space via trained Pi_i
      * cka   : pairwise CKA between all 6 models (cka_heatmap_data)
      * curves: loss/gate over steps, lifted from history.json

    The HuggingFace Gradio Space fits its own 3D UMAP at startup using these
    embeddings, and uses reducer.transform() to add live probe points.

    Fetch locally:
      modal volume get ofa-student ofa_student/viz_data.json viz_data.json
    """
    import json
    import os
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from ofa.config import OFAConfig
    from ofa.models.loader import ModelBundle
    from ofa.distill.gating import TeacherProjections
    from ofa.distill.losses import masked_mean
    from ofa.viz.embeddings import cka_heatmap_data

    out_dir = f"{VOL_PATH}/ofa_student"
    ckpt_dir = f"{out_dir}/{ckpt}"
    cfg = OFAConfig(output_dir=out_dir)
    probe = _probe_texts()[:probe_size]

    # teachers (4-bit, frozen)
    bundle = ModelBundle(cfg)
    bundle.load_teachers()

    # trained student = base + saved LoRA adapter
    s_tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    if s_tok.pad_token is None:
        s_tok.pad_token = s_tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg.student.model_id, dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True)
    student = PeftModel.from_pretrained(base, ckpt_dir)
    student.eval()

    # trained projections Pi_i (teacher dim -> student dim)
    projections = TeacherProjections(cfg)
    projections.load_state_dict(torch.load(f"{ckpt_dir}/projections.pt"))
    projections = projections.cuda().float()

    def pool(model, tok):
        enc = tok(probe, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_seq_len)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        return masked_mean(out.hidden_states[-1], enc["attention_mask"]).float().cpu()

    # pooled embeddings per model (each with its OWN tokenizer)
    model_emb = {"student": pool(student, s_tok)}
    for t in bundle.teachers:
        model_emb[t.spec.name] = pool(t.model, t.tokenizer)

    # CKA heatmap across all 6 (linear_cka handles differing feature dims)
    cka = cka_heatmap_data(model_emb)

    # Store raw embeddings projected into student space.
    # The Space fits its own 3D UMAP at startup using these.
    embeddings: dict[str, list] = {
        "student": model_emb["student"].numpy().tolist()
    }
    all_labels: list[str] = ["student"] * model_emb["student"].shape[0]
    for i, t in enumerate(bundle.teachers):
        with torch.no_grad():
            proj = projections[i](model_emb[t.spec.name].cuda().float()).cpu()
        embeddings[t.spec.name] = proj.numpy().tolist()
        all_labels += [t.spec.name] * proj.shape[0]

    # training curves from history.json
    curves = {}
    hist_path = f"{out_dir}/history.json"
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            states = json.load(f)
        curves = {
            "steps": [st["step"] for st in states],
            "task":  [st["task"] for st in states],
            "kd":    [st.get("kd", 0.0) for st in states],
            "geo":   [st["geo"] for st in states],
            "total": [st["total"] for st in states],
            "gate":  [st["gate"] for st in states],
            "teacher_names": [t.name for t in cfg.teachers],
        }

    viz = {
        "embeddings": embeddings,
        "labels": all_labels,
        "curves": curves,
        "cka": cka,
    }
    with open(f"{out_dir}/viz_data.json", "w") as f:
        json.dump(viz, f)
    volume.commit()
    print(f"wrote {out_dir}/viz_data.json "
          f"(embeddings: {len(embeddings)} models × {len(all_labels)} pts, "
          f"cka {len(cka['models'])}×{len(cka['models'])}, "
          f"curves {len(curves.get('steps', []))} steps)")


# ----------------------------------------------------------------- gguf export
# Separate CPU image: llama.cpp's HF→GGUF converter + its python deps.
# No cmake build needed — convert_hf_to_gguf.py emits q8_0 directly.
gguf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44,<5",
        "peft>=0.11",
        "accelerate>=0.30",
        "huggingface_hub>=0.22",
        "numpy",
        "sentencepiece",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "pip install -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
    )
    .add_local_python_source("ofa")
)


@app.function(
    image=gguf_image,
    volumes={VOL_PATH: volume},
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
)
def export_gguf(ckpt: str = "final", push: bool = False,
                repo: str = "build-small-hackathon/deku-gguf"):
    """
    Merge the LoRA adapter into the base student, convert to GGUF (f16 + q8_0)
    and export the gating network as numpy — everything the Space's llama.cpp
    backend (OFA_BACKEND=llamacpp) needs. CPU-only, ~minutes for a 0.5B.

    Saved to the volume under ofa_student/gguf/. Pushing stays opt-in:

        modal run ofa/modal_app.py::export_gguf
        modal run ofa/modal_app.py::export_gguf --push   # → deku-gguf repo
    """
    import os
    import subprocess
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from ofa.config import OFAConfig

    cfg = OFAConfig()
    out_dir = f"{VOL_PATH}/ofa_student"
    ckpt_dir = f"{out_dir}/{ckpt}"
    gguf_dir = f"{out_dir}/gguf"
    os.makedirs(gguf_dir, exist_ok=True)

    # 1. merge LoRA into the base (0.5B — CPU is fine)
    merged_dir = "/tmp/deku-merged"
    base = AutoModelForCausalLM.from_pretrained(
        cfg.student.model_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, ckpt_dir).merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(ckpt_dir).save_pretrained(merged_dir)
    print(f"[gguf] merged LoRA → {merged_dir}")

    # 2. convert: f16 (archival) + q8_0 (what the Space serves)
    for outtype in ("f16", "q8_0"):
        outfile = f"{gguf_dir}/deku-{outtype}.gguf"
        subprocess.run(
            ["python", "/llama.cpp/convert_hf_to_gguf.py", merged_dir,
             "--outfile", outfile, "--outtype", outtype],
            check=True,
        )
        print(f"[gguf] wrote {outfile} ({os.path.getsize(outfile) / 1e6:.0f} MB)")

    # 3. gating → numpy so the llama.cpp backend gates without torch
    sd = torch.load(f"{ckpt_dir}/gating.pt", map_location="cpu")
    np.savez(f"{gguf_dir}/gating.npz",
             weight=sd["fc.weight"].float().numpy(),
             bias=sd["fc.bias"].float().numpy())
    print(f"[gguf] wrote gating.npz (n_teachers={sd['fc.weight'].shape[0]})")
    volume.commit()

    if push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        api.upload_folder(folder_path=gguf_dir, repo_id=repo,
                          commit_message=f"GGUF export from {ckpt}")
        print(f"[gguf] pushed → https://huggingface.co/{repo}")


# ----------------------------------------------------------------- gguf validate
# Mirrors space/_probe.py's llama.cpp path so we can prove the runtime (embed
# dim, gate, streaming) against the published GGUF before flipping the Space.
gguf_run_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu",
    )
    .pip_install("huggingface_hub>=0.22", "numpy")
)


@app.function(image=gguf_run_image, timeout=60 * 15,
              secrets=[modal.Secret.from_name("huggingface")])
def validate_gguf(repo: str = "build-small-hackathon/deku-gguf"):
    import numpy as np
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    gguf = hf_hub_download(repo, "deku-q8_0.gguf")
    npz = np.load(hf_hub_download(repo, "gating.npz"))
    W, b = npz["weight"].astype(np.float32), npz["bias"].astype(np.float32)
    print(f"[validate] gate weight {W.shape} bias {b.shape}")

    # embedding instance → gate
    emb = Llama(model_path=gguf, embedding=True, n_ctx=2048, verbose=False)
    raw = np.asarray(emb.embed("Name the capital of Japan.", normalize=False),
                     dtype=np.float32)
    print(f"[validate] raw embed shape {raw.shape}")
    e = raw.mean(axis=0) if raw.ndim == 2 else raw      # pool if per-token
    print(f"[validate] pooled embed dim {e.shape} (gate expects {W.shape[1]})")
    assert e.shape[0] == W.shape[1], "embed dim != gate input dim"
    z = W @ e + b
    g = np.exp(z - z.max()); g = g / g.sum()
    print(f"[validate] gate = {[round(float(x), 3) for x in g]} sum={g.sum():.3f}")

    # generation instance → streaming
    gen = Llama(model_path=gguf, n_ctx=2048, verbose=False)
    pieces = []
    for chunk in gen.create_chat_completion(
        messages=[{"role": "user", "content": "Name the capital of Japan."}],
        max_tokens=40, temperature=0.0, stream=True,
    ):
        d = chunk["choices"][0]["delta"].get("content")
        if d:
            pieces.append(d)
    print(f"[validate] generation: {''.join(pieces)!r}")
    print("[validate] OK ✓")


# ----------------------------------------------------------------- benchmark
@app.function(
    gpu="A10G",
    volumes={VOL_PATH: volume},
    timeout=60 * 90,
    secrets=[modal.Secret.from_name("huggingface")],
)
def benchmark(
    tasks: str = "gsm8k,arc_challenge,hellaswag",
    batch_size: int = 8,
    deku_repo: str = "build-small-hackathon/deku",
):
    """
    Run lm-evaluation-harness on:
      1. Qwen2.5-0.5B-Instruct base (no adapter)
      2. Deku (base + LoRA adapter from HF Hub)

    Results are printed as a comparison table and saved to the volume.

    Usage:
        modal run ofa/modal_app.py::benchmark
        modal run ofa/modal_app.py::benchmark --tasks gsm8k,mmlu --batch-size 4
    """
    import json, os
    from lm_eval import simple_evaluate

    results_dir = f"{VOL_PATH}/ofa_student/benchmarks"
    os.makedirs(results_dir, exist_ok=True)

    BASE = "Qwen/Qwen2.5-0.5B-Instruct"
    task_list = [t.strip() for t in tasks.split(",")]

    runs = [
        ("baseline", f"pretrained={BASE}"),
        ("deku",     f"pretrained={BASE},peft={deku_repo}"),
    ]

    all_results = {}
    for name, model_args in runs:
        print(f"\n{'='*60}")
        print(f"[benchmark] {name}  model_args={model_args}")
        print(f"{'='*60}")
        r = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=task_list,
            device="cuda",
            batch_size=batch_size,
        )
        all_results[name] = r["results"]
        out = f"{results_dir}/{name}.json"
        with open(out, "w") as f:
            json.dump(r["results"], f, indent=2)
        print(f"[benchmark] saved → {out}")

    # comparison table
    print(f"\n{'task':<22} {'baseline':>10} {'deku':>10} {'delta':>10}")
    print("-" * 56)
    for task in task_list:
        for metric in ("acc_norm,none", "acc,none", "exact_match,none"):
            b = all_results.get("baseline", {}).get(task, {}).get(metric)
            d = all_results.get("deku", {}).get(task, {}).get(metric)
            if b is not None and d is not None:
                delta = d - b
                sign = "+" if delta >= 0 else ""
                print(f"{task:<22} {b:>10.4f} {d:>10.4f} {sign}{delta:>9.4f}")
                break

    volume.commit()


@app.local_entrypoint()
def main(steps: int = 5000, batch_size: int = 8, max_seq_len: int = 512,
         warmup_steps: int = 1000, geo_steps: int = 500):
    train_ofa.remote(steps=steps, batch_size=batch_size, max_seq_len=max_seq_len,
                     warmup_steps=warmup_steps, geo_steps=geo_steps)
