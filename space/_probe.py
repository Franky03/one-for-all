from __future__ import annotations
import os
import torch
import torch.nn as nn
import numpy as np

# Inlined so Space has no dependency on the ofa package.
STUDENT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_REPO = "build-small-hackathon/deku"
STUDENT_HIDDEN_DIM = 896  # Qwen2.5-0.5B hidden size

# Backend: "torch" (default) or "llamacpp" (GGUF via llama-cpp-python, CPU-friendly).
BACKEND = os.environ.get("OFA_BACKEND", "torch").lower()
GGUF_REPO = os.environ.get("OFA_GGUF_REPO", "build-small-hackathon/deku-gguf")
GGUF_FILE = os.environ.get("OFA_GGUF_FILE", "deku-q8_0.gguf")
BASE_GGUF_REPO = os.environ.get("OFA_BASE_GGUF_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
BASE_GGUF_FILE = os.environ.get("OFA_BASE_GGUF_FILE", "qwen2.5-0.5b-instruct-q8_0.gguf")
# cpu-basic Spaces give 2 vCPU, but os.cpu_count() reports the HOST core count
# (16+) — passing that to llama.cpp oversubscribes and stalls. Cap to the real
# allocation; override with OFA_THREADS if the Space is on bigger hardware.
_N_THREADS = int(os.environ.get("OFA_THREADS", "2"))


class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim: int, n_teachers: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_teachers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc(x), dim=-1)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _umap_point(reducer, vec) -> dict | None:
    """Project a (1, d) embedding into the soul space. umap-learn's transform()
    is version-fragile (sparse-index errors on some scipy combos); on failure
    return None so the caller places the probe by its dominant teacher instead."""
    try:
        c = reducer.transform(vec)
        return {"x": float(c[0, 0]), "y": float(c[0, 1]), "z": float(c[0, 2]), "label": "probe"}
    except Exception as e:
        print(f"[ofa-space] umap transform failed ({e}); probe point skipped")
        return None


def load_student(hf_token: str | None = None):
    """Load student (base + LoRA) and gating network from HF Hub.

    Returns (tokenizer, student_model, gating_network).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    token = hf_token or os.environ.get("HF_TOKEN")

    tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load on CPU — NO device_map="auto". On ZeroGPU there is no GPU at startup
    # (it attaches only inside @spaces.GPU), and device_map="auto" probes CUDA →
    # "No CUDA GPUs are available". The handler moves the model to cuda per call.
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    student = PeftModel.from_pretrained(base, ADAPTER_REPO, token=token)
    student.eval()

    gating_path = hf_hub_download(
        repo_id=ADAPTER_REPO,
        filename="gating.pt",
        repo_type="model",
        token=token,
    )
    gating = _gating_from_state(torch.load(gating_path, map_location="cpu"))

    return tok, student, gating


def _gating_from_state(state: dict) -> GatingNetwork:
    """Build the gate from a checkpoint, inferring n_teachers from the weight
    shape — keeps the Space compatible with 5- and 6-teacher runs."""
    n_teachers, hidden = state["fc.weight"].shape
    gating = GatingNetwork(hidden, n_teachers)
    gating.load_state_dict(state)
    gating.eval()
    return gating


def _chat_prompt(text: str, tok) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return text


def generate_response(
    text: str,
    student: nn.Module,
    tok,
    max_new_tokens: int = 200,
) -> str:
    """Run student generation and return decoded answer text."""
    device = next(student.parameters()).device
    enc = tok(_chat_prompt(text, tok), return_tensors="pt").to(device)
    with torch.no_grad():
        out = student.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def _decode_step(student: nn.Module, ids: torch.Tensor, past):
    """One greedy decode step. Returns (next_id, new_past, last_hidden)."""
    with torch.no_grad():
        out = student(
            input_ids=ids,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    h_last = out.hidden_states[-1][:, -1, :].float()
    return next_id, out.past_key_values, h_last


def _adapter_off(student: nn.Module):
    """disable_adapter() ctx on PeftModel; no-op for plain models (tests)."""
    disable = getattr(student, "disable_adapter", None)
    if disable is not None:
        return disable()
    import contextlib
    return contextlib.nullcontext()


def stream_generate(
    text: str,
    student: nn.Module,
    tok,
    gating: GatingNetwork,
    max_new_tokens: int = 768,   # ~30s on ZeroGPU A10G — safe under the 60s cap
    ema: float = 0.3,
):
    """Greedy decode loop that exposes the gate at every step.

    Yields (partial_text, gate_weights) per generated token. Gate weights are
    EMA-smoothed so the live bars move instead of flickering. model.generate()
    can't do this — it doesn't surface hidden states mid-stream.
    """
    device = next(student.parameters()).device
    enc = tok(_chat_prompt(text, tok), return_tensors="pt")
    ids = enc["input_ids"].to(device)
    past = None
    gate_smooth: torch.Tensor | None = None
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        next_id, past, h_last = _decode_step(student, ids, past)
        with torch.no_grad():
            g = gating(h_last).squeeze(0)
        gate_smooth = g if gate_smooth is None else ema * g + (1 - ema) * gate_smooth
        if next_id.item() == tok.eos_token_id:
            break
        out_ids.append(next_id.item())
        ids = next_id
        yield (
            tok.decode(out_ids, skip_special_tokens=True),
            gate_smooth.tolist(),
        )


def stream_pair(
    text: str,
    student: nn.Module,
    tok,
    gating: GatingNetwork,
    max_new_tokens: int = 384,
    ema: float = 0.3,
):
    """Race base (LoRA off) vs deku (LoRA on) — one token each per round.

    Same PeftModel serves both sides via disable_adapter(), so no extra
    memory. Yields (base_text, deku_text, gate_weights); gates come from the
    deku side only.
    """
    device = next(student.parameters()).device
    enc = tok(_chat_prompt(text, tok), return_tensors="pt")
    prompt_ids = enc["input_ids"].to(device)

    sides = {
        "base": {"ids": prompt_ids, "past": None, "out": [], "done": False},
        "deku": {"ids": prompt_ids, "past": None, "out": [], "done": False},
    }
    gate_smooth: torch.Tensor | None = None

    for _ in range(max_new_tokens):
        if not sides["base"]["done"]:
            s = sides["base"]
            with _adapter_off(student):
                next_id, s["past"], _ = _decode_step(student, s["ids"], s["past"])
            if next_id.item() == tok.eos_token_id:
                s["done"] = True
            else:
                s["out"].append(next_id.item())
                s["ids"] = next_id

        if not sides["deku"]["done"]:
            s = sides["deku"]
            next_id, s["past"], h_last = _decode_step(student, s["ids"], s["past"])
            with torch.no_grad():
                g = gating(h_last).squeeze(0)
            gate_smooth = g if gate_smooth is None else ema * g + (1 - ema) * gate_smooth
            if next_id.item() == tok.eos_token_id:
                s["done"] = True
            else:
                s["out"].append(next_id.item())
                s["ids"] = next_id

        if sides["base"]["done"] and sides["deku"]["done"]:
            break
        yield (
            tok.decode(sides["base"]["out"], skip_special_tokens=True),
            tok.decode(sides["deku"]["out"], skip_special_tokens=True),
            gate_smooth.tolist() if gate_smooth is not None
            else [1.0 / gating.fc.out_features] * gating.fc.out_features,
        )


def run_probe(
    text: str,
    student: nn.Module,
    tok,
    gating: GatingNetwork,
    reducer,
    max_len: int = 512,
) -> tuple[dict, list[float]]:
    """Single student forward. Returns (new_umap_point, gate_weights).

    new_umap_point : {"x": float, "y": float, "z": float, "label": str}
    gate_weights   : list of N_TEACHERS floats summing to 1.0
    """
    enc = tok([text], return_tensors="pt", truncation=True, max_length=max_len)
    device = next(student.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = student(**enc, output_hidden_states=True, use_cache=False)

    pooled = _masked_mean(out.hidden_states[-1], enc["attention_mask"]).float()
    gate_weights: list[float] = gating(pooled).squeeze(0).tolist()

    return _umap_point(reducer, pooled.cpu().numpy()), gate_weights


# ── llama.cpp backend (🦙 badge — GGUF student, runs on CPU, no torch path) ──

def _np_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def gate_from_embedding(emb: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> list[float]:
    """Gate in pure numpy: softmax(W @ e + b). Mirrors GatingNetwork.forward."""
    return _np_softmax(weight @ emb + bias).tolist()


class LlamaCppStudent:
    """deku as GGUF: one llama.cpp instance for generation, one in embedding
    mode (mean pooling) for the gate / soul-space probe, numpy gate weights.
    The base model (Arena) is downloaded lazily on first use."""

    def __init__(self, gen, emb, gate_w: np.ndarray, gate_b: np.ndarray, token=None):
        self.gen = gen
        self.emb = emb
        self.gate_w = gate_w
        self.gate_b = gate_b
        self.base = None
        self._token = token

    @property
    def n_teachers(self) -> int:
        return self.gate_w.shape[0]

    def embed(self, text: str) -> np.ndarray:
        # llama.cpp returns per-token embeddings (T, 896) for this model; mean-pool
        # to one vector to match the gate's masked-mean training input.
        raw = np.asarray(self.emb.embed(text, normalize=False), dtype=np.float32)
        return raw.mean(axis=0) if raw.ndim == 2 else raw

    def prompt_gates(self, text: str) -> list[float]:
        return gate_from_embedding(self.embed(text), self.gate_w, self.gate_b)

    def ensure_base(self) -> None:
        if self.base is None:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(BASE_GGUF_REPO, BASE_GGUF_FILE, token=self._token)
            self.base = Llama(model_path=path, n_ctx=1024, n_threads=_N_THREADS, verbose=False)


def load_student_llamacpp(hf_token: str | None = None) -> LlamaCppStudent:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    token = hf_token or os.environ.get("HF_TOKEN")
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE, token=token)
    npz = np.load(hf_hub_download(GGUF_REPO, "gating.npz", token=token))
    gen = Llama(model_path=gguf, n_ctx=1024, n_threads=_N_THREADS, verbose=False)
    emb = Llama(model_path=gguf, n_ctx=1024, n_threads=_N_THREADS, embedding=True, verbose=False)
    return LlamaCppStudent(gen, emb,
                           npz["weight"].astype(np.float32),
                           npz["bias"].astype(np.float32), token=token)


def _chat_stream(llm, text: str, max_tokens: int):
    """Yield accumulated text from a llama.cpp chat-completion stream."""
    pieces: list[str] = []
    for chunk in llm.create_chat_completion(
        messages=[{"role": "user", "content": text}],
        max_tokens=max_tokens, temperature=0.0, stream=True,
    ):
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            pieces.append(delta)
            yield "".join(pieces)


def stream_generate_llamacpp(text: str, lcs: LlamaCppStudent, max_tokens: int = 96):
    """Yields (partial_text, gate_weights). llama.cpp can't expose hidden states
    mid-generation, so no live gate here — the bars stay empty during streaming
    and the caller's final run_probe_llamacpp fills the gate + soul-space point
    in a single embedding forward (kept off the first-token path → faster TTFB)."""
    zeros = [0.0] * lcs.n_teachers
    for partial in _chat_stream(lcs.gen, text, max_tokens):
        yield partial, zeros


def stream_pair_llamacpp(text: str, lcs: LlamaCppStudent, max_tokens: int = 96):
    """Arena on llama.cpp: deku GGUF vs official base GGUF, interleaved.
    Yields (base_text, deku_text, gate_weights)."""
    import itertools
    lcs.ensure_base()
    zeros = [0.0] * lcs.n_teachers
    base_txt, deku_txt = "", ""
    for b, d in itertools.zip_longest(
        _chat_stream(lcs.base, text, max_tokens),
        _chat_stream(lcs.gen, text, max_tokens),
    ):
        base_txt = b if b is not None else base_txt
        deku_txt = d if d is not None else deku_txt
        yield base_txt, deku_txt, zeros        # gate off the hot path
    # one embedding forward at the end → the real gate
    yield base_txt, deku_txt, lcs.prompt_gates(text)


def run_probe_llamacpp(text: str, lcs: LlamaCppStudent, reducer) -> tuple[dict | None, list[float]]:
    """llama.cpp analogue of run_probe: mean-pooled embedding → gate + UMAP point."""
    pooled = lcs.embed(text)
    gates = gate_from_embedding(pooled, lcs.gate_w, lcs.gate_b)
    return _umap_point(reducer, pooled[None, :]), gates
