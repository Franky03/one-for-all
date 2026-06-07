from __future__ import annotations
import os
import torch
import torch.nn as nn
import numpy as np

# Inlined so Space has no dependency on the ofa package.
STUDENT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_REPO = "build-small-hackathon/deku"
STUDENT_HIDDEN_DIM = 896  # Qwen2.5-0.5B hidden size
N_TEACHERS = 5


class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim: int, n_teachers: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_teachers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc(x), dim=-1)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


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

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
    gating = GatingNetwork(STUDENT_HIDDEN_DIM, N_TEACHERS)
    gating.load_state_dict(torch.load(gating_path, map_location="cpu"))
    gating.eval()

    return tok, student, gating


def generate_response(
    text: str,
    student: nn.Module,
    tok,
    max_new_tokens: int = 200,
) -> str:
    """Run student generation and return decoded answer text."""
    device = next(student.parameters()).device
    try:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = text
    enc = tok(prompt, return_tensors="pt").to(device)
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

    coords3d = reducer.transform(pooled.cpu().numpy())  # (1, 3)
    return {
        "x": float(coords3d[0, 0]),
        "y": float(coords3d[0, 1]),
        "z": float(coords3d[0, 2]),
        "label": "probe",
    }, gate_weights
