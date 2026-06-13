"""
ofa/config.py
-------------
Pure configuration for One for All. No torch, no heavy deps — this module is
imported everywhere (including the Modal entrypoint and any future Gradio app)
to know model names, hyperparameters, and paths without loading the ML stack.

Everything that defines an experiment lives here, so a run is fully described by
a single OFAConfig instance (serializable to JSON for logging / reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class TeacherSpec:
    """One teacher in the One for All ensemble."""
    name: str                 # short id used in logs / fingerprints
    model_id: str             # HuggingFace model id
    hidden_dim: int           # hidden size (for the projection Pi_i)
    strength: str = ""        # human-readable note (code, multilingual, ...)


# The five teachers from the paper + Nemotron-Mini (NVIDIA, pruned+distilled
# from Nemotron-4 15B — itself a distillation artifact, which fits the theme).
# Note: Nemotron has a 256k vocab; its logits are never used (geometry-only
# teacher), but the lm_head forward costs ~4GB transient at batch 8 / seq 512.
DEFAULT_TEACHERS: list[TeacherSpec] = [
    TeacherSpec("qwen",     "Qwen/Qwen2.5-1.5B-Instruct", 1536, "code, structured reasoning"),
    TeacherSpec("smollm",   "HuggingFaceTB/SmolLM2-1.7B-Instruct", 2048, "curated quality"),
    TeacherSpec("phi",      "microsoft/Phi-3.5-mini-instruct", 3072, "instruction, CoT"),
    TeacherSpec("gemma",    "google/gemma-2-2b-it", 2304, "long context"),
    TeacherSpec("minicpm",  "openbmb/MiniCPM-2B-sft-bf16", 2304, "multilingual, efficiency"),
    TeacherSpec("nemotron", "nvidia/Nemotron-Mini-4B-Instruct", 3072, "roleplay, RAG, function calling"),
]


@dataclass
class StudentSpec:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hidden_dim: int = 896
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class LossSchedule:
    """Three-phase lambda curriculum (Eqs. 7-9 in the paper)."""
    warmup_steps: int = 1000      # T_warm: task-only before KD ramps in (was 500)
    geo_steps: int = 500          # T_geo: KD-only before geometry ramps in
    lambda3_max: float = 0.5      # cap on the geometry loss weight
    lambda_kd_max: float = 0.5    # cap on KD loss weight (was implicitly 1.0)
    temperature: float = 4.0      # KD softmax temperature tau
    gate_temp_start: float = 2.0  # gate softmax temperature at step 0, decays to 1.0
    gate_entropy_alpha: float = 0.05  # entropy regularization to prevent gate collapse

    def lambdas(self, step: int) -> tuple[float, float, float]:
        """Return (lambda_task, lambda_kd, lambda_geo) at a given step."""
        l1 = 1.0
        l2 = (self.lambda_kd_max * min(1.0, step / self.warmup_steps)
              if self.warmup_steps > 0 else self.lambda_kd_max)
        if step <= self.warmup_steps:
            l3 = 0.0
        else:
            prog = (step - self.warmup_steps) / max(1, self.geo_steps)
            l3 = min(self.lambda3_max, prog * self.lambda3_max)
        return l1, l2, l3

    def gate_temperature(self, step: int) -> float:
        """Linear decay from gate_temp_start → 1.0 over warmup_steps."""
        if self.gate_temp_start <= 1.0 or step >= self.warmup_steps:
            return 1.0
        progress = step / self.warmup_steps
        return self.gate_temp_start - (self.gate_temp_start - 1.0) * progress


@dataclass
class OFAConfig:
    """Full experiment description."""
    teachers: list[TeacherSpec] = field(default_factory=lambda: list(DEFAULT_TEACHERS))
    student: StudentSpec = field(default_factory=StudentSpec)
    schedule: LossSchedule = field(default_factory=LossSchedule)

    # training
    train_steps: int = 5000
    batch_size: int = 32
    lr: float = 2e-4
    grad_clip: float = 1.0
    monitored_layer_frac: float = 1 / 3   # last 1/3 of student layers for CKA

    # data — weighted mix: {dataset_id: probability}
    dataset_mix: dict = field(default_factory=lambda: {
        "teknium/OpenHermes-2.5": 0.70,
        "openai/gsm8k":           0.20,
        "allenai/ai2_arc":        0.10,
    })
    probe_dataset: str = "allenai/c4"
    probe_samples: int = 10_000
    max_seq_len: int = 1024

    # quantization for teachers
    teacher_load_in_4bit: bool = True

    # io
    output_dir: str = "/vol/ofa_student"      # Modal volume path
    hf_repo: Optional[str] = None             # set at push time (org or personal)
    push_private: bool = True

    def n_teachers(self) -> int:
        return len(self.teachers)

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(asdict(self), indent=2, default=str)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s
