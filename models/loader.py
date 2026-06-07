"""
ofa/models/loader.py
--------------------
Loads the five teachers (4-bit, frozen) and the student (LoRA, trainable), plus
a hidden-state extraction helper used by both the KD and geometry losses.

All heavy imports are lazy inside functions so `ofa.config` and the metric
modules stay importable without the ML stack. This file runs inside the Modal
GPU job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import OFAConfig, TeacherSpec


@dataclass
class LoadedTeacher:
    spec: TeacherSpec
    model: Any
    tokenizer: Any


class ModelBundle:
    """Holds all teachers + the student + the student tokenizer."""

    def __init__(self, cfg: OFAConfig):
        self.cfg = cfg
        self.teachers: list[LoadedTeacher] = []
        self.student: Any = None
        self.student_tokenizer: Any = None

    # ----------------------------------------------------------- teachers
    def load_teachers(self) -> None:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)

        bnb = None
        if self.cfg.teacher_load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        for spec in self.cfg.teachers:
            # trust_remote_code: MiniCPM / Phi-3.5 ship custom modeling code.
            tok = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True)
            # Path B re-tokenizes per teacher with padding=True; some teacher
            # tokenizers (e.g. Llama-based Phi-3.5) ship no pad token.
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                quantization_config=bnb,
                dtype=torch.bfloat16,   # transformers pinned <5 (see modal_app)
                device_map="auto",
                output_hidden_states=True,
                trust_remote_code=True,
            )
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.teachers.append(LoadedTeacher(spec, model, tok))

    # ----------------------------------------------------------- student
    def load_student(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        s = self.cfg.student
        self.student_tokenizer = AutoTokenizer.from_pretrained(s.model_id)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            s.model_id,
            dtype=torch.bfloat16,   # transformers pinned <5 (see modal_app)
            device_map="auto",
            output_hidden_states=True,
        )
        lora = LoraConfig(
            r=s.lora_r, lora_alpha=s.lora_alpha, lora_dropout=s.lora_dropout,
            target_modules=list(s.lora_targets), task_type="CAUSAL_LM",
        )
        self.student = get_peft_model(base, lora)

    def n_student_layers(self) -> int:
        return self.student.config.num_hidden_layers

    def monitored_layers(self) -> list[int]:
        """Last fraction of layers, used for CKA geometry alignment."""
        n = self.n_student_layers()
        k = max(1, int(n * self.cfg.monitored_layer_frac))
        return list(range(n - k, n))
