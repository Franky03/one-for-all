"""
ofa/distill/gating.py
---------------------
The two small trainable modules that make One for All novel:

  * GatingNetwork — given the student's mean hidden state for a batch, produces
    a distribution over the N teachers (Eq. 1). This is "self-referential
    routing": the student decides whose soul to absorb based on its own state.

  * TeacherProjections — one linear map Pi_i per teacher, projecting teacher
    hidden dim -> student hidden dim, so CKA compares commensurate spaces.

Both are tiny relative to the LLMs, so they add negligible parameters.
torch is imported at top: this module only lives inside the training stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import OFAConfig


class GatingNetwork(nn.Module):
    """
    Maps the student's mean-pooled hidden state -> softmax weights over teachers.

    Input:  (B, d_s)  mean hidden state per batch (or (d_s,) for a single vector)
    Output: (N,)      gating distribution g, sums to 1
    """

    def __init__(self, student_dim: int, n_teachers: int):
        super().__init__()
        self.fc = nn.Linear(student_dim, n_teachers)
        # init near-uniform so early training absorbs all teachers evenly
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, student_mean_h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if student_mean_h.dim() == 2:
            student_mean_h = student_mean_h.mean(dim=0)   # pool over batch
        logits = self.fc(student_mean_h)
        return torch.softmax(logits / max(temperature, 1e-6), dim=-1)


class TeacherProjections(nn.Module):
    """
    One Pi_i: R^{d_t_i} -> R^{d_s} per teacher. Used to align teacher hidden
    states into the student space before CKA.
    """

    def __init__(self, cfg: OFAConfig):
        super().__init__()
        d_s = cfg.student.hidden_dim
        self.projections = nn.ModuleList([
            nn.Linear(t.hidden_dim, d_s, bias=False) for t in cfg.teachers
        ])

    def forward(self, teacher_idx: int, teacher_h: torch.Tensor) -> torch.Tensor:
        return self.projections[teacher_idx](teacher_h)

    def __getitem__(self, idx: int) -> nn.Module:
        return self.projections[idx]
