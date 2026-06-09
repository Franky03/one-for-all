"""
ofa/distill/losses.py
---------------------
The composite One for All objective (Eq. 4):

    L = lambda1 * L_task + lambda2 * L_KD(g) + lambda3 * L_geo(g)

Each term is its own function for testability; `composite_loss` assembles them
with the scheduled lambdas and the gating weights. Returns both the scalar loss
and a dict of components for logging (which the Gradio dashboard will plot).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..metrics.cka import cka_loss


def qwen_kl_loss(student_logits: torch.Tensor,
                 qwen_logits: torch.Tensor,
                 temperature: float = 4.0) -> torch.Tensor:
    """
    KL divergence from Qwen2.5-1.5B (same tokenizer → same vocab, zero friction).
    tau^2 * KL(qwen^tau || student^tau) summed over non-padding positions.
    This is the richest distillation signal: full output distribution, not just geometry.
    """
    tau = temperature
    s = student_logits[:, :-1, :]   # (B, T-1, V) — predict t+1 from t
    q = qwen_logits[:, :-1, :]
    B, T, V = s.shape
    s_logp = F.log_softmax(s.reshape(-1, V) / tau, dim=-1)
    q_prob  = F.softmax(q.reshape(-1, V) / tau, dim=-1)
    return tau ** 2 * F.kl_div(s_logp, q_prob, reduction="batchmean")


def task_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy. Shifts handled by caller or here."""
    # student_logits: (B, T, V); labels: (B, T) with -100 for ignored positions
    vocab = student_logits.shape[-1]
    return F.cross_entropy(
        student_logits[:, :-1, :].reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def gated_kd_loss(student_logits: torch.Tensor,
                  teacher_logits_list: list[torch.Tensor],
                  gate: torch.Tensor,
                  temperature: float = 4.0) -> torch.Tensor:
    """
    Gated KL distillation (Eq. 5):
        sum_i g_i * tau^2 * KL(p_Ti^tau || p_S^tau)

    student_logits:        (B, T, V)
    teacher_logits_list:   list of N tensors, each (B, T, V) (already aligned vocab)
    gate:                  (N,) gating weights summing to 1
    """
    tau = temperature
    s_logp = F.log_softmax(student_logits / tau, dim=-1)
    total = student_logits.new_zeros(())
    for i, t_logits in enumerate(teacher_logits_list):
        t_prob = F.softmax(t_logits / tau, dim=-1)
        # KL(teacher || student) = sum t_prob * (log t_prob - s_logp)
        kl = F.kl_div(s_logp, t_prob, reduction="batchmean")
        total = total + gate[i] * (tau ** 2) * kl
    return total


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool over the token axis, ignoring padding.

    hidden: (B, T, d), mask: (B, T) of 0/1 -> (B, d).

    This is what makes geometry distillation tokenizer-agnostic: each sequence
    collapses to ONE vector regardless of how many tokens its tokenizer produced,
    so the B sequences correspond across the student and every teacher.
    """
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def gated_geometry_loss(student_hidden: list[torch.Tensor],
                        student_mask: torch.Tensor,
                        teacher_hidden_list: list[list[torch.Tensor]],
                        teacher_masks: list[torch.Tensor],
                        gate: torch.Tensor,
                        projections,
                        monitored_layers: list[int]) -> torch.Tensor:
    """
    Gated CKA geometry loss (Eq. 6), tokenizer-agnostic via sequence pooling:
        1 - (1/|L|) sum_l sum_i g_i * CKA(pool(H_S^l), Pi_i pool(H_Ti^map(l)))

    student_hidden:       list over layers, each (B, T_s, d_s)
    student_mask:         (B, T_s)
    teacher_hidden_list:  list over teachers, each a list over layers (B, T_ti, d_ti)
    teacher_masks:        list over teachers, each (B, T_ti)
    gate:                 (N,)
    projections:          TeacherProjections module
    monitored_layers:     student layer indices to align

    Teachers differ in depth from the student, so each monitored student layer is
    mapped to a teacher layer by depth fraction. With pooling, CKA is computed
    over B samples (the batch) — B is small, so the estimate is noisier than the
    old per-token form; accumulate over steps or raise the effective batch if the
    geometry signal looks unstable.
    """
    n_teachers = len(teacher_hidden_list)
    L_student = len(student_hidden)

    # Models run in bf16; the projections are fp32 and CKA's squared Frobenius
    # norms are precision-sensitive, so the geometry math is done in fp32.
    total_align = torch.zeros((), dtype=torch.float32, device=student_hidden[0].device)
    count = 0

    for l in monitored_layers:
        s_pool = masked_mean(student_hidden[l], student_mask).float()   # (B, d_s)
        for i in range(n_teachers):
            L_teacher = len(teacher_hidden_list[i])
            # depth-fraction alignment of student layer l -> teacher layer
            t_l = round(l * (L_teacher - 1) / (L_student - 1)) if L_student > 1 else 0

            t_pool = masked_mean(teacher_hidden_list[i][t_l], teacher_masks[i]).float()

            # cka_loss returns 1 - CKA; we want the CKA alignment, so invert
            align = 1.0 - cka_loss(s_pool, t_pool, projection=projections[i])
            total_align = total_align + gate[i] * align
            count += 1

    mean_align = total_align / max(1, count)
    return 1.0 - mean_align

@dataclass
class LossComponents:
    """For logging / the Gradio dashboard."""
    total: float
    task: float
    kd: float
    geo: float
    lambda_task: float
    lambda_kd: float
    lambda_geo: float
    gate: list[float]            # current teacher gating weights

    def as_dict(self) -> dict:
        return {
            "total": self.total, "task": self.task, "kd": self.kd, "geo": self.geo,
            "lambda_task": self.lambda_task, "lambda_kd": self.lambda_kd,
            "lambda_geo": self.lambda_geo, "gate": self.gate,
        }


def composite_loss(student_logits, labels,
                   student_hidden, student_mask,
                   teacher_hidden_list, teacher_masks,
                   gate, projections, monitored_layers,
                   lambdas: tuple[float, float, float],
                   qwen_logits=None,
                   temperature: float = 4.0,
                   gate_entropy_alpha: float = 0.0):
    """
    Assemble the objective. Returns (loss_tensor, LossComponents).

    l1 * L_task + l2 * L_KL(qwen) + l3 * L_geo - alpha * H(gate)

    H(gate) entropy term prevents gate collapse: maximizing entropy keeps the
    gate from betting everything on one teacher early in training.
    """
    l1, l2, l3 = lambdas

    lt = task_loss(student_logits, labels)

    lkd = (qwen_kl_loss(student_logits, qwen_logits, temperature)
           if (l2 > 0 and qwen_logits is not None)
           else student_logits.new_zeros(()))

    if l3 > 0:
        lgeo = gated_geometry_loss(student_hidden, student_mask,
                                   teacher_hidden_list, teacher_masks,
                                   gate, projections, monitored_layers)
    else:
        lgeo = student_logits.new_zeros(())

    loss = l1 * lt + l2 * lkd + l3 * lgeo

    # entropy regularization: H(gate) = -sum(p * log(p)) >= 0
    # subtracting it from loss = maximizing entropy = preventing collapse
    if gate_entropy_alpha > 0:
        entropy = -(gate * gate.clamp(min=1e-8).log()).sum()
        loss = loss - gate_entropy_alpha * entropy

    comps = LossComponents(
        total=float(loss.detach()), task=float(lt.detach()),
        kd=float(lkd.detach()), geo=float(lgeo.detach()),
        lambda_task=l1, lambda_kd=l2, lambda_geo=l3,
        gate=[float(x) for x in gate.detach()],
    )
    return loss, comps
