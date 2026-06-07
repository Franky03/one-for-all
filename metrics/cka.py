"""
ofa/metrics/cka.py
------------------
Centered Kernel Alignment (CKA) — the geometric similarity measure at the heart
of One for All.

Two roles in this project:
  1. As a TRAINING LOSS (differentiable): forces the student's internal
     representations to align with a gated mixture of teacher geometries.
  2. As an ANALYSIS METRIC (no_grad): measures how much of each teacher's "soul"
     survived in the trained student, and powers the UMAP/heatmap visuals.

The linear-kernel CKA is differentiable w.r.t. its inputs, so the same function
serves both roles. torch is imported at module top here because this module is
only ever used inside the training/eval stack (never in the pure-config layer).
"""

from __future__ import annotations

import torch


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix: H K H, with H = I - (1/n) 11^T."""
    n = K.shape[0]
    unit = torch.ones(n, n, device=K.device, dtype=K.dtype)
    identity = torch.eye(n, device=K.device, dtype=K.dtype)
    H = identity - unit / n
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Linear CKA between two activation matrices.

    X: (n, d_x)  — n samples, d_x features  (e.g., student hidden states)
    Y: (n, d_y)  — n samples, d_y features  (e.g., teacher hidden states)

    Returns a scalar in [0, 1]. Differentiable w.r.t. X and Y.

    Uses the Frobenius-norm form:
        CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    on column-centered X, Y — equivalent to HSIC with linear kernels.
    """
    # column-center (subtract feature means over the sample axis)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # cross- and self-similarity (Frobenius norms of covariance-like terms)
    xty = X.t() @ Y                       # (d_x, d_y)
    numerator = (xty ** 2).sum()          # ||X^T Y||_F^2

    xtx = X.t() @ X
    yty = Y.t() @ Y
    denom = torch.sqrt((xtx ** 2).sum() * (yty ** 2).sum()) + eps

    return numerator / denom


def kernel_cka(X: torch.Tensor, Y: torch.Tensor,
               sigma: float | None = None, eps: float = 1e-6) -> torch.Tensor:
    """
    RBF-kernel CKA. Heavier but captures non-linear similarity. Used in analysis,
    not in the training loop (linear_cka is the loss). sigma=None uses the median
    heuristic for the bandwidth.
    """
    def rbf(Z):
        sq = torch.cdist(Z, Z) ** 2
        if sigma is None:
            med = torch.median(sq[sq > 0])
            s = torch.sqrt(0.5 * med + eps)
        else:
            s = torch.tensor(sigma, device=Z.device, dtype=Z.dtype)
        return torch.exp(-sq / (2 * s ** 2 + eps))

    Kx = _center_gram(rbf(X))
    Ky = _center_gram(rbf(Y))
    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()
    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)


def cka_loss(student_h: torch.Tensor, teacher_h: torch.Tensor,
             projection: torch.nn.Module | None = None) -> torch.Tensor:
    """
    Geometry alignment loss = 1 - CKA. Minimizing pushes the student geometry
    toward the teacher's.

    student_h: (n, d_s)
    teacher_h: (n, d_t)
    projection: optional nn.Module mapping d_t -> d_s (the learned Pi_i). When
                provided, the teacher is projected into student space first so
                CKA compares commensurate representations.
    """
    if projection is not None:
        teacher_h = projection(teacher_h)
    return 1.0 - linear_cka(student_h, teacher_h)
