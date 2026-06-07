"""
ofa/smoke_test_core.py
----------------------
Validates the mathematical core with torch but WITHOUT downloading any models:
CKA properties, gating network shapes, and the composite loss assembly using
tiny random tensors that stand in for student/teacher logits and hidden states.

Run:  python -m ofa.smoke_test_core
"""

from __future__ import annotations

import torch

from ofa.metrics.cka import linear_cka, cka_loss
from ofa.distill.gating import GatingNetwork, TeacherProjections
from ofa.distill.losses import (task_loss, masked_mean, gated_geometry_loss,
                                composite_loss)
from ofa.config import OFAConfig, TeacherSpec, StudentSpec


def main():
    torch.manual_seed(0)
    print("=" * 60)
    print("One for All — core math smoke test (torch, no models)")
    print("=" * 60)

    # ---- 1. CKA properties ----
    X = torch.randn(64, 32)
    assert abs(float(linear_cka(X, X)) - 1.0) < 1e-4, "CKA(X,X) must be 1"
    Y = torch.randn(64, 48)
    c_xy = float(linear_cka(X, Y))
    assert 0.0 <= c_xy <= 1.0, "CKA must be in [0,1]"
    # rotation invariance: CKA(X, XQ) == 1 for orthogonal Q
    Q, _ = torch.linalg.qr(torch.randn(32, 32))
    assert abs(float(linear_cka(X, X @ Q)) - 1.0) < 1e-3, "CKA rotation-invariant"
    print(f"\n[1] CKA: self={float(linear_cka(X,X)):.4f}, "
          f"cross={c_xy:.4f}, rotation-invariant ✓")

    # ---- 2. CKA loss is differentiable ----
    Xr = torch.randn(64, 32, requires_grad=True)
    Yt = torch.randn(64, 48)
    proj = torch.nn.Linear(48, 32, bias=False)
    loss = cka_loss(Xr, Yt, projection=proj)
    loss.backward()
    assert Xr.grad is not None and proj.weight.grad is not None, "grads must flow"
    print(f"[2] CKA loss differentiable: loss={float(loss):.4f}, grads flow ✓")

    # ---- 3. Gating network ----
    n_teachers = 5
    d_s = 16
    gate_net = GatingNetwork(d_s, n_teachers)
    mean_h = torch.randn(d_s)
    g = gate_net(mean_h)
    assert g.shape == (n_teachers,)
    assert abs(float(g.sum()) - 1.0) < 1e-5, "gate must sum to 1"
    # near-uniform at init (zero weights)
    assert all(abs(float(x) - 1/n_teachers) < 1e-5 for x in g), "init ~uniform"
    print(f"[3] Gating: shape={tuple(g.shape)}, sum={float(g.sum()):.4f}, "
          f"init uniform={[round(float(x),2) for x in g]} ✓")

    # ---- 4. Teacher projections ----
    cfg = OFAConfig(
        teachers=[TeacherSpec(f"t{i}", f"fake/t{i}", 24 + i * 8) for i in range(n_teachers)],
        student=StudentSpec(hidden_dim=d_s),
    )
    projs = TeacherProjections(cfg)
    t_h = torch.randn(64, cfg.teachers[2].hidden_dim)
    projected = projs(2, t_h)
    assert projected.shape == (64, d_s), "projection maps to student dim"
    print(f"[4] Projections: teacher dim {cfg.teachers[2].hidden_dim} -> "
          f"student dim {projected.shape[-1]} ✓")

    # ---- 5. Composite loss assembly (geometry-only, tokenizer-agnostic) ----
    B, T, V = 4, 8, 50
    s_logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))

    # student: 5 hidden states (embed + 4 layers), (B, T, d_s)
    Ls = 5
    s_hidden = [torch.randn(B, T, d_s) for _ in range(Ls)]
    s_mask = torch.ones(B, T)
    # teachers: DIFFERENT token counts (T+i) and DIFFERENT depths (Ls+i) — the
    # whole point of Path B. Pooling + depth-fraction mapping must absorb both.
    t_hidden = [[torch.randn(B, T + i, cfg.teachers[i].hidden_dim)
                 for _ in range(Ls + i)]
                for i in range(n_teachers)]
    t_masks = [torch.ones(B, T + i) for i in range(n_teachers)]
    monitored = [3, 4]   # student layer indices

    # masked_mean ignores padding -> (B, d)
    pooled = masked_mean(s_hidden[0], s_mask)
    assert pooled.shape == (B, d_s), "masked_mean must pool to (B, d)"

    # task only (warmup): geometry skipped, no teacher tensors needed
    loss0, comps0_obj = composite_loss(
        s_logits, labels, s_hidden, s_mask, [], [],
        g.detach(), projs, monitored, lambdas=(1.0, 0.0, 0.0))
    comps0 = comps0_obj.as_dict()
    assert comps0["geo"] == 0.0, "geo skipped during warmup"
    assert comps0["kd"] == 0.0, "kd dropped in Path B"

    # full objective (task + gated geometry)
    loss1, comps1_obj = composite_loss(
        s_logits, labels, s_hidden, s_mask, t_hidden, t_masks,
        g.detach(), projs, monitored, lambdas=(1.0, 0.0, 0.5))
    comps1 = comps1_obj.as_dict()
    loss1.backward()
    assert s_logits.grad is not None, "composite loss grads must flow to student"
    print(f"[5] Composite loss (geometry-only):")
    print(f"    warmup -> task={comps0['task']:.3f} kd={comps0['kd']:.3f} geo={comps0['geo']:.3f}")
    print(f"    full   -> task={comps1['task']:.3f} kd={comps1['kd']:.3f} geo={comps1['geo']:.3f}")
    print(f"    gate in components: {[round(x,2) for x in comps1['gate']]}")
    print(f"    backward grads flow ✓")

    print("\n" + "=" * 60)
    print("CORE MATH TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
