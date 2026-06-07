"""
ofa/smoke_test_train.py
-----------------------
Integration test for the FULL training loop (OFATrainer.train) WITHOUT
downloading any models. Tiny fake student/teachers stand in for the LLMs so we
exercise the real plumbing end to end:

  dataloader -> student forward -> gating -> per-teacher native tokenization ->
  gated CKA geometry loss -> backward over {student, gating, projections} ->
  telemetry accumulation.

It also crosses the curriculum boundary (task-only warmup -> geometry phase) so
both code paths in train_step run, including the heterogeneous-tokenizer case
(teachers with different token counts AND different depths than the student).

Run:  python -m ofa.smoke_test_train
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from ofa.config import OFAConfig, TeacherSpec, StudentSpec, LossSchedule
from ofa.distill.gating import GatingNetwork, TeacherProjections
from ofa.distill.trainer import OFATrainer
from ofa.data.dataloader import mock_dataloader


# --------------------------------------------------------------- fakes
class _FakeStudent(nn.Module):
    """Trainable stand-in: embeds ids, a few tanh layers, an LM head."""
    def __init__(self, vocab: int, d: int, n_layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                use_cache=False):
        h = self.emb(input_ids)
        hiddens = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h))
            hiddens.append(h)
        # real LLMs run in bf16 -> hidden states are bf16 (logits kept fp32 so
        # cross_entropy works on CPU). This reproduces the dtype mix that the
        # fp32 gating/projections must handle.
        hiddens = [hh.to(torch.bfloat16) for hh in hiddens]
        return SimpleNamespace(logits=self.head(h), hidden_states=tuple(hiddens))


class _FakeTeacherModel(nn.Module):
    """Frozen stand-in: only needs to emit hidden_states of its own dim/depth."""
    def __init__(self, vocab: int, d: int, n_layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                use_cache=False):
        h = self.emb(input_ids)
        hiddens = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h))
            hiddens.append(h)
        hiddens = [hh.to(torch.bfloat16) for hh in hiddens]   # bf16 like a real teacher
        return SimpleNamespace(hidden_states=tuple(hiddens))


class _FakeTokenizer:
    """Emits its OWN token length per teacher -> heterogeneous tokenization."""
    def __init__(self, vocab: int, t_len: int):
        self.vocab, self.t_len = vocab, t_len

    def __call__(self, texts, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        B = len(texts)
        return {
            "input_ids": torch.randint(0, self.vocab, (B, self.t_len)),
            "attention_mask": torch.ones(B, self.t_len, dtype=torch.long),
        }


class _FakeBundle:
    def __init__(self, student, teachers, monitored):
        self.student, self.teachers, self._monitored = student, teachers, monitored

    def monitored_layers(self):
        return self._monitored


def main():
    torch.manual_seed(0)
    print("=" * 60)
    print("One for All — full training-loop smoke test (no models)")
    print("=" * 60)

    d_s, vocab = 16, 50
    cfg = OFAConfig(
        teachers=[TeacherSpec("t0", "fake/t0", 24),
                  TeacherSpec("t1", "fake/t1", 32)],
        student=StudentSpec(hidden_dim=d_s),
        # cross the boundary fast: warmup steps are task-only, then geometry
        schedule=LossSchedule(warmup_steps=2, geo_steps=2, lambda3_max=0.5),
        train_steps=6, batch_size=4, lr=1e-2, max_seq_len=32,
    )

    student = _FakeStudent(vocab, d_s, n_layers=3)        # 4 hidden states
    teachers = [
        SimpleNamespace(tokenizer=_FakeTokenizer(60, t_len=6),
                        model=_FakeTeacherModel(60, 24, n_layers=3)),   # depth 4
        SimpleNamespace(tokenizer=_FakeTokenizer(70, t_len=11),
                        model=_FakeTeacherModel(70, 32, n_layers=5)),   # depth 6
    ]
    bundle = _FakeBundle(student, teachers, monitored=[2, 3])

    gating = GatingNetwork(cfg.student.hidden_dim, cfg.n_teachers())
    projections = TeacherProjections(cfg)
    trainer = OFATrainer(cfg, bundle, gating, projections)

    loader = mock_dataloader(cfg, vocab_size=vocab, seq_len=8, n_batches=10)
    history = trainer.train(loader)

    # ---- assertions ----
    assert len(history.states) == cfg.train_steps, "one telemetry row per step"
    for s in history.states:
        for k in ("total", "task", "kd", "geo", "gate"):
            assert k in s, f"missing telemetry key {k}"
        assert s["kd"] == 0.0, "kd dropped in Path B"
        assert s["total"] == s["total"], "loss must be finite (not NaN)"
        assert abs(sum(s["gate"]) - 1.0) < 1e-4, "gate must sum to 1"

    geos = [s["geo"] for s in history.states]
    assert any(g == 0.0 for g in geos), "warmup steps must have geo == 0"
    assert any(g > 0.0 for g in geos), "geometry phase must produce geo > 0"

    # gating starts at zero weights -> must have learned something once geometry
    # gradients flowed
    assert float(gating.fc.weight.abs().sum()) > 0, "gating must receive grads"

    print(f"\n[1] ran {len(history.states)} steps end to end ✓")
    print(f"[2] geo per step: {[round(g, 3) for g in geos]}")
    print(f"      (0.0 during warmup, >0 once geometry ramps in) ✓")
    print(f"[3] kd == 0 every step (Path B, geometry-only) ✓")
    print(f"[4] gating learned (|W|={float(gating.fc.weight.abs().sum()):.4f} > 0) ✓")
    print(f"[5] final gate: {[round(x, 3) for x in history.gate_samples[-1]]}")

    print("\n" + "=" * 60)
    print("TRAINING-LOOP TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
