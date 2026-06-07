"""
ofa/distill/trainer.py
----------------------
The One for All training loop. Orchestrates:
  forward through student + all teachers (each with its OWN tokenizer) ->
  gate from student state -> composite loss (task + gated CKA geometry) ->
  backward over {student LoRA, gating net, projections}.

Path B is geometry-only: logit-KD is dropped because per-position KL is
ill-defined across heterogeneous tokenizers (see distill/losses.py).

Emits a TrainState each step (loss components + gate) so the Modal job can log
it and a Gradio dashboard can plot training curves live. Heavy imports lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import OFAConfig
from .losses import composite_loss, LossComponents


@dataclass
class TrainState:
    """Per-step telemetry — JSON-serializable for logging + Gradio plots."""
    step: int
    components: dict
    lr: float

    def as_dict(self) -> dict:
        return {"step": self.step, "lr": self.lr, **self.components}


@dataclass
class TrainHistory:
    """Accumulated telemetry across the run."""
    states: list[dict] = field(default_factory=list)
    gate_samples: list[list[float]] = field(default_factory=list)
    task_labels: list[str] = field(default_factory=list)

    def record(self, state: TrainState, task_label: Optional[str] = None) -> None:
        self.states.append(state.as_dict())
        self.gate_samples.append(state.components["gate"])
        if task_label is not None:
            self.task_labels.append(task_label)


class OFATrainer:
    """
    Centralizes the training step. The Modal entrypoint constructs the bundle,
    gating, projections, optimizer, dataloader and calls `train`.
    """

    def __init__(self, cfg: OFAConfig, bundle, gating, projections):
        self.cfg = cfg
        self.bundle = bundle
        self.gating = gating
        self.projections = projections
        self.monitored = bundle.monitored_layers()
        self.history = TrainHistory()
        self._opt = None

    # ----------------------------------------------------------- optimizer
    def _optimizer(self):
        import torch
        if self._opt is None:
            params = (
                [p for p in self.bundle.student.parameters() if p.requires_grad]
                + list(self.gating.parameters())
                + list(self.projections.parameters())
            )
            self._opt = torch.optim.AdamW(params, lr=self.cfg.lr,
                                          betas=(0.9, 0.95))
        return self._opt

    # ----------------------------------------------------------- one step
    def train_step(self, batch: dict, step: int) -> TrainState:
        import torch
        from .losses import masked_mean

        opt = self._optimizer()
        student = self.bundle.student
        student.train()

        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        s_mask = batch.get("attention_mask")
        if s_mask is None:
            s_mask = torch.ones_like(input_ids)

        # ---- student forward (logits + hidden states) ----
        # use_cache=False: we only do single forwards (never generate), so the KV
        # cache is dead weight — and some teachers' remote-code cache API is
        # incompatible with the pinned transformers (DynamicCache.get_usable_length).
        s_out = student(input_ids, attention_mask=s_mask,
                        output_hidden_states=True, use_cache=False)
        s_logits = s_out.logits
        s_hidden_all = s_out.hidden_states  # tuple: (L_s+1) x (B, T_s, d_s)

        # gating from the masked-mean last hidden state, per sequence -> (B, d_s);
        # the gating net pools over the batch internally. Upcast to fp32: the
        # student runs in bf16 but the gating Linear is fp32.
        gate = self.gating(masked_mean(s_hidden_all[-1], s_mask).float())   # (N,)

        lambdas = self.cfg.schedule.lambdas(step)
        l2, l3 = lambdas[1], lambdas[2]

        # ---- Qwen KL: same tokenizer → pass student input_ids directly ----
        # Qwen2.5-1.5B shares the Qwen2 tokenizer with the 0.5B student, so
        # there is no tokenizer mismatch: input_ids are identical. No re-encoding.
        qwen_logits = None
        if l2 > 0:
            qwen_teacher = self.bundle.teachers[0]  # index 0 = Qwen2.5-1.5B
            with torch.no_grad():
                qwen_logits = qwen_teacher.model(
                    input_ids, attention_mask=s_mask, use_cache=False
                ).logits

        # ---- geometry teacher forwards (no grad), each with its OWN tokenizer ----
        # Only needed once geometry kicks in (l3 > 0); skipped during warmup.
        teacher_hidden: list = []
        teacher_masks: list = []
        if l3 > 0:
            texts = batch["texts"]
            with torch.no_grad():
                for t in self.bundle.teachers:
                    enc = t.tokenizer(texts, return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.cfg.max_seq_len)
                    enc = {k: v.to(input_ids.device) for k, v in enc.items()}
                    t_out = t.model(**enc, output_hidden_states=True,
                                    use_cache=False)
                    teacher_hidden.append(t_out.hidden_states)
                    teacher_masks.append(enc["attention_mask"])

        # ---- composite loss (task + KL + geometry) ----
        loss, comps = composite_loss(
            s_logits, labels,
            s_hidden_all, s_mask,
            teacher_hidden, teacher_masks,
            gate, self.projections, self.monitored,
            lambdas,
            qwen_logits=qwen_logits,
            temperature=self.cfg.schedule.temperature,
        )

        # ---- backward ----
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad],
            self.cfg.grad_clip,
        )
        opt.step()

        return TrainState(step=step, components=comps.as_dict(), lr=self.cfg.lr)

    # ----------------------------------------------------------- full loop
    def train(self, dataloader: Callable[[], Any],
              log_fn: Optional[Callable[[TrainState], None]] = None) -> TrainHistory:
        """
        dataloader: a callable/iterator yielding batches (dicts with input_ids).
        log_fn: optional hook called each step (Modal logging / live dashboard).
        """
        step = 0
        for batch in dataloader():
            if step >= self.cfg.train_steps:
                break
            state = self.train_step(batch, step)
            task_label = batch.get("task_label")
            self.history.record(state, task_label)
            if log_fn:
                log_fn(state)
            step += 1
        return self.history
