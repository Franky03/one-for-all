"""
ofa/metrics/fingerprint.py
--------------------------
The Soul Fingerprint (Eq. 10): how much of each teacher survived in the student.

  alpha_i = E_batch[ g_i(batch) ]

Computed by running the trained gating network over an eval set and averaging
the gate weights. Task-conditional fingerprints reveal whether the student
routes to different teachers on different task types — the signature of
internalized specialization.

Outputs are plain dicts/lists, ready for the Gradio radar chart and the paper's
Figure 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SoulFingerprint:
    """Per-teacher contribution, optionally broken down by task category."""
    teacher_names: list[str]
    overall: list[float]                                  # alpha_i, sums to ~1
    by_task: dict[str, list[float]] = field(default_factory=dict)

    def dominant_teacher(self) -> str:
        idx = max(range(len(self.overall)), key=lambda i: self.overall[i])
        return self.teacher_names[idx]

    def to_radar_data(self) -> dict:
        """Shape the Gradio radar chart consumes directly."""
        return {
            "axes": self.teacher_names,
            "overall": self.overall,
            "series": [{"task": k, "values": v} for k, v in self.by_task.items()],
        }

    def as_dict(self) -> dict:
        return {
            "teacher_names": self.teacher_names,
            "overall": self.overall,
            "by_task": self.by_task,
            "dominant": self.dominant_teacher(),
        }


def aggregate_fingerprint(teacher_names: list[str],
                          gate_samples: list[list[float]],
                          task_labels: list[str] | None = None) -> SoulFingerprint:
    """
    Average gating weights into a fingerprint.

    gate_samples: list of per-batch gate vectors, each length N
    task_labels:  optional per-batch task category for the by_task breakdown
    """
    n = len(teacher_names)
    if not gate_samples:
        return SoulFingerprint(teacher_names, [1.0 / n] * n)

    # overall mean
    overall = [0.0] * n
    for g in gate_samples:
        for i in range(n):
            overall[i] += g[i]
    overall = [x / len(gate_samples) for x in overall]

    # task-conditional
    by_task: dict[str, list[float]] = {}
    if task_labels:
        buckets: dict[str, list[list[float]]] = {}
        for g, t in zip(gate_samples, task_labels):
            buckets.setdefault(t, []).append(g)
        for task, gs in buckets.items():
            vec = [0.0] * n
            for g in gs:
                for i in range(n):
                    vec[i] += g[i]
            by_task[task] = [x / len(gs) for x in vec]

    return SoulFingerprint(teacher_names, overall, by_task)
