"""
ofa/smoke_test_pure.py
----------------------
Validates the torch-free layer: config, lambda schedule, and fingerprint
aggregation. Runs with zero ML deps — the fast correctness gate.

Run:  python -m ofa.smoke_test_pure
"""

from __future__ import annotations

from ofa.config import OFAConfig, LossSchedule, DEFAULT_TEACHERS
from ofa.metrics.fingerprint import aggregate_fingerprint


def main():
    print("=" * 60)
    print("One for All — pure-layer smoke test (no torch)")
    print("=" * 60)

    # 1. config
    cfg = OFAConfig()
    assert cfg.n_teachers() == 5, "expected 5 teachers"
    js = cfg.to_json()
    assert "Qwen2.5-0.5B" in js
    print(f"\n[1] Config: {cfg.n_teachers()} teachers, "
          f"student={cfg.student.model_id}, JSON {len(js)} chars ✓")

    # 2. lambda schedule — three phases
    sch = LossSchedule(warmup_steps=500, geo_steps=500, lambda3_max=0.5)
    l_early = sch.lambdas(100)     # phase 1: task only
    l_mid = sch.lambdas(600)       # phase 2: task + kd, geo ramping
    l_late = sch.lambdas(2000)     # phase 3: full
    assert l_early[2] == 0.0, "geo must be 0 during warmup"
    assert l_mid[1] == 1.0, "kd must be full after warmup"
    assert abs(l_late[2] - 0.5) < 1e-9, "geo must reach lambda3_max"
    print(f"[2] Lambda schedule:")
    print(f"    step 100  -> task={l_early[0]:.2f} kd={l_early[1]:.2f} geo={l_early[2]:.2f}")
    print(f"    step 600  -> task={l_mid[0]:.2f} kd={l_mid[1]:.2f} geo={l_mid[2]:.2f}")
    print(f"    step 2000 -> task={l_late[0]:.2f} kd={l_late[1]:.2f} geo={l_late[2]:.2f}")

    # 3. fingerprint aggregation
    names = [t.name for t in DEFAULT_TEACHERS]
    # simulate gate samples: qwen-dominant on code, phi-dominant on reasoning
    gate_samples = [
        [0.5, 0.1, 0.2, 0.1, 0.1],   # code -> qwen
        [0.1, 0.1, 0.5, 0.2, 0.1],   # reasoning -> phi
        [0.2, 0.2, 0.2, 0.2, 0.2],   # general -> uniform
    ]
    task_labels = ["code", "reasoning", "general"]
    fp = aggregate_fingerprint(names, gate_samples, task_labels)
    assert len(fp.overall) == 5
    assert "code" in fp.by_task
    radar = fp.to_radar_data()
    assert radar["axes"] == names
    print(f"\n[3] Soul fingerprint:")
    print(f"    overall: {[round(x,2) for x in fp.overall]}")
    print(f"    dominant: {fp.dominant_teacher()}")
    print(f"    by_task code -> {[round(x,2) for x in fp.by_task['code']]}")
    print(f"    radar data ready for Gradio: {len(radar['series'])} series ✓")

    print("\n" + "=" * 60)
    print("PURE-LAYER TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
