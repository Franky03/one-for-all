"""Single test file covering _data, _fig, _html, _probe, and _three."""
import json
import numpy as np
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

import _data
import _fig
import _html
import _probe
import _three


# ── Shared fixtures ────────────────────────────────────────────────────────

def _fake_json(tmp_path, P=4, d=8, n_teachers=2):
    rng = np.random.default_rng(0)
    names = ["student"] + [f"t{i}" for i in range(n_teachers)]
    emb = {n: rng.standard_normal((P, d)).tolist() for n in names}
    labels = sum([[n] * P for n in names], [])
    data = {
        "embeddings": emb,
        "labels": labels,
        "cka": {
            "matrix": [[1.0 if i == j else 0.6 for j in range(len(names))]
                       for i in range(len(names))],
            "models": names,
        },
        "curves": {
            "steps": [0, 1, 2],
            "task": [2.0, 1.5, 1.0],
            "geo": [0.0, 0.5, 0.8],
            "total": [2.0, 2.0, 1.8],
            "gate": [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
            "teacher_names": [f"t{i}" for i in range(n_teachers)],
        },
    }
    p = tmp_path / "viz_data.json"
    p.write_text(json.dumps(data))
    return str(p)


def _make_viz_and_coords(P=4, d=8, n_teachers=2):
    rng = np.random.default_rng(1)
    names = ["student"] + [f"t{i}" for i in range(n_teachers)]
    stacked = rng.standard_normal((P * len(names), d)).astype(np.float32)
    labels = sum([[n] * P for n in names], [])
    coords3d = rng.standard_normal((P * len(names), 3)).astype(np.float32)
    viz = {
        "stacked": stacked,
        "labels": labels,
        "model_names": names,
        "teacher_names": names[1:],
        "cka": {
            "matrix": [[1.0 if i == j else 0.6 for j in range(len(names))]
                       for i in range(len(names))],
            "models": names,
        },
        "curves": {
            "steps": [0, 1, 2],
            "task": [2.0, 1.5, 1.0],
            "geo": [0.0, 0.5, 0.8],
            "total": [2.0, 2.0, 1.8],
            "gate": [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
            "teacher_names": names[1:],
        },
    }
    return viz, coords3d


# ── _data ──────────────────────────────────────────────────────────────────

def test_data_stacked_shape(tmp_path):
    viz = _data.load_from_path(_fake_json(tmp_path, P=4, d=8, n_teachers=2))
    assert viz["stacked"].shape == (12, 8)  # 3 models × 4 points


def test_data_labels_length(tmp_path):
    viz = _data.load_from_path(_fake_json(tmp_path, P=4, d=8, n_teachers=2))
    assert len(viz["labels"]) == 12


def test_data_teacher_names(tmp_path):
    viz = _data.load_from_path(_fake_json(tmp_path))
    assert viz["teacher_names"] == ["t0", "t1"]
    assert "student" not in viz["teacher_names"]


def test_data_umap_shape(tmp_path):
    viz = _data.load_from_path(_fake_json(tmp_path, P=4, d=8, n_teachers=2))
    reducer = _data.fit_umap3d(viz["stacked"])
    assert reducer.embedding_.shape == (12, 3)


def test_data_umap_min_samples():
    with pytest.raises(ValueError, match="at least 2"):
        _data.fit_umap3d(np.array([[1.0, 2.0]], dtype=np.float32))


def test_data_empty_viz():
    viz = _data.make_empty_viz()
    assert viz["stacked"].shape[0] == 0
    assert viz["teacher_names"] == []
    assert viz["cka"] == {} and viz["curves"] == {}


# ── _fig ──────────────────────────────────────────────────────────────────

def test_fig_base_traces_count():
    viz, coords3d = _make_viz_and_coords()
    assert len(_fig.build_base_traces(viz, coords3d)) == 3


def test_fig_rebuild_no_probes():
    viz, coords3d = _make_viz_and_coords()
    traces = _fig.build_base_traces(viz, coords3d)
    assert len(_fig.rebuild_fig(traces, []).data) == 3


def test_fig_rebuild_with_probe():
    viz, coords3d = _make_viz_and_coords()
    traces = _fig.build_base_traces(viz, coords3d)
    fig = _fig.rebuild_fig(traces, [{"x": 0.1, "y": 0.2, "z": 0.3, "label": "p"}])
    assert len(fig.data) == 4  # 3 base + 1 probe


def test_fig_base_traces_immutable():
    viz, coords3d = _make_viz_and_coords()
    traces = _fig.build_base_traces(viz, coords3d)
    original_len = len(traces)
    _fig.rebuild_fig(traces, [{"x": 0.0, "y": 0.0, "z": 0.0, "label": "p"}])
    assert len(traces) == original_len


def test_fig_cka_heatmap():
    viz, _ = _make_viz_and_coords()
    fig = _fig.build_cka_fig(viz["cka"])
    assert len(fig.data) == 1 and fig.data[0].type == "heatmap"


def test_fig_curves_empty():
    assert _fig.build_curves_fig({}) is not None


def test_fig_curves_with_data():
    viz, _ = _make_viz_and_coords()
    assert len(_fig.build_curves_fig(viz["curves"]).data) == 3


def test_fig_gate_area():
    viz, _ = _make_viz_and_coords()
    assert len(_fig.build_gate_area_fig(viz["curves"]).data) == 2


# ── _html ──────────────────────────────────────────────────────────────────

TEACHER_NAMES = ["qwen", "smol", "phi", "gemma", "minicpm"]
WEIGHTS = [0.64, 0.22, 0.09, 0.04, 0.01]


def test_html_gate_contains_teachers():
    html = _html.gate_html(WEIGHTS, TEACHER_NAMES)
    for name in TEACHER_NAMES:
        assert name in html


def test_html_gate_sorted_descending():
    html = _html.gate_html([0.01, 0.02, 0.03, 0.04, 0.90], TEACHER_NAMES)
    assert html.index("minicpm") < html.index("qwen")


def test_html_task_shows_argmax():
    html = _html.task_html(WEIGHTS, TEACHER_NAMES)
    assert "QWEN" in html


def test_html_task_shows_runner_up():
    html = _html.task_html(WEIGHTS, TEACHER_NAMES)
    assert "SMOL" in html and "0.22" in html


def test_html_header_brand():
    html = _html.header_html()
    assert "One for All" in html and "∀" in html


def test_html_header_pills():
    html = _html.header_html()
    assert "Qwen2.5-0.5B" in html and "PATH B" in html


# ── _probe ────────────────────────────────────────────────────────────────

D_STUDENT = 16
N_TEACHERS = 3


class _FakeTok:
    pad_token = "<pad>"
    def __call__(self, texts, return_tensors=None, truncation=None, max_length=None):
        B = len(texts)
        return {"input_ids": torch.zeros(B, 8, dtype=torch.long),
                "attention_mask": torch.ones(B, 8, dtype=torch.long)}


class _FakeStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, D_STUDENT)
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        h = self.emb(input_ids).to(torch.bfloat16)
        return SimpleNamespace(hidden_states=(h, h, h))
    def parameters(self):
        return iter([self.emb.weight])


class _FakeGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D_STUDENT, N_TEACHERS)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class _FakeReducer:
    def transform(self, X):
        return np.zeros((X.shape[0], 3), dtype=np.float32)


def test_probe_masked_mean_shape():
    h = torch.ones(2, 5, D_STUDENT)
    mask = torch.ones(2, 5, dtype=torch.long)
    assert _probe._masked_mean(h, mask).shape == (2, D_STUDENT)


def test_probe_masked_mean_ignores_padding():
    h = torch.ones(1, 4, D_STUDENT)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    h[0, 2] = h[0, 3] = 999.0
    out = _probe._masked_mean(h, mask)
    assert out.abs().max().item() < 2.0


def test_probe_run_returns_point_and_weights():
    pt, weights = _probe.run_probe(
        "hello", _FakeStudent(), _FakeTok(), _FakeGating(), _FakeReducer()
    )
    assert set(pt.keys()) == {"x", "y", "z", "label"}
    assert len(weights) == N_TEACHERS
    assert abs(sum(weights) - 1.0) < 1e-4


def test_probe_coords_are_floats():
    pt, _ = _probe.run_probe(
        "test", _FakeStudent(), _FakeTok(), _FakeGating(), _FakeReducer()
    )
    assert all(isinstance(pt[k], float) for k in ("x", "y", "z"))


# ── _three ────────────────────────────────────────────────────────────────

def test_three_empty_without_coords():
    html = _three.build_umap_html({"model_names": [], "labels": []}, None, [])
    assert "no embedding data" in html


def test_three_contains_threejs():
    viz, coords3d = _make_viz_and_coords()
    html = _three.build_umap_html(viz, coords3d, [])
    assert "three.module.min.js" in html


def test_three_embeds_model_names():
    viz, coords3d = _make_viz_and_coords()
    html = _three.build_umap_html(viz, coords3d, [])
    for name in viz["model_names"]:
        assert name in html


def test_three_probe_points_included():
    viz, coords3d = _make_viz_and_coords()
    probes = [{"x": 0.1, "y": 0.2, "z": 0.3, "label": "p"}]
    html = _three.build_umap_html(viz, coords3d, probes)
    assert '"probes"' in html
    assert "0.1" in html
