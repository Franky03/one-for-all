import json, os, tempfile
import numpy as np
import pytest

import _data  # available via conftest.py sys.path insert


def _write_fake(tmp_path, P=4, d=8, n_teachers=2):
    """Write a minimal viz_data.json for testing."""
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


def test_parse_stacked_shape(tmp_path):
    path = _write_fake(tmp_path, P=4, d=8, n_teachers=2)
    viz = _data.load_from_path(path)
    # 3 models × 4 points = 12 rows, 8 cols
    assert viz["stacked"].shape == (12, 8)


def test_parse_labels_length(tmp_path):
    path = _write_fake(tmp_path, P=4, d=8, n_teachers=2)
    viz = _data.load_from_path(path)
    assert len(viz["labels"]) == 12


def test_parse_teacher_names(tmp_path):
    path = _write_fake(tmp_path, P=4, d=8, n_teachers=2)
    viz = _data.load_from_path(path)
    assert viz["teacher_names"] == ["t0", "t1"]
    assert "student" not in viz["teacher_names"]


def test_fit_umap3d_output_shape(tmp_path):
    path = _write_fake(tmp_path, P=4, d=8, n_teachers=2)
    viz = _data.load_from_path(path)
    reducer = _data.fit_umap3d(viz["stacked"])
    # embedding_ has shape (N_total, 3) after fit
    assert reducer.embedding_.shape == (12, 3)


def test_make_empty_viz_structure():
    viz = _data.make_empty_viz()
    assert "stacked" in viz
    assert viz["stacked"].shape[0] == 0
    assert viz["teacher_names"] == []
    assert viz["cka"] == {}
    assert viz["curves"] == {}


def test_fit_umap3d_insufficient_data():
    with pytest.raises(ValueError, match="at least 2 samples"):
        _data.fit_umap3d(np.array([[1.0, 2.0]], dtype=np.float32))
