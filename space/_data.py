from __future__ import annotations
import json, os
import numpy as np


def load_and_parse(hf_token: str | None = None) -> dict:
    """Download viz_data.json from HF dataset and parse. Uses HF_TOKEN env var."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="build-small-hackathon/ofa-viz-data",
        filename="viz_data.json",
        repo_type="dataset",
        token=hf_token or os.environ.get("HF_TOKEN"),
    )
    return load_from_path(path)


def load_from_path(path: str) -> dict:
    """Parse a local viz_data.json. Used for testing and local runs."""
    with open(path) as f:
        raw = json.load(f)
    return _parse(raw)


def _parse(raw: dict) -> dict:
    model_names: list[str] = list(raw["embeddings"].keys())  # preserves order
    emb_list, labels = [], []
    for name in model_names:
        arr = np.array(raw["embeddings"][name], dtype=np.float32)
        emb_list.append(arr)
        labels.extend([name] * len(arr))
    stacked = np.concatenate(emb_list, axis=0) if emb_list else np.empty((0, 1), dtype=np.float32)
    return {
        "stacked": stacked,
        "labels": labels,
        "model_names": model_names,
        "teacher_names": [n for n in model_names if n != "student"],
        "cka": raw.get("cka", {}),
        "curves": raw.get("curves", {}),
    }


def fit_umap3d(stacked: np.ndarray, n_neighbors: int = 15):
    """Fit a 3D UMAP on stacked embeddings. Returns fitted reducer."""
    import umap as umap_lib
    if stacked.shape[0] < 2:
        raise ValueError(f"UMAP requires at least 2 samples, got {stacked.shape[0]}")
    n = min(n_neighbors, stacked.shape[0] - 1)
    reducer = umap_lib.UMAP(n_components=3, n_neighbors=n, random_state=42, verbose=False)
    reducer.fit(stacked)
    return reducer


def make_empty_viz() -> dict:
    """Return a zero-data dict for use when viz_data.json is unavailable."""
    return {
        "stacked": np.empty((0, 1), dtype=np.float32),
        "labels": [],
        "model_names": [],
        "teacher_names": [],
        "cka": {},
        "curves": {},
    }
