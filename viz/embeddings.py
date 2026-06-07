"""
ofa/viz/embeddings.py
---------------------
Turns raw embeddings into visualization-ready data for the Gradio dashboard and
the paper's Figure 3 (the four-stage UMAP of soul transfer).

Two outputs, both plain dicts (no plotting here — the front-end renders):
  * umap_projection(...)   -> 2D coords + labels per point
  * cka_heatmap_data(...)  -> pairwise CKA matrix between all models

Keeping this as data-prep (not plotting) means the same numbers feed a Gradio
Plotly chart, a matplotlib figure for the paper, or a static PNG — one source.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectionResult:
    """2D embedding coordinates ready for a scatter plot."""
    x: list[float]
    y: list[float]
    labels: list[str]          # which model each point belongs to
    stage: str = ""            # "teachers" | "student_pre" | "student_mid" | "student_post"

    def to_plotly(self) -> dict:
        """Group points by label for a colored scatter."""
        groups: dict[str, dict] = {}
        for xi, yi, lab in zip(self.x, self.y, self.labels):
            g = groups.setdefault(lab, {"x": [], "y": [], "name": lab})
            g["x"].append(xi)
            g["y"].append(yi)
        return {"stage": self.stage, "traces": list(groups.values())}


def umap_projection(embeddings, labels: list[str], stage: str = "",
                    n_neighbors: int = 15, min_dist: float = 0.1,
                    seed: int = 42) -> ProjectionResult:
    """
    Project high-dim embeddings to 2D with UMAP.

    embeddings: array-like (n_points, dim) — np.ndarray or torch.Tensor
    labels:     length-n list naming the source model of each point
    """
    import numpy as np
    try:
        import umap  # umap-learn
    except ImportError:
        # graceful fallback to PCA if umap-learn isn't installed
        return _pca_fallback(embeddings, labels, stage)

    arr = _to_numpy(embeddings)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=seed)
    coords = reducer.fit_transform(arr)
    return ProjectionResult(
        x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
        labels=labels, stage=stage,
    )


def _pca_fallback(embeddings, labels, stage):
    import numpy as np
    arr = _to_numpy(embeddings)
    arr = arr - arr.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(arr, full_matrices=False)
    coords = arr @ vh[:2].T
    return ProjectionResult(x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
                            labels=labels, stage=stage)


def cka_heatmap_data(model_embeddings: dict[str, "object"]) -> dict:
    """
    Pairwise CKA between every model's embeddings (on the same probe set).
    Returns {"models": [...], "matrix": [[...]]} for a heatmap.
    """
    import torch
    from ..metrics.cka import linear_cka

    names = list(model_embeddings.keys())
    n = len(names)
    matrix = [[0.0] * n for _ in range(n)]
    tensors = {k: _to_tensor(v) for k, v in model_embeddings.items()}
    for i in range(n):
        for j in range(n):
            matrix[i][j] = float(linear_cka(tensors[names[i]], tensors[names[j]]))
    return {"models": names, "matrix": matrix}


# ----------------------------------------------------------------- helpers
def _to_numpy(x):
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype="float32")


def _to_tensor(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)
