"""Metrics: differentiable CKA (needs torch) + soul fingerprint (pure).

`fingerprint` is torch-free and safe to import anywhere. `cka` needs torch and
should be imported directly (`from ofa.metrics.cka import linear_cka`) inside
the training/eval stack rather than eagerly here.
"""
from .fingerprint import SoulFingerprint, aggregate_fingerprint

__all__ = ["SoulFingerprint", "aggregate_fingerprint"]
