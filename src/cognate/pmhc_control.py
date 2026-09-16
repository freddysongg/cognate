"""Novelty-control analysis for the pMHC transfer gradient.

Pure analysis over committed artifacts and one predictions frame. Imports nothing from
TensorFlow or torch, reaches no network, and never writes to a shipped artifact.
"""

from __future__ import annotations

import pandas as pd

from cognate.metrics import auc01

_PREDICTION_COLUMNS = frozenset({"Allele", "Target", "Score"})


def score_per_allele_auc01(predictions: pd.DataFrame) -> dict[str, float]:
    """Standardized AUC0.1 per allele over a predictions frame."""
    missing = _PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions frame missing columns: {sorted(missing)}")
    scored: dict[str, float] = {}
    for allele, group in predictions.groupby("Allele", sort=True):
        labels = group["Target"].to_numpy(dtype=bool)
        if labels.all() or not labels.any():
            raise ValueError(f"single-class allele cannot be scored: {allele}")
        scored[str(allele)] = auc01(labels, group["Score"].to_numpy(dtype=float))
    return scored
