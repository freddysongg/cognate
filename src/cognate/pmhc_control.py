"""Novelty-control analysis for the pMHC transfer gradient.

Pure analysis over committed artifacts and one predictions frame. Imports nothing from
TensorFlow or torch, reaches no network, and never writes to a shipped artifact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from cognate.metrics import auc01

_PREDICTION_COLUMNS = frozenset({"Allele", "Target", "Score"})
CONFIDENCE_MULTIPLIER = 1.96
PINNED_REFERENCE_SLOPE = -1.0315

ControlOutcome = Literal["novelty_effect", "intrinsic_difficulty", "mixed"]


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


@dataclass(frozen=True)
class DistanceSlope:
    """An OLS fit of per-allele AUC0.1 on pseudo-sequence distance."""

    slope: float
    stderr: float
    ci_lo: float
    ci_hi: float
    intercept: float
    r_squared: float
    n: int


def fit_distance_slope(
    distances: Sequence[float], scores: Sequence[float]
) -> DistanceSlope:
    """Fit the pre-declared linear relation of AUC0.1 on distance."""
    x = np.asarray(distances, dtype=float)
    y = np.asarray(scores, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"length mismatch: {x.shape} distances, {y.shape} scores")
    if x.size < 3:
        raise ValueError(f"need at least three alleles to fit a slope, got {x.size}")
    if np.unique(x).size < 2:
        raise ValueError("distance axis is constant; no slope is identifiable")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - y.mean()) ** 2))
    degrees_of_freedom = x.size - 2
    stderr = float(
        np.sqrt(residual_sum / degrees_of_freedom / np.sum((x - x.mean()) ** 2))
    )
    return DistanceSlope(
        slope=float(slope),
        stderr=stderr,
        ci_lo=float(slope) - CONFIDENCE_MULTIPLIER * stderr,
        ci_hi=float(slope) + CONFIDENCE_MULTIPLIER * stderr,
        intercept=float(intercept),
        r_squared=0.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum,
        n=int(x.size),
    )


def classify_control_outcome(
    reference_slope: float, control: DistanceSlope
) -> ControlOutcome:
    """Apply the frozen decision rule from the pre-declaration, in its stated order."""
    is_flat = control.ci_lo <= 0.0 <= control.ci_hi
    if is_flat and reference_slope < control.ci_lo:
        return "novelty_effect"
    if control.ci_lo <= reference_slope <= control.ci_hi:
        return "intrinsic_difficulty"
    return "mixed"


def _require_field(mapping: dict[str, object], field: str, allele: str, source: str) -> object:
    """Look up `field` in `mapping`, raising the same contextual error as the arm check above."""
    if field not in mapping:
        raise ValueError(f"{source} missing {field!r} for {allele}")
    return mapping[field]


def load_reference_table(results_path: Path, arm: str) -> pd.DataFrame:
    """Per-allele distance and AUC0.1 for one arm of a committed LOAO artifact."""
    artifact = json.loads(results_path.read_text(encoding="utf-8"))
    per_allele = artifact["per_allele"]
    diagnostics = artifact["diagnostics"]
    records = []
    for allele in sorted(per_allele):
        arms = per_allele[allele]["arms"]
        if arm not in arms:
            raise ValueError(f"arm {arm!r} absent for {allele}; have {sorted(arms)}")
        if allele not in diagnostics:
            raise ValueError(f"diagnostics missing entry for {allele}")
        allele_diagnostics = diagnostics[allele]
        allele_summary = per_allele[allele]
        records.append(
            {
                "Allele": allele,
                "Distance": _require_field(
                    allele_diagnostics, "nearest_retained_pseudo_distance", allele, "diagnostics"
                ),
                "Auc01": _require_field(arms[arm], "auc01", allele, f"arm {arm!r}"),
                "NRows": _require_field(allele_summary, "n_rows", allele, "per_allele"),
                "NPositive": _require_field(allele_summary, "n_positive", allele, "per_allele"),
            }
        )
    return pd.DataFrame.from_records(records)
