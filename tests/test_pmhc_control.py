"""Tests for the pMHC novelty control analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cognate.pmhc_control import (
    PINNED_REFERENCE_SLOPE,
    DistanceSlope,
    classify_control_outcome,
    fit_distance_slope,
    load_reference_table,
    score_per_allele_auc01,
)

REFERENCE_SLOPE = PINNED_REFERENCE_SLOPE

ROOT = Path(__file__).resolve().parents[1]
PREDECLARATION = ROOT / "docs" / "pmhc" / "novelty_control_predeclaration.md"


def test_predeclaration_pins_the_reference_slope() -> None:
    text = PREDECLARATION.read_text(encoding="utf-8")
    assert "-1.0315" in text
    assert "0.1429" in text
    assert "[-1.3116, -0.7514]" in text


def test_predeclaration_names_all_three_outcome_literals() -> None:
    text = PREDECLARATION.read_text(encoding="utf-8")
    for literal in ("novelty_effect", "intrinsic_difficulty", "mixed"):
        assert literal in text


def test_predeclaration_forbids_absolute_accuracy_comparison() -> None:
    text = PREDECLARATION.read_text(encoding="utf-8")
    assert "only slopes are comparable" in text.lower()

COVERAGE_PATH = ROOT / "data" / "pmhc" / "mhcflurry_allele_coverage.json"


def test_coverage_artifact_partitions_the_frozen_cohort() -> None:
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    assert set(coverage) == {"supported", "unsupported", "mhcflurry_version"}
    supported = set(coverage["supported"])
    unsupported = set(coverage["unsupported"])
    assert not supported & unsupported
    assert len(supported | unsupported) == 47


def test_score_per_allele_auc01_is_perfect_when_scores_rank_targets_first() -> None:
    predictions = pd.DataFrame(
        {
            "Allele": ["HLA-A01:01"] * 4 + ["HLA-B07:02"] * 4,
            "Target": [True, True, False, False] * 2,
            "Score": [0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1],
        }
    )
    scored = score_per_allele_auc01(predictions)
    assert set(scored) == {"HLA-A01:01", "HLA-B07:02"}
    assert scored["HLA-A01:01"] == pytest.approx(1.0)


def test_score_per_allele_auc01_rejects_an_allele_missing_a_class() -> None:
    predictions = pd.DataFrame(
        {
            "Allele": ["HLA-A01:01"] * 3,
            "Target": [True, True, True],
            "Score": [0.9, 0.8, 0.7],
        }
    )
    with pytest.raises(ValueError, match="single-class allele"):
        score_per_allele_auc01(predictions)


def test_score_per_allele_auc01_reflects_the_ten_percent_fpr_cap() -> None:
    """A frame with enough negatives that the 10% FPR cap has real operating points.

    The degenerate four-row case above yields 1.0 under any max_fpr, so it cannot
    detect a full-AUC computation substituted for the partial one. This case can:
    the same input scores 0.75 under an uncapped AUROC.
    """
    predictions = pd.DataFrame(
        {
            "Allele": ["HLA-A02:01"] * 25,
            "Target": [True] * 5 + [False] * 20,
            "Score": (
                [0.99, 0.98, 0.97, 0.50, 0.40]
                + [0.96, 0.95]
                + [0.60] * 8
                + [0.45] * 5
                + [0.10] * 5
            ),
        }
    )
    scored = score_per_allele_auc01(predictions)
    assert scored["HLA-A02:01"] == pytest.approx(0.7894736842105263)


def test_fit_distance_slope_recovers_a_known_line() -> None:
    distances = [0.1, 0.2, 0.3, 0.4]
    scores = [0.8, 0.6, 0.4, 0.2]
    fitted = fit_distance_slope(distances, scores)
    assert fitted.slope == pytest.approx(-2.0)
    assert fitted.intercept == pytest.approx(1.0)
    assert fitted.n == 4


def test_classify_returns_novelty_effect_for_a_flat_control() -> None:
    flat = DistanceSlope(
        slope=0.01, stderr=0.05, ci_lo=-0.09, ci_hi=0.11,
        intercept=0.9, r_squared=0.0, n=47,
    )
    assert classify_control_outcome(REFERENCE_SLOPE, flat) == "novelty_effect"


def test_classify_returns_intrinsic_difficulty_when_control_matches_reference() -> None:
    steep = DistanceSlope(
        slope=-1.00, stderr=0.15, ci_lo=-1.30, ci_hi=-0.70,
        intercept=0.85, r_squared=0.5, n=47,
    )
    assert classify_control_outcome(REFERENCE_SLOPE, steep) == "intrinsic_difficulty"


def test_classify_returns_mixed_when_control_is_steep_but_shallower() -> None:
    partial = DistanceSlope(
        slope=-0.45, stderr=0.08, ci_lo=-0.61, ci_hi=-0.29,
        intercept=0.88, r_squared=0.3, n=47,
    )
    assert classify_control_outcome(REFERENCE_SLOPE, partial) == "mixed"


def test_outcome_is_invariant_to_a_constant_offset_in_control_scores() -> None:
    """The slopes-only constraint, enforced structurally.

    MHCflurry's training data overlaps these test rows, so its absolute AUC0.1 is partly
    memorization and is not comparable to ours. Shifting every control score by a constant
    must not change the outcome. This fails if the classifier is ever rewritten to consume
    absolute performance.
    """
    distances = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    scores = [0.80, 0.76, 0.71, 0.67, 0.62, 0.58]
    shifted = [score + 0.15 for score in scores]
    baseline = classify_control_outcome(REFERENCE_SLOPE, fit_distance_slope(distances, scores))
    offset = classify_control_outcome(REFERENCE_SLOPE, fit_distance_slope(distances, shifted))
    assert baseline == offset


ALLELE_ONLY_RESULTS = ROOT / "data" / "pmhc" / "loao_allele_only_results.json"


def test_load_reference_table_reproduces_the_pinned_reference_slope() -> None:
    table = load_reference_table(ALLELE_ONLY_RESULTS, "pseudo_sequence_mlp")
    assert len(table) == 47
    assert list(table.columns) == ["Allele", "Distance", "Auc01", "NRows", "NPositive"]
    fitted = fit_distance_slope(table["Distance"], table["Auc01"])
    assert fitted.slope == pytest.approx(REFERENCE_SLOPE, abs=5e-4)
    assert fitted.stderr == pytest.approx(0.1429, abs=5e-4)


CONTROL_RESULTS = ROOT / "data" / "pmhc" / "novelty_control_results.json"


def test_control_artifact_has_the_declared_top_level_contract() -> None:
    artifact = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    assert set(artifact) == {
        "config",
        "coverage",
        "reference",
        "control",
        "outcome",
        "support_nulls",
    }


def test_control_artifact_outcome_is_one_of_the_frozen_literals() -> None:
    artifact = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    assert artifact["outcome"] in {"novelty_effect", "intrinsic_difficulty", "mixed"}


def test_control_artifact_carries_no_cross_predictor_accuracy_comparison() -> None:
    artifact = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    flat = json.dumps(artifact).lower()
    for forbidden in ("auc01_difference", "accuracy_delta", "beats", "outperforms"):
        assert forbidden not in flat
