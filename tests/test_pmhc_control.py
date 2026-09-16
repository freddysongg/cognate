"""Tests for the pMHC novelty control analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cognate.pmhc_control import score_per_allele_auc01

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
