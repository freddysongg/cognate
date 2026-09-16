"""Tests for the pMHC novelty control analysis."""

from __future__ import annotations

from pathlib import Path

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
