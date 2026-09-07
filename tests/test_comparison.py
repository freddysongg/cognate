"""Tests for the paired bootstrap comparison."""

import numpy as np
import pytest

from cognate.metrics import compare_macro_auc01


def _fixture(n_peptides: int = 6, per_peptide: int = 60):
    rng = np.random.default_rng(0)
    groups = np.repeat([f"p{i}" for i in range(n_peptides)], per_peptide)
    y_true = rng.integers(0, 2, size=n_peptides * per_peptide)
    return y_true, groups, rng


def test_identical_arms_give_zero_difference() -> None:
    y_true, groups, rng = _fixture()
    score = y_true * 0.4 + rng.random(len(y_true))
    result = compare_macro_auc01(
        y_true, groups, ("a", score, None), ("b", score, None), n_boot=200, seed=0
    )
    assert result.difference.point == pytest.approx(0.0)
    assert result.p_two_sided == pytest.approx(1.0)


def test_better_arm_gives_a_positive_difference() -> None:
    y_true, groups, rng = _fixture()
    good = y_true * 1.5 + rng.random(len(y_true))
    bad = rng.random(len(y_true))
    result = compare_macro_auc01(
        y_true, groups, ("good", good, None), ("bad", bad, None), n_boot=300, seed=0
    )
    assert result.difference.point > 0.1
    assert result.p_two_sided < 0.05
    assert result.difference.lo > 0


def test_pairing_is_tighter_than_marginal_intervals() -> None:
    """The reason this function exists: shared peptide noise cancels inside a replicate."""
    from cognate.metrics import evaluate

    y_true, groups, rng = _fixture()
    a = y_true * 0.6 + rng.random(len(y_true))
    b = a + rng.normal(0, 0.05, len(y_true))

    paired = compare_macro_auc01(
        y_true, groups, ("a", a, None), ("b", b, None), n_boot=400, seed=0
    )
    marginal_a = evaluate(y_true, a, groups, n_boot=400, seed=0)
    marginal_b = evaluate(y_true, b, groups, n_boot=400, seed=0)
    naive_width = marginal_a.macro_auc01.width + marginal_b.macro_auc01.width
    assert paired.difference.width < naive_width


def test_disjoint_slices_raise_with_a_useful_message() -> None:
    """Seen vs unseen cannot be paired -- they share no peptide."""
    y_true, groups, rng = _fixture()
    score = rng.random(len(y_true))
    first_half = np.isin(groups, ["p0", "p1", "p2"])
    with pytest.raises(ValueError, match="Disjoint slices"):
        compare_macro_auc01(
            y_true,
            groups,
            ("a", score, first_half),
            ("b", score, ~first_half),
            n_boot=50,
            seed=0,
        )


def test_comparison_is_reproducible() -> None:
    y_true, groups, rng = _fixture()
    a = y_true * 0.6 + rng.random(len(y_true))
    b = rng.random(len(y_true))
    kwargs = {"n_boot": 200, "seed": 5}
    first = compare_macro_auc01(y_true, groups, ("a", a, None), ("b", b, None), **kwargs)
    second = compare_macro_auc01(y_true, groups, ("a", a, None), ("b", b, None), **kwargs)
    assert first.difference == second.difference


def test_comparison_is_symmetric_under_swapping_arms() -> None:
    """Pairing means arms over the same rows share a resample, so swapping the arms
    negates the difference and leaves the magnitude and p-value untouched.

    Before this was enforced the two orderings drew independent row resamples and
    disagreed by a factor of three on p for the project's headline comparison.
    """
    y_true, groups, rng = _fixture()
    a = y_true * 0.6 + rng.random(len(y_true))
    b = rng.random(len(y_true))

    forward = compare_macro_auc01(
        y_true, groups, ("a", a, None), ("b", b, None), n_boot=400, seed=0
    )
    reverse = compare_macro_auc01(
        y_true, groups, ("b", b, None), ("a", a, None), n_boot=400, seed=0
    )
    assert forward.difference.point == pytest.approx(-reverse.difference.point)
    assert forward.difference.lo == pytest.approx(-reverse.difference.hi)
    assert forward.difference.hi == pytest.approx(-reverse.difference.lo)
    assert forward.p_two_sided == pytest.approx(reverse.p_two_sided)
