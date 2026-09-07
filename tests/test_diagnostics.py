"""Tests for the degenerate-score guard.

The guard exists because a constant score column scores exactly 0.5 under both AUROC and
McClish AUC0.1 -- numerically identical to a genuinely random ranker. Without the guard a
broken scorer and a chance-level scorer are indistinguishable in the output.
"""

import warnings

import numpy as np
import pytest

from cognate.metrics import (
    DegenerateScoresError,
    auc01,
    auroc,
    diagnose_scores,
    evaluate,
)

GROUPS = np.array(["a"] * 6 + ["b"] * 6)
Y_TRUE = np.array([1, 0, 0, 1, 0, 0] * 2)


def test_constant_column_is_numerically_identical_to_random() -> None:
    """The motivating fact. Both are exactly 0.5, so the metric cannot tell them apart."""
    constant = np.zeros(12)
    assert auroc(Y_TRUE, constant) == pytest.approx(0.5)
    assert auc01(Y_TRUE, constant) == pytest.approx(0.5)


def test_constant_column_is_flagged() -> None:
    d = diagnose_scores(np.zeros(12), GROUPS)
    assert d.is_degenerate
    assert d.n_unique_scores == 1
    assert set(d.constant_groups) == {"a", "b"}


def test_one_constant_group_is_flagged() -> None:
    score = np.concatenate([np.linspace(0, 1, 6), np.zeros(6)])
    d = diagnose_scores(score, GROUPS)
    assert d.is_degenerate
    assert d.constant_groups == ("b",)
    assert d.n_constant_groups == 1


def test_healthy_scores_are_not_flagged() -> None:
    d = diagnose_scores(np.linspace(0, 1, 12), GROUPS)
    assert not d.is_degenerate
    assert d.n_unique_scores == 12
    assert d.constant_groups == ()


def test_non_finite_scores_are_flagged() -> None:
    score = np.linspace(0, 1, 12).copy()
    score[3] = np.nan
    score[7] = np.inf
    d = diagnose_scores(score, GROUPS)
    assert d.is_degenerate
    assert d.n_nonfinite == 2


def test_tie_heavy_is_reported_but_is_not_degeneracy() -> None:
    """Heavy ties cap achievable ranking without making the column unrankable."""
    score = np.array([0.0] * 5 + list(np.linspace(0.1, 1.0, 7)))
    d = diagnose_scores(score, GROUPS)
    assert d.is_tie_heavy
    assert not d.is_degenerate


def test_evaluate_warns_by_default() -> None:
    with pytest.warns(UserWarning, match="degenerate scores"):
        evaluate(Y_TRUE, np.zeros(12), GROUPS, n_boot=20, seed=0)


def test_evaluate_can_raise() -> None:
    with pytest.raises(DegenerateScoresError, match="indistinguishable from a random"):
        evaluate(Y_TRUE, np.zeros(12), GROUPS, n_boot=20, seed=0, on_degenerate="raise")


def test_evaluate_can_ignore() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = evaluate(
            Y_TRUE, np.zeros(12), GROUPS, n_boot=20, seed=0, on_degenerate="ignore"
        )
    assert report.macro_auc01.point == pytest.approx(0.5)
    assert report.diagnostics.is_degenerate


def test_healthy_scores_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = evaluate(Y_TRUE, np.linspace(0, 1, 12), GROUPS, n_boot=20, seed=0)
    assert not report.diagnostics.is_degenerate


def test_report_carries_diagnostics_into_its_string() -> None:
    with pytest.warns(UserWarning):
        report = evaluate(Y_TRUE, np.zeros(12), GROUPS, n_boot=20, seed=0)
    assert "DEGENERATE SCORES" in str(report)


def test_guard_catches_the_real_knn_unseen_slice() -> None:
    """The case this was built for: k-NN has no database for unseen peptides, so all
    1,066 of those rows take the default score and the 0.500 is arithmetic."""
    from cognate.baseline_knn import score_by_nearest_positive
    from cognate.data import load_solutions, load_train

    train, sol = load_train(), load_solutions()
    knn = score_by_nearest_positive(sol, train)
    is_seen = sol["Peptide"].isin(set(train["Peptide"])).to_numpy()

    unseen = diagnose_scores(knn.score[~is_seen], sol["Peptide"][~is_seen])
    assert unseen.is_degenerate
    assert unseen.n_constant_groups == 7

    seen = diagnose_scores(knn.score[is_seen], sol["Peptide"][is_seen])
    assert not seen.is_degenerate
