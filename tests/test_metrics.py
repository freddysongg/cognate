"""Hand-checked acceptance tests for the three scorers.

The 10-row toy example, worked by hand:

    y_true  = [1, 1, 0,   0,   0,   0,   0,   0,   0,   0   ]
    y_score = [.9, .5, .8, .7,  .6,  .4,  .3,  .2,  .1,  .05]

Ranked descending: P(.9) N(.8) N(.7) N(.6) P(.5) N(.4) N(.3) N(.2) N(.1) N(.05)
2 positives, 8 negatives.

AUROC  = correctly ordered (pos, neg) pairs / 16.
         P(.9) beats all 8 negatives; P(.5) beats 5. -> 13/16 = 0.8125

AUPRC  = sum over positives of precision * recall increment (sklearn average_precision).
         rank 1: precision 1/1, recall 0 -> 0.5  => 1.00 * 0.5
         rank 5: precision 2/5, recall 0.5 -> 1  => 0.40 * 0.5
         -> 0.5 + 0.2 = 0.7

AUC0.1 = partial ROC area to FPR 0.1, McClish-standardised.
         The ROC jumps to TPR 0.5 at FPR 0, then stays flat past FPR 0.1,
         so partial area = 0.5 * 0.1 = 0.05.
         min_area = 0.5 * 0.1^2 = 0.005, max_area = 0.1
         McClish: 0.5 * (1 + (0.05 - 0.005) / (0.1 - 0.005)) = 0.5 * (1 + 9/19) = 14/19
"""

import numpy as np
import pytest

from cognate.metrics import (
    auc01,
    auprc,
    auroc,
    evaluate,
    macro_auc01,
)

TOY_TRUE = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
TOY_SCORE = np.array([0.9, 0.5, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])

TOY_AUROC = 13 / 16
TOY_AUPRC = 0.7
TOY_AUC01 = 14 / 19

AUC01_FLOOR = 9 / 19


def test_auroc_matches_hand_computation() -> None:
    assert auroc(TOY_TRUE, TOY_SCORE) == pytest.approx(TOY_AUROC)


def test_auprc_matches_hand_computation() -> None:
    assert auprc(TOY_TRUE, TOY_SCORE) == pytest.approx(TOY_AUPRC)


def test_auc01_matches_hand_computation() -> None:
    assert auc01(TOY_TRUE, TOY_SCORE) == pytest.approx(TOY_AUC01)


def test_perfect_ranker_scores_one() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.linspace(1.0, 0.0, 10)
    assert auroc(y_true, y_score) == pytest.approx(1.0)
    assert auprc(y_true, y_score) == pytest.approx(1.0)
    assert auc01(y_true, y_score) == pytest.approx(1.0)


def test_random_ranker_scores_about_half() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=20_000)
    y_score = rng.random(20_000)
    assert auroc(y_true, y_score) == pytest.approx(0.5, abs=0.02)
    assert auc01(y_true, y_score) == pytest.approx(0.5, abs=0.03)


def test_auc01_floor_is_not_zero() -> None:
    """No positive inside the top 10% of negatives gives 9/19, not 0.

    McClish standardisation maps a partial area of 0 onto 0.5 * (1 - min/max-min).
    The metric therefore lives in [0.474, 1.0], and 0.47 is a floor, not a near-miss
    of random.
    """
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_score = np.linspace(1.0, 0.0, 10)
    assert auc01(y_true, y_score) == pytest.approx(AUC01_FLOOR)


def test_macro_averages_over_groups_not_rows() -> None:
    """Group A is the toy (14/19); group B is a perfect 5-row ranker (1.0).

    The macro score is the mean of the two per-group scores, 33/38, and is deliberately
    not weighted by group size -- group A has twice the rows.
    """
    y_true = np.concatenate([TOY_TRUE, [1, 0, 0, 0, 0]])
    y_score = np.concatenate([TOY_SCORE, [0.9, 0.8, 0.7, 0.6, 0.5]])
    groups = np.array(["A"] * 10 + ["B"] * 5)

    result = macro_auc01(y_true, y_score, groups)
    assert result.per_group["A"] == pytest.approx(TOY_AUC01)
    assert result.per_group["B"] == pytest.approx(1.0)
    assert result.value == pytest.approx(33 / 38)
    assert result.n_groups_scored == 2
    assert result.n_groups_skipped == 0


def test_single_class_groups_are_skipped_and_counted() -> None:
    y_true = np.concatenate([TOY_TRUE, [0, 0, 0], [1, 1]])
    y_score = np.concatenate([TOY_SCORE, [0.3, 0.2, 0.1], [0.9, 0.8]])
    groups = np.array(["A"] * 10 + ["all_neg"] * 3 + ["all_pos"] * 2)

    result = macro_auc01(y_true, y_score, groups)
    assert result.n_groups_scored == 1
    assert result.n_groups_skipped == 2
    assert set(result.skipped_groups) == {"all_neg", "all_pos"}
    assert result.value == pytest.approx(TOY_AUC01)


def test_macro_differs_from_global_auc01() -> None:
    """The distinction the task warns about: grouping changes the number."""
    y_true = np.concatenate([TOY_TRUE, [1, 0, 0, 0, 0]])
    y_score = np.concatenate([TOY_SCORE, [0.9, 0.8, 0.7, 0.6, 0.5]])
    groups = np.array(["A"] * 10 + ["B"] * 5)

    assert macro_auc01(y_true, y_score, groups).value != pytest.approx(
        auc01(y_true, y_score)
    )


def test_evaluate_point_estimates_match_direct_scorers() -> None:
    groups = np.array(["A"] * 10)
    report = evaluate(TOY_TRUE, TOY_SCORE, groups, n_boot=50, seed=0)
    assert report.macro_auc01.point == pytest.approx(TOY_AUC01)
    assert report.auroc.point == pytest.approx(TOY_AUROC)
    assert report.auprc.point == pytest.approx(TOY_AUPRC)
    assert report.n_rows == 10
    assert report.n_positive == 2


def test_bootstrap_is_reproducible_and_brackets_the_point() -> None:
    rng = np.random.default_rng(1)
    groups = np.repeat([f"p{i}" for i in range(8)], 40)
    y_true = rng.integers(0, 2, size=320)
    y_score = y_true * 0.5 + rng.random(320)

    a = evaluate(y_true, y_score, groups, n_boot=200, seed=7)
    b = evaluate(y_true, y_score, groups, n_boot=200, seed=7)
    c = evaluate(y_true, y_score, groups, n_boot=200, seed=8)

    assert a.macro_auc01 == b.macro_auc01
    assert a.macro_auc01.lo != c.macro_auc01.lo
    assert a.macro_auc01.lo <= a.macro_auc01.point <= a.macro_auc01.hi
    assert a.macro_auc01.n_replicates == 200


def test_group_resampling_widens_the_interval() -> None:
    """Peptide-level noise dominates when the average is over few peptides.

    Resampling rows within a fixed peptide set cannot see the variance introduced by
    which peptides landed in the test set, so it reports a narrower interval.
    """
    rng = np.random.default_rng(2)
    groups = np.repeat([f"p{i}" for i in range(8)], 40)
    y_true = rng.integers(0, 2, size=320)
    y_score = y_true * 0.5 + rng.random(320)

    rows_only = evaluate(y_true, y_score, groups, mode="rows", n_boot=300, seed=3)
    two_level = evaluate(y_true, y_score, groups, mode="both", n_boot=300, seed=3)
    assert two_level.macro_auc01.width > rows_only.macro_auc01.width


def test_subset_mask_scores_a_slice() -> None:
    y_true = np.concatenate([TOY_TRUE, [1, 0, 0, 0, 0]])
    y_score = np.concatenate([TOY_SCORE, [0.9, 0.8, 0.7, 0.6, 0.5]])
    groups = np.array(["A"] * 10 + ["B"] * 5)

    only_a = evaluate(
        y_true, y_score, groups, subset=groups == "A", label="A only", n_boot=50, seed=0
    )
    assert only_a.n_rows == 10
    assert only_a.label == "A only"
    assert only_a.macro_auc01.point == pytest.approx(TOY_AUC01)


def test_evaluate_without_groups_is_global() -> None:
    report = evaluate(TOY_TRUE, TOY_SCORE, n_boot=50, seed=0)
    assert report.n_groups_scored == 1
    assert report.macro_auc01.point == pytest.approx(auc01(TOY_TRUE, TOY_SCORE))


def test_empty_subset_raises() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        evaluate(TOY_TRUE, TOY_SCORE, subset=np.zeros(10, dtype=bool))
