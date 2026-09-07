"""Acceptance tests for the nearest-neighbour baseline."""

import numpy as np
import pandas as pd
import pytest

from cognate.baseline_knn import (
    build_database,
    exact_match_mask,
    score_by_nearest_positive,
)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Peptide": ["AAAAAAAAA", "AAAAAAAAA", "AAAAAAAAA", "CCCCCCCCC"],
            "CDR3b": ["CASSLGQ", "CASSLGF", "CASSLGQ", "CWWWWWW"],
            "Target": [1, 1, 1, 1],
        }
    )


def _test_rows(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Peptide": [p for p, _ in pairs], "CDR3b": [c for _, c in pairs]}
    )


def test_database_holds_distinct_positives_per_peptide() -> None:
    database = build_database(_train())
    assert set(database) == {"AAAAAAAAA", "CCCCCCCCC"}
    assert list(database["AAAAAAAAA"]) == ["CASSLGF", "CASSLGQ"]


def test_database_ignores_negative_rows() -> None:
    """Row 1 is the only occurrence of CASSLGF; CASSLGQ appears twice, so zeroing one
    copy of it would not remove it."""
    train = _train()
    train.loc[1, "Target"] = 0
    assert list(build_database(train)["AAAAAAAAA"]) == ["CASSLGQ"]


def test_exact_match_scores_one() -> None:
    result = score_by_nearest_positive(
        _test_rows([("AAAAAAAAA", "CASSLGQ")]), _train()
    )
    assert result.score[0] == pytest.approx(1.0)
    assert result.nearest_sequence[0] == "CASSLGQ"


def test_database_is_per_peptide_not_global() -> None:
    """A TCR that exactly matches a binder of a *different* peptide gets no credit.

    This is the property that makes the baseline a lookup table rather than a similarity
    model, and the reason unseen peptides are unscorable.
    """
    result = score_by_nearest_positive(
        _test_rows([("CCCCCCCCC", "CASSLGQ")]), _train()
    )
    assert result.score[0] < 1.0
    assert result.nearest_sequence[0] == "CWWWWWW"


def test_missing_peptide_takes_the_default_score() -> None:
    result = score_by_nearest_positive(
        _test_rows([("DDDDDDDDD", "CASSLGQ")]), _train()
    )
    assert result.score[0] == 0.0
    assert not result.has_database[0]
    assert result.n_without_database == 1


def test_constant_scores_give_exactly_half() -> None:
    """A peptide with no database yields one constant column, and AUC0.1 of a constant
    column is exactly 0.5. The unseen-slice result is arithmetic, not measurement."""
    from cognate.metrics import auc01

    y_true = np.array([1, 0, 0, 1, 0, 0])
    assert auc01(y_true, np.zeros(6)) == pytest.approx(0.5)


def test_leave_out_exact_matches_falls_back_to_second_nearest() -> None:
    rows = _test_rows([("AAAAAAAAA", "CASSLGQ")])
    plain = score_by_nearest_positive(rows, _train())
    held = score_by_nearest_positive(rows, _train(), leave_out_exact_matches=True)

    assert plain.score[0] == pytest.approx(1.0)
    assert held.score[0] < 1.0
    assert held.nearest_sequence[0] == "CASSLGF"


def test_leave_out_exact_matches_keeps_the_row_scorable() -> None:
    """Distinct from dropping the row: the row stays in the denominator."""
    rows = _test_rows([("AAAAAAAAA", "CASSLGQ"), ("AAAAAAAAA", "CASSXYZ")])
    held = score_by_nearest_positive(rows, _train(), leave_out_exact_matches=True)
    assert len(held.score) == 2
    assert np.isfinite(held.score).all()


def test_database_size_is_reported() -> None:
    result = score_by_nearest_positive(
        _test_rows([("AAAAAAAAA", "CASSLGQ"), ("CCCCCCCCC", "CWWWWWW")]), _train()
    )
    assert list(result.database_size) == [2, 1]


def test_exact_match_mask_requires_both_fields_to_match() -> None:
    rows = _test_rows(
        [
            ("AAAAAAAAA", "CASSLGQ"),
            ("CCCCCCCCC", "CASSLGQ"),
            ("AAAAAAAAA", "CASSXYZ"),
        ]
    )
    assert list(exact_match_mask(rows, _train())) == [True, False, False]


def test_score_ordering_prefers_closer_sequences() -> None:
    rows = _test_rows(
        [("AAAAAAAAA", "CASSLGQ"), ("AAAAAAAAA", "CASSLGW"), ("AAAAAAAAA", "WWWWWWW")]
    )
    scores = score_by_nearest_positive(rows, _train()).score
    assert scores[0] > scores[1] > scores[2]


def test_macro_auc01_is_invariant_to_within_peptide_rescaling() -> None:
    """Macro AUC0.1 cannot see a per-peptide level shift -- it is computed per peptide
    and AUC is rank-based, so any strictly monotonic within-peptide transform leaves
    every per-peptide value unchanged. Database saturation therefore cannot inflate this
    metric, only pooled ones.
    """
    import pandas as pd

    from cognate.metrics import auroc, macro_auc01

    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c"], 40)
    y_true = rng.integers(0, 2, size=120)
    score = y_true * 0.3 + rng.random(120)
    # give each group a wildly different level and spread
    score = score + np.repeat([0.0, 5.0, 20.0], 40) * np.repeat([1.0, 1.0, 1.0], 40)

    rescaled = score.copy()
    ranked = score.copy()
    for name in ["a", "b", "c"]:
        mask = groups == name
        values = score[mask]
        rescaled[mask] = (values - values.mean()) / values.std()
        ranked[mask] = pd.Series(values).rank().to_numpy() / mask.sum()

    baseline = macro_auc01(y_true, score, groups).value
    assert macro_auc01(y_true, rescaled, groups).value == pytest.approx(baseline, abs=1e-12)
    assert macro_auc01(y_true, ranked, groups).value == pytest.approx(baseline, abs=1e-12)
    assert auroc(y_true, rescaled) != pytest.approx(auroc(y_true, score))


def test_top_k_changes_the_within_peptide_ranking() -> None:
    """Unlike rescaling, averaging the k nearest is not a monotone transform of the max,
    so it can move per-peptide AUC0.1."""
    train = _train()
    rows = _test_rows([("AAAAAAAAA", "CASSLGQ"), ("AAAAAAAAA", "CASSLGF")])
    top1 = score_by_nearest_positive(rows, train, top_k=1).score
    top2 = score_by_nearest_positive(rows, train, top_k=2).score
    assert not np.allclose(top1, top2)
    assert (top2 <= top1).all()


def test_top_k_falls_back_when_database_is_smaller_than_k() -> None:
    rows = _test_rows([("CCCCCCCCC", "CWWWWWW")])
    result = score_by_nearest_positive(rows, _train(), top_k=10)
    assert result.score[0] == pytest.approx(1.0)


def test_top_k_rejects_zero() -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        score_by_nearest_positive(_test_rows([("AAAAAAAAA", "CASSLGQ")]), _train(), top_k=0)
