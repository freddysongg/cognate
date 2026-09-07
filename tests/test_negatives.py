"""Acceptance tests for the negative samplers."""

import numpy as np
import pandas as pd
import pytest

from cognate.negatives import (
    STRATEGIES,
    make_negatives,
    make_training_set,
    summarise,
)

PEPTIDES = ["AAAAAAAAA", "AAAAAAAAV", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"]
HLAS = ["HLA-A*01:01", "HLA-A*02:01", "HLA-B*07:02", "HLA-B*08:01", "HLA-A*11:01"]


def _toy_positives() -> pd.DataFrame:
    """Six TCRs across five peptides. CDR3b 'CASSX' is cross-reactive to two peptides."""
    rows = [
        ("AAAAAAAAA", "CASSA"),
        ("AAAAAAAAV", "CASSB"),
        ("CCCCCCCCC", "CASSC"),
        ("DDDDDDDDD", "CASSD"),
        ("EEEEEEEEE", "CASSE"),
        ("AAAAAAAAA", "CASSX"),
        ("CCCCCCCCC", "CASSX"),
    ]
    hla = dict(zip(PEPTIDES, HLAS))
    return pd.DataFrame(
        {
            "Peptide": [p for p, _ in rows],
            "HLA": [hla[p] for p, _ in rows],
            "CDR3b": [c for _, c in rows],
            "CDR3a": [f"CAV{c[-1]}" for _, c in rows],
            "Target": 1,
        }
    )


def _wide_positives() -> pd.DataFrame:
    """Twelve peptides, one TCR each -- enough headroom for a ratio of 5."""
    peptides = [f"{c * 8}{c}" for c in "ABCDEFGHIKLM"]
    return pd.DataFrame(
        {
            "Peptide": peptides,
            "HLA": ["HLA-A*02:01"] * len(peptides),
            "CDR3b": [f"CASS{c}" for c in "ABCDEFGHIKLM"],
            "CDR3a": [f"CAV{c}" for c in "ABCDEFGHIKLM"],
            "Target": 1,
        }
    )


@pytest.mark.parametrize("strategy", ["shuffle", "hard"])
def test_ratio_is_respected(strategy: str) -> None:
    positives = _toy_positives()
    negatives = make_negatives(positives, strategy, ratio=2.0, seed=0)
    assert len(negatives) == 2 * len(positives)
    assert (negatives["Target"] == 0).all()
    assert list(negatives.columns) == list(positives.columns)


@pytest.mark.parametrize("strategy", ["shuffle", "hard"])
def test_same_seed_reproduces_exactly(strategy: str) -> None:
    positives = _toy_positives()
    a = make_negatives(positives, strategy, ratio=2.0, seed=42)
    b = make_negatives(positives, strategy, ratio=2.0, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_shuffle_but_not_hard() -> None:
    """`hard` is deterministic by construction: nearest neighbours, ties broken by name."""
    positives = _toy_positives()
    shuffle_a = make_negatives(positives, "shuffle", ratio=2.0, seed=1)
    shuffle_b = make_negatives(positives, "shuffle", ratio=2.0, seed=2)
    assert not shuffle_a["Peptide"].equals(shuffle_b["Peptide"])

    hard_a = make_negatives(positives, "hard", ratio=2.0, seed=1)
    hard_b = make_negatives(positives, "hard", ratio=2.0, seed=2)
    pd.testing.assert_frame_equal(hard_a, hard_b)


@pytest.mark.parametrize("strategy", ["shuffle", "hard", "uniform"])
def test_no_generated_negative_is_a_known_positive(strategy: str) -> None:
    """The cognate set covers every peptide a CDR3b binds anywhere, not just its own row.

    'CASSX' binds both AAAAAAAAA and CCCCCCCCC, so neither may appear as its negative.
    """
    positives = _toy_positives()
    negatives = make_negatives(positives, strategy, ratio=2.0, seed=0)
    true_pairs = set(zip(positives["Peptide"], positives["CDR3b"]))
    generated = set(zip(negatives["Peptide"], negatives["CDR3b"]))
    assert generated & true_pairs == set()

    cross_reactive = negatives[negatives["CDR3b"] == "CASSX"]["Peptide"]
    assert not cross_reactive.isin(["AAAAAAAAA", "CCCCCCCCC"]).any()


def test_hla_travels_with_the_swapped_peptide() -> None:
    positives = _toy_positives()
    expected = dict(zip(PEPTIDES, HLAS))
    for strategy in ["shuffle", "hard"]:
        negatives = make_negatives(positives, strategy, ratio=2.0, seed=0)
        assert [expected[p] for p in negatives["Peptide"]] == list(negatives["HLA"])


def test_hard_picks_the_nearest_peptide_first() -> None:
    """AAAAAAAAA and AAAAAAAAV differ by one substitution; everything else is distance 9."""
    positives = _toy_positives()
    negatives = make_negatives(positives, "hard", ratio=1.0, seed=0)
    first_for_a = negatives[negatives["CDR3b"] == "CASSA"]["Peptide"].iloc[0]
    assert first_for_a == "AAAAAAAAV"

    first_for_b = negatives[negatives["CDR3b"] == "CASSB"]["Peptide"].iloc[0]
    assert first_for_b == "AAAAAAAAA"


def test_shuffle_and_hard_disagree() -> None:
    """The two strategies must produce materially different datasets, or the ablation
    in T4-T6 is measuring nothing."""
    positives = _toy_positives()
    shuffled = make_negatives(positives, "shuffle", ratio=2.0, seed=0)
    hard = make_negatives(positives, "hard", ratio=2.0, seed=0)
    overlap = set(zip(shuffled["Peptide"], shuffled["CDR3b"])) & set(
        zip(hard["Peptide"], hard["CDR3b"])
    )
    assert len(overlap) < len(hard)


def test_negatives_keep_source_row_index() -> None:
    positives = _toy_positives()
    negatives = make_negatives(positives, "shuffle", ratio=3.0, seed=0)
    assert set(negatives.index) <= set(positives.index)
    assert negatives.index.value_counts().unique().tolist() == [3]


def test_fractional_ratio_is_approximate() -> None:
    positives = pd.concat([_toy_positives()] * 40, ignore_index=True)
    negatives = make_negatives(positives, "shuffle", ratio=1.5, seed=0)
    assert 1.3 < len(negatives) / len(positives) < 1.7


def test_uniform_covers_every_non_cognate_pair() -> None:
    positives = _toy_positives()
    negatives = make_negatives(positives, "uniform", seed=0)
    n_tcrs = positives["CDR3b"].nunique()
    n_cognate_pairs = positives[["Peptide", "CDR3b"]].drop_duplicates().shape[0]
    assert len(negatives) == n_tcrs * len(PEPTIDES) - n_cognate_pairs


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        make_negatives(_toy_positives(), "nearest", ratio=1.0)  # type: ignore[arg-type]


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        make_negatives(pd.DataFrame({"Peptide": ["AAAAAAAAA"]}), "shuffle")


def test_training_set_is_labelled_and_balanced_to_ratio() -> None:
    """Needs a peptide pool big enough to supply 5 distinct non-cognates per TCR."""
    positives = _wide_positives()
    combined = make_training_set(positives, "shuffle", ratio=5.0, seed=0)
    assert len(combined) == 6 * len(positives)
    assert combined["Target"].mean() == pytest.approx(1 / 6)


def test_shortfall_when_the_peptide_pool_is_too_small() -> None:
    """A TCR cannot receive more negatives than there are non-cognate peptides.

    The toy set has 5 peptides, so a TCR with one cognate caps at 4 negatives and the
    cross-reactive 'CASSX' caps at 3. The sampler takes what is available rather than
    sampling with replacement, which would create duplicate rows.
    """
    positives = _toy_positives()
    negatives = make_negatives(positives, "shuffle", ratio=5.0, seed=0)
    assert len(negatives) == 5 * 4 + 2 * 3
    per_source = negatives.groupby(negatives.index).size()
    assert per_source.max() == 4


def test_summarise_reports_no_collisions() -> None:
    positives = _toy_positives()
    stats = summarise(make_negatives(positives, "shuffle", ratio=2.0, seed=0), positives)
    assert stats["collisions_with_positives"] == 0
    assert stats["realised_ratio"] == pytest.approx(2.0)


def test_every_registered_strategy_is_callable() -> None:
    assert set(STRATEGIES) == {"shuffle", "matched", "hard", "uniform"}
    assert all(callable(fn) for fn in STRATEGIES.values())


# --- matched strategy (Session 5 correction) ---------------------------------------


def test_matched_reproduces_the_positive_peptide_marginal() -> None:
    """The bug `matched` fixes: uniform sampling makes peptide identity predictive.

    With a skewed positive distribution, uniform negatives leave the frequent peptides
    hugely over-represented among positives, so a model can score on peptide identity
    alone without learning anything about binding.
    """
    # 12 peptides so sampling has real freedom; one dominates the positives.
    letters = "ACDEFGHIKLMN"
    peptides = [letters[0] * 9] * 60 + [c * 9 for c in letters[1:] for _ in range(4)]
    positives = pd.DataFrame(
        {
            "Peptide": peptides,
            "HLA": ["HLA-A*02:01"] * len(peptides),
            "CDR3b": [f"CASS{i}" for i in range(len(peptides))],
            "CDR3a": [f"CAV{i}" for i in range(len(peptides))],
            "Target": 1,
        }
    )
    positive_share = positives["Peptide"].value_counts(normalize=True)

    uniform = make_negatives(positives, "shuffle", ratio=2.0, seed=0)
    matched = make_negatives(positives, "matched", ratio=2.0, seed=0)
    uniform_share = uniform["Peptide"].value_counts(normalize=True)
    matched_share = matched["Peptide"].value_counts(normalize=True)

    dominant = "AAAAAAAAA"  # letters[0] * 9
    uniform_gap = abs(positive_share[dominant] - uniform_share.get(dominant, 0.0))
    matched_gap = abs(positive_share[dominant] - matched_share.get(dominant, 0.0))
    assert matched_gap < uniform_gap


def test_matched_shrinks_the_peptide_identity_shortcut() -> None:
    """Measured on the real training set, not a toy."""
    from sklearn.linear_model import LogisticRegression

    from cognate.data import load_train
    from cognate.metrics import auroc

    train = load_train()
    scores = {}
    for strategy in ["shuffle", "matched"]:
        rows = pd.concat(
            [train, make_negatives(train, strategy, 5.0, 0)], ignore_index=True
        )
        design = pd.get_dummies(rows["Peptide"]).to_numpy(dtype=np.float32)
        labels = rows["Target"].to_numpy()
        fitted = LogisticRegression(max_iter=200).fit(design, labels)
        scores[strategy] = auroc(labels, fitted.predict_proba(design)[:, 1])

    assert scores["shuffle"] > 0.9
    assert scores["matched"] < 0.65


def test_matched_still_excludes_cognates() -> None:
    positives = _toy_positives()
    negatives = make_negatives(positives, "matched", ratio=2.0, seed=0)
    true_pairs = set(zip(positives["Peptide"], positives["CDR3b"]))
    assert set(zip(negatives["Peptide"], negatives["CDR3b"])) & true_pairs == set()


def test_matched_is_reproducible() -> None:
    positives = _wide_positives()
    a = make_negatives(positives, "matched", ratio=3.0, seed=11)
    b = make_negatives(positives, "matched", ratio=3.0, seed=11)
    pd.testing.assert_frame_equal(a, b)
    c = make_negatives(positives, "matched", ratio=3.0, seed=12)
    assert not a["Peptide"].reset_index(drop=True).equals(c["Peptide"].reset_index(drop=True))


def test_matched_is_the_default_strategy() -> None:
    positives = _wide_positives()
    pd.testing.assert_frame_equal(
        make_negatives(positives, ratio=2.0, seed=0),
        make_negatives(positives, "matched", ratio=2.0, seed=0),
    )
