"""Tests for leakage-free component splitting."""

import numpy as np
import pandas as pd
import pytest

from cognate.split import Split, component_labels, component_split


def _positives() -> pd.DataFrame:
    """Three islands plus one bridge.

    P1-P2 are joined because TCR 'B' binds both, so any split that separates P1 from P2
    would put TCR 'B' on both sides.
    """
    rows = [
        ("P1", "A"),
        ("P1", "B"),
        ("P2", "B"),
        ("P2", "C"),
        ("P3", "D"),
        ("P4", "E"),
        ("P4", "F"),
    ]
    return pd.DataFrame(
        {
            "Peptide": [p for p, _ in rows],
            "CDR3b": [c for _, c in rows],
            "HLA": ["HLA-A*02:01"] * len(rows),
            "Target": 1,
        }
    )


def test_shared_tcr_merges_two_peptides_into_one_component() -> None:
    labels = component_labels(_positives())
    assert labels[0] == labels[2], "P1 and P2 share TCR B and must not be separable"
    assert labels[4] != labels[0]
    assert len(set(labels)) == 3


def test_split_never_shares_a_peptide_or_a_tcr() -> None:
    split = component_split(_positives(), validation_fraction=0.3, seed=0)
    assert split.shared_peptides() == set()
    assert split.shared_sequences() == set()


def test_split_is_reproducible() -> None:
    a = component_split(_positives(), validation_fraction=0.3, seed=3)
    b = component_split(_positives(), validation_fraction=0.3, seed=3)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.validation, b.validation)


def test_largest_component_is_pinned_to_training() -> None:
    """It holds 71.8% of the real training rows; splitting it is impossible and putting
    it in validation would leave nothing to train on."""
    positives = _positives()
    labels = component_labels(positives)
    largest = pd.Series(labels).value_counts().idxmax()
    largest_peptides = set(positives["Peptide"][labels == largest])

    split = component_split(positives, validation_fraction=0.5, seed=0)
    assert largest_peptides <= set(split.train["Peptide"])


def test_rows_are_conserved() -> None:
    positives = _positives()
    split = component_split(positives, validation_fraction=0.3, seed=0)
    assert len(split.train) + len(split.validation) == len(positives)
    assert "_component" not in split.train.columns


def test_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError, match="must be in"):
        component_split(_positives(), validation_fraction=0.0)


def test_real_split_is_clean_and_hits_its_target() -> None:
    from cognate.data import load_train

    split = component_split(load_train(), validation_fraction=0.2, seed=0)
    assert split.shared_peptides() == set()
    assert split.shared_sequences() == set()
    assert 0.18 < split.validation_fraction < 0.22
    assert split.n_components == 684


def test_peptide_only_split_would_leak_by_comparison() -> None:
    """The motivating measurement: holding out peptides alone is not enough."""
    from cognate.data import load_train

    train = load_train()
    rng = np.random.default_rng(0)
    peptides = rng.permutation(sorted(set(train["Peptide"])))
    held_out = set(peptides[:162])

    naive_val = train[train["Peptide"].isin(held_out)]
    naive_train = train[~train["Peptide"].isin(held_out)]
    leaked = set(naive_train["CDR3b"]) & set(naive_val["CDR3b"])
    assert len(leaked) > 0

    clean = component_split(train, validation_fraction=0.2, seed=0)
    assert clean.shared_sequences() == set()
