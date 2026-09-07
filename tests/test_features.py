"""Tests for feature-block construction."""

import numpy as np
import pandas as pd
import pytest

from cognate.embed import embed_sequences
from cognate.features import build_features, feature_width


@pytest.fixture(scope="module")
def cache():
    result, _ = embed_sequences(
        ["GILGFVFTL", "NLVPMVATV", "CASSIRSSYEQYF", "CASSLGQAYEQYF"],
        "8M",
        progress_every=0,
    )
    return result


@pytest.fixture
def rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Peptide": ["GILGFVFTL", "NLVPMVATV"],
            "CDR3b": ["CASSIRSSYEQYF", "CASSLGQAYEQYF"],
        }
    )


def test_concat_width_is_two_blocks(cache, rows) -> None:
    features = build_features(rows, cache, -1, "concat")
    assert features.shape == (2, 2 * cache.hidden_size)
    assert features.shape[1] == feature_width(cache, "concat")


def test_interaction_width_is_four_blocks(cache, rows) -> None:
    features = build_features(rows, cache, -1, "interactions")
    assert features.shape == (2, 4 * cache.hidden_size)
    assert features.shape[1] == feature_width(cache, "interactions")


def test_interaction_blocks_are_the_stated_functions(cache, rows) -> None:
    peptide = cache.lookup(rows["Peptide"], -1)
    sequence = cache.lookup(rows["CDR3b"], -1)
    features = build_features(rows, cache, -1, "interactions")
    width = cache.hidden_size

    np.testing.assert_allclose(features[:, :width], peptide, rtol=1e-6)
    np.testing.assert_allclose(features[:, width : 2 * width], sequence, rtol=1e-6)
    np.testing.assert_allclose(
        features[:, 2 * width : 3 * width], np.abs(peptide - sequence), rtol=1e-6
    )
    np.testing.assert_allclose(features[:, 3 * width :], peptide * sequence, rtol=1e-6)


def test_concat_cannot_express_a_peptide_specific_ranking(cache) -> None:
    """The structural reason interactions are not optional.

    With concat features a linear model scores w_p.p + w_t.t; inside one peptide the first
    term is constant, so the induced TCR ranking is identical for every peptide.
    """
    rows = pd.DataFrame(
        {
            "Peptide": ["GILGFVFTL", "GILGFVFTL", "NLVPMVATV", "NLVPMVATV"],
            "CDR3b": [
                "CASSIRSSYEQYF",
                "CASSLGQAYEQYF",
                "CASSIRSSYEQYF",
                "CASSLGQAYEQYF",
            ],
        }
    )
    features = build_features(rows, cache, -1, "concat")
    weights = np.random.default_rng(0).normal(size=features.shape[1])
    scores = features @ weights

    first_peptide_gap = scores[0] - scores[1]
    second_peptide_gap = scores[2] - scores[3]
    assert first_peptide_gap == pytest.approx(second_peptide_gap, abs=1e-4)


def test_interactions_can_express_a_peptide_specific_ranking(cache) -> None:
    rows = pd.DataFrame(
        {
            "Peptide": ["GILGFVFTL", "GILGFVFTL", "NLVPMVATV", "NLVPMVATV"],
            "CDR3b": [
                "CASSIRSSYEQYF",
                "CASSLGQAYEQYF",
                "CASSIRSSYEQYF",
                "CASSLGQAYEQYF",
            ],
        }
    )
    features = build_features(rows, cache, -1, "interactions")
    weights = np.random.default_rng(0).normal(size=features.shape[1])
    scores = features @ weights
    assert scores[0] - scores[1] != pytest.approx(scores[2] - scores[3], abs=1e-3)


def test_unknown_block_raises(cache, rows) -> None:
    with pytest.raises(ValueError, match="unknown feature blocks"):
        build_features(rows, cache, -1, "quadratic")  # type: ignore[arg-type]


def test_single_side_probe_widths_are_one_block(cache, rows) -> None:
    for blocks in ("peptide_only", "tcr_only"):
        features = build_features(rows, cache, -1, blocks)
        assert features.shape == (2, cache.hidden_size)
        assert features.shape[1] == feature_width(cache, blocks)


def test_probe_blocks_carry_only_their_own_side(cache, rows) -> None:
    peptide = cache.lookup(rows["Peptide"], -1)
    tcr = cache.lookup(rows["CDR3b"], -1)
    assert np.allclose(build_features(rows, cache, -1, "peptide_only"), peptide)
    assert np.allclose(build_features(rows, cache, -1, "tcr_only"), tcr)


def test_peptide_only_gives_one_constant_score_per_peptide(cache) -> None:
    """Why the peptide-only probe is arithmetic on a balanced set, not a measurement.

    Every row of a peptide gets an identical feature vector, so any model produces a constant
    column within the peptide, and a constant column scores exactly 0.5 per group. Reading
    that 0.5 as evidence of no shortcut is the same error as reading the k-NN's 0.500 on
    unseen peptides as a measurement (docs/findings.md S3a).
    """
    rows = pd.DataFrame(
        {
            "Peptide": ["GILGFVFTL"] * 2 + ["NLVPMVATV"] * 2,
            "CDR3b": ["CASSIRSSYEQYF", "CASSLGQAYEQYF"] * 2,
        }
    )
    features = build_features(rows, cache, -1, "peptide_only")
    assert np.allclose(features[0], features[1])
    assert np.allclose(features[2], features[3])
    assert not np.allclose(features[0], features[2])
