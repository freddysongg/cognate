"""The per-residue cache must be the pooled cache, unpooled.

Fork 2 depends on these residues being the same numbers the Phase 1 head averaged. The
acceptance check the work order sets is that mean-pooling the residue cache reproduces the
existing pooled cache to ``atol=1e-5``; the round-trip and BOS/EOS tests below are what make a
failure of that check interpretable rather than mysterious.
"""

import numpy as np
import pytest

from cognate.embed import (
    embed_residues,
    load_residue_cache,
    save_residue_cache,
)

SEQUENCES = ["CASSL", "CASSLGQY", "SIINFEKL"]
LAYERS = (0, 2)
MODEL_KEY = "8M"


@pytest.fixture(scope="module")
def cache():
    built, _ = embed_residues(SEQUENCES, MODEL_KEY, layer_indices=LAYERS, progress_every=0)
    return built


def test_one_row_per_residue_with_specials_dropped(cache):
    for sequence in SEQUENCES:
        assert cache.residues_for(sequence, LAYERS[0]).shape == (
            len(sequence),
            cache.hidden_size,
        )


def test_mean_pooling_matches_the_pooled_path(cache):
    """The property the 0.3 acceptance check rests on, at toy scale.

    Compared relative to layer magnitude rather than absolutely. ESM-2 layer norms differ by
    more than 10x across depth, so a fixed absolute tolerance silently becomes a far stricter
    test of the deeper layers for identical arithmetic.
    """
    from cognate.embed import embed_sequences

    pooled, _ = embed_sequences(SEQUENCES, MODEL_KEY, progress_every=0)
    for layer in LAYERS:
        expected = pooled.lookup(SEQUENCES, layer)
        scale = np.abs(expected).max()
        assert cache.mean_pooled(SEQUENCES, layer) == pytest.approx(
            expected, abs=1e-6 * scale
        )


def test_unstored_layer_raises_rather_than_returning_the_wrong_one(cache):
    with pytest.raises(KeyError, match="not stored"):
        cache.residues_for(SEQUENCES[0], 5)


def test_unknown_sequence_raises(cache):
    with pytest.raises(KeyError):
        cache.residues_for("WWWWW", LAYERS[0])


def test_save_load_round_trip(cache, tmp_path):
    path = tmp_path / "residues.npz"
    save_residue_cache(cache, path)
    restored = load_residue_cache(path)
    assert restored.sequences.tolist() == cache.sequences.tolist()
    assert restored.layer_indices.tolist() == cache.layer_indices.tolist()
    for sequence in SEQUENCES:
        assert restored.residues_for(sequence, LAYERS[0]) == pytest.approx(
            cache.residues_for(sequence, LAYERS[0])
        )
