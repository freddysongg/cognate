"""Correctness tests for the ESM-2 embedding cache.

The load-bearing one is padding invariance: if the batch a sequence lands in changes its
embedding, every downstream number is silently dependent on sort order.
"""

import numpy as np
import pytest
import torch

from cognate.embed import (
    MODELS,
    _length_batches,
    _residue_mask,
    embed_sequences,
    load_cache,
    save_cache,
)

SEQUENCES = [
    "GILGFVFTL",
    "CASSIRSSYEQYF",
    "CASSLGQAYEQYF",
    "NLVPMVATV",
    "CASSPGTGGYNEQFF",
    "CASSF",
]


@pytest.fixture(scope="module")
def cache():
    result, _ = embed_sequences(SEQUENCES, "8M", progress_every=0)
    return result


def test_residue_mask_drops_bos_and_eos() -> None:
    attention = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    mask = _residue_mask(attention)
    assert mask.tolist() == [[0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0]]
    assert mask.sum(dim=1).tolist() == [3, 5]


def test_length_batches_cover_every_sequence_once() -> None:
    sequences = [f"{'A' * n}" for n in range(3, 40)]
    batches = _length_batches(sequences, max_batch_tokens=64)
    covered = sorted(i for batch in batches for i in batch)
    assert covered == list(range(len(sequences)))
    assert all(batches)


def test_length_batches_keep_padding_waste_low() -> None:
    """The point of length-sorting. A batch straddling a length boundary is fine; what
    matters is that total padded tokens stay close to total real tokens."""
    rng = np.random.default_rng(0)
    sequences = ["A" * int(n) for n in rng.integers(5, 24, size=2000)]
    batches = _length_batches(sequences, max_batch_tokens=4096)

    real = sum(len(s) + 2 for s in sequences)
    padded = sum(
        len(batch) * max(len(sequences[i]) + 2 for i in batch) for batch in batches
    )
    assert padded / real < 1.10

    unsorted_padded = 0
    for start in range(0, len(sequences), 256):
        window = sequences[start : start + 256]
        unsorted_padded += len(window) * max(len(s) + 2 for s in window)
    assert padded < unsorted_padded


def test_cache_shape_covers_every_layer(cache) -> None:
    assert cache.n_sequences == len(SEQUENCES)
    assert cache.n_layers == 7
    assert cache.hidden_size == 320
    assert cache.layers.shape == (7, len(SEQUENCES), 320)
    assert np.isfinite(cache.layers).all()


def test_model_has_no_randomly_initialised_pooler() -> None:
    """add_pooling_layer=False. The checkpoint has no pooler, so leaving it on would
    attach randomly initialised weights and pooler_output would be noise."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODELS["8M"], add_pooling_layer=False)
    assert getattr(model, "pooler", None) is None


def test_embeddings_are_invariant_to_batching(cache) -> None:
    """Each sequence embedded alone must match the batched result.

    A mismatch means padding is leaking through attention, which would make every result
    depend on how sequences happened to be grouped.
    """
    for sequence in SEQUENCES:
        alone, _ = embed_sequences([sequence], "8M", progress_every=0)
        for layer_index in range(cache.n_layers):
            np.testing.assert_allclose(
                alone.lookup([sequence], layer_index),
                cache.lookup([sequence], layer_index),
                atol=2e-5,
            )


def test_pooling_matches_manual_mean_over_residues(cache) -> None:
    """Recompute one embedding by hand from the raw hidden states."""
    from transformers import AutoModel, AutoTokenizer

    sequence = "GILGFVFTL"
    tokenizer = AutoTokenizer.from_pretrained(MODELS["8M"])
    model = AutoModel.from_pretrained(MODELS["8M"], add_pooling_layer=False).eval()
    encoded = tokenizer([sequence], return_tensors="pt")
    with torch.no_grad():
        hidden = model(**encoded, output_hidden_states=True).hidden_states[-1]

    residues = hidden[0, 1:-1, :]
    assert residues.shape[0] == len(sequence)
    np.testing.assert_allclose(
        residues.mean(dim=0).numpy(), cache.lookup([sequence], -1)[0], atol=2e-5
    )


def test_lookup_preserves_requested_order(cache) -> None:
    wanted = ["NLVPMVATV", "CASSF", "GILGFVFTL"]
    rows = cache.lookup(wanted, -1)
    for position, sequence in enumerate(wanted):
        np.testing.assert_array_equal(rows[position], cache.lookup([sequence], -1)[0])


def test_lookup_raises_on_unknown_sequence(cache) -> None:
    with pytest.raises(KeyError, match="not in the"):
        cache.lookup(["WWWWWWWWWW"], -1)


def test_cache_round_trips_through_disk(cache, tmp_path) -> None:
    path = tmp_path / "cache.npz"
    size = save_cache(cache, path)
    assert size > 0

    restored = load_cache(path)
    assert restored.model_name == cache.model_name
    assert restored.n_layers == cache.n_layers
    np.testing.assert_array_equal(restored.sequences, cache.sequences)
    np.testing.assert_allclose(restored.layers, cache.layers)


def test_layers_are_not_all_identical(cache) -> None:
    """If the layer axis were collapsed the T6b sweep would be meaningless."""
    first, middle, last = cache.layer(0), cache.layer(3), cache.layer(-1)
    assert not np.allclose(first, middle)
    assert not np.allclose(middle, last)
