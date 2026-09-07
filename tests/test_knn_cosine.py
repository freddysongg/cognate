"""The cosine backend must swap the representation and nothing else.

The point of the ESM-2 cosine variant is to remove a confound: the edit-distance k-NN and
the logistic head differed in both algorithm and representation. These tests pin the two
properties that make the swap a fair one -- the backend computes real cosine, and routing
through it leaves the database, the operator and the bookkeeping untouched.
"""

import numpy as np
import pandas as pd
import pytest

from cognate.baseline_knn import (
    cosine_similarity,
    edit_similarity,
    score_by_nearest_positive,
)
from cognate.embed import EmbeddingCache

LAYER = 0


def make_cache(vectors: dict[str, list[float]]) -> EmbeddingCache:
    sequences = np.array(list(vectors), dtype=object)
    layers = np.array([[vectors[s] for s in sequences]], dtype=np.float32)
    return EmbeddingCache(
        model_name="toy",
        sequences=sequences,
        layers=layers,
        index={str(s): i for i, s in enumerate(sequences)},
    )


def test_cosine_backend_computes_cosine():
    cache = make_cache({"Q": [1.0, 0.0], "A": [1.0, 0.0], "B": [0.0, 1.0], "C": [1.0, 1.0]})
    similarity = cosine_similarity(cache, LAYER)
    got = similarity(np.array(["Q"], dtype=object), np.array(["A", "B", "C"], dtype=object))
    assert got.ravel() == pytest.approx([1.0, 0.0, np.sqrt(0.5)], abs=1e-6)


def test_cosine_is_blind_to_magnitude():
    """Scaling a database vector must not change its similarity; a dot product would."""
    cache = make_cache({"Q": [1.0, 1.0], "A": [1.0, 0.0], "A10": [10.0, 0.0]})
    similarity = cosine_similarity(cache, LAYER)
    got = similarity(np.array(["Q"], dtype=object), np.array(["A", "A10"], dtype=object))
    assert got[0][0] == pytest.approx(got[0][1], abs=1e-6)


def test_separate_caches_for_queries_and_database():
    """Evaluation TCRs are disjoint from training TCRs, so the two live in different caches."""
    queries = make_cache({"Q": [1.0, 0.0]})
    database = make_cache({"D": [0.0, 1.0]})
    similarity = cosine_similarity(queries, LAYER, database)
    got = similarity(np.array(["Q"], dtype=object), np.array(["D"], dtype=object))
    assert got.ravel() == pytest.approx([0.0], abs=1e-6)


def test_backend_swap_leaves_database_bookkeeping_identical():
    train = pd.DataFrame(
        {"Peptide": ["P", "P", "R"], "CDR3b": ["AAA", "AAC", "GGG"], "Target": [1, 1, 1]}
    )
    test = pd.DataFrame({"Peptide": ["P", "P", "S"], "CDR3b": ["AAG", "CCC", "TTT"]})
    cache = make_cache(
        {s: list(np.random.default_rng(i).normal(size=4))
         for i, s in enumerate(["AAA", "AAC", "GGG", "AAG", "CCC", "TTT"])}
    )

    by_edit = score_by_nearest_positive(test, train)
    by_cosine = score_by_nearest_positive(
        test, train, similarity_fn=cosine_similarity(cache, LAYER)
    )

    assert by_cosine.has_database.tolist() == by_edit.has_database.tolist()
    assert by_cosine.database_size.tolist() == by_edit.database_size.tolist()
    assert by_cosine.score[2] == by_edit.score[2] == 0.0


def test_explicit_edit_backend_equals_the_default():
    train = pd.DataFrame({"Peptide": ["P", "P"], "CDR3b": ["AAA", "AAC"], "Target": [1, 1]})
    test = pd.DataFrame({"Peptide": ["P", "P"], "CDR3b": ["AAG", "CCC"]})
    default = score_by_nearest_positive(test, train)
    explicit = score_by_nearest_positive(test, train, similarity_fn=edit_similarity)
    assert explicit.score.tolist() == default.score.tolist()
