"""Nearest-neighbour baseline. No learning: a test pair scores as high as its closest
known binder of the same peptide.

Adapted from IMMREP23's TCRbase. For a query TCR, the score is the highest similarity to
any training positive *for that peptide*, so the method is a per-peptide lookup table with
fuzzy matching. Two consequences follow directly from that definition and are findings
rather than defects:

* A peptide absent from training has an empty database. Every one of its test rows takes
  ``default_score``, and a constant score column yields exactly 0.5 under AUROC and under
  McClish AUC0.1. The baseline is structurally incapable of ranking unseen peptides; the
  0.5 is arithmetic, not measurement. The IMMREP23 README prescribes this same
  zero-prediction fallback.
* 115 test rows are verbatim ``(Peptide, CDR3b)`` pairs from the training file and score
  1.0 for free. ``leave_out_exact_matches`` forbids the database from answering with the
  query itself, which separates fuzzy matching from pure recall.

v1 uses normalised Levenshtein similarity on ``CDR3b``. The BLOSUM62 k-mer kernel of the
published spec is deliberately not implemented.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

DEFAULT_SCORE = 0.0
EXACT_MATCH_SIMILARITY = 1.0

PEPTIDE_KEY = "Peptide"
SEQUENCE_KEY = "CDR3b"
TARGET_KEY = "Target"


@dataclass(frozen=True)
class KnnResult:
    """Per-row scores plus the provenance needed to interpret them."""

    score: np.ndarray
    has_database: np.ndarray
    database_size: np.ndarray
    nearest_sequence: np.ndarray

    @property
    def n_without_database(self) -> int:
        return int((~self.has_database).sum())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "score": self.score,
                "has_database": self.has_database,
                "database_size": self.database_size,
                "nearest_sequence": self.nearest_sequence,
            }
        )


def build_database(
    train_df: pd.DataFrame,
    peptide_column: str = PEPTIDE_KEY,
    sequence_column: str = SEQUENCE_KEY,
) -> dict[str, np.ndarray]:
    """Map each peptide to the distinct TCR sequences known to bind it.

    Only rows labelled positive are eligible. Duplicates are dropped because a repeated
    sequence cannot raise a maximum, and 2,007 training rows share a ``(Peptide, CDR3b)``
    pair with another row through a differing alpha chain.
    """
    positives = (
        train_df[train_df[TARGET_KEY] == 1] if TARGET_KEY in train_df else train_df
    )
    return {
        str(peptide): np.array(sorted(set(sequences.astype(str))), dtype=object)
        for peptide, sequences in positives.groupby(peptide_column)[sequence_column]
    }


def score_by_nearest_positive(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    peptide_column: str = PEPTIDE_KEY,
    sequence_column: str = SEQUENCE_KEY,
    default_score: float = DEFAULT_SCORE,
    leave_out_exact_matches: bool = False,
    top_k: int = 1,
) -> KnnResult:
    """Score every test row by its similarity to the nearest training binder of its peptide.

    ``leave_out_exact_matches`` removes similarity-1.0 hits from the database at query
    time, so a row whose exact ``(Peptide, CDR3b)`` pair is in training falls back to its
    second-nearest neighbour instead of scoring 1.0. This asks a different question from
    dropping those rows at evaluation time: it keeps the row in the denominator and tests
    whether the method can still rank it.

    ``top_k`` averages the k highest similarities instead of taking the single maximum.
    Unlike rescaling scores within a peptide, this changes the within-peptide *ranking*
    and so can move per-peptide AUC0.1. A peptide whose database is smaller than k
    averages over everything it has.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    database = build_database(train_df, peptide_column, sequence_column)

    n_rows = len(test_df)
    score = np.full(n_rows, default_score, dtype=float)
    has_database = np.zeros(n_rows, dtype=bool)
    database_size = np.zeros(n_rows, dtype=int)
    nearest_sequence = np.full(n_rows, "", dtype=object)

    peptides = test_df[peptide_column].astype(str).to_numpy()
    sequences = test_df[sequence_column].astype(str).to_numpy()

    for peptide in np.unique(peptides):
        if peptide not in database:
            continue
        rows = np.flatnonzero(peptides == peptide)
        entries = database[peptide]
        similarity = cdist(
            sequences[rows],
            entries,
            scorer=Levenshtein.normalized_similarity,
            workers=-1,
        )
        if leave_out_exact_matches:
            similarity = np.where(
                similarity >= EXACT_MATCH_SIMILARITY, -np.inf, similarity
            )

        best = similarity.argmax(axis=1)
        best_similarity = similarity[np.arange(len(rows)), best]
        if top_k > 1:
            width = min(top_k, similarity.shape[1])
            top = np.sort(similarity, axis=1)[:, -width:]
            aggregated = np.where(
                np.isfinite(top).all(axis=1), top.mean(axis=1), best_similarity
            )
        else:
            aggregated = best_similarity
        usable = np.isfinite(aggregated)

        has_database[rows] = True
        database_size[rows] = len(entries)
        score[rows[usable]] = aggregated[usable]
        nearest_sequence[rows[usable]] = entries[best[usable]]

    return KnnResult(
        score=score,
        has_database=has_database,
        database_size=database_size,
        nearest_sequence=nearest_sequence,
    )


def exact_match_mask(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    peptide_column: str = PEPTIDE_KEY,
    sequence_column: str = SEQUENCE_KEY,
) -> np.ndarray:
    """True where a test row's ``(Peptide, CDR3b)`` pair appears verbatim in training."""
    positives = (
        train_df[train_df[TARGET_KEY] == 1] if TARGET_KEY in train_df else train_df
    )
    pairs = set(zip(positives[peptide_column], positives[sequence_column]))
    return np.array(
        [
            (p, s) in pairs
            for p, s in zip(test_df[peptide_column], test_df[sequence_column])
        ]
    )
