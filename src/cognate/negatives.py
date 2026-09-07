"""Negative example generation for TCR-epitope training data.

``VDJdb_paired_chain.csv`` contains only positives, so the negatives a model trains on
are a design decision rather than data. Two strategies are implemented behind one
interface; see docs/negative_sampling.md for which is the default and why.

Conventions shared by every strategy:

* A TCR's *cognate set* is every peptide its ``CDR3b`` is known to bind anywhere in the
  positives table, not just the peptide on its own row. 201 of 8,993 training CDR3b bind
  more than one peptide, and since the models here see only ``CDR3b``, pairing a TCR with
  a peptide another copy of it binds would be a guaranteed false negative.
* Swapping a peptide swaps its HLA too. HLA is a property of the peptide's presentation,
  not of the TCR, so carrying the original TCR row's HLA across would be incoherent.
* Generated rows keep the index of the positive row they were derived from, so a negative
  is always traceable back to its source without adding columns that could leak into
  features.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

DEFAULT_RATIO = 5.0
DEFAULT_SEED = 0
UNIFORM_MAX_PAIRS = 2_000_000

TCR_KEY = "CDR3b"
PEPTIDE_KEY = "Peptide"
HLA_KEY = "HLA"
TARGET_KEY = "Target"

NegativeStrategy = Literal["shuffle", "matched", "hard", "uniform"]


@dataclass(frozen=True)
class _SamplingContext:
    """Precomputed lookups shared by the strategies."""

    positives: pd.DataFrame
    peptides: np.ndarray
    cognates: dict[str, frozenset[str]]
    hla_by_peptide: dict[str, str]


def _build_context(positives: pd.DataFrame) -> _SamplingContext:
    cognates = {
        str(tcr): frozenset(str(p) for p in peps)
        for tcr, peps in positives.groupby(TCR_KEY)[PEPTIDE_KEY].unique().items()
    }
    hla_by_peptide = {
        str(p): str(hla.mode().iloc[0])
        for p, hla in positives.groupby(PEPTIDE_KEY)[HLA_KEY]
    }
    return _SamplingContext(
        positives=positives,
        peptides=np.array(sorted(hla_by_peptide), dtype=object),
        cognates=cognates,
        hla_by_peptide=hla_by_peptide,
    )


def _per_row_counts(n_rows: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Split a possibly fractional ratio into an integer count per positive row.

    An integral ratio gives every row exactly that many negatives. A fractional one gives
    each row the floor and distributes the remainder by Bernoulli draw, so the realised
    ratio matches in expectation rather than exactly.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    base = int(np.floor(ratio))
    remainder = ratio - base
    counts = np.full(n_rows, base, dtype=int)
    if remainder > 0:
        counts += (rng.random(n_rows) < remainder).astype(int)
    return counts


def _assemble(
    ctx: _SamplingContext,
    base: pd.DataFrame,
    source_positions: list[int],
    drawn_peptides: list[str],
) -> pd.DataFrame:
    """Build negative rows from source row positions and the peptides swapped in."""
    negatives = base.iloc[source_positions].copy()
    negatives[PEPTIDE_KEY] = drawn_peptides
    negatives[HLA_KEY] = [ctx.hla_by_peptide[p] for p in drawn_peptides]
    negatives[TARGET_KEY] = 0
    return negatives


def _sample_shuffle(
    ctx: _SamplingContext, ratio: float, seed: int
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Pair each TCR with uniformly random non-cognate peptides.

    Simple and standard. It will still manufacture false negatives, because a TCR that
    binds one peptide often binds close relatives that were never recorded in VDJdb.
    """
    rng = np.random.default_rng(seed)
    counts = _per_row_counts(len(ctx.positives), ratio, rng)
    pool_cache: dict[frozenset[str], np.ndarray] = {}

    source_positions: list[int] = []
    drawn_peptides: list[str] = []
    for position, (tcr, k) in enumerate(zip(ctx.positives[TCR_KEY], counts)):
        forbidden = ctx.cognates[str(tcr)]
        if forbidden not in pool_cache:
            pool_cache[forbidden] = ctx.peptides[
                ~np.isin(ctx.peptides, list(forbidden))
            ]
        pool = pool_cache[forbidden]
        take = min(int(k), len(pool))
        if take == 0:
            continue
        source_positions.extend([position] * take)
        drawn_peptides.extend(rng.choice(pool, size=take, replace=False))
    return ctx.positives, source_positions, drawn_peptides


def _sample_matched(
    ctx: _SamplingContext, ratio: float, seed: int
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Draw negative peptides from the empirical positive peptide distribution.

    ``shuffle`` draws uniformly over the 808 peptides, which makes the peptide marginal of
    the negatives flat while the positives stay heavily skewed -- ``GILGFVFTL`` is 16.1% of
    positives and 0.10% of uniform negatives, a 159x ratio. Peptide identity alone then
    predicts the label at 0.94 global AUROC, and a model can score well by memorising which
    peptides are common instead of learning anything about TCR-peptide compatibility.

    Sampling proportional to positive frequency removes that shortcut and reproduces the
    IMMREP23 test construction, where TCRs were swapped *among* the peptides in the set so
    every peptide keeps the same 5:1 ratio in both classes.

    Uses the Gumbel top-k trick for weighted sampling without replacement, which avoids
    renormalising a 808-entry probability vector once per row.
    """
    rng = np.random.default_rng(seed)
    counts = ctx.positives[PEPTIDE_KEY].value_counts()
    weights = np.array([counts.get(p, 0) for p in ctx.peptides], dtype=float)
    log_weights = np.log(np.where(weights > 0, weights, np.finfo(float).tiny))

    per_row = _per_row_counts(len(ctx.positives), ratio, rng)
    forbidden_cache: dict[frozenset[str], np.ndarray] = {}

    source_positions: list[int] = []
    drawn_peptides: list[str] = []
    for position, (tcr, k) in enumerate(zip(ctx.positives[TCR_KEY], per_row)):
        forbidden = ctx.cognates[str(tcr)]
        if forbidden not in forbidden_cache:
            forbidden_cache[forbidden] = ~np.isin(ctx.peptides, list(forbidden))
        allowed = forbidden_cache[forbidden]
        take = min(int(k), int(allowed.sum()))
        if take == 0:
            continue
        keys = log_weights + rng.gumbel(size=len(ctx.peptides))
        keys[~allowed] = -np.inf
        chosen = np.argpartition(-keys, take - 1)[:take]
        source_positions.extend([position] * take)
        drawn_peptides.extend(str(ctx.peptides[i]) for i in chosen)
    return ctx.positives, source_positions, drawn_peptides


def _neighbour_ranking(peptides: np.ndarray) -> dict[str, list[str]]:
    """For each peptide, the others ordered by Levenshtein distance then alphabetically."""
    distances = cdist(peptides, peptides, scorer=Levenshtein.distance, workers=-1)
    ranking: dict[str, list[str]] = {}
    for i, peptide in enumerate(peptides):
        order = sorted(
            (j for j in range(len(peptides)) if j != i),
            key=lambda j: (int(distances[i, j]), str(peptides[j])),
        )
        ranking[str(peptide)] = [str(peptides[j]) for j in order]
    return ranking


def _sample_hard(
    ctx: _SamplingContext, ratio: float, seed: int
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Pair each TCR with the peptides closest to its true partner by edit distance.

    Deterministic: ties break alphabetically, so ``seed`` affects only the fractional part
    of ``ratio``. Every TCR of a given peptide therefore receives the same negatives,
    which is the point -- the sampler is probing the decision boundary, not covering it.
    """
    rng = np.random.default_rng(seed)
    counts = _per_row_counts(len(ctx.positives), ratio, rng)
    ranking = _neighbour_ranking(ctx.peptides)

    source_positions: list[int] = []
    drawn_peptides: list[str] = []
    rows = zip(ctx.positives[TCR_KEY], ctx.positives[PEPTIDE_KEY], counts)
    for position, (tcr, true_peptide, k) in enumerate(rows):
        forbidden = ctx.cognates[str(tcr)]
        chosen = [p for p in ranking[str(true_peptide)] if p not in forbidden][: int(k)]
        source_positions.extend([position] * len(chosen))
        drawn_peptides.extend(chosen)
    return ctx.positives, source_positions, drawn_peptides


def _sample_uniform(
    ctx: _SamplingContext, ratio: float, seed: int
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Every distinct TCR against every peptide, keeping the pairs that are not positives.

    ``ratio`` is ignored: the strategy is exhaustive by definition. Guarded because the
    full cross product on the VDJdb training set is 7.3M pairs.
    """
    representatives = ctx.positives.drop_duplicates(subset=[TCR_KEY])
    n_pairs = len(representatives) * len(ctx.peptides)
    if n_pairs > UNIFORM_MAX_PAIRS:
        raise ValueError(
            f"uniform strategy would build {n_pairs:,} pairs, over the "
            f"{UNIFORM_MAX_PAIRS:,} guard. Subset the peptides or TCRs first."
        )

    source_positions: list[int] = []
    drawn_peptides: list[str] = []
    for position, tcr in enumerate(representatives[TCR_KEY]):
        forbidden = ctx.cognates[str(tcr)]
        for peptide in ctx.peptides:
            if str(peptide) not in forbidden:
                source_positions.append(position)
                drawn_peptides.append(str(peptide))
    return representatives, source_positions, drawn_peptides


_Sampler = Callable[
    [_SamplingContext, float, int], tuple[pd.DataFrame, list[int], list[str]]
]

STRATEGIES: dict[NegativeStrategy, _Sampler] = {
    "shuffle": _sample_shuffle,
    "matched": _sample_matched,
    "hard": _sample_hard,
    "uniform": _sample_uniform,
}


def make_negatives(
    positives_df: pd.DataFrame,
    strategy: NegativeStrategy = "matched",
    ratio: float = DEFAULT_RATIO,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate negative TCR-peptide pairs from a positives-only table.

    Returns rows with the same columns as ``positives_df`` and ``Target == 0``. The index
    of each returned row is the index of the positive row its TCR came from.

    ``ratio`` is negatives per positive; it defaults to 5.0 to match the 1:5 ratio the
    IMMREP23 organisers built into the test set.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}, expected one of {sorted(STRATEGIES)}"
        )
    missing = {PEPTIDE_KEY, TCR_KEY, HLA_KEY} - set(positives_df.columns)
    if missing:
        raise ValueError(f"positives_df is missing required columns: {sorted(missing)}")

    ctx = _build_context(positives_df)
    base, source_positions, drawn = STRATEGIES[strategy](ctx, ratio, seed)
    return _assemble(ctx, base, source_positions, drawn)


def make_training_set(
    positives_df: pd.DataFrame,
    strategy: NegativeStrategy = "matched",
    ratio: float = DEFAULT_RATIO,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Positives and generated negatives concatenated, shuffled, with a reset index."""
    negatives = make_negatives(positives_df, strategy, ratio, seed)
    combined = pd.concat([positives_df, negatives], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def summarise(negatives: pd.DataFrame, positives: pd.DataFrame) -> dict[str, float | int]:
    """Descriptive counts used in the strategy note and the notebooks."""
    generated = set(zip(negatives[PEPTIDE_KEY], negatives[TCR_KEY]))
    true_pairs = set(zip(positives[PEPTIDE_KEY], positives[TCR_KEY]))
    return {
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "realised_ratio": len(negatives) / len(positives),
        "unique_pairs": len(generated),
        "collisions_with_positives": len(generated & true_pairs),
        "distinct_peptides_used": negatives[PEPTIDE_KEY].nunique(),
    }


def peptide_distances(peptides: Iterable[str]) -> pd.DataFrame:
    """Pairwise Levenshtein distance matrix over peptides, for inspection."""
    unique = np.array(sorted(set(str(p) for p in peptides)), dtype=object)
    matrix = cdist(unique, unique, scorer=Levenshtein.distance, workers=-1)
    return pd.DataFrame(matrix, index=unique, columns=unique)
