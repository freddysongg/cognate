"""Feature construction from cached ESM-2 embeddings.

The block choice is not cosmetic. A linear model on plain ``concat([peptide, tcr])``
computes ``w_p . p + w_t . t + b``; within one peptide the first term is constant, so the
model produces **the same ranking of TCRs for every peptide**. Per-peptide AUC0.1 then
measures whether one universal 'this TCR looks like a binder' score works, which the
shuffled-negative construction specifically prevents -- every TCR appears as a positive
once and as a negative five times.

Interaction blocks fix that. The elementwise product gives a diagonal bilinear form,
``sum_i w_i p_i t_i``, whose TCR ranking depends on the peptide. The absolute difference
adds a per-dimension distance between the two sequences.

``peptide_only`` and ``tcr_only`` exist for the shortcut probes required before any new
evaluation set is scored (docs/findings.md S4). A head that scores well from one side alone
is reading something other than compatibility, which is how the Session 5 sampler bug went
undetected for two sessions.
"""

from typing import Literal

import numpy as np
import pandas as pd

from cognate.embed import EmbeddingCache

FeatureBlocks = Literal["concat", "interactions", "peptide_only", "tcr_only"]

PEPTIDE_KEY = "Peptide"
SEQUENCE_KEY = "CDR3b"


def build_features(
    df: pd.DataFrame,
    cache: EmbeddingCache,
    layer_index: int = -1,
    blocks: FeatureBlocks = "interactions",
    peptide_column: str = PEPTIDE_KEY,
    sequence_column: str = SEQUENCE_KEY,
) -> np.ndarray:
    """Assemble the design matrix for one layer of one embedding cache."""
    peptide = cache.lookup(df[peptide_column].astype(str), layer_index)
    sequence = cache.lookup(df[sequence_column].astype(str), layer_index)

    if blocks == "concat":
        parts = [peptide, sequence]
    elif blocks == "interactions":
        parts = [peptide, sequence, np.abs(peptide - sequence), peptide * sequence]
    elif blocks == "peptide_only":
        parts = [peptide]
    elif blocks == "tcr_only":
        parts = [sequence]
    else:
        raise ValueError(f"unknown feature blocks {blocks!r}")

    return np.concatenate(parts, axis=1).astype(np.float32)


BLOCK_MULTIPLIERS: dict[str, int] = {
    "concat": 2,
    "interactions": 4,
    "peptide_only": 1,
    "tcr_only": 1,
}


def feature_width(cache: EmbeddingCache, blocks: FeatureBlocks = "interactions") -> int:
    return cache.hidden_size * BLOCK_MULTIPLIERS[blocks]
