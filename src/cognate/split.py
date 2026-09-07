"""Leakage-free train/validation splitting for TCR-epitope data.

A peptide-held-out split is the obvious move and it is not sufficient. 201 of 8,993
training CDR3b bind more than one peptide, so holding out peptides still leaves a mean
10.9% of validation rows whose TCR also appears in training. The fix is to split the
bipartite peptide-CDR3b graph on connected components, which guarantees both arms are
disjoint in peptides *and* in TCRs.

The component structure is extremely uneven: 684 components, of which the largest holds
64 peptides, 6,026 CDR3b and 71.8% of all rows. That component is always assigned to
training, because splitting it is impossible and putting it in validation would leave
almost nothing to train on.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_SEED = 0

PEPTIDE_KEY = "Peptide"
SEQUENCE_KEY = "CDR3b"


@dataclass(frozen=True)
class Split:
    """A train/validation partition that shares no peptide and no TCR."""

    train: pd.DataFrame
    validation: pd.DataFrame
    n_components: int
    n_validation_components: int

    @property
    def validation_fraction(self) -> float:
        total = len(self.train) + len(self.validation)
        return len(self.validation) / total if total else 0.0

    def shared_peptides(self) -> set[str]:
        return set(self.train[PEPTIDE_KEY]) & set(self.validation[PEPTIDE_KEY])

    def shared_sequences(self) -> set[str]:
        return set(self.train[SEQUENCE_KEY]) & set(self.validation[SEQUENCE_KEY])

    def __str__(self) -> str:
        return (
            f"train {len(self.train):,} rows / {self.train[PEPTIDE_KEY].nunique()} peptides"
            f" | val {len(self.validation):,} rows "
            f"({self.validation_fraction:.1%}) / "
            f"{self.validation[PEPTIDE_KEY].nunique()} peptides"
            f" | shared peptides {len(self.shared_peptides())}, "
            f"shared CDR3b {len(self.shared_sequences())}"
        )


def component_labels(
    positives_df: pd.DataFrame,
    peptide_column: str = PEPTIDE_KEY,
    sequence_column: str = SEQUENCE_KEY,
) -> np.ndarray:
    """Connected-component id per row of the bipartite peptide-TCR graph."""
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for peptide, sequence in zip(
        positives_df[peptide_column], positives_df[sequence_column]
    ):
        adjacency[("p", str(peptide))].add(("c", str(sequence)))
        adjacency[("c", str(sequence))].add(("p", str(peptide)))

    component_of: dict[tuple[str, str], int] = {}
    next_id = 0
    for node in adjacency:
        if node in component_of:
            continue
        queue = deque([node])
        component_of[node] = next_id
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in component_of:
                    component_of[neighbour] = next_id
                    queue.append(neighbour)
        next_id += 1

    return np.array(
        [component_of[("p", str(p))] for p in positives_df[peptide_column]]
    )


def component_split(
    positives_df: pd.DataFrame,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
    peptide_column: str = PEPTIDE_KEY,
    sequence_column: str = SEQUENCE_KEY,
) -> Split:
    """Partition on whole components until validation reaches ``validation_fraction``.

    The largest component is pinned to training. Remaining components are shuffled and
    added to validation until the row target is met, so the split is reproducible from
    ``seed`` and never cuts a component.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {validation_fraction}"
        )

    labelled = positives_df.assign(
        _component=component_labels(positives_df, peptide_column, sequence_column)
    )
    rows_per = labelled.groupby("_component").size().sort_values(ascending=False)

    rng = np.random.default_rng(seed)
    target = validation_fraction * len(labelled)
    chosen: list[int] = []
    accumulated = 0
    for component in rng.permutation(rows_per.index[1:].to_numpy()):
        if accumulated >= target:
            break
        chosen.append(int(component))
        accumulated += int(rows_per[component])

    in_validation = labelled["_component"].isin(chosen)
    return Split(
        train=labelled[~in_validation].drop(columns="_component"),
        validation=labelled[in_validation].drop(columns="_component"),
        n_components=len(rows_per),
        n_validation_components=len(chosen),
    )
