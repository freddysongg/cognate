"""Source validation and scoring for the pMHC binding-affinity experiment.

Parses the NetMHCpan training folds, enforces the eligibility and class-balance guards
from the source contract, and implements the experiment's fixed scoring rules.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from cognate.baseline_knn import KnnResult, SimilarityFn, score_by_nearest_positive
from cognate.train_head import (
    DEFAULT_BATCH_SIZE,
    MlpHead,
    fit_logistic,
    logistic_scores,
)

FOLD_NAMES = tuple(f"c00{index}_ba" for index in range(5))
HLA_PREFIXES = ("HLA-A", "HLA-B", "HLA-C")
POSITIVE_THRESHOLD = 0.426
ARTIFICIAL_NEGATIVE_AFFINITY = 0.01
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY"
PSEUDO_SEQUENCE_LENGTH = 34
MLP_HIDDEN_SIZES = (55, 66)
MLP_SEEDS = tuple(range(5))
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20
MLP_LEARNING_RATE = 1e-3


def random_scores(n_rows: int) -> np.ndarray:
    """Return the experiment's fixed random baseline in caller row order."""
    return np.random.default_rng(0).random(n_rows)


def build_composition_features(sequences: Sequence[str]) -> np.ndarray:
    """Encode length followed by residue counts in the fixed amino-acid order."""
    return np.array(
        [
            [len(sequence), *(sequence.count(residue) for residue in AMINO_ACID_ORDER)]
            for sequence in sequences
        ],
        dtype=float,
    )


def score_composition_fold(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit and score the composition logistic baseline for one outer fold."""
    model = fit_logistic(
        build_composition_features(train["Peptide"].astype(str)),
        train["Target"].to_numpy(dtype=int),
    )
    return logistic_scores(
        model, build_composition_features(test["Peptide"].astype(str))
    )


def score_retrieval_fold(
    test: pd.DataFrame,
    train: pd.DataFrame,
    *,
    similarity_fn: SimilarityFn | None = None,
) -> KnnResult:
    """Score peptides against positive training peptides from the same allele."""
    if similarity_fn is None:
        return score_by_nearest_positive(
            test,
            train,
            peptide_column="Allele",
            sequence_column="Peptide",
        )
    return score_by_nearest_positive(
        test,
        train,
        peptide_column="Allele",
        sequence_column="Peptide",
        similarity_fn=similarity_fn,
    )


def score_pwm_fold(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit allele-specific smoothed log-odds matrices and score test peptides."""
    residue_indices = {
        residue: index for index, residue in enumerate(AMINO_ACID_ORDER)
    }
    allele_log_odds: dict[str, np.ndarray] = {}
    for allele, allele_rows in train.groupby("Allele", sort=False):
        is_positive = allele_rows["Target"].to_numpy(dtype=bool)
        encoded_peptides = np.array(
            [
                [residue_indices[residue] for residue in peptide]
                for peptide in allele_rows["Peptide"].astype(str)
            ],
            dtype=int,
        )
        positive_counts = np.zeros((9, len(AMINO_ACID_ORDER)), dtype=float)
        negative_counts = np.zeros_like(positive_counts)
        for position in range(9):
            positive_counts[position] = np.bincount(
                encoded_peptides[is_positive, position],
                minlength=len(AMINO_ACID_ORDER),
            )
            negative_counts[position] = np.bincount(
                encoded_peptides[~is_positive, position],
                minlength=len(AMINO_ACID_ORDER),
            )
        positive_probabilities = (positive_counts + 1) / (is_positive.sum() + 20)
        negative_probabilities = (negative_counts + 1) / ((~is_positive).sum() + 20)
        allele_log_odds[str(allele)] = np.log(
            positive_probabilities / negative_probabilities
        )

    scores = np.empty(len(test), dtype=float)
    for row_index, (allele, peptide) in enumerate(
        zip(test["Allele"].astype(str), test["Peptide"].astype(str))
    ):
        log_odds = allele_log_odds[allele]
        scores[row_index] = sum(
            log_odds[position, residue_indices[residue]]
            for position, residue in enumerate(peptide)
        )
    return scores


@cache
def _blosum50_rows() -> dict[str, np.ndarray]:
    matrix = substitution_matrices.load("BLOSUM50")
    return {
        residue: np.array(
            [matrix[residue, target] for target in AMINO_ACID_ORDER], dtype=float
        )
        for residue in AMINO_ACID_ORDER
    }


def encode_blosum50(sequences: Sequence[str]) -> np.ndarray:
    """Flatten raw BLOSUM50 rows in fixed residue and sequence-position order."""
    if len(sequences) == 0:
        return np.empty((0, 0), dtype=float)
    return np.array(
        [
            np.concatenate([_blosum50_rows()[residue] for residue in sequence])
            for sequence in sequences
        ],
        dtype=float,
    )


def build_mlp_features(
    rows: pd.DataFrame,
    pseudo_sequences: Mapping[str, str],
    allele_order: Sequence[str],
    *,
    use_one_hot_allele: bool,
) -> np.ndarray:
    """Build peptide plus pseudo-sequence or one-hot-allele MLP features."""
    peptide_features = encode_blosum50(rows["Peptide"].astype(str))
    if not use_one_hot_allele:
        allele_features = encode_blosum50(
            [pseudo_sequences[allele] for allele in rows["Allele"].astype(str)]
        )
        return np.concatenate((peptide_features, allele_features), axis=1)

    allele_indices = {allele: index for index, allele in enumerate(allele_order)}
    allele_features = np.zeros((len(rows), len(allele_order)), dtype=float)
    allele_features[
        np.arange(len(rows)),
        [allele_indices[allele] for allele in rows["Allele"].astype(str)],
    ] = 1.0
    return np.concatenate((peptide_features, allele_features), axis=1)


def split_fit_validation(
    rows: pd.DataFrame, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed 90/10 split stratified jointly by allele and target."""
    strata = (
        rows["Allele"].astype(str)
        + "|"
        + rows["Target"].astype(int).astype(str)
    ).to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.1, random_state=seed
    )
    return next(splitter.split(np.zeros(len(rows)), strata))


def _score_single_mlp(
    fit_features: np.ndarray,
    fit_affinities: np.ndarray,
    validation_features: np.ndarray,
    validation_affinities: np.ndarray,
    test_features: np.ndarray,
    *,
    hidden: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """Train one continuous-affinity MLP and score its held-out rows."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(fit_features)
    x_fit = torch.tensor(
        np.ascontiguousarray(scaler.transform(fit_features)),
        dtype=torch.float32,
        device=device,
    )
    y_fit = torch.tensor(fit_affinities, dtype=torch.float32, device=device)
    x_validation = torch.tensor(
        np.ascontiguousarray(scaler.transform(validation_features)),
        dtype=torch.float32,
        device=device,
    )
    y_validation = torch.tensor(
        validation_affinities, dtype=torch.float32, device=device
    )
    model = MlpHead(fit_features.shape[1], hidden=hidden, dropout=0.0).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0

    for _ in range(MLP_MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(x_fit), generator=generator).to(device)
        for start in range(0, len(order), DEFAULT_BATCH_SIZE):
            batch = order[start : start + DEFAULT_BATCH_SIZE]
            optimiser.zero_grad()
            probabilities = torch.sigmoid(model(x_fit[batch]))
            loss = criterion(probabilities, y_fit[batch])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            validation_probabilities = torch.sigmoid(model(x_validation))
            validation_loss = float(
                criterion(validation_probabilities, y_validation)
            )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= MLP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(
            np.ascontiguousarray(scaler.transform(test_features)),
            dtype=torch.float32,
            device=device,
        )
        return torch.sigmoid(model(x_test)).cpu().numpy()


def score_mlp_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pseudo_sequences: Mapping[str, str],
    allele_order: Sequence[str],
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    use_one_hot_allele: bool,
    device: str,
) -> np.ndarray:
    """Average the ten fixed hidden-size and seed models for one outer fold."""
    train_features = build_mlp_features(
        train,
        pseudo_sequences,
        allele_order,
        use_one_hot_allele=use_one_hot_allele,
    )
    test_features = build_mlp_features(
        test,
        pseudo_sequences,
        allele_order,
        use_one_hot_allele=use_one_hot_allele,
    )
    affinities = train["Affinity"].to_numpy(dtype=float)
    predictions = [
        _score_single_mlp(
            train_features[fit_indices],
            affinities[fit_indices],
            train_features[validation_indices],
            affinities[validation_indices],
            test_features,
            hidden=hidden,
            seed=seed,
            device=device,
        )
        for hidden in MLP_HIDDEN_SIZES
        for seed in MLP_SEEDS
    ]
    return np.mean(predictions, axis=0)


@dataclass(frozen=True)
class PmhcDataset:
    all_nine_mer_hla_rows: pd.DataFrame
    rows: pd.DataFrame
    eligible_alleles: tuple[str, ...]


def load_source_contract(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("pMHC source contract must be a JSON object")
    return loaded


def _require_matching_counts(
    field: str, contract: Mapping[str, object], actual_counts: Mapping[str, int]
) -> None:
    expected_counts = contract.get(field)
    if not isinstance(expected_counts, Mapping):
        raise ValueError(f"source contract {field} must be an object")
    if set(expected_counts) != set(actual_counts):
        missing = sorted(set(actual_counts) - set(expected_counts))
        unexpected = sorted(set(expected_counts) - set(actual_counts))
        raise ValueError(
            f"{field} keys mismatch: missing {missing}, unexpected {unexpected}"
        )
    for key, actual in actual_counts.items():
        expected = expected_counts.get(key)
        if expected != actual:
            raise ValueError(
                f"{field} mismatch for {key}: expected {expected}, got {actual}"
            )


def verify_source(
    source_dir: Path, archive_path: Path, contract: Mapping[str, object]
) -> None:
    expected_archive_hash = contract.get("archive_sha256")
    if not isinstance(expected_archive_hash, str):
        raise ValueError("source contract archive_sha256 must be a string")
    actual_archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual_archive_hash != expected_archive_hash:
        raise ValueError(
            "archive SHA-256 mismatch: "
            f"expected {expected_archive_hash}, got {actual_archive_hash}"
        )
    file_hashes = contract.get("file_sha256")
    if not isinstance(file_hashes, Mapping):
        raise ValueError("source contract file_sha256 must be an object")
    for filename in (*FOLD_NAMES, "MHC_pseudo.dat"):
        expected_file_hash = file_hashes.get(filename)
        if not isinstance(expected_file_hash, str):
            raise ValueError(f"missing SHA-256 for extracted file {filename}")
        actual_file_hash = hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
        if actual_file_hash != expected_file_hash:
            raise ValueError(
                f"extracted file SHA-256 mismatch for {filename}: "
                f"expected {expected_file_hash}, got {actual_file_hash}"
            )

    raw_rows = _load_raw_rows(source_dir)
    is_hla_abc = raw_rows["Allele"].str.startswith(HLA_PREFIXES)
    hla_abc_rows = raw_rows.loc[is_hla_abc]
    is_nine_mer = hla_abc_rows["Peptide"].str.len() == 9
    nine_mer_hla_rows = hla_abc_rows.loc[is_nine_mer]
    _require_matching_counts(
        "source_counts",
        contract,
        {
            "ba_rows_all_species": len(raw_rows),
            "hla_abc_rows": len(hla_abc_rows),
            "hla_abc_positive": int(hla_abc_rows["Target"].sum()),
            "hla_abc_artificial_negative": int(
                (hla_abc_rows["RowType"] == "artificial_negative").sum()
            ),
            "nine_mer_hla_rows": len(nine_mer_hla_rows),
            "nine_mer_hla_positive": int(nine_mer_hla_rows["Target"].sum()),
            "nine_mer_hla_negative": int((~nine_mer_hla_rows["Target"]).sum()),
        },
    )

    dataset = load_pmhc_dataset(source_dir)
    _require_matching_counts(
        "headline_counts",
        contract,
        {
            "alleles": len(dataset.eligible_alleles),
            "rows": len(dataset.rows),
            "positive": int(dataset.rows["Target"].sum()),
            "negative": int((~dataset.rows["Target"]).sum()),
            "measured_nonbinder": int(
                (dataset.rows["RowType"] == "measured_nonbinder").sum()
            ),
            "artificial_negative": int(
                (dataset.rows["RowType"] == "artificial_negative").sum()
            ),
        },
    )


def _parse_fold(path: Path) -> pd.DataFrame:
    records: list[tuple[str, float, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"malformed row in {path.name} at line {line_number}")
        peptide, affinity_text, allele = fields
        invalid_residues = set(peptide) - STANDARD_AMINO_ACIDS
        if invalid_residues:
            raise ValueError(
                f"invalid peptide residue in {path.name} at line {line_number}: "
                f"{sorted(invalid_residues)}"
            )
        try:
            affinity = float(affinity_text)
        except ValueError as error:
            raise ValueError(
                f"malformed row in {path.name} at line {line_number}: "
                f"invalid affinity {affinity_text!r}"
            ) from error
        if not math.isfinite(affinity) or not 0.0 <= affinity <= 1.0:
            raise ValueError(
                f"affinity outside [0, 1] in {path.name} at line {line_number}: "
                f"{affinity_text}"
            )
        records.append((peptide, affinity, allele))
    frame = pd.DataFrame(records, columns=["Peptide", "Affinity", "Allele"])
    frame["Fold"] = path.name
    return frame


def _load_raw_rows(source_dir: Path) -> pd.DataFrame:
    actual_fold_names = {path.name for path in source_dir.glob("c*_ba")}
    if actual_fold_names != set(FOLD_NAMES):
        raise ValueError(
            f"fold set mismatch: expected {list(FOLD_NAMES)}, "
            f"got {sorted(actual_fold_names)}"
        )
    raw_rows = pd.concat(
        [_parse_fold(source_dir / fold_name) for fold_name in FOLD_NAMES],
        ignore_index=True,
    )
    raw_rows["Target"] = raw_rows["Affinity"] > POSITIVE_THRESHOLD
    raw_rows["RowType"] = [
        "positive"
        if affinity > POSITIVE_THRESHOLD
        else (
            "artificial_negative"
            if affinity == ARTIFICIAL_NEGATIVE_AFFINITY
            else "measured_nonbinder"
        )
        for affinity in raw_rows["Affinity"]
    ]
    return raw_rows


def _require_both_classes_per_fold(
    rows: pd.DataFrame, eligible_alleles: tuple[str, ...]
) -> None:
    for fold_name in FOLD_NAMES:
        is_test = rows["Fold"] == fold_name
        test_rows = rows.loc[is_test]
        train_rows = rows.loc[~is_test]
        for allele in eligible_alleles:
            test_targets = set(test_rows.loc[test_rows["Allele"] == allele, "Target"])
            if test_targets != {True, False}:
                raise ValueError(
                    f"missing class in test fold {fold_name} for allele {allele}"
                )
            train_targets = set(train_rows.loc[train_rows["Allele"] == allele, "Target"])
            if train_targets != {True, False}:
                raise ValueError(
                    "missing class in training complement of fold "
                    f"{fold_name} for allele {allele}"
                )


def load_pmhc_dataset(
    source_dir: Path, *, minimum_class_rows: int = 100
) -> PmhcDataset:
    raw_rows = _load_raw_rows(source_dir)
    is_hla_abc = raw_rows["Allele"].str.startswith(HLA_PREFIXES)
    is_nine_mer = raw_rows["Peptide"].str.len() == 9
    nine_mer_hla_rows = raw_rows.loc[is_hla_abc & is_nine_mer].copy()
    peptide_fold_counts = nine_mer_hla_rows.groupby("Peptide")["Fold"].nunique()
    overlapping_peptides = peptide_fold_counts.loc[peptide_fold_counts > 1].index.tolist()
    if overlapping_peptides:
        raise ValueError(
            "peptide overlap across folds: " f"{overlapping_peptides[:5]}"
        )
    class_counts = nine_mer_hla_rows.groupby(["Allele", "Target"]).size().unstack(fill_value=0)
    eligible_alleles = tuple(
        sorted(
            allele
            for allele, counts in class_counts.iterrows()
            if counts.get(True, 0) >= minimum_class_rows
            and counts.get(False, 0) >= minimum_class_rows
        )
    )
    if not eligible_alleles:
        raise ValueError(
            f"no allele meets minimum class support of {minimum_class_rows} rows per class"
        )
    rows = nine_mer_hla_rows.loc[
        nine_mer_hla_rows["Allele"].isin(eligible_alleles)
    ].reset_index(drop=True)
    _require_both_classes_per_fold(rows, eligible_alleles)
    return PmhcDataset(
        all_nine_mer_hla_rows=nine_mer_hla_rows.reset_index(drop=True),
        rows=rows,
        eligible_alleles=eligible_alleles,
    )


def iter_pmhc_folds(
    dataset: PmhcDataset,
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield `(fold_name, train_rows, test_rows)` for each fold, held out as test in turn."""
    for fold_name in FOLD_NAMES:
        is_test = dataset.rows["Fold"] == fold_name
        yield (
            fold_name,
            dataset.rows.loc[~is_test].reset_index(drop=True),
            dataset.rows.loc[is_test].reset_index(drop=True),
        )


def load_pseudo_sequences(path: Path, alleles: Sequence[str]) -> dict[str, str]:
    by_key: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"malformed pseudo-sequence row in {path.name} at line {line_number}"
            )
        by_key[fields[0]] = fields[1]
    pseudo_sequences: dict[str, str] = {}
    for allele in alleles:
        sequence = by_key.get(allele.replace(":", ""))
        if sequence is None:
            raise ValueError(f"missing pseudo-sequence for allele {allele}")
        if len(sequence) != PSEUDO_SEQUENCE_LENGTH:
            raise ValueError(
                f"pseudo-sequence for allele {allele} has {len(sequence)} residues, "
                f"expected {PSEUDO_SEQUENCE_LENGTH} residues"
            )
        pseudo_sequences[allele] = sequence
    return pseudo_sequences
