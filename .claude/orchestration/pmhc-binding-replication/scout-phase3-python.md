file: /Users/freddy/Documents/repos/cognate/src/cognate/pmhc.py
mode: full
entries:
  - kind: import
    name: stdlib
    lines: 7-15
    signature: from __future__ import annotations; import hashlib, json, math; from collections.abc import Iterator, Mapping, Sequence; from dataclasses import dataclass; from functools import cache; from pathlib import Path
    behavior: Supplies hashing, JSON-contract parsing, numeric validation, typed collections, immutable records, cached BLOSUM loading, and paths.
  - kind: import
    name: third_party
    lines: 17-22
    signature: import numpy as np; import pandas as pd; import torch; from Bio.Align import substitution_matrices; from sklearn.model_selection import StratifiedShuffleSplit; from sklearn.preprocessing import StandardScaler
    behavior: Supplies array/dataframe operations, MLP training, BLOSUM50, stratified inner splitting, and fit-only feature scaling.
  - kind: import
    name: cognate
    lines: 24-30
    signature: from cognate.baseline_knn import KnnResult, SimilarityFn, score_by_nearest_positive; from cognate.train_head import DEFAULT_BATCH_SIZE, MlpHead, fit_logistic, logistic_scores
    behavior: Reuses the shared nearest-positive retrieval operator and training-head primitives.
  - kind: constant
    name: FOLD_NAMES
    lines: 32-32
    signature: FOLD_NAMES = tuple(f"c00{index}_ba" for index in range(5))
    behavior: Fixes the five outer-fold filenames and their evaluation order.
  - kind: constant
    name: HLA_PREFIXES
    lines: 33-33
    signature: HLA_PREFIXES = ("HLA-A", "HLA-B", "HLA-C")
    behavior: Limits the experiment to classical HLA-A/B/C rows.
  - kind: constant
    name: POSITIVE_THRESHOLD
    lines: 34-34
    signature: POSITIVE_THRESHOLD = 0.426
    behavior: Labels affinity strictly greater than 0.426 as positive, leaving 0.426 negative.
  - kind: constant
    name: ARTIFICIAL_NEGATIVE_AFFINITY
    lines: 35-35
    signature: ARTIFICIAL_NEGATIVE_AFFINITY = 0.01
    behavior: Identifies exact 0.01 rows as artificial negatives; all other nonpositive affinities are measured nonbinders.
  - kind: constant
    name: STANDARD_AMINO_ACIDS
    lines: 36-36
    signature: STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
    behavior: Defines the only accepted peptide residues during fold parsing.
  - kind: constant
    name: AMINO_ACID_ORDER
    lines: 37-37
    signature: AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    behavior: Fixes composition-count and BLOSUM feature column order.
  - kind: constant
    name: PSEUDO_SEQUENCE_LENGTH
    lines: 38-38
    signature: PSEUDO_SEQUENCE_LENGTH = 34
    behavior: Enforces 34-residue MHC pseudo-sequences.
  - kind: constant
    name: MLP_HIDDEN_SIZES
    lines: 39-39
    signature: MLP_HIDDEN_SIZES = (55, 66)
    behavior: Defines the two ensemble hidden widths.
  - kind: constant
    name: MLP_SEEDS
    lines: 40-40
    signature: MLP_SEEDS = tuple(range(5))
    behavior: Defines five seeds per hidden width, producing ten models per fold.
  - kind: constant
    name: MLP_MAX_EPOCHS
    lines: 41-41
    signature: MLP_MAX_EPOCHS = 200
    behavior: Caps each MLP member at 200 epochs.
  - kind: constant
    name: MLP_PATIENCE
    lines: 42-42
    signature: MLP_PATIENCE = 20
    behavior: Stops after 20 validation epochs without improvement.
  - kind: constant
    name: MLP_LEARNING_RATE
    lines: 43-43
    signature: MLP_LEARNING_RATE = 1e-3
    behavior: Fixes Adam learning rate for continuous-affinity MLP training.
  - kind: function
    name: random_scores
    lines: 46-48
    signature: def random_scores(n_rows: int) -> np.ndarray
    behavior: Returns shape (n_rows,) from np.random.default_rng(0).random in caller row order.
  - kind: function
    name: build_composition_features
    lines: 51-59
    signature: def build_composition_features(sequences: Sequence[str]) -> np.ndarray
    behavior: Returns float shape (n, 21): sequence length followed by counts for the 20 fixed amino acids.
  - kind: function
    name: score_composition_fold
    lines: 62-70
    signature: def score_composition_fold(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray
    behavior: Fits logistic regression only on train Peptide/Target and returns one probability per test row.
  - kind: function
    name: score_retrieval_fold
    lines: 73-93
    signature: def score_retrieval_fold(test: pd.DataFrame, train: pd.DataFrame, *, similarity_fn: SimilarityFn | None = None) -> KnnResult
    behavior: Delegates to score_by_nearest_positive with peptide_column="Allele" and sequence_column="Peptide", optionally forwarding a custom similarity function.
  - kind: function
    name: score_pwm_fold
    lines: 96-137
    signature: def score_pwm_fold(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray
    behavior: Builds allele-specific 9x20 Laplace-smoothed positive/negative log-odds matrices and returns summed position scores shape (len(test),).
  - kind: function
    name: _blosum50_rows
    lines: 140-148
    signature: @cache; def _blosum50_rows() -> dict[str, np.ndarray]
    behavior: Loads raw BLOSUM50 once and maps each standard residue to its fixed-order 20-value row.
  - kind: function
    name: encode_blosum50
    lines: 151-161
    signature: def encode_blosum50(sequences: Sequence[str]) -> np.ndarray
    behavior: Returns flattened raw BLOSUM rows shape (n, sequence_length*20), or float shape (0, 0) for no sequences.
  - kind: function
    name: build_mlp_features
    lines: 164-185
    signature: def build_mlp_features(rows: pd.DataFrame, pseudo_sequences: Mapping[str, str], allele_order: Sequence[str], *, use_one_hot_allele: bool) -> np.ndarray
    behavior: Concatenates 9-mer BLOSUM peptide features with either 34-residue pseudo-sequence BLOSUM features (860 columns total) or allele one-hot features (180+len(allele_order) columns).
  - kind: function
    name: split_fit_validation
    lines: 188-200
    signature: def split_fit_validation(rows: pd.DataFrame, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]
    behavior: Returns deterministic fit and validation integer-index arrays from one 90/10 StratifiedShuffleSplit over joint Allele|Target strata.
  - kind: function
    name: _score_single_mlp
    lines: 203-277
    signature: def _score_single_mlp(fit_features: np.ndarray, fit_affinities: np.ndarray, validation_features: np.ndarray, validation_affinities: np.ndarray, test_features: np.ndarray, *, hidden: int, seed: int, device: str) -> np.ndarray
    behavior: Fits StandardScaler only on fit features, trains MlpHead with Adam/MSE on sigmoid outputs against continuous affinities, restores best validation weights, and returns test probabilities.
  - kind: function
    name: score_mlp_fold
    lines: 280-319
    signature: def score_mlp_fold(train: pd.DataFrame, test: pd.DataFrame, pseudo_sequences: Mapping[str, str], allele_order: Sequence[str], fit_indices: np.ndarray, validation_indices: np.ndarray, *, use_one_hot_allele: bool, device: str) -> np.ndarray
    behavior: Builds train/test features once, trains all 2x5 hidden/seed members on shared fit/validation membership, and returns their row-wise mean shape (len(test),).
  - kind: dataclass
    name: PmhcDataset
    lines: 322-326
    signature: @dataclass(frozen=True) class PmhcDataset(all_nine_mer_hla_rows: pd.DataFrame, rows: pd.DataFrame, eligible_alleles: tuple[str, ...])
    behavior: Carries all filtered nine-mer HLA rows, the eligible-allele subset, and deterministic sorted allele names.
  - kind: function
    name: load_source_contract
    lines: 329-333
    signature: def load_source_contract(path: Path) -> dict[str, object]
    behavior: Parses UTF-8 JSON and rejects any non-object top level.
  - kind: function
    name: _require_matching_counts
    lines: 336-353
    signature: def _require_matching_counts(field: str, contract: Mapping[str, object], actual_counts: Mapping[str, int]) -> None
    behavior: Requires an exact count-key set and exact values, distinguishing key mismatch from per-key value mismatch.
  - kind: function
    name: verify_source
    lines: 356-419
    signature: def verify_source(source_dir: Path, archive_path: Path, contract: Mapping[str, object]) -> None
    behavior: Verifies archive and six extracted-file SHA-256 hashes, then exact source_counts and headline_counts derived through the same loaders used by the experiment.
  - kind: function
    name: _parse_fold
    lines: 422-452
    signature: def _parse_fold(path: Path) -> pd.DataFrame
    behavior: Parses exactly three whitespace fields into Peptide/Affinity/Allele plus Fold, rejecting invalid residues, non-float/nonfinite affinity, and affinity outside [0,1].
  - kind: function
    name: _load_raw_rows
    lines: 455-477
    signature: def _load_raw_rows(source_dir: Path) -> pd.DataFrame
    behavior: Requires exactly the five c00x_ba files, concatenates them, and adds boolean Target plus positive/measured_nonbinder/artificial_negative RowType.
  - kind: function
    name: _require_both_classes_per_fold
    lines: 480-498
    signature: def _require_both_classes_per_fold(rows: pd.DataFrame, eligible_alleles: tuple[str, ...]) -> None
    behavior: Requires both boolean classes for every eligible allele in each held-out fold and in every four-fold training complement.
  - kind: function
    name: load_pmhc_dataset
    lines: 501-535
    signature: def load_pmhc_dataset(source_dir: Path, *, minimum_class_rows: int = 100) -> PmhcDataset
    behavior: Filters HLA-A/B/C nine-mers, rejects peptide leakage across folds, selects alleles with at least minimum_class_rows in each class, validates fold class coverage, and resets output indices.
  - kind: function
    name: iter_pmhc_folds
    lines: 538-548
    signature: def iter_pmhc_folds(dataset: PmhcDataset) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]
    behavior: Yields five (fold_name, train_rows, test_rows) triples with reset indices and each fold held out once.
  - kind: function
    name: load_pseudo_sequences
    lines: 551-575
    signature: def load_pseudo_sequences(path: Path, alleles: Sequence[str]) -> dict[str, str]
    behavior: Parses two-field pseudo rows, resolves colonless keys for requested alleles, and rejects missing or non-34-residue sequences.

file: /Users/freddy/Documents/repos/cognate/src/cognate/metrics.py
mode: full
entries:
  - kind: import
    name: stdlib
    lines: 13-16
    signature: import warnings; from collections.abc import Callable, Iterator, Sequence; from dataclasses import dataclass; from typing import Literal
    behavior: Supplies warnings, callable/sequence typing, immutable records, and literal policy types.
  - kind: import
    name: third_party
    lines: 18-20
    signature: import numpy as np; import pandas as pd; from sklearn.metrics import average_precision_score, roc_auc_score
    behavior: Supplies array/dataframe structures and canonical ROC/AP scorers.
  - kind: constant
    name: MAX_FPR
    lines: 22-22
    signature: MAX_FPR = 0.1
    behavior: Fixes partial ROC evaluation at 10% false-positive rate.
  - kind: constant
    name: DEFAULT_N_BOOTSTRAP
    lines: 23-23
    signature: DEFAULT_N_BOOTSTRAP = 1000
    behavior: Sets the default evaluation and comparison replicate count.
  - kind: constant
    name: DEFAULT_CONFIDENCE
    lines: 24-24
    signature: DEFAULT_CONFIDENCE = 0.95
    behavior: Sets percentile interval coverage.
  - kind: constant
    name: DEFAULT_SEED
    lines: 25-25
    signature: DEFAULT_SEED = 0
    behavior: Makes bootstrap outputs deterministic by default.
  - kind: constant
    name: ResampleMode
    lines: 27-27
    signature: ResampleMode = Literal["rows", "groups", "both"]
    behavior: Selects row, peptide-group, or two-level resampling.
  - kind: constant
    name: DegenerateAction
    lines: 28-28
    signature: DegenerateAction = Literal["ignore", "warn", "raise"]
    behavior: Selects how evaluate reacts to nonfinite or constant-within-group score columns.
  - kind: constant
    name: TIE_HEAVY_FRACTION
    lines: 30-30
    signature: TIE_HEAVY_FRACTION = 0.25
    behavior: Flags a score distribution when one value covers at least one quarter of rows.
  - kind: constant
    name: Scorer
    lines: 32-32
    signature: Scorer = Callable[[np.ndarray, np.ndarray], float]
    behavior: Defines the per-group metric callable contract.
  - kind: function
    name: auroc
    lines: 35-37
    signature: def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float
    behavior: Returns sklearn full ROC AUC as float.
  - kind: function
    name: auprc
    lines: 40-42
    signature: def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float
    behavior: Returns sklearn average precision as float.
  - kind: function
    name: auc01
    lines: 45-47
    signature: def auc01(y_true: np.ndarray, y_score: np.ndarray) -> float
    behavior: Returns McClish-standardized sklearn ROC AUC truncated at FPR 0.1.
  - kind: dataclass
    name: Interval
    lines: 50-64
    signature: @dataclass(frozen=True) class Interval(point: float, lo: float, hi: float, n_replicates: int)
    behavior: Stores point/percentile bounds/retained-replicate count, exposes width, and formats as point [lo, hi].
  - kind: class
    name: DegenerateScoresError
    lines: 67-68
    signature: class DegenerateScoresError(ValueError)
    behavior: Signals score columns that cannot express a ranking when strict handling is requested.
  - kind: dataclass
    name: ScoreDiagnostics
    lines: 71-112
    signature: @dataclass(frozen=True) class ScoreDiagnostics(n_rows: int, n_unique_scores: int, n_nonfinite: int, modal_fraction: float, constant_groups: tuple[str, ...])
    behavior: Reports unique/nonfinite/tie/constant-group state with computed n_constant_groups, is_degenerate, and is_tie_heavy properties.
  - kind: function
    name: diagnose_scores
    lines: 115-140
    signature: def diagnose_scores(y_score: np.ndarray, groups: Sequence[str] | np.ndarray | pd.Series | None = None) -> ScoreDiagnostics
    behavior: Coerces scores to float, counts finite distinct/modal values, and marks any group with at most one unique score as constant; omitted groups become __all__.
  - kind: function
    name: _handle_degenerate
    lines: 143-155
    signature: def _handle_degenerate(diagnostics: ScoreDiagnostics, label: str, action: DegenerateAction) -> None
    behavior: Ignores, warns, or raises DegenerateScoresError with the slice label according to policy.
  - kind: dataclass
    name: GroupedScore
    lines: 158-169
    signature: @dataclass(frozen=True) class GroupedScore(value: float, per_group: dict[str, float], n_groups_skipped: int, skipped_groups: tuple[str, ...])
    behavior: Stores the macro mean, per-group scores, skipped groups, and derived scored-group count.
  - kind: dataclass
    name: ScoreReport
    lines: 172-198
    signature: @dataclass(frozen=True) class ScoreReport(label: str, n_rows: int, n_positive: int, n_groups_scored: int, n_groups_skipped: int, skipped_groups: tuple[str, ...], auroc: Interval, auprc: Interval, macro_auc01: Interval, per_group_auc01: dict[str, float], diagnostics: ScoreDiagnostics)
    behavior: Bundles three interval metrics, slice counts, per-peptide AUC0.1 values, and degeneracy diagnostics for one evaluated slice.
  - kind: function
    name: _both_classes_present
    lines: 201-202
    signature: def _both_classes_present(y_true: np.ndarray) -> bool
    behavior: Returns true only when positives are nonzero and fewer than total rows.
  - kind: function
    name: _group_indices
    lines: 205-210
    signature: def _group_indices(groups: np.ndarray) -> dict[str, np.ndarray]
    behavior: Returns first-appearance-ordered string group names mapped to integer row-index arrays.
  - kind: function
    name: macro_by_group
    lines: 213-242
    signature: def macro_by_group(y_true: np.ndarray, y_score: np.ndarray, groups: Sequence[str] | np.ndarray | pd.Series, scorer: Scorer = auc01) -> GroupedScore
    behavior: Scores groups with both classes, skips one-class groups, and returns their unweighted mean or NaN if none are scorable.
  - kind: function
    name: macro_auc01
    lines: 245-251
    signature: def macro_auc01(y_true: np.ndarray, y_score: np.ndarray, groups: Sequence[str] | np.ndarray | pd.Series) -> GroupedScore
    behavior: Computes IMMREP23 headline AUC0.1 independently per peptide and averages peptides equally.
  - kind: function
    name: _replicate_group_indices
    lines: 254-276
    signature: def _replicate_group_indices(indices: dict[str, np.ndarray], mode: ResampleMode, n_boot: int, rng: np.random.Generator) -> Iterator[list[np.ndarray]]
    behavior: Yields bootstrap replicate lists by resampling groups, rows within groups, or both according to mode.
  - kind: function
    name: _percentile_interval
    lines: 279-286
    signature: def _percentile_interval(point: float, samples: list[float], confidence: float) -> Interval
    behavior: Returns percentile bounds and sample count, or NaN bounds with zero replicates when no samples survive.
  - kind: function
    name: _pearson
    lines: 289-292
    signature: def _pearson(x: np.ndarray, y: np.ndarray) -> float
    behavior: Returns Pearson r or NaN for fewer than two points or zero variance.
  - kind: function
    name: correlation_interval
    lines: 295-321
    signature: def correlation_interval(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, *, n_boot: int = 5_000, confidence: float = DEFAULT_CONFIDENCE, seed: int = DEFAULT_SEED) -> Interval
    behavior: Validates equal length, bootstraps paired points, drops variance-free replicates, and returns Pearson r with percentile CI.
  - kind: function
    name: correlation_difference_interval
    lines: 324-367
    signature: def correlation_difference_interval(x: Sequence[float] | np.ndarray, y_a: Sequence[float] | np.ndarray, y_b: Sequence[float] | np.ndarray, *, n_boot: int = 5_000, confidence: float = DEFAULT_CONFIDENCE, seed: int = DEFAULT_SEED) -> tuple[Interval, float]
    behavior: Uses one shared point resample for corr(x,y_a)-corr(x,y_b), returns its interval and symmetric two-sided bootstrap tail p-value.
  - kind: function
    name: leave_one_out_correlations
    lines: 370-387
    signature: def leave_one_out_correlations(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> np.ndarray
    behavior: Returns shape (n_points,) Pearson correlations with each paired point excluded in turn.
  - kind: function
    name: evaluate
    lines: 390-464
    signature: def evaluate(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series, groups: Sequence[str] | np.ndarray | pd.Series | None = None, *, subset: np.ndarray | pd.Series | None = None, label: str = "all", mode: ResampleMode = "both", n_boot: int = DEFAULT_N_BOOTSTRAP, confidence: float = DEFAULT_CONFIDENCE, seed: int = DEFAULT_SEED, on_degenerate: DegenerateAction = "warn") -> ScoreReport
    behavior: Applies one boolean subset to labels/scores/groups, rejects empty slices, diagnoses degeneracy, computes pooled AUROC/AUPRC plus macro group AUC0.1, and bootstraps percentile intervals with skipped one-class replicates.
  - kind: dataclass
    name: Comparison
    lines: 467-484
    signature: @dataclass(frozen=True) class Comparison(label_a: str, label_b: str, point_a: float, point_b: float, difference: Interval, p_two_sided: float, n_groups: int)
    behavior: Stores paired macro-AUC0.1 arm values, their interval difference, p-value, and common peptide count.
  - kind: function
    name: compare_macro_auc01
    lines: 487-590
    signature: def compare_macro_auc01(y_true: np.ndarray | pd.Series, groups: Sequence[str] | np.ndarray | pd.Series, a: tuple[str, np.ndarray, np.ndarray | None], b: tuple[str, np.ndarray, np.ndarray | None], *, mode: ResampleMode = "both", n_boot: int = DEFAULT_N_BOOTSTRAP, confidence: float = DEFAULT_CONFIDENCE, seed: int = DEFAULT_SEED) -> Comparison
    behavior: Restricts to common peptides scorable in both (label, score, subset) arms, preserves shared row resamples for identical row sets, and returns paired macro-AUC0.1 difference with symmetric tail p-value.
  - kind: function
    name: report_table
    lines: 593-610
    signature: def report_table(reports: Sequence[ScoreReport]) -> pd.DataFrame
    behavior: Returns one index row per report label with n, pos, peptides, skipped, formatted macro AUC0.1/AUROC/AUPRC, and a degeneracy marker column.

file: /Users/freddy/Documents/repos/cognate/src/cognate/embed.py
mode: full
entries:
  - kind: import
    name: stdlib
    lines: 19-23
    signature: import os, time; from collections.abc import Iterable, Sequence; from dataclasses import dataclass; from pathlib import Path
    behavior: Supplies environment configuration, timing, iterable typing, immutable cache records, and paths.
  - kind: import
    name: third_party
    lines: 25-27
    signature: import numpy as np; import torch; from transformers import AutoModel, AutoTokenizer
    behavior: Supplies cache arrays, tensor/device execution, and Hugging Face ESM-2 loading/tokenization.
  - kind: constant
    name: MODELS
    lines: 29-32
    signature: MODELS = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D"}
    behavior: Maps experiment model keys to exact ESM-2 checkpoint names.
  - kind: constant
    name: DEFAULT_MAX_BATCH_TOKENS
    lines: 34-34
    signature: DEFAULT_MAX_BATCH_TOKENS = 16_384
    behavior: Bounds padded tokens per embedding batch.
  - kind: constant
    name: CACHE_SUFFIX
    lines: 35-35
    signature: CACHE_SUFFIX = ".npz"
    behavior: Defines the NumPy archive suffix for embedding caches.
  - kind: constant
    name: CACHE_DIR_ENV
    lines: 36-36
    signature: CACHE_DIR_ENV = "COGNATE_CACHE_DIR"
    behavior: Names the environment override for cache location.
  - kind: constant
    name: SHARED_CACHE_DIRNAME
    lines: 37-37
    signature: SHARED_CACHE_DIRNAME = "shared"
    behavior: Names the repository-root default cache directory.
  - kind: dataclass
    name: EmbeddingCache
    lines: 40-78
    signature: @dataclass(frozen=True) class EmbeddingCache(model_name: str, sequences: np.ndarray, layers: np.ndarray, index: dict[str, int])
    behavior: Stores pooled layers shape (n_layers, n_sequences, hidden), exposes dimensions/layer arrays, and lookup returns ordered shape (n_requested, hidden) while raising on missing sequences.
  - kind: function
    name: _make_cache
    lines: 81-89
    signature: def _make_cache(model_name: str, sequences: np.ndarray, layers: np.ndarray) -> EmbeddingCache
    behavior: Constructs EmbeddingCache and rebuilds the string-to-row index from sequence order.
  - kind: function
    name: pick_device
    lines: 92-93
    signature: def pick_device() -> str
    behavior: Chooses mps when available, otherwise cpu.
  - kind: function
    name: _length_batches
    lines: 96-113
    signature: def _length_batches(sequences: list[str], max_batch_tokens: int) -> list[list[int]]
    behavior: Sorts indices by length then sequence and groups them so batch_size*(max_length+2) stays within the token budget.
  - kind: function
    name: _residue_mask
    lines: 116-122
    signature: def _residue_mask(attention_mask: torch.Tensor) -> torch.Tensor
    behavior: Clones the attention mask and clears BOS plus each row's EOS position.
  - kind: function
    name: embed_sequences
    lines: 125-165
    signature: def embed_sequences(sequences: Iterable[str], model_key: str = "8M", *, device: str | None = None, max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS, progress_every: int = 20) -> tuple[EmbeddingCache, float]
    behavior: Deduplicates/sorts strings, loads ESM without pooling, mean-pools residue-only hidden states for every layer into float32 shape (layers, unique, hidden), and returns cache plus elapsed seconds.
  - kind: function
    name: default_cache_dir
    lines: 168-178
    signature: def default_cache_dir() -> Path
    behavior: Returns expanded COGNATE_CACHE_DIR when set, else the repository-root shared directory.
  - kind: function
    name: cache_path
    lines: 181-182
    signature: def cache_path(model_key: str, directory: Path | None = None) -> Path
    behavior: Returns <directory-or-default>/esm2_<model_key>.npz.
  - kind: function
    name: save_cache
    lines: 185-197
    signature: def save_cache(cache: EmbeddingCache, path: Path) -> int
    behavior: Creates parents, writes uncompressed model_name/sequences-as-string/layers NPZ arrays, and returns file bytes.
  - kind: function
    name: load_cache
    lines: 200-206
    signature: def load_cache(path: Path) -> EmbeddingCache
    behavior: Loads NPZ with allow_pickle=False and reconstructs object sequences plus the index.
  - kind: dataclass
    name: ResidueCache
    lines: 209-260
    signature: @dataclass(frozen=True) class ResidueCache(model_name: str, sequences: np.ndarray, layer_indices: np.ndarray, offsets: np.ndarray, residues: np.ndarray, index: dict[str, int])
    behavior: Stores float32 ragged residue blocks shape (stored_layers, total_residues, hidden), resolves exact stored layers, returns (sequence_length, hidden) slices, and recomputes ordered pooled rows.
  - kind: function
    name: embed_residues
    lines: 263-316
    signature: def embed_residues(sequences: Iterable[str], model_key: str = "35M", *, layer_indices: Sequence[int] = (6, 10, 12), device: str | None = None, max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS, progress_every: int = 20) -> tuple[ResidueCache, float]
    behavior: Deduplicates/sorts strings, extracts BOS/EOS-free float32 residues for requested hidden layers into offsets-based storage, and returns cache plus elapsed seconds.
  - kind: function
    name: save_residue_cache
    lines: 319-329
    signature: def save_residue_cache(cache: ResidueCache, path: Path) -> int
    behavior: Creates parents, writes uncompressed model_name/sequences/layer_indices/offsets/residues NPZ arrays, and returns file bytes.
  - kind: function
    name: load_residue_cache
    lines: 332-342
    signature: def load_residue_cache(path: Path) -> ResidueCache
    behavior: Loads residue NPZ with allow_pickle=False and rebuilds the string sequence index.

file: /Users/freddy/Documents/repos/cognate/tests/test_pmhc.py
mode: full
entries:
  - kind: import
    name: stdlib
    lines: 1-6
    signature: from __future__ import annotations; import hashlib, json; from collections.abc import Callable, Iterable; from pathlib import Path
    behavior: Supplies fixtures/helpers for synthetic source files, hashing, JSON contracts, and typed monkeypatch probes.
  - kind: import
    name: third_party
    lines: 8-11
    signature: import numpy as np; import pandas as pd; import pytest; import torch
    behavior: Supplies assertions/data fixtures, parametrization/monkeypatch/tmp_path, and train-loop probes.
  - kind: import
    name: cognate
    lines: 13-29
    signature: import cognate.pmhc as pmhc; from cognate.pmhc import PmhcDataset, build_composition_features, build_mlp_features, encode_blosum50, iter_pmhc_folds, load_pmhc_dataset, load_pseudo_sequences, load_source_contract, random_scores, score_mlp_fold, score_pwm_fold, score_retrieval_fold, split_fit_validation, verify_source
    behavior: Imports public experiment APIs plus the module object needed to monkeypatch dependencies and private training seams.
  - kind: constant
    name: AMINO_ACIDS
    lines: 31-31
    signature: AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    behavior: Drives deterministic synthetic peptide generation.
  - kind: constant
    name: FOLD_NAMES
    lines: 32-32
    signature: FOLD_NAMES = tuple(f"c00{index}_ba" for index in range(5))
    behavior: Mirrors expected source fold names in fixtures and assertions.
  - kind: constant
    name: PSEUDO_SEQUENCE
    lines: 33-33
    signature: PSEUDO_SEQUENCE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"
    behavior: Provides a valid 34-residue pseudo-sequence fixture.
  - kind: function
    name: test_random_is_seeded_uniform
    lines: 36-41
    signature: def test_random_is_seeded_uniform() -> None
    behavior: Asserts exact first three default_rng(0) values, pinning deterministic row-order random scores.
  - kind: function
    name: test_composition_columns_are_fixed
    lines: 44-52
    signature: def test_composition_columns_are_fixed() -> None
    behavior: Asserts exact 1x21 length-plus-residue-count output and column order.
  - kind: function
    name: test_composition_fold_fits_train_target_and_scores_test_only
    lines: 55-100
    signature: def test_composition_fold_fits_train_target_and_scores_test_only(monkeypatch: pytest.MonkeyPatch) -> None
    behavior: Monkeypatches fit_logistic/logistic_scores to prove one train-only fit, test-only scoring, integer targets, and identity-preserved returned score array.
  - kind: function
    name: test_retrieval_is_per_allele
    lines: 103-118
    signature: def test_retrieval_is_per_allele() -> None
    behavior: Uses a cross-allele exact peptide decoy to assert lookup remains allele-specific and returns the expected nearest sequence/zero score.
  - kind: function
    name: test_retrieval_operator_is_shared
    lines: 121-152
    signature: def test_retrieval_operator_is_shared(monkeypatch: pytest.MonkeyPatch) -> None
    behavior: Monkeypatches score_by_nearest_positive and asserts exact positional/keyword forwarding for edit and custom-similarity branches.
  - kind: function
    name: test_pwm_matches_hand_calculation
    lines: 155-173
    signature: def test_pwm_matches_hand_calculation() -> None
    behavior: Pins Laplace denominators and summed 9-position allele-specific log-odds against a hand calculation.
  - kind: function
    name: test_blosum50_values_are_raw
    lines: 176-205
    signature: def test_blosum50_values_are_raw() -> None
    behavior: Asserts exact raw A/W BLOSUM50 rows plus pseudo-feature shape (1,860), one-hot shape (1,227) for 47 alleles, and one-hot start at column 180.
  - kind: function
    name: test_validation_membership_is_shared
    lines: 208-231
    signature: def test_validation_membership_is_shared() -> None
    behavior: Asserts deterministic seed-0 36/4 split and one validation row for every Allele/Target stratum.
  - kind: function
    name: _mlp_rows
    lines: 234-247
    signature: def _mlp_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], tuple[str, ...]]
    behavior: Returns a four-row one-allele train fixture, one-row test fixture, pseudo mapping, and allele order.
  - kind: function
    name: test_scaler_uses_fit_partition_only
    lines: 250-284
    signature: def test_scaler_uses_fit_partition_only(monkeypatch: pytest.MonkeyPatch) -> None
    behavior: Replaces StandardScaler with a recorder and asserts every ensemble scaler sees only fit-partition features.
  - kind: function
    name: test_mlp_uses_adam_mse_with_sigmoid_and_affinity_targets
    lines: 287-357
    signature: def test_mlp_uses_adam_mse_with_sigmoid_and_affinity_targets(monkeypatch: pytest.MonkeyPatch) -> None
    behavior: Uses probe head/optimizer/loss seams to assert Adam lr=1e-3, sigmoid-before-MSE, continuous fit affinities, and validation affinity targets.
  - kind: function
    name: test_mlp_restores_best_weights
    lines: 360-411
    signature: def test_mlp_restores_best_weights(monkeypatch: pytest.MonkeyPatch) -> None
    behavior: Uses a deterministic stepping optimizer and probe head to assert the best validation state is reloaded before test prediction.
  - kind: function
    name: test_ensemble_uses_all_ten_models
    lines: 414-446
    signature: def test_ensemble_uses_all_ten_models(monkeypatch: pytest.MonkeyPatch) -> None
    behavior: Replaces _score_single_mlp and asserts exact (55,66)x(seed 0..4) call order and arithmetic mean output.
  - kind: function
    name: _peptide
    lines: 449-455
    signature: def _peptide(number: int, length: int = 9) -> str
    behavior: Deterministically encodes an integer into a standard-amino-acid sequence for collision-controlled fixtures.
  - kind: function
    name: _write_valid_source
    lines: 458-542
    signature: def _write_valid_source(root: Path) -> tuple[Path, Path, Path, str, str, str, str]
    behavior: Writes five synthetic BA folds, ignored EL file, pseudo file, archive bytes, and matching JSON contract while returning paths plus boundary peptides.
  - kind: function
    name: _sha256
    lines: 545-546
    signature: def _sha256(path: Path) -> str
    behavior: Returns a file SHA-256 hex digest for fixture contracts.
  - kind: function
    name: _valid_contract
    lines: 549-575
    signature: def _valid_contract(source_dir: Path, archive_path: Path) -> dict[str, object]
    behavior: Builds exact archive/file hashes, seven source counts, and six headline counts for the synthetic dataset.
  - kind: function
    name: _replace_first_line
    lines: 578-581
    signature: def _replace_first_line(path: Path, transform: Callable[[list[str]], list[str]]) -> None
    behavior: Mutates the first whitespace-split fixture row through a supplied transform.
  - kind: function
    name: test_valid_source_is_parsed_and_partitioned
    lines: 584-638
    signature: def test_valid_source_is_parsed_and_partitioned(tmp_path: Path) -> None
    behavior: End-to-end synthetic integration test verifies hashes/counts, eligibility, exact target/row-type boundaries, 180/45 fold complements, fold membership, and colon-normalized pseudo lookup.
  - kind: function
    name: _run_corruption
    lines: 641-720
    signature: def _run_corruption(case: str, root: Path) -> None
    behavior: Dispatches isolated corruptions for archive/file hashes, fold set/rows/affinity/residues/leakage/support/classes, and pseudo presence/length/shape.
  - kind: constant
    name: CORRUPTION_CASES
    lines: 723-743
    signature: CORRUPTION_CASES = ((case_name, expected_error_fragment), ... x16)
    behavior: Defines the 16 corruption/error-message integration cases.
  - kind: function
    name: test_contract_corruption_is_rejected
    lines: 746-755
    signature: @pytest.mark.parametrize(("case", "message"), CORRUPTION_CASES, ids=...); def test_contract_corruption_is_rejected(case: str, message: str, tmp_path: Path) -> None
    behavior: Requires ValueError with the case-specific message fragment for every corruption path.
  - kind: constant
    name: CONTRACT_COUNT_FIELDS
    lines: 758-772
    signature: CONTRACT_COUNT_FIELDS = (("source_counts", <7 fields>), ("headline_counts", <6 fields>))
    behavior: Enumerates all 13 contract count fields that must match exactly.
  - kind: constant
    name: CONTRACT_COUNT_FIELD_CASES
    lines: 775-779
    signature: CONTRACT_COUNT_FIELD_CASES = all count fields with increment plus source_counts.ba_rows_all_species delete/extra
    behavior: Defines 15 exact-count schema/value mismatch cases.
  - kind: function
    name: test_contract_count_field_mismatch_is_rejected
    lines: 782-804
    signature: @pytest.mark.parametrize(("parent", "field", "mode"), CONTRACT_COUNT_FIELD_CASES, ids=...); def test_contract_count_field_mismatch_is_rejected(parent: str, field: str, mode: str, tmp_path: Path) -> None
    behavior: Mutates a contract count value/key set and asserts distinct mismatch versus keys-mismatch ValueErrors.

file: /Users/freddy/Documents/repos/cognate/scripts/run_knn_esm_cosine.py
mode: full
entries:
  - kind: import
    name: stdlib
    lines: 17-19
    signature: import json, warnings; from pathlib import Path
    behavior: Supplies output serialization, warning suppression around known metric edge cases, and file paths.
  - kind: import
    name: third_party
    lines: 21-21
    signature: import pandas as pd
    behavior: Loads evaluation CSV and writes per-peptide CSV.
  - kind: import
    name: cognate
    lines: 23-32
    signature: from cognate.baseline_knn import cosine_similarity, score_by_nearest_positive; from cognate.data import load_train; from cognate.embed import cache_path, default_cache_dir, load_cache; from cognate.metrics import auc01, compare_macro_auc01, diagnose_scores, evaluate, report_table
    behavior: Reuses shared training data, pooled-cache APIs, retrieval operator/similarity adapter, and evaluation/comparison APIs.
  - kind: constant
    name: ROOT
    lines: 34-34
    signature: ROOT = Path(__file__).resolve().parents[1]
    behavior: Anchors data inputs and outputs at repository root.
  - kind: constant
    name: EVAL_CSV
    lines: 35-35
    signature: EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
    behavior: Requires evaluation columns Label, Peptide, and retrieval-default columns including CDR3b.
  - kind: constant
    name: OUT_JSON
    lines: 36-36
    signature: OUT_JSON = ROOT / "data" / "knn_esm_cosine.json"
    behavior: Receives configuration, score blocks, degeneracy probes, and paired comparison blocks.
  - kind: constant
    name: OUT_PER_PEPTIDE
    lines: 37-37
    signature: OUT_PER_PEPTIDE = ROOT / "data" / "knn_esm_cosine_per_peptide.csv"
    behavior: Receives one row per peptide with peptide/seen/n_rows/n_positive plus one <scorer>_auc01 column per scorer.
  - kind: constant
    name: SEED
    lines: 38-38
    signature: SEED = 0
    behavior: Fixes evaluation and comparison bootstrap randomness.
  - kind: constant
    name: CONFIGS
    lines: 43-49
    signature: CONFIGS = [("35M",12,"35M last"), ("35M",10,"35M layer10 (headline)"), ("35M",6,"35M middle"), ("8M",6,"8M last"), ("8M",3,"8M middle")]
    behavior: Sweeps five exact model/layer/label combinations while keeping the nearest-positive operator fixed.
  - kind: function
    name: vdjdb_cache_path
    lines: 52-53
    signature: def vdjdb_cache_path(model_key: str) -> Path
    behavior: Returns <default-cache-dir>/esm2_<model_key>_vdjdb.npz for query embeddings.
  - kind: function
    name: interval_dict
    lines: 56-61
    signature: def interval_dict(interval) -> dict
    behavior: Serializes point/lo/hi rounded to four decimals and omits n_replicates.
  - kind: function
    name: score_block
    lines: 64-72
    signature: def score_block(report) -> dict
    behavior: Serializes macro_auc01/auroc/auprc interval dicts plus n_rows, n_peptides, and is_degenerate.
  - kind: function
    name: main
    lines: 75-165
    signature: def main() -> None
    behavior: Loads training/eval data and paired 8M/35M query/database caches, scores edit plus five cosine variants, evaluates seen/unseen slices, compares each cosine arm to edit, writes per-peptide CSV and JSON {config,scores,degeneracy,comparisons}.

file: /Users/freddy/Documents/repos/cognate/scripts/run_b3.py
mode: full
entries:
  - kind: import
    name: stdlib
    lines: 15-17
    signature: import json, warnings; from pathlib import Path
    behavior: Supplies serialization, controlled metric-warning suppression, and paths.
  - kind: import
    name: third_party
    lines: 19-23
    signature: import numpy as np; import pandas as pd; from rapidfuzz.distance import Levenshtein; from rapidfuzz.process import cdist; from scipy.stats import rankdata
    behavior: Supplies arrays/tables, parallel edit-distance matrices, and within-peptide rank normalization.
  - kind: import
    name: cognate
    lines: 25-43
    signature: from cognate.baseline_knn import score_by_nearest_positive; from cognate.data import load_train; from cognate.embed import cache_path, default_cache_dir, load_cache; from cognate.features import build_features; from cognate.metrics import auc01, auroc, compare_macro_auc01, correlation_difference_interval, correlation_interval, diagnose_scores, evaluate, leave_one_out_correlations, macro_auc01, report_table; from cognate.negatives import make_negatives; from cognate.split import component_split; from cognate.train_head import fit_logistic, fit_mlp, logistic_scores, mlp_scores
    behavior: Integrates shared data, caches/features, negative generation, split/head training, nearest-neighbor scoring, and statistical evaluation.
  - kind: constant
    name: ROOT
    lines: 45-45
    signature: ROOT = Path(__file__).resolve().parents[1]
    behavior: Anchors data files at repository root.
  - kind: constant
    name: CACHE_DIR
    lines: 46-46
    signature: CACHE_DIR = default_cache_dir()
    behavior: Resolves pooled embedding storage once at import.
  - kind: constant
    name: VDJDB_CACHE
    lines: 47-47
    signature: VDJDB_CACHE = CACHE_DIR / "esm2_35M_vdjdb.npz"
    behavior: Fixes the evaluation query cache path.
  - kind: constant
    name: EVAL_CSV
    lines: 48-48
    signature: EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
    behavior: Supplies evaluation Label/Peptide/CDR3b rows.
  - kind: constant
    name: OUT_JSON
    lines: 49-49
    signature: OUT_JSON = ROOT / "data" / "b3_results.json"
    behavior: Receives all Phase B aggregate outputs.
  - kind: constant
    name: OUT_PER_PEPTIDE
    lines: 50-50
    signature: OUT_PER_PEPTIDE = ROOT / "data" / "b3_per_peptide.csv"
    behavior: Receives per-peptide support/distance/database/score diagnostics.
  - kind: constant
    name: MODEL_KEY
    lines: 52-52
    signature: MODEL_KEY = "35M"
    behavior: Fixes ESM-2 model/cache selection.
  - kind: constant
    name: LAYER
    lines: 53-53
    signature: LAYER = 10
    behavior: Fixes the headline embedding layer.
  - kind: constant
    name: STRATEGY
    lines: 54-54
    signature: STRATEGY = "matched"
    behavior: Fixes negative generation strategy.
  - kind: constant
    name: RATIO
    lines: 55-55
    signature: RATIO = 5.0
    behavior: Generates five negatives per positive in train and validation partitions.
  - kind: constant
    name: SEED
    lines: 56-56
    signature: SEED = 0
    behavior: Fixes split, negatives, heads, random baseline, and bootstraps.
  - kind: constant
    name: WELL_SUPPORTED_MIN_POSITIVES
    lines: 57-57
    signature: WELL_SUPPORTED_MIN_POSITIVES = 100
    behavior: Defines the support-sensitivity correlation subset threshold.
  - kind: function
    name: nearest_training_distance
    lines: 60-67
    signature: def nearest_training_distance(frame: pd.DataFrame, train: pd.DataFrame) -> np.ndarray
    behavior: Computes unique-query versus unique-training CDR3b normalized Levenshtein similarities and returns row-order minimum distances shape (len(frame),).
  - kind: function
    name: interval_dict
    lines: 70-75
    signature: def interval_dict(interval) -> dict
    behavior: Serializes point/lo/hi rounded to four decimals.
  - kind: function
    name: correlation_block
    lines: 78-110
    signature: def correlation_block(label: str, x: np.ndarray, y: np.ndarray, support: np.ndarray) -> dict
    behavior: Returns label/n_points, interval r/spans_zero, leave-one-out min/max, >=100-positive refit, and support-weighted Pearson r.
  - kind: function
    name: main
    lines: 113-349
    signature: def main() -> None
    behavior: Trains matched-negative logistic/MLP heads, scores random/k-NN/heads, evaluates all/seen/unseen, runs paired comparisons/correlations/invariance, writes CSV columns for support/distance/database and per-model AUC/means, and JSON {config,composition,scores,degeneracy,comparisons,correlations,correlation_difference,invariance}.
