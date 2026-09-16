file: /Users/freddy/Documents/repos/cognate/src/cognate/pmhc.py
mode: all
entries:
  - kind: import
    name: __future__.annotations
    lines: 8-8
    signature: from __future__ import annotations
    behavior: Enables postponed annotation evaluation.
  - kind: import
    name: hashlib
    lines: 10-10
    signature: import hashlib
    behavior: Supplies SHA-256 hashing for source verification.
  - kind: import
    name: json
    lines: 11-11
    signature: import json
    behavior: Parses the persisted source contract.
  - kind: import
    name: math
    lines: 12-12
    signature: import math
    behavior: Validates finite affinity values.
  - kind: import
    name: collections.abc
    lines: 13-13
    signature: from collections.abc import Iterator, Mapping, Sequence
    behavior: Provides public collection protocols used in callable contracts.
  - kind: import
    name: dataclasses.dataclass
    lines: 14-14
    signature: from dataclasses import dataclass
    behavior: Declares the immutable pMHC dataset container.
  - kind: import
    name: pathlib.Path
    lines: 15-15
    signature: from pathlib import Path
    behavior: Types source, archive, and pseudo-sequence paths.
  - kind: import
    name: pandas
    lines: 17-17
    signature: import pandas as pd
    behavior: Represents and partitions pMHC rows.
  - kind: constant
    name: FOLD_NAMES
    lines: 19-19
    signature: FOLD_NAMES = tuple(f"c00{index}_ba" for index in range(5))
    behavior: Defines the five accepted NetMHCpan binding-affinity fold filenames.
  - kind: constant
    name: HLA_PREFIXES
    lines: 20-20
    signature: HLA_PREFIXES = ("HLA-A", "HLA-B", "HLA-C")
    behavior: Restricts dataset rows to classical HLA class-I alleles.
  - kind: constant
    name: POSITIVE_THRESHOLD
    lines: 21-21
    signature: POSITIVE_THRESHOLD = 0.426
    behavior: Marks affinity values strictly above 0.426 as positive.
  - kind: constant
    name: ARTIFICIAL_NEGATIVE_AFFINITY
    lines: 22-22
    signature: ARTIFICIAL_NEGATIVE_AFFINITY = 0.01
    behavior: Identifies artificial-negative source rows.
  - kind: constant
    name: STANDARD_AMINO_ACIDS
    lines: 23-23
    signature: STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
    behavior: Defines the accepted peptide residue alphabet.
  - kind: constant
    name: PSEUDO_SEQUENCE_LENGTH
    lines: 24-24
    signature: PSEUDO_SEQUENCE_LENGTH = 34
    behavior: Enforces the allele pseudo-sequence width used by scoring features.
  - kind: dataclass
    name: PmhcDataset
    lines: 28-31
    signature: "@dataclass(frozen=True) class PmhcDataset(all_nine_mer_hla_rows: pd.DataFrame, rows: pd.DataFrame, eligible_alleles: tuple[str, ...])"
    behavior: Holds all nine-mer HLA rows, the eligibility-filtered scoring rows, and eligible allele names.
  - kind: function
    name: load_source_contract
    lines: 34-38
    signature: "def load_source_contract(path: Path) -> dict[str, object]"
    behavior: Loads a JSON object from path and rejects non-object roots.
  - kind: function
    name: _require_matching_counts
    lines: 41-58
    signature: "def _require_matching_counts(field: str, contract: Mapping[str, object], actual_counts: Mapping[str, int]) -> None"
    behavior: Requires exact key and integer-count equality for one contract field.
  - kind: function
    name: verify_source
    lines: 61-124
    signature: "def verify_source(source_dir: Path, archive_path: Path, contract: Mapping[str, object]) -> None"
    behavior: Verifies archive/file hashes and source/headline row counts against the contract.
  - kind: function
    name: _parse_fold
    lines: 127-157
    signature: "def _parse_fold(path: Path) -> pd.DataFrame"
    behavior: Parses one three-column fold file into Peptide, Affinity, Allele, and Fold columns with residue and range validation.
  - kind: function
    name: _load_raw_rows
    lines: 160-182
    signature: "def _load_raw_rows(source_dir: Path) -> pd.DataFrame"
    behavior: Loads exactly the five folds and adds boolean Target plus positive/measured_nonbinder/artificial_negative RowType labels.
  - kind: function
    name: _require_both_classes_per_fold
    lines: 185-203
    signature: "def _require_both_classes_per_fold(rows: pd.DataFrame, eligible_alleles: tuple[str, ...]) -> None"
    behavior: Requires positive and negative examples for every eligible allele in each held-out fold and its training complement.
  - kind: function
    name: load_pmhc_dataset
    lines: 206-240
    signature: "def load_pmhc_dataset(source_dir: Path, *, minimum_class_rows: int = 100) -> PmhcDataset"
    behavior: Builds the nine-mer HLA dataset, rejects peptide overlap across folds, filters alleles by per-class support, and validates fold coverage.
  - kind: function
    name: iter_pmhc_folds
    lines: 243-253
    signature: "def iter_pmhc_folds(dataset: PmhcDataset) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]"
    behavior: Yields each fold name with reset-index training-complement and held-out DataFrames.
  - kind: function
    name: load_pseudo_sequences
    lines: 256-280
    signature: "def load_pseudo_sequences(path: Path, alleles: Sequence[str]) -> dict[str, str]"
    behavior: Resolves requested colon-form allele names to validated 34-residue pseudo-sequences keyed by the original allele names.

file: /Users/freddy/Documents/repos/cognate/src/cognate/baseline_knn.py
mode: all
entries:
  - kind: import
    name: collections.abc.Callable
    lines: 22-22
    signature: from collections.abc import Callable
    behavior: Types pluggable pairwise sequence-similarity functions.
  - kind: import
    name: dataclasses.dataclass
    lines: 23-23
    signature: from dataclasses import dataclass
    behavior: Declares immutable k-nearest-neighbour results.
  - kind: import
    name: numpy
    lines: 25-25
    signature: import numpy as np
    behavior: Supplies array storage, similarity aggregation, and masks.
  - kind: import
    name: pandas
    lines: 26-26
    signature: import pandas as pd
    behavior: Supplies row-grouping and tabular inputs/outputs.
  - kind: import
    name: rapidfuzz.distance.Levenshtein
    lines: 27-27
    signature: from rapidfuzz.distance import Levenshtein
    behavior: Provides normalized edit similarity.
  - kind: import
    name: rapidfuzz.process.cdist
    lines: 28-28
    signature: from rapidfuzz.process import cdist
    behavior: Computes query-by-entry similarity matrices.
  - kind: import
    name: cognate.embed.EmbeddingCache
    lines: 30-30
    signature: from cognate.embed import EmbeddingCache
    behavior: Supplies cached mean-pooled embeddings for cosine scoring.
  - kind: constant
    name: DEFAULT_SCORE
    lines: 32-32
    signature: DEFAULT_SCORE = 0.0
    behavior: Sets the score for queries with no usable peptide database.
  - kind: constant
    name: EXACT_MATCH_SIMILARITY
    lines: 33-33
    signature: EXACT_MATCH_SIMILARITY = 1.0
    behavior: Defines the threshold removed by the leave-out-exact-match mode.
  - kind: constant
    name: PEPTIDE_KEY
    lines: 35-35
    signature: PEPTIDE_KEY = "Peptide"
    behavior: Names the default peptide DataFrame column.
  - kind: constant
    name: SEQUENCE_KEY
    lines: 36-36
    signature: SEQUENCE_KEY = "CDR3b"
    behavior: Names the default receptor-sequence DataFrame column.
  - kind: constant
    name: TARGET_KEY
    lines: 37-37
    signature: TARGET_KEY = "Target"
    behavior: Names the optional training-label column used to retain binders.
  - kind: constant
    name: SimilarityFn
    lines: 40-40
    signature: "SimilarityFn = Callable[[np.ndarray, np.ndarray], np.ndarray]"
    behavior: Types a function returning a query-by-entry similarity matrix.
  - kind: function
    name: edit_similarity
    lines: 42-46
    signature: "def edit_similarity(queries: np.ndarray, entries: np.ndarray) -> np.ndarray"
    behavior: Computes normalized Levenshtein similarity for every query-entry pair.
  - kind: function
    name: _unit_rows
    lines: 48-49
    signature: "def _unit_rows(matrix: np.ndarray) -> np.ndarray"
    behavior: L2-normalizes each matrix row for cosine similarity.
  - kind: function
    name: cosine_similarity
    lines: 51-70
    signature: "def cosine_similarity(query_cache: EmbeddingCache, layer_index: int, database_cache: EmbeddingCache | None = None) -> SimilarityFn"
    behavior: Returns a closure that looks up one embedding layer in separate or shared caches and computes pairwise cosine scores.
  - kind: dataclass
    name: KnnResult
    lines: 73-93
    signature: "@dataclass(frozen=True) class KnnResult(score: np.ndarray, has_database: np.ndarray, database_size: np.ndarray, nearest_sequence: np.ndarray)"
    behavior: Carries per-row scores and nearest-neighbour provenance.
  - kind: function
    name: KnnResult.n_without_database
    lines: 82-83
    signature: "@property def n_without_database(self) -> int"
    behavior: Counts rows whose peptide had no training database.
  - kind: function
    name: KnnResult.to_frame
    lines: 85-93
    signature: "def to_frame(self) -> pd.DataFrame"
    behavior: Converts all result arrays to score, has_database, database_size, and nearest_sequence columns.
  - kind: function
    name: build_database
    lines: 96-113
    signature: "def build_database(train_df: pd.DataFrame, peptide_column: str = PEPTIDE_KEY, sequence_column: str = SEQUENCE_KEY) -> dict[str, np.ndarray]"
    behavior: Maps each peptide to sorted distinct positive receptor sequences, or all sequences when Target is absent.
  - kind: function
    name: score_by_nearest_positive
    lines: 116-191
    signature: "def score_by_nearest_positive(test_df: pd.DataFrame, train_df: pd.DataFrame, *, peptide_column: str = PEPTIDE_KEY, sequence_column: str = SEQUENCE_KEY, default_score: float = DEFAULT_SCORE, leave_out_exact_matches: bool = False, top_k: int = 1, similarity_fn: SimilarityFn | None = None) -> KnnResult"
    behavior: Scores each test row from the top-k similarities to same-peptide training binders, with optional exact-match removal and provenance.
  - kind: function
    name: exact_match_mask
    lines: 194-210
    signature: "def exact_match_mask(test_df: pd.DataFrame, train_df: pd.DataFrame, peptide_column: str = PEPTIDE_KEY, sequence_column: str = SEQUENCE_KEY) -> np.ndarray"
    behavior: Returns a boolean mask for test peptide-sequence pairs present among positive training pairs or all pairs when Target is absent.

file: /Users/freddy/Documents/repos/cognate/src/cognate/embed.py
mode: all
entries:
  - kind: import
    name: os
    lines: 19-19
    signature: import os
    behavior: Reads the embedding-cache directory override.
  - kind: import
    name: time
    lines: 20-20
    signature: import time
    behavior: Measures embedding wall-clock duration.
  - kind: import
    name: collections.abc
    lines: 21-21
    signature: from collections.abc import Iterable, Sequence
    behavior: Types input sequences and selected layer indices.
  - kind: import
    name: dataclasses.dataclass
    lines: 22-22
    signature: from dataclasses import dataclass
    behavior: Declares immutable pooled and residue cache containers.
  - kind: import
    name: pathlib.Path
    lines: 23-23
    signature: from pathlib import Path
    behavior: Types cache paths and resolves the repository anchor.
  - kind: import
    name: numpy
    lines: 25-25
    signature: import numpy as np
    behavior: Stores embeddings, offsets, indices, and serialized arrays.
  - kind: import
    name: torch
    lines: 26-26
    signature: import torch
    behavior: Runs ESM-2 inference and device-specific tensor operations.
  - kind: import
    name: transformers
    lines: 27-27
    signature: from transformers import AutoModel, AutoTokenizer
    behavior: Loads ESM-2 tokenizer and encoder checkpoints.
  - kind: constant
    name: MODELS
    lines: 29-32
    signature: "MODELS = {\"8M\": \"facebook/esm2_t6_8M_UR50D\", \"35M\": \"facebook/esm2_t12_35M_UR50D\"}"
    behavior: Maps short model keys to Hugging Face ESM-2 checkpoint names.
  - kind: constant
    name: DEFAULT_MAX_BATCH_TOKENS
    lines: 34-34
    signature: DEFAULT_MAX_BATCH_TOKENS = 16_384
    behavior: Caps padded tokens per embedding batch.
  - kind: constant
    name: CACHE_SUFFIX
    lines: 35-35
    signature: CACHE_SUFFIX = ".npz"
    behavior: Defines the NumPy cache filename suffix.
  - kind: constant
    name: CACHE_DIR_ENV
    lines: 36-36
    signature: CACHE_DIR_ENV = "COGNATE_CACHE_DIR"
    behavior: Names the environment variable overriding cache storage.
  - kind: constant
    name: SHARED_CACHE_DIRNAME
    lines: 37-37
    signature: SHARED_CACHE_DIRNAME = "shared"
    behavior: Names the repository-local default cache directory.
  - kind: dataclass
    name: EmbeddingCache
    lines: 41-78
    signature: "@dataclass(frozen=True) class EmbeddingCache(model_name: str, sequences: np.ndarray, layers: np.ndarray, index: dict[str, int])"
    behavior: Stores mean-pooled embeddings shaped by layer, sequence, and hidden dimension with string lookup indices.
  - kind: function
    name: EmbeddingCache.n_sequences
    lines: 50-51
    signature: "@property def n_sequences(self) -> int"
    behavior: Returns the number of cached unique sequences.
  - kind: function
    name: EmbeddingCache.n_layers
    lines: 54-56
    signature: "@property def n_layers(self) -> int"
    behavior: Returns stored hidden-state count including embedding layer zero.
  - kind: function
    name: EmbeddingCache.hidden_size
    lines: 59-60
    signature: "@property def hidden_size(self) -> int"
    behavior: Returns the embedding width from layers axis two.
  - kind: function
    name: EmbeddingCache.layer
    lines: 62-63
    signature: "def layer(self, layer_index: int) -> np.ndarray"
    behavior: Returns all cached sequence rows for one layer index.
  - kind: function
    name: EmbeddingCache.lookup
    lines: 65-78
    signature: "def lookup(self, sequences: Iterable[str], layer_index: int = -1) -> np.ndarray"
    behavior: Returns requested sequence rows in input order from one layer and raises KeyError for missing strings.
  - kind: function
    name: _make_cache
    lines: 81-89
    signature: "def _make_cache(model_name: str, sequences: np.ndarray, layers: np.ndarray) -> EmbeddingCache"
    behavior: Constructs an EmbeddingCache and derives its string-to-row index.
  - kind: function
    name: pick_device
    lines: 92-93
    signature: "def pick_device() -> str"
    behavior: Selects mps when available and cpu otherwise.
  - kind: function
    name: _length_batches
    lines: 96-113
    signature: "def _length_batches(sequences: list[str], max_batch_tokens: int) -> list[list[int]]"
    behavior: Sorts positions by sequence length and packs batches under the padded-token cap.
  - kind: function
    name: _residue_mask
    lines: 116-122
    signature: "def _residue_mask(attention_mask: torch.Tensor) -> torch.Tensor"
    behavior: Clones an attention mask and clears BOS plus each row's EOS position.
  - kind: function
    name: embed_sequences
    lines: 125-165
    signature: "def embed_sequences(sequences: Iterable[str], model_key: str = \"8M\", *, device: str | None = None, max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS, progress_every: int = 20) -> tuple[EmbeddingCache, float]"
    behavior: Deduplicates sequences, embeds them with every ESM-2 hidden state, mean-pools residues without BOS/EOS, and returns cache plus elapsed seconds.
  - kind: function
    name: default_cache_dir
    lines: 168-178
    signature: "def default_cache_dir() -> Path"
    behavior: Returns COGNATE_CACHE_DIR when set or the repository-root shared directory.
  - kind: function
    name: cache_path
    lines: 181-182
    signature: "def cache_path(model_key: str, directory: Path | None = None) -> Path"
    behavior: Builds esm2_<model_key>.npz beneath an explicit or default cache directory.
  - kind: function
    name: save_cache
    lines: 185-197
    signature: "def save_cache(cache: EmbeddingCache, path: Path) -> int"
    behavior: Creates parent directories, writes an uncompressed pooled NPZ, and returns file size in bytes.
  - kind: function
    name: load_cache
    lines: 200-206
    signature: "def load_cache(path: Path) -> EmbeddingCache"
    behavior: Loads a pooled NPZ without pickle and rebuilds the cache lookup index.
  - kind: dataclass
    name: ResidueCache
    lines: 210-260
    signature: "@dataclass(frozen=True) class ResidueCache(model_name: str, sequences: np.ndarray, layer_indices: np.ndarray, offsets: np.ndarray, residues: np.ndarray, index: dict[str, int])"
    behavior: Stores selected per-residue layers in concatenated ragged form with offsets and sequence indices.
  - kind: function
    name: ResidueCache.n_sequences
    lines: 232-233
    signature: "@property def n_sequences(self) -> int"
    behavior: Returns the number of cached unique sequences.
  - kind: function
    name: ResidueCache.hidden_size
    lines: 236-237
    signature: "@property def hidden_size(self) -> int"
    behavior: Returns per-residue embedding width.
  - kind: function
    name: ResidueCache._layer_row
    lines: 239-245
    signature: "def _layer_row(self, layer_index: int) -> int"
    behavior: Maps a model layer index to its cache row and raises KeyError when absent.
  - kind: function
    name: ResidueCache.residues_for
    lines: 247-254
    signature: "def residues_for(self, sequence: str, layer_index: int) -> np.ndarray"
    behavior: Returns one sequence's length-by-hidden residue block for a stored layer and raises for missing sequences or layers.
  - kind: function
    name: ResidueCache.mean_pooled
    lines: 256-260
    signature: "def mean_pooled(self, sequences: Iterable[str], layer_index: int) -> np.ndarray"
    behavior: Mean-pools residue blocks into sequence rows in caller-supplied order.
  - kind: function
    name: embed_residues
    lines: 263-316
    signature: "def embed_residues(sequences: Iterable[str], model_key: str = \"35M\", *, layer_indices: Sequence[int] = (6, 10, 12), device: str | None = None, max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS, progress_every: int = 20) -> tuple[ResidueCache, float]"
    behavior: Deduplicates sequences, embeds selected ESM-2 layers, stores residue-only float32 blocks, and returns the ragged cache plus elapsed seconds.
  - kind: function
    name: save_residue_cache
    lines: 319-329
    signature: "def save_residue_cache(cache: ResidueCache, path: Path) -> int"
    behavior: Creates parent directories, writes all residue-cache arrays to NPZ, and returns file size in bytes.
  - kind: function
    name: load_residue_cache
    lines: 332-342
    signature: "def load_residue_cache(path: Path) -> ResidueCache"
    behavior: Loads a residue NPZ without pickle and rebuilds the string-to-row index.

file: /Users/freddy/Documents/repos/cognate/src/cognate/train_head.py
mode: all
entries:
  - kind: import
    name: dataclasses
    lines: 14-14
    signature: from dataclasses import dataclass, field
    behavior: Declares training history and list field factories.
  - kind: import
    name: numpy
    lines: 16-16
    signature: import numpy as np
    behavior: Supplies feature, label, group, and score arrays.
  - kind: import
    name: torch
    lines: 17-17
    signature: import torch
    behavior: Supplies neural-network training, tensors, devices, and inference.
  - kind: import
    name: sklearn.linear_model.LogisticRegression
    lines: 18-18
    signature: from sklearn.linear_model import LogisticRegression
    behavior: Provides the L2 logistic scoring head.
  - kind: import
    name: sklearn.pipeline
    lines: 19-19
    signature: from sklearn.pipeline import Pipeline, make_pipeline
    behavior: Types and constructs the scaler-plus-logistic pipeline.
  - kind: import
    name: sklearn.preprocessing.StandardScaler
    lines: 20-20
    signature: from sklearn.preprocessing import StandardScaler
    behavior: Standardizes features before either trained head.
  - kind: import
    name: torch.nn
    lines: 21-21
    signature: from torch import nn
    behavior: Defines the MLP module, layers, dropout, and loss.
  - kind: import
    name: cognate.metrics.macro_auc01
    lines: 23-23
    signature: from cognate.metrics import macro_auc01
    behavior: Records per-epoch validation macro partial AUC.
  - kind: constant
    name: DEFAULT_SEED
    lines: 25-25
    signature: DEFAULT_SEED = 0
    behavior: Defines deterministic model and batch-order seeding.
  - kind: constant
    name: DEFAULT_HIDDEN
    lines: 26-26
    signature: DEFAULT_HIDDEN = 64
    behavior: Sets the default MLP hidden width.
  - kind: constant
    name: DEFAULT_DROPOUT
    lines: 27-27
    signature: DEFAULT_DROPOUT = 0.2
    behavior: Sets the default MLP dropout probability.
  - kind: constant
    name: DEFAULT_LEARNING_RATE
    lines: 28-28
    signature: DEFAULT_LEARNING_RATE = 1e-3
    behavior: Sets the default AdamW learning rate.
  - kind: constant
    name: DEFAULT_WEIGHT_DECAY
    lines: 29-29
    signature: DEFAULT_WEIGHT_DECAY = 1e-4
    behavior: Sets the default AdamW weight decay.
  - kind: constant
    name: DEFAULT_BATCH_SIZE
    lines: 30-30
    signature: DEFAULT_BATCH_SIZE = 512
    behavior: Sets the default MLP training batch size.
  - kind: constant
    name: DEFAULT_MAX_EPOCHS
    lines: 31-31
    signature: DEFAULT_MAX_EPOCHS = 60
    behavior: Caps default MLP training epochs.
  - kind: constant
    name: DEFAULT_PATIENCE
    lines: 32-32
    signature: DEFAULT_PATIENCE = 8
    behavior: Sets validation-loss early-stopping patience in epochs.
  - kind: function
    name: fit_logistic
    lines: 35-51
    signature: "def fit_logistic(features: np.ndarray, labels: np.ndarray, *, seed: int = DEFAULT_SEED, max_iter: int = 1000, regularisation: float = 1.0) -> Pipeline"
    behavior: Fits StandardScaler followed by seeded L2 LogisticRegression with C equal to regularisation.
  - kind: function
    name: logistic_scores
    lines: 54-55
    signature: "def logistic_scores(model: Pipeline, features: np.ndarray) -> np.ndarray"
    behavior: Returns the fitted pipeline's positive-class probabilities.
  - kind: class
    name: MlpHead
    lines: 58-73
    signature: class MlpHead(nn.Module)
    behavior: Implements a linear-ReLU-dropout-linear binary-logit head.
  - kind: function
    name: MlpHead.__init__
    lines: 61-70
    signature: "def __init__(self, n_features: int, hidden: int = DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT) -> None"
    behavior: Builds the two-layer sequential network with one scalar output.
  - kind: function
    name: MlpHead.forward
    lines: 72-73
    signature: "def forward(self, features: torch.Tensor) -> torch.Tensor"
    behavior: Returns one squeezed logit per feature row.
  - kind: dataclass
    name: TrainingHistory
    lines: 77-87
    signature: "@dataclass class TrainingHistory(train_loss: list[float] = field(default_factory=list), val_loss: list[float] = field(default_factory=list), val_macro_auc01: list[float] = field(default_factory=list), best_epoch: int = -1)"
    behavior: Stores epoch-wise losses, validation macro AUC0.1, and best epoch.
  - kind: function
    name: TrainingHistory.n_epochs
    lines: 86-87
    signature: "@property def n_epochs(self) -> int"
    behavior: Returns the number of recorded training-loss epochs.
  - kind: function
    name: _to_tensor
    lines: 90-91
    signature: "def _to_tensor(array: np.ndarray, device: str) -> torch.Tensor"
    behavior: Converts a contiguous NumPy array to float32 on the requested device.
  - kind: function
    name: mlp_scores
    lines: 95-106
    signature: "@torch.no_grad() def mlp_scores(model: MlpHead, scaler: StandardScaler, features: np.ndarray) -> np.ndarray"
    behavior: Standardizes features, runs the model on its own device, and returns sigmoid probabilities as NumPy rows.
  - kind: function
    name: fit_mlp
    lines: 109-181
    signature: "def fit_mlp(train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray, val_groups: np.ndarray, *, hidden: int = DEFAULT_HIDDEN, dropout: float = DEFAULT_DROPOUT, learning_rate: float = DEFAULT_LEARNING_RATE, weight_decay: float = DEFAULT_WEIGHT_DECAY, batch_size: int = DEFAULT_BATCH_SIZE, max_epochs: int = DEFAULT_MAX_EPOCHS, patience: int = DEFAULT_PATIENCE, seed: int = DEFAULT_SEED, device: str | None = None) -> tuple[MlpHead, StandardScaler, TrainingHistory]"
    behavior: Fits a standardized MLP with shuffled mini-batches and validation-loss early stopping, records grouped macro AUC0.1, restores best weights, and returns model, scaler, and history.

file: /Users/freddy/Documents/repos/cognate/src/cognate/features.py
mode: all
entries:
  - kind: import
    name: typing.Literal
    lines: 20-20
    signature: from typing import Literal
    behavior: Restricts accepted feature-block names.
  - kind: import
    name: numpy
    lines: 22-22
    signature: import numpy as np
    behavior: Builds and concatenates feature arrays.
  - kind: import
    name: pandas
    lines: 23-23
    signature: import pandas as pd
    behavior: Types the scoring-row input frame.
  - kind: import
    name: cognate.embed.EmbeddingCache
    lines: 25-25
    signature: from cognate.embed import EmbeddingCache
    behavior: Supplies sequence-to-embedding lookup and hidden width.
  - kind: constant
    name: FeatureBlocks
    lines: 27-27
    signature: "FeatureBlocks = Literal[\"concat\", \"interactions\", \"peptide_only\", \"tcr_only\"]"
    behavior: Defines the four supported design-matrix layouts.
  - kind: constant
    name: PEPTIDE_KEY
    lines: 29-29
    signature: PEPTIDE_KEY = "Peptide"
    behavior: Names the default peptide column.
  - kind: constant
    name: SEQUENCE_KEY
    lines: 30-30
    signature: SEQUENCE_KEY = "CDR3b"
    behavior: Names the default receptor-sequence column.
  - kind: function
    name: build_features
    lines: 33-56
    signature: "def build_features(df: pd.DataFrame, cache: EmbeddingCache, layer_index: int = -1, blocks: FeatureBlocks = \"interactions\", peptide_column: str = PEPTIDE_KEY, sequence_column: str = SEQUENCE_KEY) -> np.ndarray"
    behavior: Looks up peptide and receptor rows, assembles concat/interactions/single-side blocks, and returns a float32 design matrix.
  - kind: constant
    name: BLOCK_MULTIPLIERS
    lines: 59-64
    signature: "BLOCK_MULTIPLIERS = {\"concat\": 2, \"interactions\": 4, \"peptide_only\": 1, \"tcr_only\": 1}"
    behavior: Maps each block layout to its embedding-width multiplier.
  - kind: function
    name: feature_width
    lines: 67-68
    signature: "def feature_width(cache: EmbeddingCache, blocks: FeatureBlocks = \"interactions\") -> int"
    behavior: Returns hidden size multiplied by the selected feature-block width.
