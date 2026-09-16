file: /Users/freddy/Documents/repos/cognate/src/cognate/data.py
mode: full
entries:
  - kind: import
    name: pathlib.Path
    lines: 3-3
    signature: from pathlib import Path
    behavior: Builds repository-relative dataset paths and accepts typed filesystem paths.
  - kind: import
    name: pandas
    lines: 5-5
    signature: import pandas as pd
    behavior: Provides CSV loading, dataframe, series, and vectorized string operations.
  - kind: constant
    name: DATA_DIR
    lines: 7-7
    signature: DATA_DIR: Path
    behavior: Resolves the IMMREP23 data directory relative to the module location.
  - kind: constant
    name: TRAIN_CSV
    lines: 9-9
    signature: TRAIN_CSV: Path
    behavior: Names the paired-chain training CSV under DATA_DIR.
  - kind: constant
    name: TEST_CSV
    lines: 10-10
    signature: TEST_CSV: Path
    behavior: Names the unlabelled test CSV under DATA_DIR.
  - kind: constant
    name: SOLUTIONS_CSV
    lines: 11-11
    signature: SOLUTIONS_CSV: Path
    behavior: Names the labelled solutions CSV under DATA_DIR.
  - kind: constant
    name: SEQUENCE_COLUMNS
    lines: 13-13
    signature: 'SEQUENCE_COLUMNS = ["Peptide", "CDR3a", "CDR3b", "CDR1a", "CDR2a", "CDR1b", "CDR2b"]'
    behavior: Defines the canonical peptide and receptor sequence column order.
  - kind: function
    name: _read
    lines: 16-19
    signature: 'def _read(path: Path) -> pd.DataFrame'
    behavior: Fails with an actionable FileNotFoundError when absent, then delegates CSV parsing to pandas.
  - kind: function
    name: load_train
    lines: 22-24
    signature: 'def load_train() -> pd.DataFrame'
    behavior: Loads the configured positive-only paired-chain training dataframe through the shared reader.
  - kind: function
    name: load_test
    lines: 27-29
    signature: 'def load_test() -> pd.DataFrame'
    behavior: Loads the configured unlabelled ID-keyed test dataframe through the shared reader.
  - kind: function
    name: load_solutions
    lines: 32-34
    signature: 'def load_solutions() -> pd.DataFrame'
    behavior: Loads the configured labelled leaderboard-split solutions dataframe through the shared reader.
  - kind: function
    name: train_peptides
    lines: 37-38
    signature: 'def train_peptides() -> set[str]'
    behavior: Returns distinct training peptides as a set for fast membership checks.
  - kind: function
    name: seen_mask
    lines: 41-43
    signature: 'def seen_mask(test_df: pd.DataFrame) -> pd.Series'
    behavior: Produces a vectorized boolean series indicating peptides observed in training.
  - kind: function
    name: to_immrep_cdr3
    lines: 46-58
    signature: 'def to_immrep_cdr3(junction: str) -> str'
    behavior: Normalizes case and whitespace and removes conserved leading C and terminal F or W flanks.

file: /Users/freddy/Documents/repos/cognate/tests/test_vdjdb_eval.py
mode: full
entries:
  - kind: import
    name: json
    lines: 14-14
    signature: import json
    behavior: Loads persisted evaluation statistics fixtures.
  - kind: import
    name: pathlib.Path
    lines: 15-15
    signature: from pathlib import Path
    behavior: Builds repository-relative artifact paths.
  - kind: import
    name: pandas
    lines: 17-17
    signature: import pandas as pd
    behavior: Loads tabular fixtures and constructs deliberately corrupted dataframes.
  - kind: import
    name: pytest
    lines: 18-18
    signature: import pytest
    behavior: Supplies module fixtures, skip gates, parameter checks, and approximate assertions.
  - kind: import
    name: rapidfuzz.distance.Levenshtein
    lines: 19-19
    signature: from rapidfuzz.distance import Levenshtein
    behavior: Computes peptide edit-distance violations for negative validation.
  - kind: import
    name: sklearn.metrics.roc_auc_score
    lines: 20-20
    signature: from sklearn.metrics import roc_auc_score
    behavior: Tests whether peptide identity alone can predict labels.
  - kind: import
    name: cognate.data
    lines: 22-22
    signature: from cognate.data import load_train, to_immrep_cdr3
    behavior: Reuses production training loading and CDR3 normalization in test fixtures.
  - kind: constant
    name: REPO_ROOT
    lines: 24-24
    signature: REPO_ROOT: Path
    behavior: Resolves the repository root from the test file location.
  - kind: constant
    name: EVAL_CSV
    lines: 25-25
    signature: EVAL_CSV: Path
    behavior: Names the row-level evaluation dataframe artifact.
  - kind: constant
    name: PEPTIDES_CSV
    lines: 26-26
    signature: PEPTIDES_CSV: Path
    behavior: Names the per-peptide summary dataframe artifact.
  - kind: constant
    name: STATS_JSON
    lines: 27-27
    signature: STATS_JSON: Path
    behavior: Names the persisted evaluation statistics artifact.
  - kind: constant
    name: VDJDB_SLIM
    lines: 28-28
    signature: VDJDB_SLIM: Path
    behavior: Names the release-pinned raw VDJdb input table.
  - kind: constant
    name: NEGATIVES_PER_POSITIVE
    lines: 30-30
    signature: NEGATIVES_PER_POSITIVE = 5
    behavior: Pins the required per-peptide negative sampling ratio.
  - kind: constant
    name: MIN_PEPTIDE_EDIT_DISTANCE
    lines: 31-31
    signature: MIN_PEPTIDE_EDIT_DISTANCE = 3
    behavior: Pins the distance threshold used to reject near-cognate negatives.
  - kind: constant
    name: EXPECTED_POSITIVE_RATE
    lines: 32-32
    signature: EXPECTED_POSITIVE_RATE = 1 / (1 + NEGATIVES_PER_POSITIVE)
    behavior: Derives the expected class marginal from the sampling ratio.
  - kind: constant
    name: pytestmark
    lines: 34-36
    signature: pytestmark = pytest.mark.skipif(not EVAL_CSV.exists(), reason=...)
    behavior: Skips the module when generated evaluation artifacts are unavailable.
  - kind: function
    name: evalset
    lines: 40-41
    signature: def evalset()
    behavior: Module-scoped fixture that loads the row-level evaluation CSV once.
  - kind: function
    name: per_peptide
    lines: 45-46
    signature: def per_peptide()
    behavior: Module-scoped fixture that loads the per-peptide summary CSV once.
  - kind: function
    name: stats
    lines: 50-51
    signature: def stats()
    behavior: Module-scoped fixture that deserializes the persisted statistics JSON once.
  - kind: function
    name: training_pairs
    lines: 55-57
    signature: def training_pairs()
    behavior: Module-scoped fixture that uppercases and materializes training peptide-CDR3b pairs as a set.
  - kind: function
    name: training_tcrs
    lines: 61-62
    signature: def training_tcrs()
    behavior: Module-scoped fixture that uppercases and materializes training CDR3b values as a set.
  - kind: function
    name: vdjdb_cognates
    lines: 66-78
    signature: def vdjdb_cognates()
    behavior: Filters human TRB rows, drops incomplete records, normalizes sequences, and groups cognate peptides by CDR3b.
  - kind: function
    name: shared_pairs
    lines: 81-82
    signature: 'def shared_pairs(frame: pd.DataFrame, reference: set[tuple[str, str]]) -> int'
    behavior: Counts exact dataframe pair overlap using set intersection.
  - kind: function
    name: known_binder_pairs
    lines: 85-92
    signature: 'def known_binder_pairs(frame: pd.DataFrame, cognates: dict[str, set[str]]) -> list[tuple[str, str]]'
    behavior: Returns dataframe pairs whose peptide is a known cognate for the row CDR3b.
  - kind: function
    name: edit_distance_violations
    lines: 95-105
    signature: 'def edit_distance_violations(frame: pd.DataFrame, cognates: dict[str, set[str]]) -> list[tuple[str, str]]'
    behavior: Returns candidate negatives within the pinned edit-distance threshold of any known cognate.
  - kind: function
    name: a_tcr_with_its_own_cognate
    lines: 108-110
    signature: 'def a_tcr_with_its_own_cognate(cognates: dict[str, set[str]]) -> pd.DataFrame'
    behavior: Constructs a deterministic one-row known-binder dataframe for positive-control corruption.
  - kind: function
    name: test_to_immrep_cdr3_strips_imgt_flanks
    lines: 116-119
    signature: 'def test_to_immrep_cdr3_strips_imgt_flanks() -> None'
    behavior: Asserts normalization across terminal F, terminal W, whitespace, and lowercase inputs.
  - kind: function
    name: test_to_immrep_cdr3_leaves_already_stripped_cores_that_lack_flanks
    lines: 122-123
    signature: 'def test_to_immrep_cdr3_leaves_already_stripped_cores_that_lack_flanks() -> None'
    behavior: Asserts normalization is stable for an already core-form sequence.
  - kind: function
    name: test_raw_vdjdb_junctions_look_disjoint_from_training_without_normalisation
    lines: 126-133
    signature: def test_raw_vdjdb_junctions_look_disjoint_from_training_without_normalisation(training_tcrs, vdjdb_cognates) -> None
    behavior: Demonstrates the units bug by contrasting zero raw overlap with substantial normalized overlap.
  - kind: function
    name: test_normalisation_recovers_the_real_training_overlap
    lines: 136-150
    signature: def test_normalisation_recovers_the_real_training_overlap(training_pairs, training_tcrs) -> None
    behavior: Pins the exact and approximate normalized overlap after explicit species, gene, class, and null filtering.
  - kind: function
    name: test_no_eval_pair_appears_in_immrep23_training
    lines: 156-157
    signature: 'def test_no_eval_pair_appears_in_immrep23_training(evalset, training_pairs) -> None'
    behavior: Asserts zero exact peptide-CDR3b leakage into evaluation data.
  - kind: function
    name: test_pair_leakage_check_fires_on_a_planted_training_pair
    lines: 160-166
    signature: 'def test_pair_leakage_check_fires_on_a_planted_training_pair(evalset, training_pairs) -> None'
    behavior: Positive control proving the pair leakage assertion detects one appended training pair.
  - kind: function
    name: test_no_eval_tcr_appears_in_immrep23_training
    lines: 169-170
    signature: 'def test_no_eval_tcr_appears_in_immrep23_training(evalset, training_tcrs) -> None'
    behavior: Asserts zero receptor-level leakage via set intersection.
  - kind: function
    name: test_tcr_leakage_check_fires_on_a_planted_training_tcr
    lines: 173-179
    signature: 'def test_tcr_leakage_check_fires_on_a_planted_training_tcr(evalset, training_tcrs) -> None'
    behavior: Positive control proving receptor leakage detection catches an appended training CDR3b.
  - kind: function
    name: test_negatives_are_not_known_vdjdb_binders
    lines: 185-186
    signature: 'def test_negatives_are_not_known_vdjdb_binders(evalset, vdjdb_cognates) -> None'
    behavior: Asserts the Label-zero dataframe slice contains no known cognate pairs.
  - kind: function
    name: test_known_binder_check_fires_on_a_planted_cognate_pair
    lines: 189-194
    signature: 'def test_known_binder_check_fires_on_a_planted_cognate_pair(evalset, vdjdb_cognates) -> None'
    behavior: Positive control proving cognate exclusion detects one appended known binder.
  - kind: function
    name: test_negatives_respect_the_edit_distance_rule
    lines: 197-198
    signature: 'def test_negatives_respect_the_edit_distance_rule(evalset, vdjdb_cognates) -> None'
    behavior: Asserts no negative lies within the prohibited edit-distance neighborhood.
  - kind: function
    name: test_edit_distance_check_fires_on_a_planted_cognate_pair
    lines: 201-206
    signature: 'def test_edit_distance_check_fires_on_a_planted_cognate_pair(evalset, vdjdb_cognates) -> None'
    behavior: Positive control proving distance validation detects an appended exact cognate.
  - kind: function
    name: test_positive_rate_is_identical_for_every_peptide
    lines: 212-215
    signature: 'def test_positive_rate_is_identical_for_every_peptide(evalset) -> None'
    behavior: Groups by peptide and asserts identical minimum and maximum positive rates with tight tolerance.
  - kind: function
    name: test_peptide_identity_alone_scores_exactly_half
    lines: 218-222
    signature: 'def test_peptide_identity_alone_scores_exactly_half(evalset) -> None'
    behavior: Asserts per-peptide label rates yield chance ROC AUC using an exact-tolerance approximate check.
  - kind: function
    name: test_peptide_identity_probe_detects_an_imbalanced_set
    lines: 225-234
    signature: 'def test_peptide_identity_probe_detects_an_imbalanced_set() -> None'
    behavior: Positive control constructs a skewed dataframe and asserts the identity-only probe rises above 0.85 AUC.
  - kind: function
    name: test_headline_composition
    lines: 240-246
    signature: 'def test_headline_composition(evalset, per_peptide) -> None'
    behavior: Pins peptide counts, seen-unseen split, class counts, and unique receptor count.
  - kind: function
    name: test_negative_ratio_holds_per_peptide
    lines: 249-251
    signature: 'def test_negative_ratio_holds_per_peptide(evalset) -> None'
    behavior: Uses grouped counts and unstacking to assert the sampling ratio for every peptide.
  - kind: function
    name: test_per_peptide_support_bounds
    lines: 254-259
    signature: 'def test_per_peptide_support_bounds(per_peptide) -> None'
    behavior: Pins minimum, median, maximum, capped, and low-support peptide summary counts.
  - kind: function
    name: test_confidence_filter_cannot_support_the_analysis
    lines: 262-266
    signature: 'def test_confidence_filter_cannot_support_the_analysis(per_peptide) -> None'
    behavior: Pins high-confidence support thresholds to guard against an unsupported analysis slice.
  - kind: function
    name: test_hla_concentration
    lines: 269-271
    signature: 'def test_hla_concentration(per_peptide) -> None'
    behavior: Pins the dominant HLA count and total unique HLA count.
  - kind: function
    name: test_stats_json_agrees_with_the_written_artefacts
    lines: 274-282
    signature: 'def test_stats_json_agrees_with_the_written_artefacts(evalset, per_peptide, stats) -> None'
    behavior: Cross-validates persisted statistics against dataframe-derived counts and the release identifier.

file: /Users/freddy/Documents/repos/cognate/tests/test_frozen_contract_hashes.py
mode: full
entries:
  - kind: import
    name: hashlib
    lines: 13-13
    signature: import hashlib
    behavior: Computes SHA-256 digests for frozen artifacts.
  - kind: import
    name: json
    lines: 14-14
    signature: import json
    behavior: Loads the recorded hash manifest and verification verdict.
  - kind: import
    name: pathlib.Path
    lines: 15-15
    signature: from pathlib import Path
    behavior: Resolves repository-relative manifest and artifact paths.
  - kind: import
    name: pytest
    lines: 17-17
    signature: import pytest
    behavior: Parameterizes the frozen-artifact assertion across manifest entries.
  - kind: constant
    name: ROOT
    lines: 19-19
    signature: ROOT: Path
    behavior: Resolves the repository root from the test module.
  - kind: constant
    name: RECORD
    lines: 20-20
    signature: RECORD: Path
    behavior: Names the JSON manifest containing the frozen contract and recorded hashes.
  - kind: function
    name: recorded
    lines: 23-26
    signature: 'def recorded() -> dict[str, str]'
    behavior: Selects the file-to-digest mapping for the pinned commit from the manifest.
  - kind: function
    name: test_frozen_artifact_matches_its_recorded_hash
    lines: 30-36
    signature: 'def test_frozen_artifact_matches_its_recorded_hash(relative_path: str) -> None'
    behavior: Hashes each parameterized artifact as bytes and compares it with an actionable mismatch message.
  - kind: function
    name: test_the_verification_itself_passed
    lines: 39-40
    signature: 'def test_the_verification_itself_passed() -> None'
    behavior: Asserts the persisted cross-worktree verification verdict is PASS.

file: /Users/freddy/Documents/repos/cognate/tests/test_reference_constants.py
mode: full
entries:
  - kind: import
    name: ast
    lines: 19-19
    signature: import ast
    behavior: Parses driver source without importing or executing it.
  - kind: import
    name: json
    lines: 20-20
    signature: import json
    behavior: Loads authoritative metric and erratum artifacts.
  - kind: import
    name: pathlib.Path
    lines: 21-21
    signature: from pathlib import Path
    behavior: Resolves repository-relative data and driver paths.
  - kind: import
    name: pytest
    lines: 23-23
    signature: import pytest
    behavior: Supplies skip gates, parameterization, and exception assertions.
  - kind: constant
    name: REPO_ROOT
    lines: 25-25
    signature: REPO_ROOT: Path
    behavior: Resolves the repository root from the test module.
  - kind: constant
    name: DATA
    lines: 26-26
    signature: DATA: Path
    behavior: Names the repository data directory.
  - kind: constant
    name: SCRIPTS
    lines: 27-27
    signature: SCRIPTS: Path
    behavior: Names the repository driver-script directory.
  - kind: constant
    name: KNN_ESM_COSINE
    lines: 29-29
    signature: KNN_ESM_COSINE: Path
    behavior: Names the authoritative KNN and ESM cosine result artifact.
  - kind: constant
    name: B3_RESULTS
    lines: 30-30
    signature: B3_RESULTS: Path
    behavior: Names the authoritative B3 result artifact.
  - kind: constant
    name: PROBES
    lines: 31-31
    signature: PROBES: Path
    behavior: Names the authoritative VDJdb probe artifact.
  - kind: constant
    name: REFERENCE_KEY
    lines: 33-33
    signature: REFERENCE_KEY = "reference"
    behavior: Centralizes the driver dictionary key located by AST parsing.
  - kind: constant
    name: SUPERSEDED_ESM_COSINE
    lines: 34-34
    signature: SUPERSEDED_ESM_COSINE = 0.5364
    behavior: Pins the known stale literal used by positive controls and erratum checks.
  - kind: function
    name: _macro_point
    lines: 37-38
    signature: 'def _macro_point(path: Path, slice_name: str) -> float'
    behavior: Reads a nested macro AUC point estimate from a JSON artifact.
  - kind: function
    name: _sources
    lines: 41-51
    signature: 'def _sources() -> dict[str, float]'
    behavior: Maps each restated driver constant to its authoritative artifact value.
  - kind: function
    name: parse_reference_block
    lines: 54-75
    signature: 'def parse_reference_block(source: str) -> dict[str, float]'
    behavior: Walks parsed source for the reference dictionary literal and returns literal key-value pairs without execution.
  - kind: function
    name: mismatches
    lines: 78-84
    signature: 'def mismatches(block: dict[str, float], sources: dict[str, float]) -> dict[str, tuple]'
    behavior: Returns shared reference names whose driver literal differs from its authoritative source.
  - kind: constant
    name: DRIVERS
    lines: 87-87
    signature: 'DRIVERS = ["run_fork1.py", "run_fork2.py"]'
    behavior: Enumerates the driver sources checked through pytest parameterization.
  - kind: constant
    name: missing_artifacts
    lines: 89-91
    signature: missing_artifacts = [p.name for p in required_paths if not p.exists()]
    behavior: Collects absent authoritative artifact names for the module skip reason.
  - kind: constant
    name: pytestmark
    lines: 92-94
    signature: pytestmark = pytest.mark.skipif(bool(missing_artifacts), reason=...)
    behavior: Skips reference validation when its authoritative artifacts are unavailable.
  - kind: function
    name: test_every_reference_constant_matches_its_source
    lines: 98-110
    signature: 'def test_every_reference_constant_matches_its_source(driver: str) -> None'
    behavior: Rejects uncheckable driver keys and any literal-to-artifact mismatch with actionable assertion messages.
  - kind: function
    name: test_driver_carries_the_esm_cosine_constant
    lines: 114-117
    signature: 'def test_driver_carries_the_esm_cosine_constant(driver: str) -> None'
    behavior: Prevents the consistency check from passing vacuously if the required key disappears.
  - kind: function
    name: test_check_fires_on_the_superseded_value
    lines: 120-129
    signature: 'def test_check_fires_on_the_superseded_value() -> None'
    behavior: Positive control injects the known stale value and pins the detected mismatch tuple.
  - kind: function
    name: test_parse_reference_block_rejects_a_file_without_one
    lines: 132-134
    signature: 'def test_parse_reference_block_rejects_a_file_without_one() -> None'
    behavior: Asserts via pytest.raises that source lacking a reference block is rejected.
  - kind: constant
    name: FORK_ARTIFACTS
    lines: 137-137
    signature: 'FORK_ARTIFACTS = ["fork1_results.json", "fork2_results.json"]'
    behavior: Enumerates committed fork outputs checked against the erratum.
  - kind: function
    name: test_committed_fork_artifacts_still_carry_the_erratum
    lines: 141-153
    signature: 'def test_committed_fork_artifacts_still_carry_the_erratum(artifact: str) -> None'
    behavior: Pins each stale artifact literal to the erratum while asserting the authoritative replacement matches live sources.
