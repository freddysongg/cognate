file: tests/test_pmhc.py
mode: full
entries:
  - kind: import
    name: __future__.annotations
    lines: 1-1
    signature: from __future__ import annotations
    behavior: Enables deferred annotation evaluation.
  - kind: import
    name: hashlib
    lines: 3-3
    signature: import hashlib
    behavior: Computes fixture archive and extracted-file SHA-256 values.
  - kind: import
    name: json
    lines: 4-4
    signature: import json
    behavior: Serializes the synthetic source contract fixture.
  - kind: import
    name: collections.abc.Callable
    lines: 5-5
    signature: from collections.abc import Callable
    behavior: Types the first-line mutation callback.
  - kind: import
    name: pathlib.Path
    lines: 6-6
    signature: from pathlib import Path
    behavior: Represents temporary fixture and contract paths.
  - kind: import
    name: pandas
    lines: 8-8
    signature: import pandas as pd
    behavior: Supplies dataframe types and dataframe assertions.
  - kind: import
    name: pytest
    lines: 9-9
    signature: import pytest
    behavior: Supplies parametrization and exception assertions.
  - kind: import
    name: cognate.pmhc
    lines: 11-18
    signature: from cognate.pmhc import PmhcDataset, iter_pmhc_folds, load_pmhc_dataset, load_pseudo_sequences, load_source_contract, verify_source
    behavior: Imports the Phase 1 dataset, fold, pseudo-sequence, contract-loading, and source-verification surface.
  - kind: constant
    name: AMINO_ACIDS
    lines: 20-20
    signature: AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    behavior: Defines the valid residue alphabet used by deterministic synthetic peptides.
  - kind: constant
    name: FOLD_NAMES
    lines: 21-21
    signature: FOLD_NAMES = tuple(f"c00{index}_ba" for index in range(5))
    behavior: Defines the five expected binding-affinity fold filenames.
  - kind: constant
    name: PSEUDO_SEQUENCE
    lines: 22-22
    signature: PSEUDO_SEQUENCE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"
    behavior: Defines one valid 34-residue MHC pseudo-sequence fixture.
  - kind: function
    name: _peptide
    lines: 25-31
    signature: _peptide(number: int, length: int = 9) -> str
    behavior: Encodes an integer deterministically into a valid amino-acid peptide of the requested length.
  - kind: function
    name: _write_valid_source
    lines: 34-118
    signature: _write_valid_source(root: Path) -> tuple[Path, Path, Path, str, str, str, str]
    behavior: Builds five synthetic fold files, pseudo-sequence data, an archive placeholder, and a matching JSON contract with exact class and row counts.
  - kind: function
    name: _sha256
    lines: 121-122
    signature: _sha256(path: Path) -> str
    behavior: Returns the SHA-256 hex digest of fixture file bytes.
  - kind: function
    name: _valid_contract
    lines: 125-151
    signature: _valid_contract(source_dir: Path, archive_path: Path) -> dict[str, object]
    behavior: Constructs the expected source metadata, per-file hashes, source counts, and filtered headline counts.
  - kind: function
    name: _replace_first_line
    lines: 154-157
    signature: _replace_first_line(path: Path, transform: Callable[[list[str]], list[str]]) -> None
    behavior: Mutates the first fixture row through a supplied field-level transform for corruption cases.
  - kind: function
    name: test_valid_source_is_parsed_and_partitioned
    lines: 160-214
    signature: test_valid_source_is_parsed_and_partitioned(tmp_path: Path) -> None
    behavior: Verifies the full synthetic contract, threshold boundary labels, row-type counts, five train/test complements, and normalized pseudo-sequence lookup with exact structural assertions.
  - kind: function
    name: _run_corruption
    lines: 217-296
    signature: _run_corruption(case: str, root: Path) -> None
    behavior: Applies one named archive, hash, fold, row, class-support, or pseudo-sequence corruption and invokes the relevant public loader or verifier.
  - kind: constant
    name: CORRUPTION_CASES
    lines: 299-319
    signature: CORRUPTION_CASES = ((case, expected_error_pattern), ...)
    behavior: Maps 16 corruption scenarios to stable ValueError message fragments.
  - kind: function
    name: test_contract_corruption_is_rejected
    lines: 327-331
    signature: test_contract_corruption_is_rejected(case: str, message: str, tmp_path: Path) -> None
    behavior: Parametrically asserts every corruption case raises ValueError matching its expected message fragment.
  - kind: constant
    name: CONTRACT_COUNT_FIELDS
    lines: 334-348
    signature: CONTRACT_COUNT_FIELDS = ((parent_key, count_field), ...)
    behavior: Enumerates every required field in source_counts and headline_counts.
  - kind: constant
    name: CONTRACT_COUNT_FIELD_CASES
    lines: 351-355
    signature: CONTRACT_COUNT_FIELD_CASES = ((*increment_cases, delete_case, extra_case))
    behavior: Expands count validation across every increment mismatch plus missing-key and extra-key schema cases.
  - kind: function
    name: test_contract_count_field_mismatch_is_rejected
    lines: 363-380
    signature: test_contract_count_field_mismatch_is_rejected(parent: str, field: str, mode: str, tmp_path: Path) -> None
    behavior: Mutates loaded contract dictionaries in memory and asserts verification rejects value or key-set mismatches with targeted messages.

file: tests/test_baseline_knn.py
mode: full
entries:
  - kind: import
    name: numpy
    lines: 3-3
    signature: import numpy as np
    behavior: Supplies deterministic random arrays, finite checks, and closeness assertions.
  - kind: import
    name: pandas
    lines: 4-4
    signature: import pandas as pd
    behavior: Builds compact train and test dataframes.
  - kind: import
    name: pytest
    lines: 5-5
    signature: import pytest
    behavior: Supplies approximate numeric and exception assertions.
  - kind: import
    name: cognate.baseline_knn
    lines: 7-11
    signature: from cognate.baseline_knn import build_database, exact_match_mask, score_by_nearest_positive
    behavior: Imports the shared database, exact-match, and nearest-positive scoring operators.
  - kind: function
    name: _train
    lines: 14-21
    signature: _train() -> pd.DataFrame
    behavior: Returns a four-row positive training fixture with duplicate and distinct TCRs across two peptides.
  - kind: function
    name: _test_rows
    lines: 24-27
    signature: _test_rows(pairs: list[tuple[str, str]]) -> pd.DataFrame
    behavior: Converts peptide/TCR pairs into a scoring dataframe while preserving order.
  - kind: function
    name: test_database_holds_distinct_positives_per_peptide
    lines: 30-33
    signature: test_database_holds_distinct_positives_per_peptide() -> None
    behavior: Asserts database keys are peptide-scoped and each sequence collection is deduplicated and sorted.
  - kind: function
    name: test_database_ignores_negative_rows
    lines: 36-41
    signature: test_database_ignores_negative_rows() -> None
    behavior: Flips the only occurrence of one TCR negative and asserts it is excluded while duplicated positive support remains.
  - kind: function
    name: test_exact_match_scores_one
    lines: 44-49
    signature: test_exact_match_scores_one() -> None
    behavior: Uses pytest.approx to pin exact-match score 1.0 and its nearest sequence.
  - kind: function
    name: test_database_is_per_peptide_not_global
    lines: 52-62
    signature: test_database_is_per_peptide_not_global() -> None
    behavior: Asserts a TCR match under another peptide is not treated as exact and routing stays within the query peptide database.
  - kind: function
    name: test_missing_peptide_takes_the_default_score
    lines: 65-71
    signature: test_missing_peptide_takes_the_default_score() -> None
    behavior: Pins unseen-peptide score 0.0, false database availability, and unscorable-row count.
  - kind: function
    name: test_constant_scores_give_exactly_half
    lines: 74-80
    signature: test_constant_scores_give_exactly_half() -> None
    behavior: Imports auc01 locally and uses pytest.approx to establish constant-score AUC0.1 as exactly 0.5.
  - kind: function
    name: test_leave_out_exact_matches_falls_back_to_second_nearest
    lines: 83-90
    signature: test_leave_out_exact_matches_falls_back_to_second_nearest() -> None
    behavior: Compares the same shared scoring operator with and without exact-match exclusion and pins second-nearest fallback identity.
  - kind: function
    name: test_leave_out_exact_matches_keeps_the_row_scorable
    lines: 93-98
    signature: test_leave_out_exact_matches_keeps_the_row_scorable() -> None
    behavior: Asserts leave-out exclusion preserves output length and produces finite scores rather than dropping rows.
  - kind: function
    name: test_database_size_is_reported
    lines: 101-105
    signature: test_database_size_is_reported() -> None
    behavior: Pins per-query database_size bookkeeping for two peptide databases.
  - kind: function
    name: test_exact_match_mask_requires_both_fields_to_match
    lines: 108-116
    signature: test_exact_match_mask_requires_both_fields_to_match() -> None
    behavior: Asserts exact-match masking is the conjunction of peptide and TCR equality.
  - kind: function
    name: test_score_ordering_prefers_closer_sequences
    lines: 119-124
    signature: test_score_ordering_prefers_closer_sequences() -> None
    behavior: Uses strict inequalities to pin monotonic edit-similarity ranking across exact, near, and distant TCRs.
  - kind: function
    name: test_macro_auc01_is_invariant_to_within_peptide_rescaling
    lines: 127-155
    signature: test_macro_auc01_is_invariant_to_within_peptide_rescaling() -> None
    behavior: Uses seeded random scores and 1e-12 approximate assertions to prove macro AUC0.1 invariance under within-group standardization and ranking while pooled AUROC changes.
  - kind: function
    name: test_top_k_changes_the_within_peptide_ranking
    lines: 158-166
    signature: test_top_k_changes_the_within_peptide_ranking() -> None
    behavior: Runs the shared scorer at k=1 and k=2, asserting arrays differ and averaged scores do not exceed maxima.
  - kind: function
    name: test_top_k_falls_back_when_database_is_smaller_than_k
    lines: 169-172
    signature: test_top_k_falls_back_when_database_is_smaller_than_k() -> None
    behavior: Uses pytest.approx to pin correct exact score when requested k exceeds database size.
  - kind: function
    name: test_top_k_rejects_zero
    lines: 175-177
    signature: test_top_k_rejects_zero() -> None
    behavior: Asserts zero-valued top_k raises ValueError with a stable validation message.

file: tests/test_knn_cosine.py
mode: full
entries:
  - kind: import
    name: numpy
    lines: 9-9
    signature: import numpy as np
    behavior: Builds cache arrays and supplies deterministic vectors and numeric expectations.
  - kind: import
    name: pandas
    lines: 10-10
    signature: import pandas as pd
    behavior: Builds shared-operator train and test tables.
  - kind: import
    name: pytest
    lines: 11-11
    signature: import pytest
    behavior: Supplies tolerance-aware cosine assertions.
  - kind: import
    name: cognate.baseline_knn
    lines: 13-17
    signature: from cognate.baseline_knn import cosine_similarity, edit_similarity, score_by_nearest_positive
    behavior: Imports both interchangeable similarity backends and their shared nearest-positive operator.
  - kind: import
    name: cognate.embed.EmbeddingCache
    lines: 18-18
    signature: from cognate.embed import EmbeddingCache
    behavior: Supplies the in-memory cache type used as a deterministic test double without mocks.
  - kind: constant
    name: LAYER
    lines: 20-20
    signature: LAYER = 0
    behavior: Pins all toy-cache cosine tests to the single available layer.
  - kind: function
    name: make_cache
    lines: 23-31
    signature: make_cache(vectors: dict[str, list[float]]) -> EmbeddingCache
    behavior: Constructs a one-layer EmbeddingCache directly from supplied vectors, avoiding model calls and mocking frameworks.
  - kind: function
    name: test_cosine_backend_computes_cosine
    lines: 34-38
    signature: test_cosine_backend_computes_cosine()
    behavior: Pins orthogonal, identical, and 45-degree cosine values with absolute tolerance 1e-6.
  - kind: function
    name: test_cosine_is_blind_to_magnitude
    lines: 41-46
    signature: test_cosine_is_blind_to_magnitude()
    behavior: Uses two collinear database vectors at different scales to distinguish cosine from dot product.
  - kind: function
    name: test_separate_caches_for_queries_and_database
    lines: 49-55
    signature: test_separate_caches_for_queries_and_database()
    behavior: Builds disjoint query and database caches and pins cross-cache cosine routing at tolerance 1e-6.
  - kind: function
    name: test_backend_swap_leaves_database_bookkeeping_identical
    lines: 58-75
    signature: test_backend_swap_leaves_database_bookkeeping_identical()
    behavior: Compares edit and cosine through the same scoring operator and asserts identical has_database, database_size, and missing-peptide default behavior.
  - kind: function
    name: test_explicit_edit_backend_equals_the_default
    lines: 78-83
    signature: test_explicit_edit_backend_equals_the_default()
    behavior: Asserts explicitly injecting edit_similarity produces the exact same score list as the operator default.

file: tests/test_embed.py
mode: full
entries:
  - kind: import
    name: numpy
    lines: 7-7
    signature: import numpy as np
    behavior: Supplies seeded fixtures and strict array comparison helpers.
  - kind: import
    name: pytest
    lines: 8-8
    signature: import pytest
    behavior: Supplies module-scoped fixtures and exception assertions.
  - kind: import
    name: torch
    lines: 9-9
    signature: import torch
    behavior: Builds attention tensors and manually recomputes hidden-state pooling.
  - kind: import
    name: cognate.embed
    lines: 11-18
    signature: from cognate.embed import MODELS, _length_batches, _residue_mask, embed_sequences, load_cache, save_cache
    behavior: Imports embedding internals, the real embedding operator, and cache persistence surface.
  - kind: constant
    name: SEQUENCES
    lines: 20-27
    signature: SEQUENCES = [six peptide and TCR sequence literals]
    behavior: Defines mixed-length sequences shared by the module-scoped real-model embedding cache.
  - kind: function
    name: cache
    lines: 31-33
    signature: cache()
    behavior: Module-scoped pytest fixture that embeds SEQUENCES once with the real 8M model and progress disabled.
  - kind: function
    name: test_residue_mask_drops_bos_and_eos
    lines: 36-40
    signature: test_residue_mask_drops_bos_and_eos() -> None
    behavior: Pins exact residue-mask rows and per-row counts for padded and unpadded attention patterns.
  - kind: function
    name: test_length_batches_cover_every_sequence_once
    lines: 43-48
    signature: test_length_batches_cover_every_sequence_once() -> None
    behavior: Flattens returned batch indices and asserts exact single coverage plus nonempty batches.
  - kind: function
    name: test_length_batches_keep_padding_waste_low
    lines: 51-68
    signature: test_length_batches_keep_padding_waste_low() -> None
    behavior: Uses seeded 2,000-sequence lengths to assert padded-token overhead below 10 percent and lower than unsorted windows.
  - kind: function
    name: test_cache_shape_covers_every_layer
    lines: 71-76
    signature: test_cache_shape_covers_every_layer(cache) -> None
    behavior: Pins 8M cache sequence count, seven layers, width 320, full shape, and finite values.
  - kind: function
    name: test_model_has_no_randomly_initialised_pooler
    lines: 79-85
    signature: test_model_has_no_randomly_initialised_pooler() -> None
    behavior: Loads the configured model directly with pooling disabled and asserts no pooler is attached.
  - kind: function
    name: test_embeddings_are_invariant_to_batching
    lines: 88-101
    signature: test_embeddings_are_invariant_to_batching(cache) -> None
    behavior: Re-embeds every sequence alone and compares every layer to batched cache lookup with atol 2e-5.
  - kind: function
    name: test_pooling_matches_manual_mean_over_residues
    lines: 104-119
    signature: test_pooling_matches_manual_mean_over_residues(cache) -> None
    behavior: Runs tokenizer and model hidden states directly, excludes BOS/EOS, and matches manual residue mean to cache output at atol 2e-5.
  - kind: function
    name: test_lookup_preserves_requested_order
    lines: 122-126
    signature: test_lookup_preserves_requested_order(cache) -> None
    behavior: Uses exact array equality to assert arbitrary lookup ordering is preserved.
  - kind: function
    name: test_lookup_raises_on_unknown_sequence
    lines: 129-131
    signature: test_lookup_raises_on_unknown_sequence(cache) -> None
    behavior: Asserts unknown sequence lookup raises KeyError with a stable message fragment.
  - kind: function
    name: test_cache_round_trips_through_disk
    lines: 134-143
    signature: test_cache_round_trips_through_disk(cache, tmp_path) -> None
    behavior: Saves to a temporary NPZ, checks positive byte size, reloads, and compares metadata, exact sequences, and approximate layer values.
  - kind: function
    name: test_layers_are_not_all_identical
    lines: 146-150
    signature: test_layers_are_not_all_identical(cache) -> None
    behavior: Asserts first, middle, and final layer arrays are not numerically collapsed.

file: tests/test_features.py
mode: full
entries:
  - kind: import
    name: numpy
    lines: 3-3
    signature: import numpy as np
    behavior: Supplies seeded linear weights and array closeness comparisons.
  - kind: import
    name: pandas
    lines: 4-4
    signature: import pandas as pd
    behavior: Builds shared peptide/TCR row fixtures.
  - kind: import
    name: pytest
    lines: 5-5
    signature: import pytest
    behavior: Supplies fixtures, approximate scalar comparisons, and exception assertions.
  - kind: import
    name: cognate.embed.embed_sequences
    lines: 7-7
    signature: from cognate.embed import embed_sequences
    behavior: Creates a real embedding cache for feature integration tests.
  - kind: import
    name: cognate.features
    lines: 8-8
    signature: from cognate.features import build_features, feature_width
    behavior: Imports the shared feature-construction operator and its width calculation.
  - kind: function
    name: cache
    lines: 12-18
    signature: cache()
    behavior: Module-scoped pytest fixture embedding four peptide/TCR sequences once with the real 8M model and progress disabled.
  - kind: function
    name: rows
    lines: 22-28
    signature: rows() -> pd.DataFrame
    behavior: Function-scoped fixture returning two aligned peptide/TCR pairs.
  - kind: function
    name: test_concat_width_is_two_blocks
    lines: 31-34
    signature: test_concat_width_is_two_blocks(cache, rows) -> None
    behavior: Pins concat output to two hidden-width blocks and cross-checks feature_width.
  - kind: function
    name: test_interaction_width_is_four_blocks
    lines: 37-40
    signature: test_interaction_width_is_four_blocks(cache, rows) -> None
    behavior: Pins interaction output to four hidden-width blocks and cross-checks feature_width.
  - kind: function
    name: test_interaction_blocks_are_the_stated_functions
    lines: 43-54
    signature: test_interaction_blocks_are_the_stated_functions(cache, rows) -> None
    behavior: Slice-checks peptide, TCR, absolute-difference, and elementwise-product blocks against direct cache operations at rtol 1e-6.
  - kind: function
    name: test_concat_cannot_express_a_peptide_specific_ranking
    lines: 57-80
    signature: test_concat_cannot_express_a_peptide_specific_ranking(cache) -> None
    behavior: Uses seeded linear weights and atol 1e-4 to show concat produces the same TCR score gap under both peptides.
  - kind: function
    name: test_interactions_can_express_a_peptide_specific_ranking
    lines: 83-98
    signature: test_interactions_can_express_a_peptide_specific_ranking(cache) -> None
    behavior: Uses the same crossed rows and seeded weights to assert interaction features yield peptide-specific TCR score gaps.
  - kind: function
    name: test_unknown_block_raises
    lines: 101-103
    signature: test_unknown_block_raises(cache, rows) -> None
    behavior: Asserts an unsupported feature-block literal raises ValueError with a stable message.
  - kind: function
    name: test_single_side_probe_widths_are_one_block
    lines: 106-110
    signature: test_single_side_probe_widths_are_one_block(cache, rows) -> None
    behavior: Loops through peptide_only and tcr_only and pins one hidden-width block plus feature_width agreement.
  - kind: function
    name: test_probe_blocks_carry_only_their_own_side
    lines: 113-117
    signature: test_probe_blocks_carry_only_their_own_side(cache, rows) -> None
    behavior: Compares each single-side feature output directly to its corresponding cache lookup.
  - kind: function
    name: test_peptide_only_gives_one_constant_score_per_peptide
    lines: 120-137
    signature: test_peptide_only_gives_one_constant_score_per_peptide(cache) -> None
    behavior: Asserts peptide-only rows are identical within peptide and distinct across peptides, establishing constant within-group scores.
