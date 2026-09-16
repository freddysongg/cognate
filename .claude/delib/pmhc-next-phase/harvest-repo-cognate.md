repo: /Users/freddy/Documents/repos/cognate
origin: freddysongg/cognate
fetch: ok
branches:
  - name: origin/main
    last_commit: "2026-09-09T19:28:28-07:00"
    status: active
    ahead_of_main: 0
    ahead_subjects: []
  - name: origin/fork/crossattn
    last_commit: "2026-09-07T17:12:24-07:00"
    status: active
    ahead_of_main: 0
    ahead_subjects: []
  - name: origin/fork/contrastive
    last_commit: "2026-09-07T16:36:39-07:00"
    status: active
    ahead_of_main: 0
    ahead_subjects: []
pr_state: ok
open_prs: []
modules:
  - path: docs/final_state.md
    purpose: Closed-state record for the TCR-epitope project, including the research question, results, metric contract, frozen boundaries, and linked evidence.
  - path: docs/lessons.md
    purpose: Transferable methodological lessons from the closed TCR project, labeled by the repository evidence scale.
  - path: docs/belief_list.md
    purpose: Locked TCR-project claims, overturn conditions, supporting evidence, and appended B3 outcomes.
  - path: docs/redteam_curve.md
    purpose: Read-only audit of the TCR deduplication sensitivity curve, bootstrap inference, and claim boundaries.
  - path: docs/pmhc/brief.md
    purpose: Draft scope for a pMHC class I presentation replication that reuses the TCR baseline, embeddings, metrics, and interval machinery after a literature gate.
  - path: src/cognate/baseline_knn.py
    purpose: Per-group nearest-positive retrieval baseline with normalized edit similarity or mean-pooled ESM-2 cosine and exact-match controls.
  - path: src/cognate/embed.py
    purpose: ESM-2 mean-pooled and per-residue embedding generation, lookup, and on-disk cache handling.
  - path: src/cognate/metrics.py
    purpose: AUROC, AUPRC, macro groupwise AUC0.1, score diagnostics, percentile bootstrap intervals, and paired comparisons.
  - path: tests/test_baseline_knn.py
    purpose: Tests nearest-positive retrieval, alternate similarity functions, exact-match behavior, and macro-versus-pooled scoring properties.
  - path: tests/test_embed.py
    purpose: Tests embedding cache lookup, pooling masks, batching, serialization, and per-residue cache behavior.
  - path: tests/test_metrics.py
    purpose: Tests point metrics, grouped macro AUC0.1, diagnostics, reports, and bootstrap interval behavior.
  - path: tests/test_comparison.py
    purpose: Tests paired macro AUC0.1 bootstrap comparisons, including symmetry and shared resampling.
  - path: tests/test_frozen_contract_hashes.py
    purpose: Verifies recorded hashes for the frozen metric, split, feature, and evaluation-contract files.
key_symbols:
  - name: edit_similarity
    kind: function
    lines: 42-47
    behavior: Computes the query-by-entry matrix of normalized Levenshtein similarities.
  - name: cosine_similarity
    kind: function
    lines: 51-72
    behavior: Builds a similarity function that unit-normalizes cached ESM-2 vectors and returns their cosine matrix.
  - name: KnnResult
    kind: class
    lines: 73-95
    behavior: Carries per-row nearest-neighbor scores, database availability and size, and nearest-sequence provenance.
  - name: build_database
    kind: function
    lines: 96-115
    behavior: Maps each group key to its distinct positive training sequences.
  - name: score_by_nearest_positive
    kind: function
    lines: 116-193
    behavior: Scores each test row against positive sequences in its matching group using maximum or top-k similarity with optional exact-match exclusion.
  - name: exact_match_mask
    kind: function
    lines: 194-210
    behavior: Marks test rows whose group-and-sequence pair occurs among positive training rows.
  - name: EmbeddingCache
    kind: class
    lines: 41-80
    behavior: Stores mean-pooled ESM-2 vectors for all layers and performs ordered sequence lookup with missing-key rejection.
  - name: pick_device
    kind: function
    lines: 92-95
    behavior: Selects Apple MPS when available and CPU otherwise.
  - name: embed_sequences
    kind: function
    lines: 125-167
    behavior: Embeds distinct sequences with ESM-2, removes special tokens from pooling, and returns an all-layer cache plus elapsed time.
  - name: default_cache_dir
    kind: function
    lines: 168-180
    behavior: Resolves the cache directory from COGNATE_CACHE_DIR or the repository shared directory.
  - name: cache_path
    kind: function
    lines: 181-184
    behavior: Builds the mean-pooled ESM-2 cache path for a model key.
  - name: save_cache
    kind: function
    lines: 185-199
    behavior: Writes an uncompressed mean-pooled embedding cache and returns its byte size.
  - name: load_cache
    kind: function
    lines: 200-209
    behavior: Loads a mean-pooled embedding cache without pickle and rebuilds its sequence index.
  - name: ResidueCache
    kind: class
    lines: 210-262
    behavior: Stores ragged per-residue embeddings for selected layers and supports residue lookup and recomputed mean pooling.
  - name: embed_residues
    kind: function
    lines: 263-318
    behavior: Embeds distinct sequences with ESM-2 and stores unpadded residue vectors for selected layers.
  - name: save_residue_cache
    kind: function
    lines: 319-331
    behavior: Writes an uncompressed per-residue cache and returns its byte size.
  - name: load_residue_cache
    kind: function
    lines: 332-342
    behavior: Loads a per-residue cache without pickle and rebuilds its sequence index.
  - name: auroc
    kind: function
    lines: 35-39
    behavior: Computes full ROC AUC through scikit-learn.
  - name: auprc
    kind: function
    lines: 40-44
    behavior: Computes average precision through scikit-learn.
  - name: auc01
    kind: function
    lines: 45-50
    behavior: Computes McClish-standardized partial ROC AUC through false-positive rate 0.1.
  - name: Interval
    kind: class
    lines: 51-66
    behavior: Represents a point estimate, percentile bounds, and retained bootstrap replicate count.
  - name: DegenerateScoresError
    kind: class
    lines: 67-71
    behavior: Signals a score column that cannot express a ranking.
  - name: ScoreDiagnostics
    kind: class
    lines: 72-114
    behavior: Records global and per-group score degeneracy, non-finite values, and tie concentration.
  - name: diagnose_scores
    kind: function
    lines: 115-142
    behavior: Measures unique scores, non-finite counts, modal share, and constant groups.
  - name: GroupedScore
    kind: class
    lines: 159-172
    behavior: Holds a macro groupwise point score, per-group scores, and skipped-group provenance.
  - name: ScoreReport
    kind: class
    lines: 173-200
    behavior: Aggregates slice counts, interval estimates, per-group AUC0.1 values, and score diagnostics.
  - name: macro_by_group
    kind: function
    lines: 213-244
    behavior: Computes a scorer independently in each group with both classes and averages the scorable group values.
  - name: macro_auc01
    kind: function
    lines: 245-253
    behavior: Applies groupwise McClish-standardized AUC0.1 and returns its macro average.
  - name: _replicate_group_indices
    kind: function
    lines: 254-278
    behavior: Generates row, group, or two-level group-plus-row bootstrap index replicates.
  - name: _percentile_interval
    kind: function
    lines: 279-288
    behavior: Converts bootstrap samples into a central percentile interval around a supplied point estimate.
  - name: correlation_interval
    kind: function
    lines: 295-323
    behavior: Bootstraps a Pearson correlation by resampling paired points and dropping zero-variance replicates.
  - name: correlation_difference_interval
    kind: function
    lines: 324-369
    behavior: Paired-bootstraps the difference between two correlations and returns an interval and two-sided tail probability.
  - name: leave_one_out_correlations
    kind: function
    lines: 370-389
    behavior: Recomputes Pearson correlation after dropping each point in turn.
  - name: evaluate
    kind: function
    lines: 390-467
    behavior: Computes pooled and macro metrics, diagnostics, and configurable row/group bootstrap intervals for a score slice.
  - name: Comparison
    kind: class
    lines: 468-486
    behavior: Represents two macro scores and their paired-bootstrap difference, p-value, and common-group count.
  - name: compare_macro_auc01
    kind: function
    lines: 487-592
    behavior: Compares two score arms on common scorable groups with paired group and shared-row bootstrap draws where row sets match.
  - name: report_table
    kind: function
    lines: 593-610
    behavior: Renders score reports into a tabular slice-by-metric summary.
import_edges:
  - from: src/cognate/baseline_knn.py
    imports: collections.abc
    mechanism: stdlib
  - from: src/cognate/baseline_knn.py
    imports: dataclasses
    mechanism: stdlib
  - from: src/cognate/baseline_knn.py
    imports: numpy
    mechanism: third-party
  - from: src/cognate/baseline_knn.py
    imports: pandas
    mechanism: third-party
  - from: src/cognate/baseline_knn.py
    imports: rapidfuzz
    mechanism: third-party
  - from: src/cognate/baseline_knn.py
    imports: cognate.embed
    mechanism: editable-path
  - from: src/cognate/embed.py
    imports: os
    mechanism: stdlib
  - from: src/cognate/embed.py
    imports: time
    mechanism: stdlib
  - from: src/cognate/embed.py
    imports: collections.abc
    mechanism: stdlib
  - from: src/cognate/embed.py
    imports: dataclasses
    mechanism: stdlib
  - from: src/cognate/embed.py
    imports: pathlib
    mechanism: stdlib
  - from: src/cognate/embed.py
    imports: numpy
    mechanism: third-party
  - from: src/cognate/embed.py
    imports: torch
    mechanism: third-party
  - from: src/cognate/embed.py
    imports: transformers
    mechanism: third-party
  - from: src/cognate/metrics.py
    imports: warnings
    mechanism: stdlib
  - from: src/cognate/metrics.py
    imports: collections.abc
    mechanism: stdlib
  - from: src/cognate/metrics.py
    imports: dataclasses
    mechanism: stdlib
  - from: src/cognate/metrics.py
    imports: typing
    mechanism: stdlib
  - from: src/cognate/metrics.py
    imports: numpy
    mechanism: third-party
  - from: src/cognate/metrics.py
    imports: pandas
    mechanism: third-party
  - from: src/cognate/metrics.py
    imports: scikit-learn
    mechanism: third-party
