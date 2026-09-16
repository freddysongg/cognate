repo: /Users/freddy/Documents/repos/cognate
origin: freddysongg/cognate
fetch: ok
branches:
  - name: origin/main
    last_commit: 2026-09-09
    status: active
    ahead_of_main: 0
    ahead_subjects: []
  - name: origin/fork/crossattn
    last_commit: 2026-09-07
    status: active
    ahead_of_main: 0
    ahead_subjects: []
  - name: origin/fork/contrastive
    last_commit: 2026-09-07
    status: active
    ahead_of_main: 0
    ahead_subjects: []
pr_state: ok
open_prs: []
modules:
  - path: data/pmhc/source_contract.json
    purpose: Records the NetMHCpan archive and extracted-file SHA-256 values, source counts, and the 47-allele headline cohort counts.
  - path: scripts/fetch_pmhc_data.sh
    purpose: Downloads the NetMHCpan training archive and verifies its archive and extracted-file hashes before retaining raw rows.
  - path: src/cognate/pmhc.py
    purpose: Parses the five binding-affinity folds, validates the source contract, builds the eligible pMHC cohort, and implements fixed scoring arms.
  - path: scripts/run_pmhc.py
    purpose: Scores supplied outer folds, evaluates aggregate per-allele metrics and paired comparisons, and writes data/pmhc/results.json.
  - path: data/pmhc/results.json
    purpose: Stores aggregate pMHC contract, configuration, scores, per-allele outputs, comparisons, negative-source slices, diagnostics, and criteria.
  - path: tests/test_pmhc.py
    purpose: Tests pMHC source validation, fold and score assignment guards, MLP feature/training behavior, metric configuration, and result-artifact outputs.
  - path: src/cognate/baseline_knn.py
    purpose: Provides the shared nearest-positive retrieval database and edit or embedding-cosine similarity operators used by TCR and pMHC scoring.
  - path: src/cognate/embed.py
    purpose: Builds and loads mean-pooled ESM-2 embedding caches keyed by sequence string.
  - path: src/cognate/metrics.py
    purpose: Computes standardized partial AUROC, group-macro metrics, bootstrap intervals, diagnostics, and paired arm comparisons.
  - path: src/cognate/data.py
    purpose: Loads the IMMREP23 TCR data and identifies test peptides present in TCR training data.
  - path: scripts/run_baseline_knn.py
    purpose: Evaluates the TCR nearest-positive baseline on seen and unseen peptide slices, including exact-match controls.
  - path: scripts/run_knn_esm_cosine.py
    purpose: Evaluates TCR edit and ESM cosine retrieval with the same operator on seen and unseen peptide slices.
  - path: docs/contrast.md
    purpose: Records the descriptive TCR/pMHC retrieval contrast and the endpoint, query, group, partition, and metric differences.
  - path: docs/pmhc/experiment.md
    purpose: Records the pMHC source, cohort, scoring arms, supplied-fold evaluation contract, and negative-source analysis.
  - path: docs/pmhc/postmortem.md
    purpose: Records the completed seen-allele pMHC experiment results and its stated leave-one-allele-out follow-up boundary.
  - path: .gitignore
    purpose: Ignores pMHC raw and derived rows and the shared embedding-cache directory.
key_symbols:
  - name: score_by_nearest_positive
    kind: function
    lines: 116-191
    behavior: Scores each query against positive training sequences sharing its group key, using edit similarity by default or a supplied similarity function.
  - name: cosine_similarity
    kind: function
    lines: 51-70
    behavior: Returns a similarity function over selected mean-pooled embedding-cache layers.
  - name: EmbeddingCache
    kind: class
    lines: 41-78
    behavior: Stores model name, unique sequence keys, per-layer embeddings, and sequence lookup indices.
  - name: embed_sequences
    kind: function
    lines: 125-165
    behavior: Embeds distinct sequences with ESM-2 and residue-mean-pools every hidden layer into an EmbeddingCache.
  - name: PmhcDataset
    kind: class
    lines: 323-326
    behavior: Holds all nine-mer HLA rows, the eligible analysis rows, and the eligible allele tuple.
  - name: verify_source
    kind: function
    lines: 356-419
    behavior: Validates archive and extracted-file hashes and compares parsed source and headline counts with the source contract.
  - name: load_pmhc_dataset
    kind: function
    lines: 501-535
    behavior: Restricts parsed rows to HLA-A/B/C nine-mers, rejects cross-fold peptide overlap, selects alleles with at least 100 rows of each class, and checks every supplied fold.
  - name: _require_both_classes_per_fold
    kind: function
    lines: 480-498
    behavior: Requires both target classes for every eligible allele in each held-out fold and its training complement.
  - name: iter_pmhc_folds
    kind: function
    lines: 538-548
    behavior: Yields each supplied c000_ba through c004_ba partition as test with the other four partitions as training.
  - name: load_pseudo_sequences
    kind: function
    lines: 551-573
    behavior: Maps each requested allele to a required 34-residue pseudo-sequence from MHC_pseudo.dat.
  - name: score_retrieval_fold
    kind: function
    lines: 73-93
    behavior: Applies the shared nearest-positive operator with Allele as the group key and Peptide as the scored sequence.
  - name: score_pwm_fold
    kind: function
    lines: 96-138
    behavior: Fits one smoothed nine-position positive-versus-negative log-odds matrix per training allele and scores test peptides by summed position log odds.
  - name: build_mlp_features
    kind: function
    lines: 164-185
    behavior: Concatenates peptide BLOSUM50 features with either mapped allele pseudo-sequence BLOSUM50 features or a fixed allele-order one-hot vector.
  - name: score_mlp_fold
    kind: function
    lines: 280-320
    behavior: Fits and averages ten continuous-affinity MLP predictions per outer fold across two hidden sizes and five seeds.
  - name: score_dataset
    kind: function
    lines: 188-259
    behavior: Produces random, composition, edit retrieval, ESM retrieval, PWM, pseudo-sequence MLP, and one-hot MLP out-of-fold scores for every pMHC row.
  - name: evaluate_predictions
    kind: function
    lines: 327-436
    behavior: Evaluates every pMHC arm grouped by Allele, computes paired comparisons against random and pseudo-sequence MLP versus PWM, and partitions measured and artificial negatives.
  - name: evaluate
    kind: function
    lines: 390-465
    behavior: Computes grouped macro AUC0.1, pooled AUROC/AUPRC, score diagnostics, and two-level bootstrap intervals.
  - name: compare_macro_auc01
    kind: function
    lines: 487-590
    behavior: Computes paired bootstrap macro-AUC0.1 differences over groups scorable in both arms.
  - name: load_tcr_retrieval_contrast
    kind: function
    lines: 470-490
    behavior: Reads closed TCR aggregate artifacts and calculates edit and ESM seen-slice uplifts over their observed random score.
  - name: seen_mask
    kind: function
    lines: 41-43
    behavior: Marks IMMREP23 TCR test rows whose peptide is present in the TCR training data.
import_edges:
  - from: src/cognate/pmhc.py
    imports: hashlib, json, math, collections.abc, dataclasses, functools, pathlib
    mechanism: stdlib
  - from: src/cognate/pmhc.py
    imports: numpy, pandas, torch, Bio, sklearn
    mechanism: third-party
  - from: src/cognate/baseline_knn.py
    imports: numpy, pandas, rapidfuzz
    mechanism: third-party
  - from: src/cognate/embed.py
    imports: numpy, torch, transformers
    mechanism: third-party
  - from: src/cognate/metrics.py
    imports: numpy, pandas, sklearn
    mechanism: third-party
  - from: scripts/run_pmhc.py
    imports: numpy, pandas
    mechanism: third-party
  - from: src/cognate/data.py
    imports: pandas
    mechanism: third-party
