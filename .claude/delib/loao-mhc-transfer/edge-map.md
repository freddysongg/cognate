# Edge Map — Can MHC pseudo-sequence features improve binding prediction for HLA alleles not seen in training, beyond peptide-only, PWM, and non-transfer controls?

## Scope

Can MHC pseudo-sequence features improve binding prediction for HLA alleles not seen in training, beyond peptide-only, PWM, and non-transfer controls?

## Components / nodes

- **NetMHCpan training archive** — External archive used as the pMHC training-data source. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/fetch_pmhc_data.sh`)_
- **pMHC source contract (`data/pmhc/source_contract.json`)** — Records archive and extracted-file hashes, source counts, and 47-allele cohort counts. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `data/pmhc/source_contract.json`)_
- **pMHC data fetcher (`scripts/fetch_pmhc_data.sh`)** — Downloads the archive and verifies archive and extracted-file hashes before retaining raw rows. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/fetch_pmhc_data.sh`)_
- **pMHC scoring module (`src/cognate/pmhc.py`)** — Parses the five binding-affinity folds, validates the source contract, builds the eligible cohort, and implements scoring arms. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/pmhc.py`)_
- **MHC pseudo-sequence file (`MHC_pseudo.dat`)** — Supplies required 34-residue pseudo-sequences for requested alleles. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — key_symbols > `load_pseudo_sequences`)_
- **Shared retrieval module (`src/cognate/baseline_knn.py`)** — Provides nearest-positive retrieval and edit or embedding-cosine similarity operators for TCR and pMHC scoring. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/baseline_knn.py`)_
- **Embedding module (`src/cognate/embed.py`)** — Builds and loads mean-pooled ESM-2 embedding caches keyed by sequence string. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/embed.py`)_
- **Metrics module (`src/cognate/metrics.py`)** — Computes partial AUROC, group-macro metrics, bootstrap intervals, diagnostics, and paired arm comparisons. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/metrics.py`)_
- **pMHC runner (`scripts/run_pmhc.py`)** — Scores supplied outer folds, evaluates aggregate per-allele metrics and paired comparisons, and writes the results artifact. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_pmhc.py`)_
- **pMHC results artifact (`data/pmhc/results.json`)** — Stores aggregate pMHC contract, configuration, scores, per-allele outputs, comparisons, negative-source slices, diagnostics, and criteria. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `data/pmhc/results.json`)_
- **pMHC tests (`tests/test_pmhc.py`)** — Tests pMHC validation, fold and assignment guards, MLP behavior, metric configuration, and result-artifact outputs. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `tests/test_pmhc.py`)_
- **TCR data module (`src/cognate/data.py`)** — Loads IMMREP23 TCR data and identifies test peptides present in TCR training data. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/data.py`)_
- **TCR baseline runner (`scripts/run_baseline_knn.py`)** — Evaluates TCR nearest-positive retrieval on seen and unseen peptide slices, including exact-match controls. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_baseline_knn.py`)_
- **TCR edit/ESM runner (`scripts/run_knn_esm_cosine.py`)** — Evaluates TCR edit and ESM cosine retrieval with the same operator on seen and unseen peptide slices. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_knn_esm_cosine.py`)_
- **TCR/pMHC contrast document (`docs/contrast.md`)** — Records the descriptive retrieval contrast and endpoint, query, group, partition, and metric differences. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `docs/contrast.md`)_
- **pMHC experiment document (`docs/pmhc/experiment.md`)** — Records source, cohort, scoring arms, supplied-fold evaluation contract, and negative-source analysis. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `docs/pmhc/experiment.md`)_
- **pMHC postmortem (`docs/pmhc/postmortem.md`)** — Records the completed seen-allele experiment and its stated leave-one-allele-out follow-up boundary. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `docs/pmhc/postmortem.md`)_
- **pMHC raw/cache ignore rules (`.gitignore`)** — Ignores pMHC raw and derived rows and the shared embedding-cache directory. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `.gitignore`)_
- **Python standard-library dependencies** — `hashlib`, `json`, `math`, `collections.abc`, `dataclasses`, `functools`, and `pathlib` imported by the pMHC scoring module. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/pmhc.py` / stdlib)_
- **pMHC scientific dependencies** — `numpy`, `pandas`, `torch`, `Bio`, and `sklearn` imported by the pMHC scoring module. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/pmhc.py` / third-party)_
- **Retrieval dependency (`rapidfuzz`)** — Imported by the shared retrieval module. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/baseline_knn.py`)_
- **Embedding dependencies** — `numpy`, `torch`, and `transformers` imported by the embedding module. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/embed.py`)_
- **Metrics dependencies** — `numpy`, `pandas`, and `sklearn` imported by the metrics module. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/metrics.py`)_
- **pMHC runner dependencies** — `numpy` and `pandas` imported by the pMHC runner. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `scripts/run_pmhc.py`)_
- **TCR data dependency (`pandas`)** — Imported by the TCR data module. _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/data.py`)_

## Edges / data flow

| Source | Edge type | Target | Cite |
|---|---|---|---|
| pMHC data fetcher (`scripts/fetch_pmhc_data.sh`) | emits-consumes | NetMHCpan training archive | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/fetch_pmhc_data.sh`: “Downloads the NetMHCpan training archive” |
| pMHC scoring module (`src/cognate/pmhc.py`) | shares-schema | pMHC source contract (`data/pmhc/source_contract.json`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/pmhc.py`: “validates the source contract”; key_symbols > `verify_source` |
| pMHC scoring module (`src/cognate/pmhc.py`) | emits-consumes | MHC pseudo-sequence file (`MHC_pseudo.dat`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — key_symbols > `load_pseudo_sequences`: maps each requested allele from `MHC_pseudo.dat` |
| pMHC scoring module (`src/cognate/pmhc.py`) | calls | Shared retrieval module (`src/cognate/baseline_knn.py`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — key_symbols > `score_retrieval_fold`: “Applies the shared nearest-positive operator” |
| pMHC runner (`scripts/run_pmhc.py`) | emits-consumes | pMHC results artifact (`data/pmhc/results.json`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_pmhc.py`: “writes `data/pmhc/results.json`” |
| pMHC tests (`tests/test_pmhc.py`) | calls | pMHC scoring module (`src/cognate/pmhc.py`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `tests/test_pmhc.py`: tests pMHC validation, scoring, MLP, metrics, and outputs |
| TCR baseline runner (`scripts/run_baseline_knn.py`) | calls | Shared retrieval module (`src/cognate/baseline_knn.py`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_baseline_knn.py`: evaluates the TCR nearest-positive baseline; modules > `src/cognate/baseline_knn.py`: used by TCR and pMHC scoring |
| TCR edit/ESM runner (`scripts/run_knn_esm_cosine.py`) | calls | Shared retrieval module (`src/cognate/baseline_knn.py`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_knn_esm_cosine.py`: evaluates retrieval “with the same operator”; modules > `src/cognate/baseline_knn.py` |
| pMHC scoring module (`src/cognate/pmhc.py`) | imports | Python standard-library dependencies | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/pmhc.py` / stdlib |
| pMHC scoring module (`src/cognate/pmhc.py`) | imports | pMHC scientific dependencies | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/pmhc.py` / third-party |
| Shared retrieval module (`src/cognate/baseline_knn.py`) | imports | Retrieval dependency (`rapidfuzz`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/baseline_knn.py` |
| Embedding module (`src/cognate/embed.py`) | imports | Embedding dependencies | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/embed.py` |
| Metrics module (`src/cognate/metrics.py`) | imports | Metrics dependencies | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/metrics.py` |
| pMHC runner (`scripts/run_pmhc.py`) | imports | pMHC runner dependencies | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `scripts/run_pmhc.py` |
| TCR data module (`src/cognate/data.py`) | imports | TCR data dependency (`pandas`) | `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — import_edges > `src/cognate/data.py` |

## Status

- **NetMHCpan training archive** — unknown _(src: no lifecycle fact)_
- **pMHC source contract (`data/pmhc/source_contract.json`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `data/pmhc/source_contract.json`)_
- **pMHC data fetcher (`scripts/fetch_pmhc_data.sh`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/fetch_pmhc_data.sh`)_
- **pMHC scoring module (`src/cognate/pmhc.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/pmhc.py`; `docs/pmhc/postmortem.md` records a completed experiment)_
- **MHC pseudo-sequence file (`MHC_pseudo.dat`)** — unknown _(src: no lifecycle fact)_
- **Shared retrieval module (`src/cognate/baseline_knn.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/baseline_knn.py`)_
- **Embedding module (`src/cognate/embed.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/embed.py`)_
- **Metrics module (`src/cognate/metrics.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/metrics.py`)_
- **pMHC runner (`scripts/run_pmhc.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_pmhc.py`)_
- **pMHC results artifact (`data/pmhc/results.json`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `data/pmhc/results.json`)_
- **pMHC tests (`tests/test_pmhc.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `tests/test_pmhc.py`)_
- **TCR data module (`src/cognate/data.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `src/cognate/data.py`)_
- **TCR baseline runner (`scripts/run_baseline_knn.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_baseline_knn.py`)_
- **TCR edit/ESM runner (`scripts/run_knn_esm_cosine.py`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `scripts/run_knn_esm_cosine.py`)_
- **TCR/pMHC contrast document (`docs/contrast.md`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `docs/contrast.md`)_
- **pMHC experiment document (`docs/pmhc/experiment.md`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `docs/pmhc/experiment.md`)_
- **pMHC postmortem (`docs/pmhc/postmortem.md`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `docs/pmhc/postmortem.md`)_
- **pMHC raw/cache ignore rules (`.gitignore`)** — built _(src: `.claude/delib/loao-mhc-transfer/harvest-repo-cognate.md` — modules > `.gitignore`)_
- **Python standard-library dependencies** — unknown _(src: no lifecycle fact)_
- **pMHC scientific dependencies** — unknown _(src: no lifecycle fact)_
- **Retrieval dependency (`rapidfuzz`)** — unknown _(src: no lifecycle fact)_
- **Embedding dependencies** — unknown _(src: no lifecycle fact)_
- **Metrics dependencies** — unknown _(src: no lifecycle fact)_
- **pMHC runner dependencies** — unknown _(src: no lifecycle fact)_
- **TCR data dependency (`pandas`)** — unknown _(src: no lifecycle fact)_

## Open edges (unverified)

- **A leave-one-allele-out partitioner** → **pMHC scoring module (`src/cognate/pmhc.py`)** — no capture records an allele-held-out partition; `iter_pmhc_folds` records only supplied `c000_ba`–`c004_ba` test partitions.
- **pMHC runner (`scripts/run_pmhc.py`)** → **pMHC scoring module (`src/cognate/pmhc.py`)** — both roles are recorded, but no first-party import or call edge is present in the captures.
- **pMHC scoring module (`src/cognate/pmhc.py`)** → **Embedding module (`src/cognate/embed.py`)** — the captures record ESM retrieval and embedding-cache roles, but no direct edge between these modules.
- **pMHC scoring module (`src/cognate/pmhc.py`)** → **Metrics module (`src/cognate/metrics.py`)** — the captures record pMHC evaluation and the shared metrics functions, but no direct edge between these modules.
