# Phase 3: Run, evaluate, document, and stop

## Target files

- `scripts/run_pmhc.py`
- `tests/test_pmhc.py`
- `data/pmhc/results.json`
- `docs/pmhc/experiment.md`
- `docs/contrast.md`
- `docs/pmhc/postmortem.md`

## Off-limits

- `src/cognate/metrics.py`
- `src/cognate/baseline_knn.py`
- `src/cognate/embed.py`
- `data/vdjdb_eval.csv`
- closed TCR documents
- `docs/next_candidates.md`
- `docs/pmhc/src/cognate/data.py`

## Boundaries

- Do not edit the frozen TCR contract: `src/cognate/metrics.py`, `src/cognate/baseline_knn.py`, `src/cognate/embed.py`, `data/vdjdb_eval.csv`, or closed TCR documents.
- Reuse `baseline_knn.py`, `embed.py`, `metrics.py`, and `train_head.py`; do not wrap them in new abstraction layers.
- Do not run or redistribute the NetMHCpan executable. Fetch only the public training archive.
- Score the fixed 47-allele, 112,128-row cohort before looking at results.
- Report six model families and seven rows: random, length/composition, edit retrieval, ESM-2 retrieval, PWM, pseudo-sequence MLP, and one-hot MLP control.
- The two MLP encodings each use hidden sizes 55 and 66, seeds 0–4, and five folds: 50 fits per encoding, 100 total.
- Keep raw rows, derived row-level predictions, and embeddings out of git.
- Keep only four durable documents: `docs/pmhc/brief.md`, `docs/pmhc/experiment.md`, `docs/contrast.md`, and `docs/pmhc/postmortem.md`.
- Preserve existing user changes in `docs/next_candidates.md` and `docs/pmhc/src/cognate/data.py`; do not import, move, or edit the latter.
- Each phase ends with a fresh `reviewer-py` pass. Reviews must report `<N> candidates, <K> refuted, <S> survived (<b> blocker, <m> major, <n> minor)` and resolve all blockers and majors before continuing.

## Minimal File Set

| Path | Action |
|---|---|
| `.gitignore` | ignore `data/pmhc/raw/` and `data/pmhc/derived/` |
| `docs/pmhc/brief.md` | replace with the reviewed brief |
| `docs/contrast.md` | create the shared-contract table before scoring; append results later |
| `data/pmhc/source_contract.json` | pin source URL, hashes, and counts |
| `scripts/fetch_pmhc_data.sh` | non-overwriting fetch and extraction |
| `src/cognate/pmhc.py` | all pMHC-only data, PWM, BLOSUM, MLP, and fold-scoring logic |
| `tests/test_pmhc.py` | contract corruptions and focused numerical tests |
| `scripts/run_pmhc.py` | one fixed experiment runner |
| `data/pmhc/results.json` | aggregate results and per-allele points; no row-level predictions |
| `docs/pmhc/experiment.md` | method, provenance, results, and criteria |
| `docs/pmhc/postmortem.md` | final evidence-bounded closeout |

No separate baseline, PWM, MLP, evaluation, renderer, or per-allele CSV modules are needed. They would each have one caller.

## Instructions

- [ ] Before implementing the runner/evaluator, add and individually run red-then-green selectors for `each_row_scored_once`, `invalid_scores_stop_evaluation`, `cache_rejects_wrong_model_layer_or_peptide_set`, `headline_uses_alleles_and_20000_draws`, `predictive_requires_lower_bound_above_half`, `pwm_sufficiency_requires_both_conditions`, `comparisons_are_paired_against_random`, `negative_sources_are_separate`, and `tcr_uplift_uses_observed_random`.

- [ ] Implement `scripts/run_pmhc.py` as one fixed runner, not a configurable framework:

  1. Load and verify the archive, extracted files, counts, folds, cohort, and pseudo-sequences.
  2. Load or build `shared/pmhc/esm2_35M_peptides.npz`; reuse it only when model name, layer availability, and exact peptide set match.
  3. Preallocate seven score arrays in dataset row order.
  4. For each supplied fold, fit on four folds and fill only that test fold.
  5. Reject unfilled or non-finite scores before evaluation.
  6. Write only aggregate output to `data/pmhc/results.json`; do not persist row-level predictions.

The JSON has only these top-level keys: `contract`, `config`, `scores`, `per_allele`, `comparisons`, `negative_sources`, `retrieval_diagnostics`, and `criteria`.

- [ ] Evaluate with existing functions:

  - `evaluate(... groups=Allele, mode="both", n_boot=20_000, seed=0)` for macro AUC0.1, pooled AUROC, and pooled AUPRC intervals;
  - `macro_by_group(..., auroc)` for the macro-AUROC point diagnostic; do not build a second bootstrap engine;
  - `compare_macro_auc01` for every non-random arm minus random and for pseudo-sequence MLP minus PWM;
  - positive plus measured-nonbinder rows, and positive plus artificial-negative rows, as separate fixed-cohort slices;
  - nearest-positive edit distance and exact-duplicate counts by negative source.

Predictive means the macro-AUC0.1 interval lower bound is above `0.5`. PWM is sufficient relative to the MLP only when PWM is predictive and the paired upper bound for `mlp_pseudo_sequence - pwm` is below `0.02`.

- [ ] Read TCR points from their actual artifacts:

  - edit and ESM-2: `data/knn_esm_cosine.json` → `scores["edit/seen"]` and `scores["35M layer10 (headline)/seen"]`;
  - random: `data/b3_results.json` → `scores["random/seen"]`.

Report point uplift over observed random for each task. Do not invent a cross-task paired interval or substitute theoretical `0.5`.

- [ ] Add one integration test with tiny five-fold fixtures and mocked embedding/MLP calls. It must prove every row is assigned once, all seven arms are emitted, invalid scores stop evaluation, bootstrap arguments are fixed, and both negative-source slices are reported.

- [ ] Run the real experiment:

```bash
bash scripts/fetch_pmhc_data.sh
uv run python scripts/run_pmhc.py
```

Before scoring, it must print the verified contract: five BA folds, 47 alleles, 112,128 rows, 28,538 positives, 83,590 negatives, and zero cross-fold peptide overlap.

- [ ] Write `docs/pmhc/experiment.md` from the observed JSON. Include source/hash, filters, fold/cohort counts, exact arm settings, seven score rows, 20,000-draw intervals, negative-source diagnostics, criteria verdicts, the regeneration command, and descriptive/non-causal limits.

- [ ] Append observed point results to `docs/contrast.md` without changing its pre-result table or interpretation boundary.

- [ ] Write `docs/pmhc/postmortem.md` answering only from observed outputs: which arms were predictive, whether PWM was sufficient, whether the retrieval contrast survived removal of artificial negatives, what was construction-sensitive, and whether any follow-up deserves a new project.

- [ ] Run final verification:

```bash
uv run pytest -q
uv run python -m compileall -q src scripts
git diff --check
git ls-files data/pmhc/raw data/pmhc/derived shared/pmhc
```

Expected: tests and compile pass, diff check is clean, and no raw/derived/cache paths are tracked.

- [ ] Run a fresh final `reviewer-py` pass, resolve all blockers and majors, rerun verification, and leave the worktree uncommitted for the user.

## Validation

```bash
uv run pytest -q
uv run python -m compileall -q src scripts
git diff --check
git ls-files data/pmhc/raw data/pmhc/derived shared/pmhc
```

## Stop Condition

The project is complete when all seven score rows, the negative-source analysis, the TCR/pMHC retrieval contrast, and the postmortem exist and pass review. Weak results trigger contract diagnosis, not a new architecture, second dataset, cross-attention arm, fine-tuning, or presentation endpoint.
