# pMHC Binding Replication Implementation Plan

> **For the implementation chat:** Use `orchestrate` with `superpowers:executing-plans`. Do not commit, stage, push, open a pull request, or write to GitHub.

**Goal:** Run the reviewed pMHC binding experiment, compare its two retrieval arms with the closed TCR results, and stop after the contrast and postmortem.

**Spec:** [reviewed brief](/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/brief.md), promoted in Phase 1 to [docs/pmhc/brief.md](/Users/freddy/Documents/repos/cognate/docs/pmhc/brief.md). Tracking epic: [#31](https://github.com/freddysongg/cognate/issues/31).

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

---

## Phase 1: Freeze the data and comparison contract

**Files:** `.gitignore`, `docs/pmhc/brief.md`, `docs/contrast.md`, `data/pmhc/source_contract.json`, `scripts/fetch_pmhc_data.sh`, `src/cognate/pmhc.py`, `tests/test_pmhc.py`

- [ ] Promote the reviewed brief, then verify it exactly:

```bash
cmp .claude/delib/pmhc-next-phase/brief.md docs/pmhc/brief.md
```

- [ ] Create `docs/contrast.md` before model code. Its shared-contract table must state:

| Axis | Closed TCR task | pMHC task |
|---|---|---|
| Endpoint | TCR–peptide binding label | normalized pMHC affinity; `>0.426` only for classification |
| Query | CDR3β | nine-residue peptide |
| Retrieval group | peptide | allele |
| Retrieval operator | maximum similarity to a positive | same |
| Edit representation | normalized Levenshtein | same |
| ESM representation | ESM-2 35M, layer 10, residue mean pool, cosine | same |
| Partition | closed seen-peptide slice | supplied common-motif BA folds with disjoint peptide sets |
| Headline metric | macro standardized AUC0.1 by peptide | same implementation, grouped by allele |
| Cross-task quantity | within-task uplift over observed random | same; shown side by side, never paired across tasks |

End the file with: “This comparison is descriptive and dataset-specific. The two headline values are not one matched estimand, and a difference does not identify a biological cause.” Do not add an empty results section.

- [ ] Add these ignore rules:

```gitignore
# pMHC source rows and regenerable row-level outputs
data/pmhc/raw/
data/pmhc/derived/
```

`/shared/` already ignores the pMHC embedding cache.

- [ ] Create `data/pmhc/source_contract.json` with the inspected source:

```json
{
  "source_url": "https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz",
  "retrieval_date": "2026-09-10",
  "archive_sha256": "06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4",
  "file_sha256": {
    "c000_ba": "a5704007e127c8c0e2a3b0043c52ad2f92be5efd01bd2b6a0f7e5424d25d06ef",
    "c001_ba": "754c5fecf1c8387b48fc8cf77473fc14c9e89adb2694ac977e51e7028b0304f1",
    "c002_ba": "f1ff916e3bd4ed01352b7fa4c6a1852fd9294104f6b344c5dd2eba7961a57aef",
    "c003_ba": "9cde2dd51abf1cde03383c5f8120e88243c7b6f486f341a7566bb4ad6f60eb69",
    "c004_ba": "a2a28d2565fb8a3e6c76e1f7d2be5b1a3aca28acbabe13c3db4e2063c404f2e5",
    "MHC_pseudo.dat": "f46d95dee821db6c139d6cee6f6bf72328468c1d8f6e9741daf21562752ad39a"
  },
  "source_counts": {
    "ba_rows_all_species": 208093,
    "hla_abc_rows": 170107,
    "hla_abc_positive": 42001,
    "hla_abc_artificial_negative": 10895,
    "nine_mer_hla_rows": 126375,
    "nine_mer_hla_positive": 30869,
    "nine_mer_hla_negative": 95506
  },
  "headline_counts": {
    "alleles": 47,
    "rows": 112128,
    "positive": 28538,
    "negative": 83590,
    "measured_nonbinder": 82448,
    "artificial_negative": 1142
  }
}
```

- [ ] Make `scripts/fetch_pmhc_data.sh` refuse destructive behavior:

  - Reuse an existing archive only after its SHA-256 passes.
  - Download to a new `.part` path, verify it, then rename it only when the final archive is absent.
  - Reuse an existing extraction only after all six file hashes pass.
  - Never delete, truncate, or overwrite an existing path.

- [ ] Add the data path to `src/cognate/pmhc.py`:

```python
@dataclass(frozen=True)
class PmhcDataset:
    all_nine_mer_hla_rows: pd.DataFrame
    rows: pd.DataFrame
    eligible_alleles: tuple[str, ...]

def load_source_contract(path: Path) -> dict[str, object]: ...
def verify_source(source_dir: Path, archive_path: Path, contract: Mapping[str, object]) -> None: ...
def load_pmhc_dataset(source_dir: Path, *, minimum_class_rows: int = 100) -> PmhcDataset: ...
def iter_pmhc_folds(dataset: PmhcDataset) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]: ...
def load_pseudo_sequences(path: Path, alleles: Sequence[str]) -> dict[str, str]: ...
```

The loader must:

- require exactly `c000_ba` through `c004_ba`;
- parse three whitespace-separated columns as `Peptide`, `Affinity`, and `Allele`;
- require targets in `[0,1]` and the standard 20-amino-acid alphabet;
- retain only HLA-A/B/C nine-mers;
- set binary `Target = Affinity > 0.426`;
- label positives, measured non-binders, and exact-`0.01` artificial negatives;
- choose alleles with at least 100 positives and 100 negatives across all folds;
- require both classes for every retained allele in each test fold and its training complement;
- reject exact peptide overlap across folds;
- normalize `HLA-A02:01` to pseudo key `HLA-A0201` and require 34 residues.

- [ ] In `tests/test_pmhc.py`, write one parametrized `test_contract_corruption_is_rejected` with named cases for archive hash, extracted-file hash, fold set, malformed row, invalid affinity, invalid residue, fold overlap, minimum class support, missing class in a test fold, missing class in its training complement, source counts, headline counts, missing pseudo-sequence, and short pseudo-sequence. Run each case against a callable non-rejecting implementation first, observe its own red failure, then add the guard and rerun green.

- [ ] Run and review:

```bash
uv run pytest tests/test_pmhc.py -q
```

The corrupt-fixture cases observed red then green are `failure-proven`; the valid parse is `observed`. Run the Phase 1 `reviewer-py` gate and resolve blockers/majors.

---

## Phase 2: Implement only the scoring logic the experiment uses

**Files:** `src/cognate/pmhc.py`, `tests/test_pmhc.py`

- [ ] Before implementing each scoring rule, add its named test, run that selector against a wrong-but-callable implementation, observe the numerical or spy assertion fail, implement the rule, and rerun green. Required selectors: `random_is_seeded_uniform`, `composition_columns_are_fixed`, `retrieval_is_per_allele`, `retrieval_operator_is_shared`, `pwm_matches_hand_calculation`, `blosum50_values_are_raw`, `validation_membership_is_shared`, `scaler_uses_fit_partition_only`, `mlp_uses_adam_mse`, `mlp_restores_best_weights`, and `ensemble_uses_all_ten_models`.

- [ ] Generate random scores once, outside the fold loop, in final dataset row order:

```python
random_scores = np.random.default_rng(0).random(len(dataset.rows))
```

- [ ] Add fixed composition features: peptide length plus counts in `ACDEFGHIKLMNPQRSTVWY`. Reuse `train_head.fit_logistic` and `logistic_scores`; fit on binary `Target` within each training fold.

- [ ] Reuse the existing nearest-positive operator directly:

```python
edit = score_by_nearest_positive(
    test,
    train,
    peptide_column="Allele",
    sequence_column="Peptide",
)
esm = score_by_nearest_positive(
    test,
    train,
    peptide_column="Allele",
    sequence_column="Peptide",
    similarity_fn=cosine_similarity(cache, 10),
)
```

Do not add a retrieval wrapper class. Retain the edit result’s nearest sequence and database size for diagnostics.

- [ ] Add one PWM function. For every allele, position, and residue:

```python
positive_probability = (positive_count + 1) / (positive_rows + 20)
negative_probability = (negative_count + 1) / (negative_rows + 20)
log_odds = np.log(positive_probability / negative_probability)
```

The peptide score is the sum of its nine cells. Fit only on the current training folds.

- [ ] Reuse `train_head.MlpHead` with `dropout=0.0`; do not create another network class. Add only:

```python
def encode_blosum50(sequences: Sequence[str]) -> np.ndarray: ...
def build_mlp_features(
    rows: pd.DataFrame,
    pseudo_sequences: Mapping[str, str],
    allele_order: Sequence[str],
    *,
    use_one_hot_allele: bool,
) -> np.ndarray: ...
def split_fit_validation(rows: pd.DataFrame, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]: ...
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
) -> np.ndarray: ...
```

MLP contract:

- raw BLOSUM50 vectors in the fixed 20-residue order;
- 860 pseudo-sequence features or 227 one-hot features;
- `split_fit_validation(..., seed=0)` creates one fixed 10% allele-plus-label-stratified split per outer fold; the runner passes those same indices to both encodings and all seeds;
- scaler fit on the 90% training partition only;
- `MlpHead(..., hidden=55|66, dropout=0.0)`;
- `torch.sigmoid(model(features))`, MSE on continuous `Affinity`, Adam `1e-3`, `train_head.DEFAULT_BATCH_SIZE`, maximum 200 epochs, patience 20;
- use `embed.pick_device()` once for the run and seed NumPy/PyTorch plus batch order per network;
- monitor validation MSE, clone weights at every improvement, stop after 20 epochs without improvement, and restore the best weights before prediction;
- average ten predictions per encoding per outer fold.

- [ ] The focused tests must cover:

  - 21 composition columns in fixed order;
  - retrieval isolation by allele and identical edit/cosine operator arguments;
  - one hand-computed PWM cell and nine-cell sum;
  - BLOSUM50 feature values and widths;
  - fixed validation membership and train-only scaling;
  - a spy proving hidden sizes 55/66 × seeds 0–4 are all used once.

Use mocks for expensive model fitting; do not download ESM or train 100 networks in unit tests.

- [ ] Run and review:

```bash
uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q
```

Run the Phase 2 `reviewer-py` gate and resolve blockers/majors.

---

## Phase 3: Run, evaluate, document, and stop

**Files:** `scripts/run_pmhc.py`, `tests/test_pmhc.py`, `data/pmhc/results.json`, `docs/pmhc/experiment.md`, `docs/contrast.md`, `docs/pmhc/postmortem.md`

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

## Stop Condition

The project is complete when all seven score rows, the negative-source analysis, the TCR/pMHC retrieval contrast, and the postmortem exist and pass review. Weak results trigger contract diagnosis, not a new architecture, second dataset, cross-attention arm, fine-tuning, or presentation endpoint.
