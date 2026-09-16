# Phase 2: Implement only the scoring logic the experiment uses

## Target files

- `src/cognate/pmhc.py`
- `tests/test_pmhc.py`

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

Do not add a retrieval wrapper class. Retain the edit result's nearest sequence and database size for diagnostics.

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

## Validation

```bash
uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q
```
