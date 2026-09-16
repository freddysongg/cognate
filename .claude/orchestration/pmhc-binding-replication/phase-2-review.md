# Phase 2 Python review

## 1. Summary

The Phase 2 scoring implementation matches the specified experiment logic: fixed random scores, fixed-order composition features, per-allele nearest-positive retrieval, allele-specific PWM log odds, raw BLOSUM50 encoding, a shared stratified validation split, train-only scaling, Adam/MSE affinity training, best-weight restoration, and the 55/66 by 0–4 ten-model ensemble.

Verdict: **pass with minor test follow-ups**. No blocker or major correctness defect survived review.

**8 candidates, 6 refuted, 2 survived (0 blocker, 0 major, 2 minor)**

Verification: **observed** — `uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q` completed with `83 passed in 21.30s` on the reviewed working tree.

## 2. Detailed feedback by category

### Testing

#### MINOR — the composition scorer has no dedicated behavior test (`claimed`)

References: `src/cognate/pmhc.py:62`, `tests/test_pmhc.py:44`

`test_composition_columns_are_fixed` proves the 21-column encoding, but `score_composition_fold` is not imported or called anywhere in `tests/test_pmhc.py`. The Phase 2 contract also decides that the model must be fit on the training fold's binary `Target` and that scoring must use the test fold through the reused `fit_logistic`/`logistic_scores` pair. A regression that fits on `Affinity`, fits on test rows, or returns training predictions would leave the current composition test green.

Suggested improvement: add a focused spy test around `score_composition_fold` that verifies the exact training features and binary labels passed to `fit_logistic`, then verifies that only test features are passed to `logistic_scores`. This tests decisions owned by this adapter rather than scikit-learn internals.

#### MINOR — the MLP optimizer/loss test does not prove several experiment-defining decisions (`claimed`)

References: `src/cognate/pmhc.py:232`, `src/cognate/pmhc.py:246`, `src/cognate/pmhc.py:253`, `tests/test_pmhc.py:238`

`test_mlp_uses_adam_mse` only counts construction of `Adam` and `MSELoss`. It would still pass if the learning rate changed, if MSE were applied to logits instead of `torch.sigmoid(model(...))`, if validation used a different transform, or if the continuous `Affinity` arrays were replaced before loss evaluation. Those are experiment-defining choices in the Phase 2 contract, not library behavior.

Suggested improvement: strengthen this test with lightweight probe objects or spies that assert `lr == 1e-3`, capture the tensors passed to the training and validation losses, and demonstrate that they are sigmoid probabilities paired with the supplied continuous affinities. Keep the existing best-weight and ten-model tests separate; they already document those decisions well.

### Correctness and maintainability

No blocker or major finding survived. The following candidates were explicitly refuted:

- Empty BLOSUM feature width: refuted because the fixed dataset contract and fold guards make both fit and test partitions non-empty; this is not a reachable experiment path.
- Missing PWM model for a test allele: refuted by the retained-allele guard requiring both classes in every test fold and its training complement.
- Retrieval grouping or argument drift: refuted by the per-allele numerical test and the shared-operator spy at `tests/test_pmhc.py:55` and `tests/test_pmhc.py:73`.
- Feature-order or width drift: refuted by the raw BLOSUM values and exact 860/227 widths at `tests/test_pmhc.py:128`.
- Best-state restoration failure: refuted by the controlled optimizer trajectory at `tests/test_pmhc.py:272`.
- Seeded CPU repeatability: **observed** — two consecutive two-epoch calls with identical inputs and seeds returned the same prediction (`[0.3996374]`). This does not claim cross-platform bitwise determinism; PyTorch documents that full reproducibility can require deterministic-algorithm controls beyond seeding: <https://docs.pytorch.org/docs/stable/notes/randomness.html>.

### What's done well

- The implementation reuses `score_by_nearest_positive`, `MlpHead`, `fit_logistic`, and `logistic_scores` without introducing a parallel model hierarchy.
- The scaler is fitted inside each individual network using only `fit_features`, and the existing spy checks all ten invocations.
- Best weights are cloned at improvement time and restored before test prediction; the test distinguishes the best epoch from the final epoch.
- The ensemble spy proves every `(hidden, seed)` pair appears exactly once and verifies the mean, which is the right level of unit testing for this orchestration decision.
- The Phase 2 suite uses `uv`, consistent with the repository's `pyproject.toml` dependency setup.

## 3. Recommendations

Add the two focused tests above before relying on this module as the fixed experiment contract. No implementation change is otherwise recommended from this review.

## 4. Review metadata

Reviewed local working tree on [`codex/pmhc-binding-replication`](https://github.com/freddysongg/cognate/tree/codex/pmhc-binding-replication). The two reviewed files are currently untracked, so there is no commit or pull request that durably identifies this exact snapshot.
