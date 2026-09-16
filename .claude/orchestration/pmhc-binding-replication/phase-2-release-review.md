# Phase 2 release review

## 1. Summary

Verdict: **PASS**. The Phase 2 scoring implementation matches the reviewed plan, the required named selectors are selectable, and the focused plus adjacent regression checks pass. No blocker, major, or minor issue survived refutation.

**6 candidates, 6 refuted, 0 survived (0 blocker, 0 major, 0 minor)**

The change adds the fixed pMHC scoring rules to `src/cognate/pmhc.py`: seeded uniform scores, fixed peptide composition features, allele-isolated nearest-positive retrieval, allele-specific PWM scoring, raw BLOSUM50 feature construction, allele-plus-label-stratified fit/validation membership, train-only scaling, and the ten-model MLP ensemble. `tests/test_pmhc.py` adds focused numerical and spy tests for those decisions. This is the intended Phase 2 scope; no product files were edited during this review.

## 2. Detailed feedback by category

### Issues found

No surviving issues.

### Correctness

- **Observed:** the fixed random sequence and the 21 composition columns pass their focused assertions (`src/cognate/pmhc.py:46`, `src/cognate/pmhc.py:51`; `tests/test_pmhc.py:36`, `tests/test_pmhc.py:44`).
- **Observed:** the composition arm fits `fit_logistic` only on training features and binary `Target`, then scores only test features (`src/cognate/pmhc.py:62`; `tests/test_pmhc.py:55`).
- **Observed:** edit and cosine retrieval share `score_by_nearest_positive` with `Allele` as the grouping column and `Peptide` as the sequence column, while preserving the returned `KnnResult` diagnostics (`src/cognate/pmhc.py:73`; `tests/test_pmhc.py:103`, `tests/test_pmhc.py:121`).
- **Observed:** the PWM test verifies both a hand-computed cell and the nine-position sum under the required add-one smoothing (`src/cognate/pmhc.py:96`; `tests/test_pmhc.py:155`).
- **Observed:** raw BLOSUM50 values and both required widths, 860 and 227, pass (`src/cognate/pmhc.py:140`, `src/cognate/pmhc.py:151`, `src/cognate/pmhc.py:164`; `tests/test_pmhc.py:176`).
- **Observed:** validation membership is stable for seed 0 and preserves the allele-plus-label strata (`src/cognate/pmhc.py:188`; `tests/test_pmhc.py:208`). Scikit-learn documents that an integer `random_state` produces reproducible output across calls, which is the mechanism used here: [StratifiedShuffleSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html).
- **Observed:** the scaler sees only the fit partition; training uses sigmoid outputs against continuous affinity with Adam at `1e-3` and MSE; best weights are cloned and restored; all hidden-size/seed pairs `(55, 66) × (0..4)` are invoked exactly once and averaged (`src/cognate/pmhc.py:203`, `src/cognate/pmhc.py:280`; `tests/test_pmhc.py:250`, `tests/test_pmhc.py:287`, `tests/test_pmhc.py:360`, `tests/test_pmhc.py:414`).

### Readability and maintainability

The implementation stays in the one planned module and reuses the existing logistic head, `MlpHead`, and nearest-positive operator. Constants make the experiment-defining hyperparameters visible at `src/cognate/pmhc.py:32-43`. The helper boundaries correspond to independently testable experiment decisions and do not introduce a new model or retrieval class.

The cached BLOSUM lookup at `src/cognate/pmhc.py:140` is appropriate because the loaded matrix-derived mapping is invariant for the process. Python documents `functools.cache` as a lightweight unbounded memoization wrapper; here the cache has only one no-argument entry: [Python `functools.cache` documentation](https://docs.python.org/3/library/functools.html#functools.cache).

### Testing

**Observed checks:**

- `uv run pytest tests/test_pmhc.py --collect-only -q`: 44 tests collected; all eleven required Phase 2 selector strings appear in collected test names. The longer `test_mlp_uses_adam_mse_with_sigmoid_and_affinity_targets` remains selectable by the required `mlp_uses_adam_mse` selector.
- Focused required selectors via `pytest -k`: 11 passed, 33 deselected.
- Required Phase 2 suite: `uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q`: 84 passed.
- `uv run python -m compileall -q src/cognate/pmhc.py tests/test_pmhc.py`: passed.
- `git diff --check -- src/cognate/pmhc.py tests/test_pmhc.py`: passed.

The tests assert decisions owned by this module rather than framework behavior: exact feature values/order, adapter arguments, partition membership, fitting boundaries, optimizer/loss/target semantics, best-state restoration, and ensemble membership.

### Performance and security

No concrete performance or security issue was found in the Phase 2 surface. BLOSUM matrix loading is cached, PWM counting is vectorized by allele/position, and expensive MLP coverage uses probes or bounded epochs rather than running the full 100-fit experiment.

### Dependency management

No dependency change was made. Verification used the repository's existing `uv` workflow and `pyproject.toml` dependency declarations.

### Candidate refutations

1. **Claimed candidate:** `score_pwm_fold` could fail for a test allele absent from training. **Refuted:** the accepted dataset requires both classes for every retained allele in every test fold and each training complement (`src/cognate/pmhc.py:480-530`), so the candidate is unreachable through the experiment data path.
2. **Claimed candidate:** `encode_blosum50([])` returns width zero, which would not match a trained scaler. **Refuted:** every retained allele must occur with both classes in every fold, making each experiment train/test frame non-empty; empty-frame behavior is outside the Phase 2 data contract (`src/cognate/pmhc.py:480-548`).
3. **Claimed candidate:** the retrieval helper might alter the shared operator. **Refuted, observed:** it forwards the same frames and fixed columns for both arms and only adds `similarity_fn` for cosine; the spy assertion passed (`src/cognate/pmhc.py:73-93`; `tests/test_pmhc.py:121-152`).
4. **Claimed candidate:** `score_mlp_fold` has more positional parameters than the general Python style preference. **Refuted:** its signature is the exact Phase 2 API contract, and the two experiment switches remain keyword-only (`src/cognate/pmhc.py:280-290`). Changing it would diverge from the reviewed plan without a correctness gain.
5. **Claimed candidate:** the longer MLP test name makes the required `mlp_uses_adam_mse` selector unavailable. **Refuted, observed:** collection includes the test and the combined required-selector run selected and passed all eleven cases.
6. **Claimed candidate:** explicit seeding is insufficient to promise bitwise equality across devices and releases. **Refuted as a Phase 2 defect:** the implementation fulfills the specified NumPy, PyTorch, and batch-order seeding contract at `src/cognate/pmhc.py:215-235`; the plan does not claim cross-platform bitwise identity. PyTorch itself states that identical seeds do not guarantee reproducibility across releases or CPU/GPU platforms: [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## 3. Recommendations

Proceed to Phase 3. No Phase 2 code or test change is recommended from this review.

## 4. Review metadata

- Reviewed scope: uncommitted working-tree versions of `src/cognate/pmhc.py` and `tests/test_pmhc.py`, against Phase 2 of `docs/superpowers/plans/2026-09-10-pmhc-binding-replication.md`.
- Branch: [`codex/pmhc-binding-replication`](https://github.com/freddysongg/cognate/tree/codex/pmhc-binding-replication).
- Base commit: [`71aa9fe5956f0a5607f052ea39f2018d221844e3`](https://github.com/freddysongg/cognate/commit/71aa9fe5956f0a5607f052ea39f2018d221844e3). The reviewed target files are uncommitted, so this commit link identifies the base, not an immutable snapshot of the reviewed changes.
- Evidence scale: source-only statements are `claimed`; commands whose output was observed during this review are `observed`. No red-then-green run was performed here, so this review does not independently elevate prior TDD claims to `failure-proven`.

Sources:

- [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html) (May 2026)
- [Scikit-learn `StratifiedShuffleSplit` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)
- [Python `functools` documentation](https://docs.python.org/3/library/functools.html)

Web-search result file: `/tmp/pmhc-python-review.json`
