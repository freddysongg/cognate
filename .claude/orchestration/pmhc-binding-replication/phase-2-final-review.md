# Phase 2 final review

## 1. Summary

Verdict: **PASS WITH MINOR**. The current Phase 2 scoring implementation matches the specified random, composition, allele-isolated retrieval, PWM, BLOSUM50, validation-split, train-only-scaling, MLP-training, best-weight-restoration, and ten-model ensemble behaviors. No blocker or major correctness issue survived refutation.

**8 candidates, 7 refuted, 1 survived (0 blocker, 0 major, 1 minor)**

The intended Phase 2 gate is observed green: `uv run pytest tests/test_pmhc.py tests/test_baseline_knn.py tests/test_knn_cosine.py tests/test_embed.py tests/test_features.py -q` completed with 84 passing tests in 18.79 seconds. `uv run python -m compileall -q src/cognate/pmhc.py tests/test_pmhc.py` also completed successfully.

## 2. Detailed feedback by category

### Issues found

#### MINOR — required `mlp_uses_adam_mse` selector selects no test (observed)

- Reference: `tests/test_pmhc.py:287`
- Phase 2 explicitly requires a selector named `mlp_uses_adam_mse`, but the implemented test is named `test_mlp_uses_adam_and_sigmoid_mse_with_affinity_targets`. The extra `and_sigmoid` between `adam` and `mse` means the required substring is absent.
- Observed evidence: `uv run pytest tests/test_pmhc.py -q -k mlp_uses_adam_mse` exited 5 with `44 deselected`; it ran no test.
- Impact: the numerical/spy test itself is sound and the full gate passes, so this is not a scoring-logic defect. It does break the plan's named-selector workflow and prevents the required focused red/green command from being rerun verbatim.
- Suggested improvement: rename the test so `mlp_uses_adam_mse` is a contiguous part of its node name, while keeping its current sigmoid, affinity-target, Adam-rate, and MSE assertions.

### Correctness

- Random-score candidate refuted: `src/cognate/pmhc.py:46-48` implements the plan's exact `default_rng(0).random(n_rows)` rule, and `tests/test_pmhc.py:36-41` fixes the expected sequence. Generating it once in final dataset order is a Phase 3 runner responsibility, not an omission in this Phase 2 helper.
- Composition candidate refuted: `src/cognate/pmhc.py:51-70` fixes length plus counts in `ACDEFGHIKLMNPQRSTVWY`, fits only training `Target`, and scores only test features. `tests/test_pmhc.py:44-100` constrains both the 21-column order and train/test boundary.
- Retrieval candidate refuted: `src/cognate/pmhc.py:73-93` delegates to the existing nearest-positive operator with allele as the grouping key and peptide as the compared sequence, preserving the returned nearest-sequence and database-size diagnostics. `tests/test_pmhc.py:103-152` constrains allele isolation and identical edit/cosine operator arguments.
- PWM candidate refuted: `src/cognate/pmhc.py:96-137` applies the specified add-one denominators separately to positive and negative rows for each allele and sums nine positional cells. `tests/test_pmhc.py:155-173` independently hand-computes one cell and the nine-cell sum. The Phase 1 dataset guards ensure every retained allele has both classes in each training complement, so the apparent empty-class edge is unreachable for the authorized caller.
- BLOSUM/feature-width candidate refuted: `src/cognate/pmhc.py:140-185` uses raw BLOSUM50 rows in the fixed residue order and produces the required 860- and 227-column designs. `tests/test_pmhc.py:176-205` checks raw A/W values, widths, and the first one-hot allele position.
- Validation/scaling candidate refuted: `src/cognate/pmhc.py:188-200` creates the fixed allele-plus-label stratified 90/10 split and `src/cognate/pmhc.py:217-230` fits the scaler only on passed fit features. `tests/test_pmhc.py:208-284` observes stable membership and excludes validation rows from scaler fitting. The use of an integer `random_state` for repeatable splits and a placeholder `X` is consistent with the current [scikit-learn `StratifiedShuffleSplit` contract](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html).
- MLP/ensemble candidate refuted: `src/cognate/pmhc.py:203-319` uses `MlpHead`, dropout 0, sigmoid-MSE on continuous affinity, Adam at 1e-3, the shared batch size, 200 epochs, patience 20, cloned best weights, restoration before prediction, and all 55/66 by seed 0-4 combinations. `tests/test_pmhc.py:287-446` constrains the optimizer/loss target, restoration, combination set, and mean. `torch.manual_seed` seeds all devices according to the current [PyTorch API](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html), while the explicit `torch.Generator` controls batch permutations.

### Readability and maintainability

- No actionable issue survived. The Phase 2 logic is contained in the existing pMHC module, uses the existing retrieval/logistic/MLP primitives, and avoids introducing new model or wrapper classes.
- The constants at `src/cognate/pmhc.py:32-43` keep the experimental choices visible and auditable.
- The private `_score_single_mlp` function isolates the training loop without creating a hypothetical reusable abstraction.

### Testing

- The tests mostly assert decisions owned by this module rather than library internals: fixed feature mappings, adapter arguments, split membership, leakage boundaries, loss targets, best-state restoration, and ensemble membership.
- The expensive 100-network workload is correctly replaced by focused spies and short training probes.
- The sole surviving test issue is the required selector mismatch above.

### Performance, security, and dependencies

- No obvious performance or security defect survived. Repeated scaler fitting across ensemble members is redundant, but it is small relative to the specified 200-epoch network training and does not change behavior, so it was refuted as non-actionable for this phase.
- Dependency usage matches the existing `pyproject.toml`; verification used `uv`, as configured for this Python project.

## 3. Recommendations

1. Rename `test_mlp_uses_adam_and_sigmoid_mse_with_affinity_targets` so the required `mlp_uses_adam_mse` selector matches it.
2. Rerun that exact selector, then rerun the existing 84-test Phase 2 gate.

## 4. Review metadata

- Reviewed target: local uncommitted working tree on branch `codex/pmhc-binding-replication`.
- Files reviewed: `src/cognate/pmhc.py`, `tests/test_pmhc.py`.
- Contract: Phase 2 of `docs/superpowers/plans/2026-09-10-pmhc-binding-replication.md`.
- Stable PR/commit link: unavailable because both reviewed files are currently untracked and no PR or commit was supplied.
- Review scope excluded `.claude/orchestration/**` bookkeeping, prior reviews, ledgers, and git-log messages as required.

### Sources

- [PyTorch `torch.manual_seed`](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [scikit-learn `StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)

Web-search result record: `/tmp/pmhc-python-practices.json`.
