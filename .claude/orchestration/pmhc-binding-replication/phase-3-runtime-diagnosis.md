# Phase 3 pMHC runtime diagnosis

## Conclusion

- **Observed:** PID 47749 is computing normally, not deadlocked or blocked. At the final snapshot it was `R+`, using 98.4% CPU, and had accumulated 183:16 CPU time over 5:08:42 wall time. Its command is `.venv` Python 3.11.13 launched by `uv run python scripts/run_pmhc.py`.
- **Observed:** Two macOS samples found the main thread repeatedly in NumPy sorting and cumulative-array code (`array_argsort`, `_new_argsortlike`, `atimsort_double`, `array_cumsum`). This matches scikit-learn ROC/Average Precision calculation reached from `evaluate()`.
- **Observed:** `data/pmhc/results.json` does not exist. The runner writes it only after every evaluation and comparison finishes, so there is no on-disk evaluation checkpoint.
- **Claimed from code, supported by observed benchmark:** The exact root cause is a serial, real-scale use of the general three-metric bootstrap API: 21 calls to `evaluate(..., mode="both", n_boot=20_000)` plus seven calls to `compare_macro_auc01(..., mode="both", n_boot=20_000)`. This is 560,000 bootstrap replicates and about 33.74 million scikit-learn metric invocations.
- **Observed:** A 30-replicate benchmark on the actual 112,128-row/47-allele cohort projects about 13.45 single-core hours for the evaluation phase alone under current machine conditions. The >5-hour wall time is therefore expected for this call pattern; it is not evidence of a hang. The call pattern is computationally unsuitable for a serial real run even though it produces the declared statistics.

## Exact call path and workload

`scripts/run_pmhc.py` has seven arms. `_evaluate_arm()` always calls `evaluate()` with `mode="both"` and `N_BOOTSTRAP = 20_000` (`scripts/run_pmhc.py:299-319`). `evaluate_predictions()` performs, in order:

1. seven full-cohort arm evaluations (`scripts/run_pmhc.py:341-351`);
2. six arm-minus-random paired comparisons plus MLP-minus-PWM (`scripts/run_pmhc.py:367-394`);
3. seven evaluations for positives plus measured non-binders;
4. seven evaluations for positives plus artificial negatives (`scripts/run_pmhc.py:396-426`).

All three real slices have 47 present and scorable allele groups:

| Slice | Rows | Positive | Negative | `evaluate()` calls |
|---|---:|---:|---:|---:|
| headline | 112,128 | 28,538 | 83,590 | 7 |
| measured non-binder | 110,986 | 28,538 | 82,448 | 7 |
| artificial negative | 29,680 | 28,538 | 1,142 | 7 |

For every `evaluate()` replicate, `_replicate_group_indices()` draws 47 allele groups and resamples all rows within each drawn group. `evaluate()` then:

- calls partial ROC AUC once per drawn allele, normally 47 calls (`src/cognate/metrics.py:439-446`);
- concatenates the resampled groups;
- calls pooled AUROC and pooled AUPRC over the entire active replicate (`src/cognate/metrics.py:447-450`).

Each paired comparison draws the same 47 groups, resamples their rows, and calls partial ROC AUC for both arms: normally 94 calls per replicate (`src/cognate/metrics.py:545-569`).

The resulting approximate ledger is:

- 420,000 `evaluate()` replicates and 140,000 comparison replicates;
- 19,740,000 per-allele partial-AUC calls in `evaluate()`;
- 13,160,000 per-allele partial-AUC calls in comparisons;
- 420,000 pooled AUROC plus 420,000 pooled AUPRC calls;
- about 70.8 billion row appearances supplied to the two pooled sorting metrics, before accounting for their `n log n` sorting cost;
- about 66.8 billion row appearances supplied to per-allele partial-AUC sorts.

There is no accidental nested 20,000-loop inside an individual scorer. The multiplication comes from the intended 20,000-replicate loop being invoked 28 times serially, with dozens of sorting scorers inside each replicate.

## Reproduction and timing evidence

- **Observed:** Environment: Python 3.11.13, NumPy 2.4.6, scikit-learn 1.9.0, using the repository `.venv`.
- **Observed:** The actual cohort loaded as 112,128 rows, 28,538 positives, 47 alleles; the two negative-source slices loaded as 110,986 and 29,680 rows, both with 47 scorable alleles.
- **Observed:** A 20-replicate component timing for a headline replicate averaged 0.0873 s/replicate: 60.0% in the 47 group partial-AUC calls, 22.2% in pooled AUROC, 16.6% in pooled AUPRC, and about 1.2% in resampling/concatenation/class checks. Thus pooled sorting is 38.8% in aggregate and each pooled call is the single longest sort; the group loop is the largest aggregate cost.
- **Observed:** A 30-replicate benchmark measured:

| Call | Seconds/replicate | Projected time per 20,000-draw call |
|---|---:|---:|
| headline `evaluate` | 0.0929 | 31.0 min |
| measured-negative `evaluate` | 0.0950 | 31.7 min |
| artificial-negative `evaluate` | 0.0496 | 16.5 min |
| full-cohort paired comparison | 0.1084 | 36.1 min |

Seven of each project to approximately 13.45 hours for evaluation. Shorter three- and ten-replicate measurements projected 15-19 hours because fixed point-estimate/setup work was divided across too few replicates; 13-15 hours is the better current estimate.

- **Observed:** `tests/test_pmhc.py::test_headline_uses_alleles_and_20000_draws` passes. It replaces `evaluate`, `macro_by_group`, and `compare_macro_auc01` with fakes. It verifies 20,000 draws only on the first seven headline calls, so it exercises the call contract but provides no performance coverage. Existing metric tests use only 20-400 bootstrap draws on 10-320 rows.

## Current stage and likely remaining work

- **Observed:** The process is inside an `evaluate()` call because current samples show the pooled ROC/AP native hot path. There is no progress output inside `evaluate_predictions()`, and Python frame locals are not exposed, so the exact arm/replicate cannot be identified non-invasively.
- **Inference:** It is most likely still in the initial seven headline evaluations. The runner must finish all seven headline calls and all seven comparisons before reaching a negative-source `evaluate()`. The process has only 183 CPU-minutes total, including all source verification and model scoring. Under the observed 31-minute headline and 36-minute comparison costs, it cannot have completed both earlier blocks. At the most optimistic bound it is no later than roughly the sixth headline arm; because scoring consumed CPU, it is likely earlier.
- **Inference:** Roughly 10-14 additional wall-clock hours remain if the process continues at one saturated core. After the current/remaining headline arms, all seven paired comparisons, all fourteen negative-source evaluations, criteria assembly, and the final JSON write remain. The final write itself is negligible.

## Contract impact

- **Claimed:** The statistics being computed match the present runner/output contract: 20,000 two-level draws for macro AUC0.1, pooled AUROC, pooled AUPRC, and each paired comparison. Skipping pooled intervals, changing `mode`, reducing draws, or omitting negative-source evaluations would change the artifact contract.
- **Claimed:** The runtime is expected from the implementation but indicates an unsuitable integration/call pattern: `evaluate()` is a convenient all-metrics API, and the runner calls it 21 times serially at a scale far beyond the tests. This is a performance design defect in the runner path, not a correctness defect proven in frozen `metrics.py`.

## Smallest contract-preserving recovery options

1. **Continue PID 47749.** This is the only zero-change option and the only option that retains the already-computed in-memory row scores. Expected remaining time is about 10-14 hours. It preserves the frozen metrics, 20,000 draws, no prediction persistence, and the exact artifact.
2. **Pause and resume the same PID if machine availability is the issue.** OS-level suspension/resumption preserves in-memory work but does not reduce total compute. Do not terminate it if the current scores must be retained.
3. **For a rerun only, schedule the 21 independent arm/slice evaluations and seven independent comparisons across a small bounded process pool in `scripts/run_pmhc.py`.** Each task can still call the unchanged frozen metric functions with the same seed, mode, and 20,000 draws, so results remain deterministic and contract-equivalent. This reduces wall time but not CPU work and requires recomputing all model scores because none are persisted. It also needs memory/process-start validation on macOS/MPS before use. Under the stated no-architecture-expansion constraint, this is the only plausible runner-local acceleration; batching/vectorizing the bootstrap would be a larger metric implementation change.

Not viable under the constraints: lowering `N_BOOTSTRAP`, using group-only resampling, dropping pooled confidence intervals, persisting predictions for a separate evaluation job, or rewriting/vectorizing the frozen bootstrap implementation. Killing the current run without first accepting full rescoring loses all progress because `results.json` is written only at the end and row predictions exist only in process memory.

## Safety record

No repository files were modified except this requested diagnosis. PID 47749 was sampled read-only and was neither signaled nor terminated.
