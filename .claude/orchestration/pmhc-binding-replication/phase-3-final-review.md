# Phase 3 final review

## 1. Summary

Verdict: PASS. No actionable Phase 3 defects survived refutation.

10 candidates, 10 refuted, 0 survived (0 blocker, 0 major, 0 minor)

The fixed runner, saved aggregate artifact, experiment report, cross-task contrast, and postmortem agree with the Phase 3 contract. The targeted Phase 3 suite passed, the full suite passed, bytecode compilation succeeded, `git diff --check` was clean, and the raw/derived/cache locations are ignored and untracked.

The real 100-fit experiment was not rerun during this review. The saved `results.json` was instead checked directly for schema, counts, finiteness, bootstrap settings, criteria, comparisons, per-allele coverage, and agreement with the prose.

## 2. Detailed feedback by category

### Issues found

None.

### Correctness and contract refutations

1. **Result schema drift — refuted (observed).** `data/pmhc/results.json` has exactly the eight required top-level keys, seven required score arms, seven comparisons, two negative-source slices, two retrieval-diagnostic slices, and 47 per-allele records. The artifact contains 64,994 bytes of aggregate material and no row-level prediction field.

2. **Cohort or per-allele count mismatch — refuted (observed).** The artifact reports 112,128 rows, 28,538 positives, 83,590 negatives, 82,448 measured non-binders, and 1,142 artificial negatives. Summing the 47 per-allele entries independently reproduced 112,128 rows and 28,538 positives. Every headline arm reports 47 scored alleles and zero skipped alleles.

3. **Frozen bootstrap contract drift — refuted (observed).** The artifact config records mode `both`, 20,000 draws, seed 0, grouped by allele. Every headline interval, negative-source interval, and paired comparison contains 20,000 replicates. The runner passes those same constants to `evaluate` and `compare_macro_auc01` at `scripts/run_pmhc.py:309-318` and `scripts/run_pmhc.py:373-390`; `tests/test_pmhc.py:976-1002` observes the headline arguments.

4. **Incomplete, duplicate, or non-finite prediction acceptance — refuted (observed).** `scripts/run_pmhc.py:90-118` rejects arm-set drift, wrong shapes, assignment counts other than one, and non-finite scores before evaluation. The integration path preallocates all seven arrays and assigns fold positions at `scripts/run_pmhc.py:188-259`. The focused rejection tests and tiny five-fold integration passed.

5. **Embedding-cache boundary too permissive relative to the plan — refuted (observed).** `scripts/run_pmhc.py:121-142` requires the fixed ESM-2 35M checkpoint name, layer 10 availability, the exact deduplicated peptide set, and finite headline-layer values. `tests/test_pmhc.py:959-973` passed for wrong model, unavailable layer, and wrong peptide set. This matches the explicitly frozen cache acceptance boundary; adding a new cache format or content hash would expand Phase 3 scope.

6. **Predictive or PWM-sufficiency rule drift — refuted (observed).** `scripts/run_pmhc.py:439-466` uses the strict lower-bound-above-0.5 rule and requires both predictive PWM and `MLP - PWM` upper bound below 0.02. Boundary tests at exactly 0.5 and 0.02 passed (`tests/test_pmhc.py:1005-1037`). The saved verdicts follow those rules: six non-random arms predictive, random not predictive, and PWM not sufficient because the paired upper bound is 0.1008406816.

7. **Unpaired or wrong-reference comparisons — refuted (observed).** The runner calls the shared paired `compare_macro_auc01` implementation for each non-random arm against the same observed random scores and separately for pseudo-sequence MLP minus PWM (`scripts/run_pmhc.py:367-394`). The saved comparison points equal their corresponding arm-point differences, all use 47 alleles and 20,000 draws, and the focused comparison test passed.

8. **Negative sources mixed or miscounted — refuted (observed).** `scripts/run_pmhc.py:396-426` constructs each slice as all positives plus exactly one negative source. The artifact reports 110,986 rows for 28,538 positives plus 82,448 measured non-binders, and 29,680 rows for 28,538 positives plus 1,142 artificial negatives. Both slices include all seven arms and separate edit-distance diagnostics.

9. **TCR contrast using theoretical 0.5 instead of observed random — refuted (observed).** The source artifacts report TCR edit 0.5654, ESM-2 0.5358, and observed random 0.5009. The arithmetic uplifts in `docs/contrast.md:21-24` and `docs/pmhc/experiment.md:160-164` are therefore +0.0645 and +0.0349. `scripts/run_pmhc.py:470-489` reads the specified artifact keys, and its focused test passed.

10. **Prose/result or ignored-data inconsistency — refuted (observed).** All rounded headline scores, intervals, comparisons, criteria, negative-source counts, negative-source scores, retrieval diagnostics, and cross-task points in `docs/pmhc/experiment.md:95-168`, `docs/contrast.md:17-30`, and `docs/pmhc/postmortem.md:7-58` agree with `data/pmhc/results.json` and the closed TCR artifacts. `git check-ignore -v` confirmed `data/pmhc/raw/`, `data/pmhc/derived/`, and `/shared/`; `git ls-files` returned no tracked files under those locations.

### Readability and maintainability

The Phase 3 code remains a single fixed runner rather than a configurable framework. Constants make the frozen contract visible (`scripts/run_pmhc.py:52-81`), scoring and evaluation are separated, and artifact serialization is explicit. The strict `allow_nan=False` write at `scripts/run_pmhc.py:594-597` complements the earlier finite-value guards and follows Python's standard-library mechanism for rejecting non-standard `NaN`/infinity JSON values.

The helper boundaries are proportionate to the decisions being tested: score validation, cache validation, criteria derivation, and TCR artifact loading are individually testable without introducing a second evaluation engine.

### Testing

Observed verification:

- `uv run pytest tests/test_pmhc.py -q`: 58 passed in 4.14s.
- `uv run pytest -q`: 251 passed, 1 existing warning in 104.95s.
- `uv run python -m compileall -q src scripts`: passed.
- `git diff --check`: passed.
- `git ls-files data/pmhc/raw data/pmhc/derived shared/pmhc`: no output.

The tests cover owned decisions rather than library behavior: all seven fold outputs, single assignment, invalid-score rejection, cache identity, fixed bootstrap arguments, strict criteria boundaries, paired reference selection, negative-source separation, and observed-random TCR uplift. The tiny integration test mocks expensive embedding and MLP work while exercising the runner's fold orchestration and aggregate output structure.

### Performance, security, and dependencies

No new actionable concern. The expensive work is intentionally cached, the cache is accepted only at the frozen boundary, row-level outputs are not serialized, JSON rejects non-finite extensions, and the existing `uv` workflow is used for tests and reproduction.

### What's done well

- The artifact is self-describing enough to audit the frozen settings without row-level predictions.
- Criteria are calculated from saved intervals instead of manually asserted in prose.
- The negative-source analysis keeps cohorts separate and avoids implying a paired edit-versus-ESM test that was not run.
- The postmortem respects the stop condition and clearly limits the result to seen alleles and a dataset-specific descriptive contrast.

## 3. Recommendations

No required changes. Do not expand this phase with another dataset, architecture, cache schema, or inferential contrast.

## 4. Review metadata

- Scope: current uncommitted Phase 3 working-tree artifacts only.
- Branch: `codex/pmhc-binding-replication`.
- Base commit: [71aa9fe5956f0a5607f052ea39f2018d221844e3](https://github.com/freddysongg/cognate/commit/71aa9fe5956f0a5607f052ea39f2018d221844e3).
- Pull request: none identified; the reviewed files are uncommitted working-tree changes.
