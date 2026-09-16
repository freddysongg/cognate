# Phase 3 implementer evidence

## 2026-09-10 23:22 PDT recovery checkpoint

- Assigned Phase 3 files were inspected from their current worktree state. `scripts/run_pmhc.py` and the Phase 3 tests already existed; `data/pmhc/results.json`, `docs/pmhc/experiment.md`, and `docs/pmhc/postmortem.md` did not yet exist.
- Focused fixed-contract selectors ran with `uv run pytest -q tests/test_pmhc.py -k 'each_row_scored_once or invalid_scores_stop_evaluation or cache_rejects_wrong_model_layer_or_peptide_set or headline_uses_alleles_and_20000_draws or predictive_requires_lower_bound_above_half or pwm_sufficiency_requires_both_conditions or comparisons_are_paired_against_random or negative_sources_are_separate or tcr_uplift_uses_observed_random or tiny_five_fold_integration'`: 14 passed, 44 deselected, exit 0.
- Full focused file ran with `uv run pytest -q tests/test_pmhc.py`: 58 passed, exit 0.
- `bash scripts/fetch_pmhc_data.sh` verified the existing public pMHC source archive and extraction, exit 0.
- A new invocation of `uv run python scripts/run_pmhc.py` printed the fixed contract before scoring: five BA folds, 47 alleles, 112,128 rows, 28,538 positives, 83,590 negatives, and zero cross-fold peptide overlap. It then accepted and reused `shared/pmhc/esm2_35M_peptides.npz` with exactly 22,055 peptides after the runner's model/layer/peptide-set validation.
- Process inspection then revealed the earlier real runner was still alive as PID 47749 on `ttys006`, launched at 18:45 PDT. The accidental duplicate was interrupted with Ctrl-C after it reached `c001_ba`; only the newly started duplicate was stopped.
- At this checkpoint the original PID 47749 remained running at approximately 95-99% CPU with a 3.4 GB physical footprint and no `data/pmhc/results.json` written yet. A one-second process sample showed active NumPy sorting/cumulative metric work, consistent with bootstrap evaluation rather than an idle or wedged process.

