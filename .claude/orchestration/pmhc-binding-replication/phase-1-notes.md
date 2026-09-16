# Phase 1 implementer notes — closing the seven gaps

Starting point: state-delta doc reported 8 passed (`test_valid_source_is_parsed_and_partitioned`
+ 7 of 14 named corruption branches wired into `CORRUPTION_CASES`). Ending point: 15 passed
(1 valid-parse + all 14 corruption branches).

## Correction to the state-delta doc

Gap #1 ("`scripts/fetch_pmhc_data.sh` — missing shebang") did not reproduce. `od -c` on the
file's first bytes showed `#!/usr/bin/env bash\n` already present as line 1 (`ls -la` mode
`755`, `wc -l` 80 lines). A first `cat -n ... | head -20` in this session had shown a shebang-less
first line, which was an artifact of that pipeline, not the file's real content — `Read` and `od
-c` on the same path both confirmed the shebang is there. No edit made to this file; nothing else
in the gap list mentioned this file.

## Prep changes (non-guard)

- `tests/test_pmhc.py`: `PSEUDO_SEQUENCE` extended from 30 to 34 residues
  (`"ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"`, verified 34 chars, all in `STANDARD_AMINO_ACIDS`).
  Confirmed the existing 8 tests still passed after this change alone, before touching any guard.
- Recomputed the synthetic fixture's `source_counts`/`headline_counts` by hand against
  `_write_valid_source`'s row layout (20 positive + 10 measured-negative + 10 artificial-negative
  nine-mers per fold, plus one off-length HLA-A 10-mer and one non-HLA-ABC 9-mer per fold, over 5
  folds). All values in `_valid_contract` already matched the actual synthetic data
  (`ba_rows_all_species=210`, `hla_abc_rows=205`, `hla_abc_positive=105`,
  `hla_abc_artificial_negative=50`, `nine_mer_hla_rows=200`, `nine_mer_hla_positive=100`,
  `nine_mer_hla_negative=100`; headline `alleles=1`, `rows=200`, `positive=100`, `negative=100`,
  `measured_nonbinder=50`, `artificial_negative=50`) — no numeric edits needed there.

## Design note followed

Per the state-delta's suggested shape: factored the fold-set check + concat + `Target`/`RowType`
assignment out of `load_pmhc_dataset` into a new private `_load_raw_rows(source_dir)` returning
the all-species frame with those two columns already attached. `load_pmhc_dataset` now calls it
and filters to HLA-A/B/C nine-mers as before. `verify_source` calls `_load_raw_rows` directly for
`source_counts` (computed at the HLA-ABC level and the nine-mer-HLA level, pre-eligibility-filter)
and calls `load_pmhc_dataset(source_dir)` a second time for `headline_counts` (post-eligibility).
No new public function, dataclass field, or config object added — `_load_raw_rows`,
`_require_matching_counts`, and `_require_both_classes_per_fold` are all private (`_`-prefixed)
helpers. Folds get parsed twice per full `verify_source` call, which the design note accepted.

## Per-guard red/green evidence

Procedure per guard: add its `CORRUPTION_CASES` entry with a specific match string, run that one
selector (observe red), add the minimal guard, rerun the same selector (observe green), then run
the full suite to confirm no regression.

### 1. `minimum-class-support`
- Guard: in `load_pmhc_dataset`, after computing `eligible_alleles`, raise if it is empty.
- Red: `uv run pytest tests/test_pmhc.py -q -k minimum-class-support` → `Failed: DID NOT RAISE
  ValueError` (1 failed).
- Green (after adding the guard): `1 passed, 8 deselected`.

### 2. `missing-class-test-fold`
- Guard: new private `_require_both_classes_per_fold`, called from `load_pmhc_dataset` after the
  eligibility filter. First implementation checked only that each retained allele has both
  `Target` values present within each fold's own test rows.
- Red: `-k missing-class-test-fold` → `Failed: DID NOT RAISE ValueError` (1 failed).
- Green: `1 passed, 9 deselected`.

### 3. `missing-class-training-complement`
- Guard: extended `_require_both_classes_per_fold` with a second check — both `Target` values
  present in the fold's 4-fold training complement (`rows.loc[~is_test]`).
- Red (before this check existed): `-k missing-class-training-complement` → a `ValueError` *was*
  raised, but for the wrong reason: `AssertionError: Regex pattern did not match. Expected regex:
  'missing class in training complement' / Actual message: 'missing class in test fold c001_ba
  for allele HLA-A02:01'`. This corruption's fixture edits folds `c001_ba`–`c004_ba` (leaving
  `c000_ba` untouched), so with only the test-fold check in place, the run happened to fail first
  on `c001_ba`'s own test-fold check rather than on `c000_ba`'s training-complement check — a
  legitimate red because the guard under test had not yet fired for the right reason.
- Green (after adding the complement check): `1 passed, 10 deselected`. Trace: for fold
  `c000_ba` (checked first, in `FOLD_NAMES` order), its own test rows are untouched (both classes
  present, passes), but its training complement (`c001_ba`–`c004_ba`, all edited to remove every
  positive row) has no positive rows left, so the complement check raises the correct message
  before the loop ever reaches `c001_ba`.

### 4. `source-counts`
- Guard: new private `_require_matching_counts(field, contract, actual_counts)` plus, in
  `verify_source`, a call comparing `contract["source_counts"]` against counts computed from
  `_load_raw_rows` (`ba_rows_all_species`, `hla_abc_rows`, `hla_abc_positive`,
  `hla_abc_artificial_negative`, `nine_mer_hla_rows`, `nine_mer_hla_positive`,
  `nine_mer_hla_negative`).
- Red: `-k "source-counts and not headline"` → `Failed: DID NOT RAISE ValueError` (1 failed).
- Green: `1 passed, 11 deselected`; full suite `12 passed`.

### 5. `headline-counts`
- Guard: in `verify_source`, after the `source_counts` check, call `load_pmhc_dataset(source_dir)`
  and compare `contract["headline_counts"]` against `alleles`, `rows`, `positive`, `negative`,
  `measured_nonbinder`, `artificial_negative` computed from the returned `PmhcDataset.rows`.
- Red: `-k headline-counts` → `Failed: DID NOT RAISE ValueError` (1 failed).
- Green: `1 passed, 12 deselected`; full suite `13 passed`.

### 6. `missing-pseudo-sequence`
- Guard: rewrote `load_pseudo_sequences` from a dict-comprehension that silently dropped
  unmatched alleles to an explicit loop that raises `ValueError(f"missing pseudo-sequence for
  allele {allele}")` when an allele's pseudo key is absent.
- Red: `-k missing-pseudo-sequence` → `Failed: DID NOT RAISE ValueError` (1 failed).
- Green: `1 passed, 13 deselected`; full suite `14 passed`.

### 7. `short-pseudo-sequence`
- Guard: added module constant `PSEUDO_SEQUENCE_LENGTH = 34` and a length check in the same loop,
  raising `ValueError` when a resolved sequence's length isn't 34.
- Red: `-k short-pseudo-sequence` → `Failed: DID NOT RAISE ValueError` (1 failed).
- Green: `1 passed, 14 deselected`; full suite `15 passed`.

## Final validation

```
uv run pytest tests/test_pmhc.py -v
```
15 passed (1 valid-parse + all 14 named corruption branches: archive-hash, extracted-file-hash,
fold-set, malformed-row, invalid-affinity, invalid-residue, fold-overlap, minimum-class-support,
missing-class-test-fold, missing-class-training-complement, source-counts, headline-counts,
missing-pseudo-sequence, short-pseudo-sequence).

## Judgment calls

- Match strings for the new cases are exact-phrase substrings unique to their own guard (e.g.
  `"missing class in test fold"` vs `"missing class in training complement"` — neither is a
  substring of the other, so `pytest.raises(match=...)` can't cross-match).
- Kept `_require_matching_counts` generic over the field name (`"source_counts"` /
  `"headline_counts"`) rather than writing two near-identical checks — this is the smallest
  reasonable shape, not a new abstraction layer for a single caller since it has two call sites
  with materially identical logic.
- No linter is configured for this project (`pyproject.toml` has no `[tool.ruff]`/pylint section,
  neither `ruff` nor `pylint` is an installed dependency, and there's no `scripts/local_ci.sh`),
  so no lint gate was run. Only `pytest` was used for verification.

## Not touched

`.gitignore`, `docs/pmhc/brief.md`, `docs/contrast.md`, `data/pmhc/source_contract.json` — all
already correct per the state-delta and left alone. `scripts/fetch_pmhc_data.sh` — inspected only,
no edit (see correction above). Worktree left dirty; nothing staged, committed, or pushed.
