# Phase 1 fix round 3 — mutation-testing minors closeout

Method for every finding below: shadow `cognate.pmhc` with a mutated copy via
`PYTHONPATH=<scratch-dir> uv run pytest tests/test_pmhc.py -q` (the scratch dir
contains only `cognate/__init__.py` + a mutated `cognate/pmhc.py`; confirmed this
resolves ahead of the editable install because the venv's `cognate` package is
registered via a plain `.pth` path entry, not an import-hook finder). Baseline
"survives" were re-confirmed against a reconstructed copy of the pre-fix
`tests/test_pmhc.py` (this file was untracked, so there was no git blob to diff
against — the pre-fix content used below is the file as read at task start,
saved to scratch for the baseline runs).

All baseline-survival and post-fix-death runs are `observed` (command run, output
pasted below), giving `failure-proven` status for every finding except MINOR-6's
two named loop-level mutants (see that section).

## MINOR-6 (+ MINOR-5, subsumed)

**Fix**: `src/cognate/pmhc.py`, `_require_matching_counts` — added, before the
per-key loop:

```python
if set(expected_counts) != set(actual_counts):
    missing = sorted(set(actual_counts) - set(expected_counts))
    unexpected = sorted(set(expected_counts) - set(actual_counts))
    raise ValueError(
        f"{field} keys mismatch: missing {missing}, unexpected {unexpected}"
    )
```

**Tests**: extended `test_contract_count_field_mismatch_is_rejected` from a
13-way `(parent, field)` parametrize to a 15-way `(parent, field, mode)`
parametrize — the original 13 as `mode="increment"`, plus one `"delete"`
(`del counts[field]`) and one `"extra"` (`counts["unexpected_count_field"] = 0`)
case, both on `source_counts.ba_rows_all_species`.

**Baseline (pre-fix code, no guard), `-extra` case** — reproduces the exact
defect described in the finding: an extra contract field is silently ignored.

```
PYTHONPATH=.../pre_fix_no_guard uv run pytest "tests/test_pmhc.py::test_contract_count_field_mismatch_is_rejected[source_counts.ba_rows_all_species-extra]" -q
...
E       Failed: DID NOT RAISE ValueError
1 failed in 0.43s
```

**Baseline (pre-fix code, no guard) + the MINOR-5 mutant (`.get(key, actual)`),
`-delete` case** — reproduces the MINOR-5 defect exactly: with no guard, the
plain pre-fix loop already raises on a deleted field (`expected_counts.get(key)`
→ `None`), but the `.get(key, actual)` mutant defeats that safety net silently.

```
PYTHONPATH=.../pre_fix_getdefault uv run pytest "...[source_counts.ba_rows_all_species-delete]" -q
...
E       Failed: DID NOT RAISE ValueError
1 failed in 0.47s
```

**Post-fix, real code** — both new cases pass with the guard's message:

```
uv run pytest tests/test_pmhc.py -q
................................                                         [100%]
32 passed in 0.41s
```

**The two loop-level mutants named in the finding
(`.get(key) → .get(key, actual)` and "iterate expected instead of actual") —
re-run against the *current, fixed* source with only that one line mutated:**

```
PYTHONPATH=.../minor6_getdefault        uv run pytest tests/test_pmhc.py -q  → 32 passed
PYTHONPATH=.../minor6_iterate_expected  uv run pytest tests/test_pmhc.py -q  → 32 passed
```

Both survive as standalone single-line mutations of the *current* code — this
is not a test gap. Once the symmetric key-set guard exists and runs first, both
mutations become mathematically unreachable: for any input where the key sets
match, `.get(key, actual)` and `.get(key)` return the same value (default is
never used when the key is present), and iterating `expected_counts.items()`
vs. `actual_counts.items()` visits the identical set of `(key, expected,
actual)` triples in either order. For any input where the key sets differ, the
guard raises before the loop is ever reached, so the loop's mutation never
executes. I confirmed this isn't a paperwork excuse by running the delete/extra
cases directly against each mutant and inspecting *which code path* raised —
in both cases it's the guard's own (unmutated) line, with the guard's own
message, at `tests/test_pmhc.py:379`. This is `observed`, not `claimed`. Label:
the underlying defects (MINOR-5 and MINOR-6) are `failure-proven` closed
(red on pre-fix code, green after); the two named loop-line mutants are
provably equivalent post-fix, not separately killed by a test — a strictly
stronger closure (they can never resurface under any future test) than
"killed by this specific test."

## MINOR-1

**Fix**: `tests/test_pmhc.py`, `_run_corruption`, `"missing-class-test-fold"`
case — corruption target changed from `source_dir / "c000_ba"` to
`source_dir / FOLD_NAMES[3]` (`c003_ba`), which has no boundary-row
special-casing, so all 25 of that fold's `HLA-A02:01` positive rows genuinely
flip to negative.

**Baseline (original test, unmutated `pmhc.py`)**: 29 passed (sanity).

**Baseline survives truncation mutant** (`for fold_name in FOLD_NAMES:` →
`FOLD_NAMES[:1]` in `_require_both_classes_per_fold`), run against the
*original* (pre-fix) `tests/test_pmhc.py`:

```
PYTHONPATH=.../minor1_truncate_fold uv run pytest <original test file> -q
.............................                                            [100%]
29 passed in 0.38s
```

**Post-fix test, same mutant, dies:**

```
PYTHONPATH=.../minor1_truncate_fold uv run pytest tests/test_pmhc.py -q
...
FAILED tests/test_pmhc.py::test_contract_corruption_is_rejected[missing-class-test-fold]
E       Failed: DID NOT RAISE ValueError
1 failed, 31 passed in 0.53s
```

`failure-proven`.

## MINOR-2

**Fix**: `tests/test_pmhc.py`, `_write_valid_source` — two of fold 0's twelve
`HLA-A02:01` sub-threshold rows (previously all `0.200`) now use `0.005` and
`0.100` at indices 1 and 2 (index 0 stays the existing `0.426` boundary-negative
row). Both peptides are threaded back through the function's return tuple and
asserted `RowType == "measured_nonbinder"` in
`test_valid_source_is_parsed_and_partitioned`. RowType classification is
unchanged for both (still `measured_nonbinder`), so none of the 13 contract
counts move — confirmed independently below.

**Baseline survives both mutants** (`== ARTIFICIAL_NEGATIVE_AFFINITY` →
`<=`, and → `< 0.05`), against the *original* test file: 29 passed each.

**Post-fix test, both mutants die:**

```
PYTHONPATH=.../minor2_le   uv run pytest tests/test_pmhc.py -q
...
FAILED tests/test_pmhc.py::test_valid_source_is_parsed_and_partitioned
E   AssertionError: Regex pattern did not match.
E   Expected regex: 'headline_counts mismatch'
E   Actual message: 'source_counts mismatch for hla_abc_artificial_negative: expected 115, got 116'
8 failed, 24 passed in 0.42s

PYTHONPATH=.../minor2_lt05 uv run pytest tests/test_pmhc.py -q
8 failed, 24 passed in 0.42s   (same failure set — 0.005 < 0.05 is enough to trip it)
```

`failure-proven`.

## MINOR-3

**Fix**: `tests/test_pmhc.py`, `_write_valid_source` — the one BoLA-1:00901 row
per fold changed from affinity `0.500` to `0.010`. BoLA is never HLA-A/B/C, so
this row is excluded from `hla_abc_rows` under correct code either way; none of
the 13 contract counts change (confirmed below). It does make the row
`RowType == "artificial_negative"` when (incorrectly) counted over `raw_rows`.

**Baseline survives** (`hla_abc_rows["RowType"]` → `raw_rows["RowType"]` in the
`hla_abc_artificial_negative` computation), against the original test file: 29
passed.

**Post-fix test, mutant dies:**

```
PYTHONPATH=.../minor3_raw_rows uv run pytest tests/test_pmhc.py -q
...
FAILED tests/test_pmhc.py::test_valid_source_is_parsed_and_partitioned
E   Actual message: 'source_counts mismatch for hla_abc_artificial_negative: expected 115, got 120'
7 failed, 25 passed in 0.41s
```

`failure-proven`.

## MINOR-4

**Fix**: `tests/test_pmhc.py`, `_write_valid_source` — writes a `c000_el` file
into the fixture directory (never parsed by correct code; the loader iterates
the hardcoded `FOLD_NAMES` tuple for parsing, and only uses the glob result to
validate the fold *set*).

**Baseline survives** (`source_dir.glob("c*_ba")` → `glob("c*")` in
`_load_raw_rows`), against the original test file (no `c000_el` present): 29
passed.

**Post-fix test, mutant dies** — the widened glob now also matches `c000_el`,
so `actual_fold_names` gains an extra entry the hardcoded `FOLD_NAMES` set
doesn't have, and `verify_source`/`load_pmhc_dataset` should raise "fold set
mismatch" but instead every downstream count computation runs off a fold set
that no longer matches the contract's assumptions, cascading into count
mismatches everywhere:

```
PYTHONPATH=.../minor4_widen_glob uv run pytest tests/test_pmhc.py -q
...
24 failed, 8 passed in 0.43s
```

`failure-proven`.

## MINOR-7

**Fix**: `tests/test_pmhc.py` — added `"long-pseudo-sequence"` corruption case
using `PSEUDO_SEQUENCE + "A"` (35 residues), alongside the existing
`"short-pseudo-sequence"` case, both asserting `"expected 34 residues"`.

**Baseline survives** (`len(sequence) != PSEUDO_SEQUENCE_LENGTH` → `<`), against
the original test file (no over-length case existed): 29 passed.

**Post-fix test, mutant dies:**

```
PYTHONPATH=.../minor7_lt uv run pytest tests/test_pmhc.py -q
...
FAILED tests/test_pmhc.py::test_contract_corruption_is_rejected[long-pseudo-sequence]
E       Failed: DID NOT RAISE ValueError
1 failed, 31 passed in 0.43s
```

`failure-proven`.

## Fixture-count bookkeeping (MINOR-2/3/4)

Independent recount (plain Python, no pandas, no `cognate.pmhc` import — reads
the fixture files `_write_valid_source` writes and recomputes all 13 numbers
from the raw rows) against the fixture *after* all three fixture edits:

```
source_counts:
  ba_rows_all_species 536
  hla_abc_rows 531
  hla_abc_positive 281
  hla_abc_artificial_negative 115
  nine_mer_hla_rows 526
  nine_mer_hla_positive 276
  nine_mer_hla_negative 250
headline_counts:
  alleles 1 ['HLA-A02:01']
  rows 225
  positive 125
  negative 100
  measured_nonbinder 60
  artificial_negative 40
```

Identical to the pre-existing `_valid_contract` values — the MINOR-2 rows
(0.005, 0.100) replace 0.200 rows with the same `RowType`, the MINOR-3 BoLA row
is excluded from every HLA-scoped count either way, and the MINOR-4 `c000_el`
file is never parsed by correct code. `_valid_contract` was left unchanged.
Also confirms `HLA-A02:01` stays 125 positive / 100 negative (60 measured / 40
artificial), the boundary rows, the shared-peptide row, and the
per-allele/per-fold asymmetries all remain exactly as they were.

## INFO-1

**Fix**: `scripts/fetch_pmhc_data.sh` — mirrored the extraction rename's
nested-path guard onto the archive rename:

```bash
mv -n "$archive_part" "$ARCHIVE_PATH"
nested_archive_path="$ARCHIVE_PATH/$(basename "$archive_part")"
if [[ -e "$archive_part" ]]; then
  echo "Archive appeared during download; leaving partial archive at $archive_part" >&2
  exit 1
elif [[ -e "$nested_archive_path" ]]; then
  echo "Archive directory appeared during download; mv nested the verified archive inside it at $nested_archive_path instead of $ARCHIVE_PATH" >&2
  exit 1
fi
```

Verified with `bash -n scripts/fetch_pmhc_data.sh` (syntax only — no
`shellcheck` installed in this environment, and this path isn't exercised by
`data/pmhc/raw/`, which is off-limits to populate for this task).

## Full validation gate

```
uv run pytest tests/test_pmhc.py -q
................................                                         [100%]
32 passed in 0.41s

uv run pytest -q
........................................................................ [ 32%]
........................................................................ [ 64%]
........................................................................ [ 96%]
.........                                                                [100%]
225 passed, 1 warning in 116.33s
```

225 = 222 (prior baseline) + 3 net new `test_pmhc.py` cases (29 → 32: +1
`long-pseudo-sequence` corruption case, +2 `delete`/`extra` count-field cases).
No regressions. The one warning (`test_findings.py::test_knn_headline_scores`,
degenerate-scores) is pre-existing and unrelated to this change.
