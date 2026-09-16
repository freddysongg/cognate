# Phase 1 fix round 2 — mutation-testing findings

All 8 findings fixed. Method: for each finding, shadowed `cognate.pmhc` with a
mutated copy via `PYTHONPATH=<shadow-dir>` (a directory containing only
`cognate/__init__.py` (empty) and a mutated `cognate/pmhc.py`), confirmed with
a marker-attribute smoke test that the shadow import wins over the installed
package, then ran `PYTHONPATH="$SHADOW" uv run pytest tests/test_pmhc.py -q`
against the *fixed* test suite and confirmed the mutant fails (dies). All
commands and output below are `observed` (re-run for this report), not
recalled from memory.

## MAJOR-1 — HLA-A02:01 desymmetrized

`tests/test_pmhc.py` `_write_valid_source`: changed the retained allele's
per-fold blocks from 20/10/10 (positive/measured/artificial) to 25/12/8.
Totals: positive=125, negative=100 (60 measured + 40 artificial), both ≥100
eligibility floor, positive != negative, measured != artificial. `HLA-B07:02`
stays 100/50, `HLA-C07:01` stays ~50/100 (51/100 after the MINOR-6 shared
peptide, see below) — both still ineligible and asymmetric in opposite
directions. All-allele 9-mer totals: nine_mer_hla_positive=276 !=
nine_mer_hla_negative=250.

All 13 `_valid_contract` numbers were recomputed by an independent counter
(plain `str.split()` over the written fold files, no pandas, no import of
`cognate.pmhc`) rather than by scaling the old numbers — script and full
output below.

```
$ uv run python independent_count.py
eligible alleles: ['HLA-A02:01']
source_counts {'ba_rows_all_species': 536, 'hla_abc_rows': 531, 'hla_abc_positive': 281,
  'hla_abc_artificial_negative': 115, 'nine_mer_hla_rows': 526, 'nine_mer_hla_positive': 276,
  'nine_mer_hla_negative': 250}
headline_counts {'alleles': 1, 'rows': 225, 'positive': 125, 'negative': 100,
  'measured_nonbinder': 60, 'artificial_negative': 40}
per-fold retained row counts: {'c000_ba': 45, 'c001_ba': 45, 'c002_ba': 45, 'c003_ba': 45, 'c004_ba': 45}
```

These numbers are now in `tests/test_pmhc.py::_valid_contract` and in the
updated assertions of `test_valid_source_is_parsed_and_partitioned`.

Mutant-death evidence (each mutant applied to the shadow copy, real fixed
test file unchanged):

1. Drop `~` from headline `"negative"`:
   `ValueError: headline_counts mismatch for negative: expected 100, got 125` — 1 failed.
2. Swap headline `positive`/`negative` expressions:
   `ValueError: headline_counts mismatch for positive: expected 125, got 100` — 1 failed.
3. Swap headline `measured_nonbinder`/`artificial_negative` expressions:
   `ValueError: headline_counts mismatch for measured_nonbinder: expected 60, got 40` — 1 failed.
4. Swap `nine_mer_hla_positive`/`nine_mer_hla_negative` expressions:
   fails on `test_valid_source_is_parsed_and_partitioned` with
   `source_counts mismatch for nine_mer_hla_positive: expected 276, got 250`
   (the 6 `headline_counts.*` parametrized failures alongside it are cascading
   from the same root mismatch, not independent kills).

All 4 die.

## MAJOR-2 — fold identity asserted

`tests/test_pmhc.py::test_valid_source_is_parsed_and_partitioned`: added, per
yielded fold, `assert set(test_rows["Fold"]) == {name}` and
`assert set(train_rows["Fold"]) == set(FOLD_NAMES) - {name}`. No production
change — `iter_pmhc_folds` was already correct; the test just didn't check.

Mutant (`is_test = dataset.rows["Fold"] == fold_name` → `== FOLD_NAMES[0]`):

```
AssertionError: assert {'c000_ba'} == {'c001_ba'}
1 failed, 28 passed
```

Dies.

## MINOR-1 — fetch script nested-extraction guard

`scripts/fetch_pmhc_data.sh`: after `mv -n "$extraction_part" "$SOURCE_DIR"`,
split the single combined `if` into two branches — `$extraction_part` still
existing (genuine race) vs. the nested path
`$SOURCE_DIR/$(basename "$extraction_part")` existing (macOS `mv -n dir
existing-dir` nesting) — each with its own accurate message. No change to the
archive-rename `mv -n` block.

Verified the macOS nesting behavior directly (`mv -n src dst` with `dst`
already a directory nests `src` inside `dst` and exits 0):

```
$ mv -n mv_test/src mv_test/dst; echo "exit:$?"
exit:0
$ ls mv_test/dst/src
file.txt
```

Then ran the fixed fragment in isolation against a pre-nested layout:

```
Extraction directory appeared during unpacking; mv nested the verified
extraction inside it at .../SOURCE_DIR/SOURCE_DIR.123.part instead of
.../SOURCE_DIR
script exit:1
```

Reports the real (nested) path instead of the vanished `$extraction_part`.
No automated test exists for this shell script (none existed before this
round either); verified by direct execution as shown.

## MINOR-2 — all 13 contract fields covered

Replaced the two hand-written `source-counts`/`headline-counts` cases with
`CONTRACT_COUNT_FIELDS` (13 `(parent, field)` pairs) and a new parametrized
`test_contract_count_field_mismatch_is_rejected`, bumping each field by 1 and
expecting `f"{parent} mismatch"`.

Mutant: in `_require_matching_counts`, skip one named key from the
comparison loop — repeated for all 13 keys individually. Representative
output (all 13 shown; each is its own isolated mutant/run):

```
source_counts.hla_abc_rows            -> DID NOT RAISE ValueError (1 failed)
source_counts.ba_rows_all_species      -> DID NOT RAISE ValueError (1 failed)
source_counts.hla_abc_positive         -> DID NOT RAISE ValueError (1 failed)
source_counts.hla_abc_artificial_negative -> DID NOT RAISE ValueError (1 failed)
source_counts.nine_mer_hla_rows        -> DID NOT RAISE ValueError (1 failed)
source_counts.nine_mer_hla_positive    -> DID NOT RAISE ValueError (1 failed)
source_counts.nine_mer_hla_negative    -> DID NOT RAISE ValueError (1 failed)
headline_counts.alleles                -> DID NOT RAISE ValueError (1 failed)
headline_counts.rows                   -> DID NOT RAISE ValueError (1 failed)
headline_counts.positive               -> DID NOT RAISE ValueError (1 failed)
headline_counts.negative               -> DID NOT RAISE ValueError (1 failed)
headline_counts.measured_nonbinder     -> DID NOT RAISE ValueError (1 failed)
headline_counts.artificial_negative    -> DID NOT RAISE ValueError (1 failed)
```

All 13 die (each dropped key's own parametrized case fails to see the raise).

## MINOR-3 — MHC_pseudo.dat covered by extracted-file-hash

Added `extracted-file-hash-pseudo` corruption case (appends a bogus line to
`MHC_pseudo.dat`, expects `"extracted file SHA-256 mismatch"`).

Mutant (`for filename in (*FOLD_NAMES, "MHC_pseudo.dat")` →
`for filename in FOLD_NAMES`):

```
Failed: DID NOT RAISE ValueError
1 failed
```

Dies.

## MINOR-4 — malformed pseudo-sequence rows raise

`src/cognate/pmhc.py::load_pseudo_sequences`: now raises
`ValueError(f"malformed pseudo-sequence row in {path.name} at line {line_number}")`
for any non-empty line with `len(fields) != 2`, matching `_parse_fold`'s
line-context style; blank lines are still skipped. Added
`malformed-pseudo-row` corruption case (appends a 3-field line).

Mutant (revert to the original `if len(fields) == 2: ... ` silent-skip):

```
Failed: DID NOT RAISE ValueError
1 failed
```

Dies.

## MINOR-5 — 4-column row arity

Split the single `malformed-row` case into `malformed-row-too-few` (existing
2-field line) and `malformed-row-too-many` (new 4-field line).

Mutant (`if len(fields) != 3:` → `< 3`):

```
malformed-row-too-many -> AssertionError: Regex pattern did not match.
  Expected regex: 'malformed row'
  Actual message: 'too many values to unpack (expected 3)'
malformed-row-too-few  -> passed (unaffected, as expected — `< 3` still catches 2 fields)
1 failed, 1 passed
```

The new too-many case dies; the pre-existing too-few case correctly still
passes under this mutant, confirming it's the new case doing the killing.

## MINOR-6 — one-peptide-many-alleles fixture coverage

Added a peptide shared between `HLA-A02:01` and `HLA-C07:01` inside fold
`c000_ba` only (`shared_peptide`, captured at fold0 index 1 of the A02:01
positive block, reused with allele `HLA-C07:01`). Folded into the MAJOR-1
recount above (it contributes 1 extra `HLA-C07:01` row in fold0, all-allele
totals above already include it).

Mutant (`.groupby("Peptide")["Fold"].nunique()` → `.count()`):

```
test_valid_source_is_parsed_and_partitioned ->
  ValueError: peptide overlap across folds: ['CAAAAAAAA']
(plus 9 cascading failures in other cases that share the same fixture)
10 failed, 19 passed
```

Dies — the shared same-fold peptide now has `Fold`-count 2 under the mutant
(spurious overlap) vs. `Fold`-nunique 1 under correct code (no overlap),
exactly pinning the two implementations apart.

## Full-suite validation (real, unmutated source)

```
$ uv run pytest tests/test_pmhc.py -q
29 passed in 0.37s

$ uv run pytest -q
222 passed, 1 warning in 116.44s
```

No regressions. The one warning is a pre-existing, unrelated degenerate-score
warning in `tests/test_findings.py`.
