# Phase 1 fix notes

Fixes for the four review findings (MAJOR-1, MINOR-1, MINOR-3, MINOR-4). Scope held to
exactly these four items; no other files touched.

## MAJOR-1 -- minimum-class-support guard was untested for the AND conjunction

`src/cognate/pmhc.py`'s eligibility predicate already reads:

```python
if counts.get(True, 0) >= minimum_class_rows
and counts.get(False, 0) >= minimum_class_rows
```

This is correct production code. The gap was purely in the fixture: with one allele at
exactly 100/100, deleting either arm of the `and` still passes every existing assertion.

Fix: `tests/test_pmhc.py::_write_valid_source` now emits two more alleles per fold --
`HLA-B07:02` (100 positive / 50 negative total) and `HLA-C07:01` (50 positive / 100
negative total) -- alongside the unchanged `HLA-A02:01` (100/100). Neither new allele
meets `minimum_class_rows=100` on both arms, so `eligible_alleles` stays
`("HLA-A02:01",)`; that assertion (already present at line 153, now load-bearing) is
what the mutants trip.

All `source_counts` in `_valid_contract` were recomputed from first principles against
the new fixture (headline_counts is unchanged, since eligibility still resolves to the
same single allele and its rows are untouched):

| key | old | new |
|---|---|---|
| ba_rows_all_species | 210 | 510 |
| hla_abc_rows | 205 | 505 |
| hla_abc_positive | 105 | 255 |
| hla_abc_artificial_negative | 50 | 125 |
| nine_mer_hla_rows | 200 | 500 |
| nine_mer_hla_positive | 100 | 250 |
| nine_mer_hla_negative | 100 | 250 |

`dataset.all_nine_mer_hla_rows` length assertion updated 200 -> 500 (includes the two
new alleles pre-eligibility-filter); `dataset.rows` length stays 200 (post-filter, only
`HLA-A02:01`).

### Red-then-green

Weakened the predicate to the positive-only arm:

```python
if counts.get(True, 0) >= minimum_class_rows
```

```
$ uv run pytest tests/test_pmhc.py -q -k "test_valid_source_is_parsed_and_partitioned or minimum-class-support"
...
E   ValueError: headline_counts mismatch for alleles: expected 1, got 2
1 failed, 1 passed, 13 deselected in 0.47s
```

RED confirmed (`HLA-B07:02` incorrectly became eligible; `minimum-class-support` alone
still passed, exactly as the finding describes).

Weakened to the negative-only arm:

```python
if counts.get(False, 0) >= minimum_class_rows
```

```
E   ValueError: headline_counts mismatch for alleles: expected 1, got 2
1 failed, 1 passed, 13 deselected in 0.35s
```

RED confirmed (`HLA-C07:01` incorrectly became eligible).

Reverted to the real `and` conjunction:

```
$ uv run pytest tests/test_pmhc.py -q
...............
15 passed in 0.45s
```

GREEN confirmed.

### Bonus kill (incidental)

Reducing `HLA_PREFIXES` to `("HLA-A",)`:

```
E   ValueError: source_counts mismatch for hla_abc_rows: expected 505, got 205
1 failed, 14 deselected in 0.39s
```

Caught by the recounted `source_counts`, as the finding predicted. Reverted; full suite
green (`uv run pytest tests/test_pmhc.py -q` -> 15 passed).

## MINOR-1 -- 0.426 threshold unpinned

Two existing fixture rows in fold `c000_ba` (chosen so as not to collide with the
`missing-class-*` corruption cases, see below) are pinned to the exact boundary instead
of arbitrary values: the first `HLA-A02:01` positive row's affinity is set to `0.427`
(must classify positive) and the first negative row's affinity to `0.426` (must classify
negative). Classification outcomes (`Target`/`RowType`) for both rows are unchanged from
before the edit -- only the literal affinity string changed -- so no other count in the
contract moved. `_write_valid_source` now returns the two boundary peptides so the test
can look up their rows directly:

```python
assert bool(boundary_positive_row["Target"]) is True
assert boundary_positive_row["RowType"] == "positive"
assert bool(boundary_negative_row["Target"]) is False
assert boundary_negative_row["RowType"] == "measured_nonbinder"
```

Why fold `c000_ba` and not elsewhere: `missing-class-test-fold` blanket-replaces every
literal `" 0.500 "` in `c000_ba` to wipe out that fold's positives, and
`missing-class-training-complement` does the same across folds 1-4. Had the boundary
positive row kept the literal `0.500` string it would have been swept by whichever case
touches its fold, defeating that case's own wipe. Using `0.427` for the boundary row
means it is immune to the `" 0.500 "` string replace either way, so I confirmed neither
existing corruption case regresses: with three alleles in play at `minimum_class_rows=2`
(both new MAJOR-1 alleles become eligible at that low bar), the wipe still knocks out
`HLA-B07:02`'s or `HLA-C07:01`'s positives in the swept fold(s), so the expected
`ValueError` still fires -- confirmed by running the full corruption matrix after the
change (all 15 tests, including these two cases, pass; see full green run above).

### Red-then-green

Mutated `POSITIVE_THRESHOLD` 0.426 -> 0.462:

```
$ uv run pytest tests/test_pmhc.py -q -k test_valid_source_is_parsed_and_partitioned
...
E   ValueError: source_counts mismatch for hla_abc_positive: expected 255, got 254
1 failed, 14 deselected in 0.41s
```

RED confirmed (the 0.427 boundary row flips to negative under the raised threshold).
Reverted.

Flipped both `>` comparisons against `POSITIVE_THRESHOLD` to `>=`:

```
$ uv run pytest tests/test_pmhc.py -q -k test_valid_source_is_parsed_and_partitioned
...
E   ValueError: source_counts mismatch for hla_abc_positive: expected 255, got 256
1 failed, 14 deselected in 0.48s
```

RED confirmed (the 0.426 boundary row flips to positive under `>=`). Reverted both
comparisons.

```
$ uv run pytest tests/test_pmhc.py -q
...............
15 passed in 0.46s
```

GREEN confirmed. Diffed `src/cognate/pmhc.py` against the pre-mutation copy after
reverting -- byte-identical except for the two added docstrings (MINOR-4), confirming no
mutation leaked into the final state.

## MINOR-3 -- TOCTOU in `scripts/fetch_pmhc_data.sh`

Both `if [[ -e dest ]]; then error; fi; mv part dest` sequences replaced with
`mv -n part dest` followed by a post-hoc check, closing the window between the check and
the act (they are no longer two separate steps racing against a concurrent invocation).

Archive case (dest is always a plain file, so `-n` alone is sufficient -- it either
renames or declines without touching the existing file):

```bash
mv -n "$archive_part" "$ARCHIVE_PATH"
if [[ -e "$archive_part" ]]; then
  echo "Archive appeared during download; leaving partial archive at $archive_part" >&2
  exit 1
fi
```

Extraction case needed an extra check. I verified empirically (not from the man page)
that BSD `mv -n srcdir destdir`, when `destdir` already exists as a directory, does
*not* decline -- it silently nests `srcdir` inside `destdir` (`destdir/srcdir`), exit 0,
no error, and does not touch `destdir`'s existing content:

```
$ mv -n src dest   # dest already exists as a directory
exit=0
./dest/marker.txt        # dest's own pre-existing file, untouched
./dest/src/marker.txt    # src silently nested one level inside dest
```

So checking only "does `$extraction_part` still exist" is not enough for a directory
target -- a successful-but-nested move would make it disappear from its original path
without actually landing at `$SOURCE_DIR`. The post-check also looks for the nested
sibling path:

```bash
mv -n "$extraction_part" "$SOURCE_DIR"
if [[ -e "$extraction_part" || -e "$SOURCE_DIR/$(basename "$extraction_part")" ]]; then
  echo "Extraction appeared during unpacking; leaving partial extraction at $extraction_part" >&2
  exit 1
fi
```

Existing refuse-to-clobber behavior and error message text are unchanged in both cases;
only the check-then-act ordering changed.

### Runnable check

Network/tar are out of scope to exercise directly, so the two guard blocks (archive
rename, extraction rename) were extracted verbatim into a standalone harness
(`/private/tmp/.../scratchpad/test_fetch_race_guard.sh`, not part of the repo) and run
against four cases: no-race archive, raced archive (dest pre-empted), no-race extraction,
raced extraction (dest dir pre-empted, exercising the BSD nesting quirk above). All four
passed:

```
case 1: archive rename, no race -> succeeds, part is gone, dest has our content
PASS
case 2: archive rename, raced -> destination pre-empted by another run, guard rejects, no clobber
PASS
case 3: extraction rename, no race -> succeeds, part is gone, dest has our content
PASS
case 4: extraction rename, raced -> destination dir pre-empted by another run, guard rejects, no silent nesting left unnoticed
PASS
all cases passed
```

`bash -n scripts/fetch_pmhc_data.sh` also passes (syntax only, no functional coverage).

## MINOR-4 -- missing docstrings

Added a module docstring to `src/cognate/pmhc.py` matching the one/two-sentence style
used by the other 11 modules in `src/cognate` (e.g. `data.py`, `split.py`), and a
one-line docstring on `iter_pmhc_folds` stating the yielded tuple order, matching the
single-line docstring style used for simple functions elsewhere (e.g.
`data.py::load_train`):

```python
"""Source contract verification and fold partitioning for the pMHC binding-affinity dataset.

Parses the NetMHCpan training folds, enforces the eligibility and class-balance guards
from the source contract, and exposes the result as five leave-one-fold-out train/test
splits.
"""
```

```python
def iter_pmhc_folds(...) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield `(fold_name, train_rows, test_rows)` for each fold, held out as test in turn."""
```

No other functions in the file were touched.

## Full validation

```
$ uv run pytest tests/test_pmhc.py -q
...............
15 passed in 0.45s

$ uv run pytest -q
........................................................................ [ 34%]
........................................................................ [ 69%]
................................................................         [100%]
208 passed, 1 warning in 120.19s
```

The one warning is pre-existing (`test_findings.py::test_knn_headline_scores`, unrelated
to this change, not touched).

Worktree left dirty; nothing staged or committed.
