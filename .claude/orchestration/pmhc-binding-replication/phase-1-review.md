# Phase 1 review — pMHC binding replication

`37 candidates, 23 refuted, 14 survived (0 blocker, 1 major, 5 minor, 8 info)`

## 1. Summary

Phase 1 delivers what the spec's file table asks for, and the ten loader requirements are all
present in `src/cognate/pmhc.py`. `uv run pytest tests/test_pmhc.py -q` is green at 15 passed
(`observed`, 0.52s, run from the repo root).

The guard suite is genuinely better than it looks from a read. I did not trust the read: I
shadowed `cognate.pmhc` with a mutated copy on `PYTHONPATH` and re-ran the suite once per
mutation. Nine of eleven guard mutations were killed by the existing tests, which upgrades those
guards from `claimed` to `failure-proven` — the check was shown able to fail, then shown green.

Two mutations survived the whole suite, and one of them is the finding that matters. The
`minimum-class-support` corruption case does not constrain the predicate it appears to constrain:
deleting *either* half of the eligibility condition leaves all 15 tests passing. The spec's
instruction was that each corruption case be run against a non-rejecting implementation and
observed red first. For this one case that procedure exercised the `raise`, not the predicate.

No blocker. One major, five minor, eight info. The major is a six-line fixture change and I
verified the fix works before recommending it.

## 2. Method

Everything below labelled `observed` or `failure-proven` came from a run, not a read.

Mutant harness: `src/cognate/pmhc.py` copied to a scratch package, mutated one anchor at a time,
imported ahead of the installed package via `PYTHONPATH`, full suite re-run per mutant. Import
precedence confirmed first (`cognate.pmhc.__file__` resolved to the scratch copy).

| Mutation | Result |
|---|---|
| M1 drop negative-class arm of eligibility | **15 passed — survived** |
| M2 drop positive-class arm of eligibility | **15 passed — survived** |
| M3 disable missing-class-in-test-fold guard | 1 failed — killed |
| M4 disable missing-class-in-training-complement guard | 1 failed — killed |
| M5 compute `source_counts` post-nine-mer-filter | 2 failed — killed |
| M7 remove artificial-negative labelling | 2 failed — killed |
| M8 pseudo-key normalization made a no-op | 2 failed — killed |
| M10 `POSITIVE_THRESHOLD` 0.426 → 0.462 | **15 passed — survived** |
| M11 `>` → `>=` at the cut | **15 passed — survived** |
| M12 `HLA_PREFIXES` reduced to `("HLA-A",)` | **15 passed — survived** |
| M15 disable fold-overlap guard | 1 failed — killed |
| M17 disable 34-residue guard | 1 failed — killed |
| M18b desync `RowType` threshold from `Target` | 1 failed — killed |

M18 (first attempt, threshold 0.3) survived, but the fixture affinities (0.500 / 0.200 / 0.010)
do not straddle 0.3, so that mutant never expressed the fault. Re-run at 0.6 it was killed. I am
recording this because a survivor from a non-expressing mutant is not evidence of anything, and
reporting it as one would have been a false finding.

## 3. What the spec asked me to confirm

**`source_counts` pre-filter, `headline_counts` post-filter — do the counted populations match
the key names?** Yes. `verify_source` counts `hla_abc_*` on the species-filtered frame before the
length filter and `nine_mer_hla_*` after it; `headline_counts` is computed on
`dataset.rows`, which is post-eligibility. This is not just a read: the fixture plants one 10-mer
HLA positive and one nine-mer BoLA positive per fold, so `hla_abc_rows` (205) and
`hla_abc_positive` (105) differ from `nine_mer_hla_rows` (200) and `nine_mer_hla_positive` (100)
by exactly the planted rows. M5 moving the `hla_abc_*` population behind the nine-mer filter
failed two tests. The pre-filter property is `failure-proven`.

One asymmetry, low value: `hla_abc_artificial_negative` is 50 either side of the length filter in
this fixture, so that single key would not discriminate on its own. Its two siblings do, so the
population is pinned. Refuted.

**Can each of the four partition guards fire, and only on its own condition?**

- Fold overlap: fires. M15 killed. Computed on `nine_mer_hla_rows`, a superset of the scored
  `dataset.rows`, so it is strictly stronger than the leakage question it protects. Uses
  `Fold.nunique() > 1`, so within-fold duplicates are correctly not flagged. Refuted as a concern.
- Minimum class support: fires on the all-empty case only. See MAJOR-1.
- Missing class in test fold: fires. M3 killed.
- Missing class in training complement: fires. M4 killed. The two share a loop but not a message,
  and the two `match` strings (`missing class in test fold` / `missing class in training
  complement`) are mutually non-matching, so neither case can be satisfied by the other's error.
  The `missing-class-training-complement` fixture leaves fold `c000_ba` internally two-class
  precisely so the loop reaches the train branch rather than short-circuiting on the test branch.
  That is deliberate and correct.

`set(bool Series) != {True, False}` compares `numpy.bool_` against Python `bool`; the equality
holds and the green valid-parse test proves the comparison does not spuriously fire (`observed`).
The guard also fires when an eligible allele has zero rows in a fold, which is the right
behaviour, not an over-fire.

**`load_pseudo_sequences` normalization and the 34-residue rule.** Both `failure-proven`. M8
(no-op normalization) and M17 (disabled length check) were each killed. The fixture keys the file
as `HLA-A0201` and asks for `HLA-A02:01`, so a missing `replace(":", "")` cannot pass. The
returned dict is keyed by the original allele, which is what `eligible_alleles` carries — correct
and asserted.

**Would any corruption test pass for the wrong reason?** One does: `minimum-class-support`. See
MAJOR-1. I checked the others for cross-satisfiable `match` strings and found none —
`archive SHA-256 mismatch` is not a substring of the extracted-file message, and the
`malformed row` match is not satisfied by the `ValueError` a removed field-count guard would
raise (`not enough values to unpack`). `pytest.raises` uses `re.search`, so substring
containment was the right thing to check, and it is clean.

## 4. Findings

### MAJOR-1 — the `minimum-class-support` case does not constrain the eligibility predicate

`src/cognate/pmhc.py:207-218`, `tests/test_pmhc.py:165-166`, `tests/test_pmhc.py:213`

The predicate is `counts.get(True, 0) >= minimum_class_rows and counts.get(False, 0) >=
minimum_class_rows`. The corruption case drives it with `minimum_class_rows=101` against a
fixture whose only allele has exactly 100 positives and exactly 100 negatives. Because the
fixture is symmetric, either arm alone is sufficient to make the allele ineligible, so the
`not eligible_alleles` raise still fires with either arm deleted.

`observed`: M1 (negative arm replaced with `and True`) → 15 passed. M2 (positive arm replaced
with `if True`) → 15 passed. The full suite, both times.

The spec's requirement is "choose alleles with at least 100 positives **and** 100 negatives". The
conjunction is the requirement, and nothing in the suite falsifies it. What the case does prove
is the boundary, and that part is good: 100 ≥ 100 admits at default, 100 ≥ 101 excludes, so an
off-by-one between `>` and `>=` is covered.

Partial refutation, which is why this is MAJOR and not BLOCKER: on the real archive, dropping
either arm changes which alleles qualify, so `headline_counts.alleles` (47) and
`headline_counts.rows` (112128) would mismatch and `verify_source` would raise. The contract does
back-stop this in production. But that back-stop is off the unit-test path, it only fires if a
caller chooses to call `verify_source`, and `load_pmhc_dataset` is independently callable and
independently exported. Phase 1's deliverable is the guard suite itself, and this guard is
unfalsified.

Suggested fix, and I ran it before suggesting it. Make the fixture asymmetric with two extra
ineligible alleles and assert the tuple, not just the raise:

- `HLA-B07:02` at 100 positives / 50 negatives — excluded only by the negative arm
- `HLA-C07:01` at 50 positives / 100 negatives — excluded only by the positive arm
- assert `dataset.eligible_alleles == ("HLA-A02:01",)` at the default threshold

`observed` against a scratch build of exactly that fixture: baseline gives
`('HLA-A02:01',)`; under M1 it gives `('HLA-A02:01', 'HLA-B07:02')`; under M2 it gives
`('HLA-A02:01', 'HLA-C07:01')`. Both mutants die. The same change also kills M12 (MINOR-2) for
free, since the B and C rows then have to be counted. The fixture's `source_counts` and
`headline_counts` need updating alongside.

### MINOR-1 — the 0.426 classification cut is unpinned by the suite

`src/cognate/pmhc.py:14`, `src/cognate/pmhc.py:158`

The fixture's affinities are 0.500, 0.200 and 0.010. Nothing sits near the cut, so neither the
constant's value nor the comparison direction is constrained.

`observed`: M10 (`0.426` → `0.462`, a plausible digit transposition) → 15 passed. M11 (`>` →
`>=`) → 15 passed.

Refutation that lowers this to MINOR: on the real archive a wrong constant moves
`headline_counts.positive` off 28538 and `verify_source` raises, and `>=` versus `>` only
matters for a row whose stored affinity is exactly 0.426, which is a rounded presentation of the
500 nM cut (1 − ln(500)/ln(50000) ≈ 0.42562) and so is unlikely to appear verbatim. The endpoint
definition is nonetheless the single most load-bearing number in this experiment, and
`docs/contrast.md` states `>0.426` in prose with nothing asserting the code agrees.

Suggestion: add two fixture rows at 0.426 and 0.427 for an otherwise-eligible allele and assert
their `Target` and `RowType`. Two lines, and it pins both the constant and the direction.

### MINOR-2 — single-allele fixture leaves the HLA-B and HLA-C prefixes unpinned

`src/cognate/pmhc.py:13`, `tests/test_pmhc.py:34-53`

`observed`: M12 reducing `HLA_PREFIXES` to `("HLA-A",)` → 15 passed. The fixture's only HLA
allele is an A, and its only non-HLA allele is BoLA, so the suite proves that A is included and
BoLA excluded, and nothing more. Silently dropping B or C would halve the cohort.

Refuted down to MINOR by the same real-data back-stop: `hla_abc_rows` (170107) would not
reproduce. Fixed for free by the MAJOR-1 fixture change.

### MINOR-3 — `mv` in the fetch script can still clobber, despite the pre-check

`scripts/fetch_pmhc_data.sh:51-55`, `scripts/fetch_pmhc_data.sh:73-77`

The spec asks that the rename happen "only when the final archive is absent", and the `[[ -e ]]`
check immediately before `mv` implements that as written. It is still check-then-act: a file
appearing in the window between test and `mv` is overwritten, against the script's own "never
delete, truncate, or overwrite an existing path" contract.

Suggestion: `mv -n`, supported by both BSD (macOS) and GNU coreutils, which makes the
non-overwriting behaviour the kernel's problem rather than the window's. The directory `mv` is
already safe by accident — `mv dir existing_dir` nests rather than clobbers — but `-n` there too
would make the intent uniform.

### MINOR-4 — `pmhc.py` is the only module in `src/cognate` with no docstrings at all

`src/cognate/pmhc.py`

`observed` via an AST survey of `src/cognate/*.py`: every other module has a module docstring, and
function coverage runs from 43% to 92% (`negatives.py` 12/13, `metrics.py` 21/33,
`baseline_knn.py` 6/10). `pmhc.py` has no module docstring and 0 of 10 definitions documented.

This is a house-style deviation, not a style preference of mine. The two places it actually costs
something: `verify_source` returning `None` and communicating only by raising, and
`iter_pmhc_folds` yielding `(name, train, test)` where the tuple order is load-bearing for every
Phase 2 caller and is currently documented only by an assertion in the test file.

### MINOR-5 — `docs/superpowers/plans/2026-09-10-pmhc-binding-replication.md` sits outside the four-document rule

The spec's boundary is "keep only four durable documents: `docs/pmhc/brief.md`,
`docs/pmhc/experiment.md`, `docs/contrast.md`, `docs/pmhc/postmortem.md`". This plan file is a
fifth, it is under the tracked `docs/` tree, and it is not covered by any `.gitignore` rule, so a
`git add .` would commit it. Either ignore it or move it out of `docs/`.

### INFO

1. `_require_matching_counts` (`src/cognate/pmhc.py:40`) iterates `actual_counts`, so a contract
   carrying an *extra* key is silently accepted. A renamed or missing key is still caught
   (`.get` returns `None`, which mismatches an `int`), so the exposure is narrow. `claimed`.
2. The `[[ -e "$archive_part" ]]` and `[[ -e "$extraction_part" ]]` guards
   (`scripts/fetch_pmhc_data.sh:45`, `:66`) are keyed on `$$`, so they can only fire against a
   stale part file from a prior run that held the same PID. Close to unfireable, and the repo
   rule is not to handle cases that cannot happen.
3. `sha256()` uses `shasum -a 256`. Present on macOS; many minimal Linux images ship only
   `sha256sum`. A two-branch fallback would make the script portable.
4. `iter_pmhc_folds` returns `(name, train, test)`. The order is pinned by
   `tests/test_pmhc.py:122-123` (160 train / 40 test) but not by the signature or a docstring.
5. `overlapping_peptides[:5]` (`src/cognate/pmhc.py:201-205`) truncates without reporting the
   total, so an operator cannot tell 5 collisions from 5000. Include `len(...)` in the message.
   The stray `f` on the second half of the split literal is cosmetic and rides along.
6. The pinned counts in `data/pmhc/source_contract.json` (208093 / 47 / 112128) cannot be
   reproduced in this checkout — `data/pmhc/` holds only the contract file. They are internally
   consistent (126375 = 30869 + 95506; 112128 = 28538 + 83590; 83590 = 82448 + 1142), which is a
   real check and it passes, but the numbers themselves remain `claimed` until someone runs
   `scripts/fetch_pmhc_data.sh` and `verify_source`. Several refutations above lean on this
   back-stop, so it is worth someone actually exercising it once in Phase 2.
7. The repo configures no linter or type checker — no `ruff`, `pylint`, `mypy` or `flake8` in
   `pyproject.toml` or the dev group, and no `scripts/local_ci.sh`. The standing "pylint 10.00"
   bar has nothing to hang on here. I am reporting the absence rather than quoting a score I did
   not produce.
8. **The project has no `CLAUDE.md` at all**, so there is no `## Verification anchors` section to
   align against. `find . -name CLAUDE.md` outside `.venv`/`.git` returns nothing (`observed`).
   The alignment pass was therefore skipped for lack of an anchor, not silently. Given this repo
   is measurement infrastructure with a frozen TCR contract and a newly frozen pMHC contract, the
   verification-gate and coding-invariant categories in particular would earn their keep.

## 5. Refuted candidates worth recording

These were live suspicions that died, listed so the next reviewer does not re-open them.

- `verify_source` parses all five folds twice (once directly, once inside `load_pmhc_dataset`).
  Measured at real scale on a synthetic 208,093-row source: 0.19s per parse, ~0.4s total
  (`observed`). Not worth a refactor.
- Empty-after-filter (a source dir with no HLA rows at all) does not crash in
  `groupby(...).unstack()`; it raises the intended
  `no allele meets minimum class support of 100 rows per class` (`observed`).
- Float equality on `affinity == 0.01` for artificial negatives is exact —
  `float("0.010")` and the literal `0.01` are the same double — and M7 killed the mutant.
- `RowType` re-derives the threshold in a list comprehension rather than reusing `Target`. It
  looked like a desync waiting to happen; the fixture's `value_counts` assertions catch a
  divergence (M18b killed). Duplication remains, the risk does not.
- `load_pseudo_sequences` silently skips lines without exactly two fields. Any allele actually
  needed that got skipped raises `missing pseudo-sequence`, and the file's hash is pinned.
- `data/pmhc/source_contract.json` parses equal to the spec's JSON block (`observed`).
- `docs/contrast.md` carries all nine spec rows verbatim, ends with the required sentence, and
  adds no results section.
- `.gitignore` adds both required rules; `/shared/` is already present at line 35; and
  `git check-ignore` confirms `data/pmhc/source_contract.json` is still trackable (`observed`).
- `cmp .claude/delib/pmhc-next-phase/brief.md docs/pmhc/brief.md` exits 0 (`observed`).
- `docs/next_candidates.md` is modified despite being off-limits, but the diff is next-project
  screening prose about TCR repertoire reporting, unrelated to pMHC — this is the user change the
  spec says to preserve, not an incursion.
- `PmhcDataset` is `frozen=True` while holding mutable DataFrames. The signature is spec-mandated.

## 6. Recommendations

1. Fix MAJOR-1 before Phase 2. The asymmetric three-allele fixture is verified to kill both
   surviving mutants, and it subsumes MINOR-2.
2. Add the two boundary rows at the 0.426 cut (MINOR-1). Cheapest possible protection for the
   experiment's defining number.
3. Take MINOR-3 (`mv -n`) and MINOR-5 (ignore or relocate the plan file) as one-line changes.
4. Add a module docstring and one line on `iter_pmhc_folds`'s tuple order (MINOR-4).
5. Consider adopting mutation-style checks for the remaining contract guards as they land in
   Phase 2. The spec's "run it red first" instruction is the right instinct; running it red
   against the *predicate* rather than the *raise* is what catches this class of defect.
6. Add a project `CLAUDE.md` with a verification gate and the pMHC coding invariants (INFO 8).

## 7. What is done well

- The fixture's design is better than most. Planting one 10-mer HLA row and one nine-mer BoLA row
  per fold means the species filter and the length filter are independently observable in the
  counts, and M5 confirms that is not accidental.
- Splitting the two missing-class messages so their `match` strings cannot cross-satisfy, and
  constructing the training-complement fixture so the loop has to reach the second branch, is
  exactly the discipline this review was asked to look for. It held up under mutation.
- Nine of eleven expressing mutants killed is a strong result for a first phase.
- The fetch script's refusal to touch non-file and non-directory paths, and its download-to-`.part`
  then verify-then-rename ordering, implement the spec's non-destructive requirement faithfully.
- The contract JSON and `docs/contrast.md` are exact against the spec, and the brief promotion
  verifies with `cmp`. Provenance discipline is intact.

## 8. Review metadata

- Repo: `/Users/freddy/Documents/repos/cognate`
- Branch: `codex/pmhc-binding-replication`, HEAD `71aa9fe`
- No PR and no commit for this artifact: every reviewed file except `.gitignore` is untracked, so
  the review is against the working tree, not a diff.
- Reviewed: `src/cognate/pmhc.py`, `tests/test_pmhc.py`, `scripts/fetch_pmhc_data.sh`,
  `data/pmhc/source_contract.json`, `docs/contrast.md`, `.gitignore`
- Validation run: `uv run pytest tests/test_pmhc.py -q` → 15 passed in 0.52s (`observed`)
- Mutation runs: 14 total, one per anchor, full suite each (`observed`)
- Evidence: mutation-killed guards are `failure-proven` (red under mutation, green at baseline).
  Everything drawn from reading alone is labelled `claimed`. Nothing here is
  `independently-verified`.
