# Phase 1 starting state (verified 2026-09-10, before implementer dispatch)

A prior context completed part of Phase 1. `uv run pytest tests/test_pmhc.py -q` currently
reports **8 passed**. Do not redo the finished items; close the gaps.

## Already done — leave alone

| Item | State |
|---|---|
| `.gitignore` | `data/pmhc/raw/` + `data/pmhc/derived/` rules present under a `# pMHC source rows and regenerable row-level outputs` comment |
| `docs/pmhc/brief.md` | promoted; `cmp .claude/delib/pmhc-next-phase/brief.md docs/pmhc/brief.md` is byte-identical |
| `docs/contrast.md` | shared-contract table + closing interpretation sentence present, no empty results section |
| `data/pmhc/source_contract.json` | matches the spec JSON exactly |
| `scripts/fetch_pmhc_data.sh` | non-overwriting fetch/extract logic implemented and correct |
| `src/cognate/pmhc.py` | `PmhcDataset`, `load_source_contract`, `verify_source` (hashes only), `_parse_fold`, `load_pmhc_dataset`, `iter_pmhc_folds`, `load_pseudo_sequences` all exist |
| `tests/test_pmhc.py` | `test_valid_source_is_parsed_and_partitioned` + `_run_corruption` with 14 named branches |

## Gaps to close

### 1. `scripts/fetch_pmhc_data.sh` — missing shebang
File is mode `755` but starts at `set -euo pipefail` with no `#!` line. It uses `[[ ]]` and
`local`, so executing it directly runs it under `sh` and it breaks. Add `#!/usr/bin/env bash`
as line 1. Nothing else in this file changes.

### 2. `src/cognate/pmhc.py` — seven guards missing
`_run_corruption` in the test file already has a branch for each of these, but the production
code does not reject them, so they are absent from the `CORRUPTION_CASES` parametrize tuple.

| Corruption branch | Missing guard |
|---|---|
| `minimum-class-support` | when no allele clears `minimum_class_rows` in both classes, `eligible_alleles` silently becomes empty instead of raising |
| `missing-class-test-fold` | no check that every retained allele has both classes present in each test fold |
| `missing-class-training-complement` | no check that every retained allele has both classes present in each fold's 4-fold training complement |
| `source-counts` | `contract["source_counts"]` is never compared against the parsed source |
| `headline-counts` | `contract["headline_counts"]` is never compared against the retained cohort |
| `missing-pseudo-sequence` | `load_pseudo_sequences` silently drops alleles with no pseudo entry (dict comprehension filters them out) — must raise instead |
| `short-pseudo-sequence` | no 34-residue length requirement |

### 3. Design note on the two count guards
Both `_run_corruption` branches call `verify_source(source_dir, archive_path, contract)` with
only those three arguments, so **`verify_source` must compute the counts itself** — the plan
freezes its signature and `PmhcDataset` fields, so counts cannot be threaded in from the caller.

`source_counts` keys (`ba_rows_all_species`, `hla_abc_rows`, `hla_abc_positive`,
`hla_abc_artificial_negative`) are pre-nine-mer-filter, which `PmhcDataset.all_nine_mer_hla_rows`
does not expose. Suggested shape, which adds no public API and no abstraction layer:

- factor the concat of `_parse_fold` over `FOLD_NAMES` into one private `_load_raw_rows(source_dir)`
  returning the all-species frame with `Target` / `RowType` already assigned;
- `load_pmhc_dataset` filters that frame as it does today;
- `verify_source` uses it for `source_counts` and calls `load_pmhc_dataset` for `headline_counts`.

Take a different shape if you find a smaller one, but do not add a new public function, a new
dataclass field, or a config object. Parsing the folds twice per run is acceptable.

### 4. `tests/test_pmhc.py` — fixture and parametrize gaps
- `PSEUDO_SEQUENCE` is currently 30 residues (`"ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"`). The spec
  requires 34, and `short-pseudo-sequence` corrupts by truncating one residue, so the valid
  fixture must be exactly 34.
- `CORRUPTION_CASES` lists only 7 of the 14 branches. Add the 7 above with match strings that
  pin the specific rejection, not a generic substring.
- The synthetic `_valid_contract` `source_counts` / `headline_counts` must actually match the
  synthetic fixture data once `verify_source` starts checking them. Recount against the fixture
  rather than trusting the numbers already written there.

## Red-then-green requirement
Every one of the seven guards must be observed failing before it is added. The corruption branch
already exists for each, so the procedure per guard is: add its entry to `CORRUPTION_CASES`, run
that single selector, record the observed red (the guard does not fire, so `pytest.raises` fails),
add the guard, rerun the same selector green. Report the red and green command output per guard —
a guard that was only ever seen green is `observed`, not `failure-proven`, and the spec requires
`failure-proven`.
