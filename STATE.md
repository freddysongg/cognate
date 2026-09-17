# STATE

Current build/test/issue status for `cognate`. Refresh with `state-sync` at phase boundaries.

**Updated:** 2026-09-17

## Current phase

**Closed.** Both lines of work are complete and merged to `main` at `99f1125`, pushed to
`origin/main`. No implementation work is open and none is planned here. The successor project is
[`mimicry`](https://github.com/freddysongg/mimicry), on TCR cross-reactivity evaluation
methodology; the decision to stop is on the `cognate` vault project page under
"Decision 2026-09-17".

337 tests pass. Local CI is `uv run pytest -q` then `uv run python -m compileall -q src scripts`
— there is no `scripts/local_ci.sh` and no Makefile.

## What's built

| Component | Status | Notes |
|---|---|---|
| TCR–epitope baselines | Complete, closed | k-NN, ESM-2 logistic and MLP heads, contrastive, cross-attention. Retrieval wins on seen peptides; nothing beats chance on unseen |
| VDJdb evaluation set | Complete | 88 peptides, 48 seen / 40 unseen. Superseded the 20-peptide IMMREP23 set |
| pMHC binding replication | Complete | 47 alleles, 112,128 nine-mer rows, own source contract with archive hashes |
| LOAO transfer study | Complete, 3 tracks | allele-only 0.7487, joint 0.7040, cluster 0.6218 |
| MHCflurry no-novelty control | Complete | Outcome `novelty_effect`. Decision rule frozen in a commit provably preceding the run |
| Frozen contract | Enforced | `metrics.py`, `split.py`, `features.py`, `data/vdjdb_eval.csv` hash-pinned by `tests/test_frozen_contract_hashes.py` |

## What's not built

| Component | Issue | Blocked on |
|---|---|---|
| #33 allele/peptide confound repair | Open by choice | Nothing. Cheap and correct, but will not publish. Would need its own brief |
| Predictor-dependence across 4–5 predictors | Declined | Deliberately not pursued — bounding one published claim is a comment on someone else's paper |
| Antigen presentation, MHC Class II, external validation | Out of scope | Forbidden by the LOAO brief, and each is a new project in a dense field |

## Active work

| Item | Owner | Status |
|---|---|---|
| None | — | Repository is closed |

## Pending decisions

| Decision | Context | Preference |
|---|---|---|
| Whether to keep `data/pmhc/cohort_rows.csv` tracked | 2.9 MB derived intermediate, deterministic output of `scripts/export_pmhc_cohort.py`. Already in history, so removing it reclaims nothing | Leave it. Removal would not shrink history and would break the isolated MHCflurry path out of the box |

## Open design questions

| Question | Notes |
|---|---|
| Does a per-allele performance predictor transfer across models? | Our two arms slope −1.0315 and −0.1674 on an identical axis, so probably not. Recorded in `docs/next_candidates.md` as evidence against predictor-agnosticism rather than as a closed candidate |
| Is MHCflurry's flat slope absence of intrinsic difficulty, or memorization masking it? | Not separable with what is here. Stated as a limit in `docs/pmhc/novelty_control.md` |
| Are the 47 cohort alleles in MHCflurry's training set? | Unverified. `supported_alleles` returns 14,884 entries, which proves model support, not training exposure. Labelled `claimed` in the writeup |

## Resolved decisions

| Decision | Resolution | Date | Source |
|---|---|---|---|
| TCR line | Closed. Nothing beat chance on unseen peptides; a ten-line edit-distance k-NN at 0.5654 beat every learned representation | 2026-09 | `docs/findings.md`, `docs/belief_list.md` |
| pMHC LOAO interpretation | Novelty, not intrinsic per-allele difficulty. Survived the no-novelty control | 2026-09-17 | `docs/pmhc/novelty_control.md` |
| Whether to extend the pMHC line | No. Concluded by decision, after a literature gate killed two of four candidates | 2026-09-17 | `docs/next_candidates.md`, vault `cognate.md` |
| Next direction | TCR cross-reactivity, in the separate `mimicry` repo | 2026-09-17 | vault `cognate.md`, "Decision 2026-09-17" |

## External services status

| Service | Transport | Auth | Client | Local runnable? |
|---|---|---|---|---|
| MHCflurry 2.2.1 | Local inference, isolated interpreter | None | `uv run --no-project --with mhcflurry` | Yes. Model files must be fetched once with `mhcflurry-downloads fetch models_class1_pan` |
| NetMHCpan training archive | HTTPS, one-time retrieval | None | `scripts/` + `data/pmhc/source_contract.json` | Yes, archive and per-file hashes pinned |
