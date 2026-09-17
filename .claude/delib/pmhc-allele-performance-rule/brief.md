# Brief — is the pMHC transfer gradient novelty, or intrinsic allele difficulty?

**Status:** shaped, reviewed (`spec-reviewer` + `redteam`, both passes applied). Not a plan.
No code until a plan exists via `superpowers:writing-plans`.
**Shaped:** 2026-09-16 via `forge`. Revision 2 — revision 1 proposed a 109-allele cohort
expansion and was dismantled by both review passes; see the ledger at the end.

## North star

Our shipped result says transfer degrades as an allele gets farther from the training set. Find
out whether that gradient is a novelty effect at all, or whether far alleles are simply harder
alleles.

Every element below is cut against that sentence.

## Problem

The shipped LOAO study reports macro standardized AUC0.1 falling 0.7487 → 0.7040 → 0.6218 across
allele-only, joint, and cluster holdout, and reads that as novelty-driven degradation. Per allele,
the relation is strong and already computed in committed artifacts:

| quantity, over the 47-allele cohort | value |
|---|---|
| `pseudo_sequence_mlp` AUC0.1 vs nearest-retained distance | Pearson −0.733, Spearman −0.746 |
| `nearest_pwm` AUC0.1 vs nearest-retained distance | Pearson −0.677, Spearman −0.735 |
| the two arms' per-allele AUC0.1, against each other | Pearson +0.883 |
| AUC0.1 vs log `n_rows` (mlp / pwm) | −0.032 / +0.032 |

`observed`, 2026-09-16, computed from `data/pmhc/loao_allele_only_results.json`.

The support null is already dead inside this cohort. But a second confound is untested and is not
addressed anywhere in the shipped docs: **distance is correlated with rarity.** Far alleles are
rare alleles — fewer distinct peptides, thinner assay coverage, possibly different motif
complexity. If rare alleles are intrinsically harder to predict, the gradient would appear exactly
as observed even with no novelty penalty at all, and the shipped interpretation would be
partly wrong.

Nothing in the shipped study can separate these, because every arm in it was trained under
holdout. Both explanations predict the same curve.

## Chosen approach

**Add one predictor that has already seen every allele, and check whether it shows the same
gradient.**

MHCflurry 2.0 (`pip install mhcflurry`, ~355,841 parameters, trained on ~4.3M peptides across 95
HLA-I alleles) has no novelty penalty on our cohort — it has seen these alleles. Run it, inference
only, over the same per-allele test partitions, and regress its per-allele AUC0.1 on the same
distance axis.

### The discriminating prediction, to be frozen before the run

| If MHCflurry's gradient is | Then | Consequence for the shipped study |
|---|---|---|
| flat (slope ≈ 0) | the gradient in our arms is a genuine novelty effect | shipped interpretation holds, and is now controlled |
| as steep as ours | the gradient is intrinsic allele difficulty | shipped interpretation is confounded; the docs need a correction |
| intermediate | both contribute; report the decomposition | shipped interpretation needs a stated bound |

All three outcomes are publishable as a short methods note, and the middle one is a correction to
our own delivered work. That is the point: this brief is aimed at our result, not at a new claim.

### Why this earns the word "agnostic"

MHCflurry's mechanism is not our Hamming metric. This matters because of a defect found in
review: `select_nearest_pwm_source` (`src/cognate/pmhc_transfer.py:241`) selects its source by the
**same** pseudo-sequence Hamming distance used as the x-axis —
`selected_pwm_source_distance == nearest_retained_pseudo_distance` for **47 of 47** alleles.
`observed`. So `nearest_pwm` agreeing with a distance-indexed curve is near-tautological and
carries no weight on transferability. It stays in the artifact as a shipped arm; it is **excluded
from the transferability claim**.

### Cohort: the frozen 47, unchanged

Revision 1 proposed expanding to 109 alleles. That is not constructible:
`load_pmhc_dataset` requires ≥100 rows **per class** and `_require_both_classes_per_fold` enforces
both classes per fold, and **13 of the 62 unused alleles have zero positives in the entire
archive** — the estimand does not exist for them. The 47 *is* that filter's output. The largest
cohort the existing loader builds is 72 at `minimum_class_rows=30`, and the ~25 alleles that buys
carry per-allele AUC0.1 standard errors comparable to the effect being measured. `observed`.

Keeping the 47 preserves comparability with all three shipped results, requires no loader change,
and requires no new LOAO run. Arms and the five-name scoring contract
(`TRANSFER_SCORE_NAMES`, `validate_transfer_result`) are untouched — MHCflurry is scored outside
that contract, as an added analysis, not as a sixth arm inside it.

### Covariates, split by availability

Frozen as two sets, because the north star concerns alleles with no data:

- **Deployment-available** (may enter a predictive rule): nearest-retained pseudo-sequence
  distance, retained-neighbour support.
- **Null-test-only** (may enter the confound tests, never a deployable rule): target `n_rows`,
  target `n_positive`.

### Freeze mechanism

Step order is enforced by commit boundary, matching the shipped study's convention: the
pre-declaration — functional form, covariate split, the three-way discriminating prediction above,
and the slope-equality tolerance — is committed **before** the commit that runs MHCflurry. Review
can check the order in `git log`. Without this the brief has no integrity device.

## Scope

- One MHCflurry 2.0 inference pass over the 47 existing per-allele test partitions.
- One pre-declaration commit, then one analysis: distance → per-allele AUC0.1 for
  `pseudo_sequence_mlp` and MHCflurry, with slope comparison and the support nulls.
- One results artifact under `data/pmhc/`, one writeup under `docs/pmhc/`, and a correction to the
  three shipped docs if the middle or third outcome lands.

Runtime: MHCflurry inference on ~112k rows is minutes. The analysis is seconds. `claimed` — a
timing probe on one allele precedes the full pass.

## Non-goals

- **No cohort expansion.** Not 109, not 72. Frozen 47.
- **No new LOAO run and no retraining of any arm.** Everything for our side is already committed.
- **No sixth arm inside the frozen scoring contract.** No fork of `score_transfer_partition`.
- **No conformal abstention rule.** Exchangeability across 47 alleles is not defensible.
- **No IEDB cross-check.** Dropped: the live benchmark page exposes only overall server rankings,
  no per-allele AUC/SRCC, no bulk download, no API. `observed`, 2026-09-16.
- **No new architecture, fine-tuning, or hyperparameter search.**
- **No MHC Class II, no antigen-presentation endpoint, no NetMHCpan executable.**
- **No claim that the shipped numbers are arithmetically wrong.** They were verified against the
  artifacts. Only their *interpretation* is under test.

## Open questions

1. **TensorFlow is a new, heavy dependency.** MHCflurry needs TF ≥2.2 and it is not installed.
   Whether it goes in the main environment or an isolated one has to be settled in the plan; it
   must not perturb the environment the shipped results were produced in.
2. **MHCflurry's training data overlaps our test rows.** It was trained on public affinity data
   that includes this archive, so its absolute per-allele numbers are partly memorization and are
   **not** comparable to our arms. Only the *slope* against distance is a valid comparand. The
   plan must state this and must not report a head-to-head accuracy table.
3. **Which of the 47 are in MHCflurry's 95 training alleles?** Its curated training data is
   downloadable (`mhcflurry-downloads info`). Any of our 47 that MHCflurry has *not* seen breaks
   the no-novelty-control assumption for that allele and must be identified and reported before
   the regression, not after.
4. **Slope-equality tolerance is undecided.** What counts as "as steep as ours" must be a number,
   frozen in the pre-declaration commit. Deciding it after seeing MHCflurry's scatter voids the
   design.

## Review ledger

Both passes run. `spec-reviewer`: 9 candidates, 5 refuted, 4 survived (2 MAJOR, 2 MINOR/INFO).
`redteam`: 10 raised, 3 refuted by the reviewer, 7 survived; 1 of those refuted on verification.
Every objection terminal.

| # | source | objection | state | evidence / rationale |
|---|---|---|---|---|
| 1 | redteam | 109-allele cohort is not constructible; 13 unused alleles have zero positives | accepted | cohort reverts to the frozen 47; expansion deleted. Verified against the loader, `observed` |
| 2 | redteam | "62 held-out alleles" is really ~25 usable | accepted | moot — no expansion |
| 3 | redteam | adding alleles narrows the distance axis | refuted | at the constructible 72-allele cohort median stays 3, max stays 10, far-allele count rises 28→42. Computed by redteam over the non-constructible 109 set. `observed` |
| 4 | redteam | the primary fit is answerable free, today | accepted | it is now the design; correlations above are computed, not planned |
| 5 | redteam | `nearest_pwm` is definitionally distance-driven, so two arms cannot test transferability | accepted | confirmed 47/47; arm excluded from the transferability claim, external predictor added to carry it. This is the objection that reshaped the brief |
| 6 | redteam | low-data bucket counts miscounted (41+29=70 > 62), tagged `observed` | accepted | corrected to 2 / 31 / 29; the sentence that justified the cohort override is deleted |
| 7 | redteam | target-side covariates are unavailable in the deployed case | accepted | covariate set split into deployment-available and null-test-only |
| 8 | spec-reviewer | dropping two arms breaks the frozen five-name scoring contract that all three shipped tracks depend on | accepted | all five arms retained; MHCflurry scored outside the contract |
| 9 | spec-reviewer | the ≥100-per-class eligibility gate makes the 62 inadmissible | accepted | moot — no expansion |
| 10 | spec-reviewer | step-3 freeze mechanism unspecified, only its content | accepted | commit-boundary mechanism specified above |
| 11 | spec-reviewer | "held out" used for two different things | accepted | the meta-split is gone; term now has one meaning |
| 12 | self | IEDB secondary check has no per-allele bulk data | accepted | dropped from scope, `observed` |
