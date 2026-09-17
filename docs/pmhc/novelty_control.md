# Novelty control for the LOAO degradation curve

**Status.** Observed run completed on 2026-09-16. Outcome: **`novelty_effect`**.

## Question

The shipped `loao_allele_only` study (`docs/pmhc/loao_allele_only.md`) reports per-allele
AUC0.1 falling as pseudo-sequence distance to the nearest retained allele grows. Every arm in
that study, including the pseudo-sequence MLP, was trained under holdout, so a novelty effect
(the model has never seen anything like this allele) and intrinsic per-allele difficulty (this
allele is just harder to predict, for reasons unrelated to holdout) predict the same falling
curve. The shipped study's design cannot tell the two apart.

MHCflurry 2.2.1 breaks the tie: its training data already includes every allele in this
cohort, so it carries no novelty penalty. If its own AUC0.1 still falls with distance, that
fall cannot be a holdout artifact, and some of the shipped gradient is intrinsic difficulty
instead. If it does not fall, the shipped gradient survives as a novelty reading. The decision
rule comparing the two slopes was frozen in `docs/pmhc/novelty_control_predeclaration.md`
before any MHCflurry prediction existed in this repository. The pre-declaration commit
`6d4a7d6` (`add frozen pre-declaration for pmhc novelty control`) precedes the control-run
commit `5a5d623` (`run mhcflurry novelty control, write comparison artifact`) in `git log`,
checkable with `git merge-base --is-ancestor 6d4a7d6 5a5d623`. `observed`. That ordering is
what makes the rule a genuine pre-declaration rather than a threshold picked after seeing the
scatter.

## Results

All figures below are from `data/pmhc/novelty_control_results.json`. `observed`.

Ordinary-least-squares fit of per-allele AUC0.1 on nearest-retained normalized-Hamming
distance (`config.functional_form`: `ols_linear`), `n_alleles_primary` = 47, MHCflurry
version `2.2.1`.

| Fit | Slope | 95% CI | r-squared | n |
|---|---:|---:|---:|---:|
| Reference (`pseudo_sequence_mlp`, pinned -1.0315) | -1.0315 | [-1.3116, -0.7514] | 0.5366 | 47 |
| Control (MHCflurry 2.2.1) | -0.1674 | [-0.3957, +0.0609] | 0.0439 | 47 |

`coverage.unsupported` is empty: all 47 cohort alleles are supported by MHCflurry's released
pan-allele models, so no allele was excluded from the primary fit and no secondary fit was
needed. `observed`. This is membership in `Class1AffinityPredictor.supported_alleles`
(14,884 entries as of 2.2.1, recorded as `coverage.supported_allele_count`), which is the set
of alleles the released models can emit a prediction for by pseudo-sequence similarity — not
the set of alleles MHCflurry's curated training data actually contains. That the cohort's
alleles were in MHCflurry's *training* data, and therefore carry no novelty penalty, is
inferred from these being high-data HLA-A/B/C alleles drawn from the same public
binding-affinity literature MHCflurry was trained on, not verified directly. `claimed`.
Verifying it would mean intersecting the cohort against MHCflurry's curated training-allele
list, which this study did not do; with 14,884 supported alleles against 47 cohort alleles,
the supported-set intersection is uninformative on its own.

The control was checked for skill, not just assumed skillful because its slope is flat: every
one of the 47 per-allele AUC0.1 values exceeds the frozen competence floor of 0.6
(`scripts/run_pmhc_novelty_control.py`'s `CONTROL_COMPETENCE_FLOOR`), enforced by an assertion
that fails naming the offending alleles if any allele does not clear it. On this cohort the
per-allele mean is 0.8123 and the minimum is 0.6690 (HLA-A30:02). `observed`. This rules out
the failure mode where a broken control (wrong alleles, shuffled peptides, an inverted sign)
would also read as flat and thus also yield `novelty_effect` — a genuinely broken control would
fail this floor. The per-allele values are recorded in `control.per_allele_auc01` in the
result artifact, so the check is reproducible from the artifact alone. This floor asserts only
that the control beats chance on this cohort; it never sets the control's absolute AUC0.1
beside a reference arm's, so it does not breach the slopes-only comparison rule below.

Applying the frozen decision rule (control CI spans zero, and the reference point estimate
falls below the control's CI lower bound) yields `outcome: novelty_effect`.

## What this does and does not show

The control's slope confidence interval spans zero (`[-0.3957, +0.0609]`), so under the frozen
rule the shipped novelty reading survives. But the interval's upper bound only just clears
zero, and the point estimate itself is negative. This is "not distinguishable from flat at
95%," not "flat." A small intrinsic-difficulty contribution in the shipped gradient cannot be
excluded — bound it by the control slope's magnitude, about 16% of the reference slope's
magnitude (0.1674 / 1.0315). `claimed`.

The study's original motivating story — that distant alleles are also data-poor alleles, so
an intrinsic-difficulty effect could just be reflecting undertrained models — is weak in this
cohort. From `support_nulls`: `distance_vs_log_n_rows` = -0.1685 and
`distance_vs_log_n_positive` = -0.2257 (Pearson correlation of nearest-retained distance
against log training row count and log positive count, computed in
`scripts/run_pmhc_novelty_control.py`). `observed`. Both are small; the rarity-drives-distance
pathway is measurably near-closed in this cohort. `claimed`. Consequently, if intrinsic difficulty
contributes at all to the shipped gradient, it is not explained by training-data volume here
— it would have to come from something this study does not measure, such as motif degeneracy,
binding promiscuity, or assay noise. `claimed`.

## Limits

- **Only slopes are comparable, not absolute performance.** MHCflurry 2.2.1's training data
  overlaps these test rows, so its absolute AUC0.1 is partly memorization and is not a
  no-novelty measurement of predictive skill. No table in this document, or elsewhere, sets
  MHCflurry's absolute AUC0.1 beside our arms' absolute AUC0.1; only the two fitted
  degradation slopes are compared.
- **Memorization can flatten the control's slope for reasons unrelated to novelty, not only
  inflate its accuracy.** The limit above covers memorization's effect on the control's
  absolute AUC0.1. It does not by itself cover memorization's effect on the *slope* that the
  decision rule actually consumes: a predictor that has memorized these exact measurements can
  bypass intrinsic per-allele difficulty entirely, regardless of distance, which would also
  flatten its slope. So the control's flatness is consistent both with an absence of intrinsic
  difficulty and with memorization masking it, and `novelty_effect` should be read as "the
  novelty interpretation is not contradicted," not as "novelty is established." `claimed`.
- **The distance axis is coarse.** Nearest-retained normalized Hamming distance carries only 8
  unique values across the 47 alleles, spanning 0.0294 to 0.2941 (from
  `data/pmhc/loao_allele_only_results.json`'s `diagnostics[*].nearest_retained_pseudo_distance`;
  `observed`; per the frozen pre-declaration's choice of a linear, untransformed functional
  form). Both fits are linear regressions on that 8-point support, not a smooth curve.
- **`nearest_pwm` is excluded from this comparison.** `select_nearest_pwm_source` picks its
  source by the same Hamming metric used as the x-axis, so its slope is tautologically tied to
  distance and carries no evidence either way; see the pre-declaration's exclusion clause.
- **The drift guard was never exercised against a real exclusion.** `coverage.unsupported` is
  empty, so the reference-slope drift assertion in `run_pmhc_novelty_control.py` ran on the
  same 47-allele set the pinned slope was fit on — it could only ever pass tautologically here.
  It has not yet been tested against a genuine unsupported-allele subset, which is the case it
  exists to catch. Anyone running this study against a coverage set with real exclusions should
  treat the guard as unverified for that path until it is.
- **This bounds one cohort's control comparison, not a general claim.** The result applies to
  this frozen 47-allele cohort and this reference arm; it does not extend to other predictors,
  other cohorts, or alleles outside MHCflurry's curated training set.

## Verify

```bash
uv run pytest -q tests/test_pmhc_control.py
uv run python -m compileall -q src scripts
```

Reproduce the comparison artifact with:

```bash
uv run python scripts/run_pmhc_novelty_control.py
```

`data/pmhc/mhcflurry_predictions.csv` is committed to the repository (112,128 rows, one per
cohort row). The per-allele AUC0.1 values used above are recorded in the result artifact's
`control.per_allele_auc01`, so the fit is reproducible from the artifact alone; the CSV is the
row-level record those values are computed from.
