# Pre-declaration — pMHC novelty control

Frozen 2026-09-16, before any MHCflurry code exists in this repository. Committed alone so
that `git log` shows this commit strictly precedes the commit that runs the control. This is
the integrity device for the whole study; if the order is not visible in history, the result
is not trustworthy and should be discarded.

## Question

The shipped LOAO study reports per-allele AUC0.1 falling with pseudo-sequence distance to the
nearest retained allele. Every arm in that study was trained under holdout, so a novelty
effect and intrinsic per-allele difficulty predict the same curve. MHCflurry 2.0 has already
seen these alleles, so it carries no novelty penalty. Its gradient discriminates the two.

## Functional form, frozen

Ordinary least squares of per-allele standardized AUC0.1 on nearest-retained normalized
Hamming distance. One fit per predictor. Linear, untransformed, unweighted, no interaction
terms, no polynomial expansion. Chosen because the shipped relation is close to linear
(r = -0.733, r-squared 0.537) and because the distance axis carries only 8 unique values,
which will not support a more flexible form.

## Covariate split, frozen

- **Deployment-available**, may enter a predictive rule: nearest-retained pseudo-sequence
  distance, retained-neighbour support.
- **Null-test-only**, may enter confound tests and never a deployable rule: target `n_rows`,
  target `n_positive`.

## Reference slope, frozen

`pseudo_sequence_mlp`, from `data/pmhc/loao_allele_only_results.json`: slope -1.0315,
standard error 0.1429, 95% confidence interval [-1.3116, -0.7514], n = 47.

`nearest_pwm` is excluded from this comparison. `select_nearest_pwm_source` picks its source
by the same Hamming metric used as the x-axis, and its source distance equals the x-axis for
47 of 47 alleles, so its slope (-1.0114) is tautological and carries no evidence.

## Decision rule, frozen

Let `C` be MHCflurry's fitted slope with 95% confidence interval `[C_lo, C_hi]`, and let
`R = -1.0315` be the reference slope point estimate.

| Condition | Outcome literal | Reading |
|---|---|---|
| `C_lo <= 0 <= C_hi` and `R < C_lo` | `novelty_effect` | control is flat and distinguishable from the reference; the shipped gradient is novelty |
| `C_lo <= R <= C_hi` | `intrinsic_difficulty` | control is as steep as the reference; the shipped gradient is confounded with allele difficulty |
| neither | `mixed` | both contribute; report the decomposition and bound the shipped interpretation |

Evaluated in that order. The interval is the tolerance; no separate epsilon is used, because
an interval-based rule cannot be tuned after seeing the scatter the way a hand-picked
threshold can.

## Exclusions, frozen

Any cohort allele absent from MHCflurry's curated training alleles is not a no-novelty
control for that allele. Such alleles are identified before the regression is fitted, are
excluded from the primary fit, and are reported with their count and names. A secondary fit
including them is reported alongside, labelled as such.

## What would make this study invalid

- This commit not preceding the control run in `git log`.
- Any change to the functional form, covariate split, reference slope, or decision rule after
  MHCflurry predictions exist on disk.
- Comparing MHCflurry's absolute AUC0.1 to any of our arms. Its training data overlaps these
  test rows; only slopes are comparable.
