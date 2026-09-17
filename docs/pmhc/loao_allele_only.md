# LOAO allele-only transfer evaluation (#32)

**Status.** Observed run completed on 2026-09-12. The fixed experiment held
out all 112,128 rows for each of the 47 target alleles in turn, kept every
other allele's rows in training including peptides shared with the held-out
allele's test rows, and scored the five compatible arms on each partition.

**Output.** `data/pmhc/loao_allele_only_results.json` contains the aggregate
config, per-arm macro-AUC0.1 intervals, paired comparisons, 47 per-target
peptide-overlap and nearest-retained-pseudo-sequence diagnostics, 47
per-allele point records, and the predeclared `criteria`. It contains no
row-level predictions.

**Reproduce.** The observed run used:

```bash
bash scripts/fetch_pmhc_data.sh
uv run python scripts/run_pmhc_loao_allele.py
```

This branches from the committed LOAO foundation (`fa86a76`) and reuses its
partition, preflight, scoring, and evaluation primitives unchanged.

## Source and contract

Same public NetMHCpan training archive as the pMHC binding replication and
the LOAO foundation, verified against the same recorded hashes:

- URL: `https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz`
- archive SHA-256: `06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4`

The verified, eligible cohort was unchanged from the pMHC binding
replication: 47 alleles, 112,128 rows, 28,538 positives, 83,590 negatives.

## Allele-only schedule and diagnostics

For each of the 47 target alleles, `build_allele_only_schedule` held out only
that allele's own rows; every other allele's rows stayed in training,
including rows whose peptide also appears in the held-out allele's test set.
This isolates direct allele transfer without the joint peptide exclusion the
allele-and-peptide track applies.

- **Retained allele count:** 46 for every target.
- **Training rows:** ranged 102,605–111,856 per target, mean 109,742.
- **Peptide overlap (`diagnostics.<allele>`):** every one of the 47 targets
  had at least one test peptide already present in training (0 of 47 had
  zero overlap). The overlapping-peptide fraction of each target's unique
  test peptides ranged 0.61–0.99, mean 0.92 — most held-out alleles' peptides
  were also measured against other alleles in this cohort. This overlap is
  reported, not removed; it is what separates this track's estimand from the
  allele-and-peptide track's zero-overlap-by-construction design.
- **Nearest retained pseudo-sequence distance** (normalized Hamming, 0–1):
  ranged 0.029–0.294, mean 0.108, identical to the selected nearest-PWM
  source's distance for every target — the nearest retained allele by
  pseudo-sequence distance always had both target classes and was therefore
  also PWM-eligible. 31 distinct alleles were used as a nearest-PWM source
  across the 47 targets.

The shared all-or-fail preflight ran before any partition was scored; all 47
passed, including the deliberate target-allele-leakage corruption test in the
track's test suite.

## Arms and settings

The five arms fixed by the foundation, unchanged: random, peptide-only MLP,
shuffled-pseudo-mapping MLP, per-allele nearest-PWM-source retrieval, and the
pseudo-sequence MLP. Each MLP encoding used hidden sizes 55 and 66, seeds 0
through 4 (10 fits per arm per partition), Adam at 0.001, mean-squared error
on continuous normalized affinity, a fixed classification-target-stratified
90/10 fit/validation split within the retained training rows, a 200-epoch
cap, and patience 20. The observed device was CPU with
`torch.set_num_threads(1)`.

## Results

Intervals are 95% percentile intervals from 20,000 draws with two-level row
and allele resampling, seed 0, group `Allele`. The headline is macro
standardized AUC0.1. All 47 alleles were scorable for every arm; none were
skipped.

| Arm | Macro AUC0.1, 20,000 draws |
|---|---:|
| Random | 0.5003 [0.4980, 0.5034] |
| Peptide-only MLP | 0.6349 [0.6084, 0.6649] |
| Shuffled-mapping MLP | 0.5633 [0.5336, 0.5969] |
| Nearest PWM | 0.6672 [0.6349, 0.7009] |
| Pseudo-sequence MLP | 0.7487 [0.7177, 0.7801] |

## Paired comparisons and criteria

| Comparison | Difference, 20,000 draws |
|---|---:|
| Pseudo-sequence MLP minus peptide-only MLP | +0.1138 [+0.0891, +0.1393] |
| Pseudo-sequence MLP minus shuffled-mapping MLP | +0.1854 [+0.1504, +0.2204] |
| Pseudo-sequence MLP minus nearest PWM (secondary) | +0.0815 [+0.0656, +0.0974] |

`criteria.primary_criterion_met: true` — both required paired lower bounds
(pseudo-sequence MLP minus peptide-only MLP, and minus shuffled-mapping MLP)
exceed zero (0.089 and 0.150 respectively). Per the frozen contract, this
declares a correct-mapping result for the allele-only estimand. The
nearest-PWM comparison is reported as a secondary, descriptive
sequence-neighbour baseline, not a criterion input; its lower bound also
exceeds zero here (0.066) but does not add to the primary declaration.

## Limits

This is direct allele-transfer evaluation with training peptide overlap
present and reported, not removed. The mean 92% peptide-overlap fraction
means most held-out alleles' peptides were also seen (against other alleles)
during training, so this result does not isolate allele transfer from
peptide familiarity the way the allele-and-peptide track's zero-overlap
design does; see `loao_allele_peptide.md` for that joint-novelty
decomposition on the same cohort. The nearest-PWM arm is a sequence-neighbour
baseline, not a non-transfer control. Each target allele is unseen only to
its own model; this result does not extend to alleles outside this frozen
47-allele cohort or to poorly characterized alleles in general.

The novelty reading behind this degradation curve was tested against a no-novelty control
(MHCflurry 2.2.1, which has seen every cohort allele) under a decision rule frozen before the
control ran, and survived: the control is not distinguishable from flat rather than confirmed
flat, and a residual intrinsic contribution up to about 16% of the reference slope's magnitude
is not excluded. `observed`; see `novelty_control.md`.

## Verify

```bash
uv run pytest -q tests/test_pmhc_loao_allele.py tests/test_pmhc_transfer.py
uv run python -m compileall -q src scripts
```

Raw rows and row-level predictions remain local and ignored. Only the
aggregate artifact above is durable.
