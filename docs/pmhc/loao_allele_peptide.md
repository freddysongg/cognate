# LOAO joint allele-and-peptide novelty evaluation (#33)

**Status.** Observed run completed on 2026-09-12. The fixed experiment scored
all five compatible arms across 47 joint-exclusion partitions, one per target
allele, with target-only validation.

**Output.** `data/pmhc/loao_allele_peptide_results.json` contains the
aggregate config, per-arm macro-AUC0.1 intervals, paired comparisons, 47
per-target deletion/retention diagnostics, and the `estimand`/
`claim_boundary` labels. It contains no row-level predictions.

**Reproduce.** The observed run used:

```bash
bash scripts/fetch_pmhc_data.sh
uv run python scripts/run_pmhc_loao_allele_peptide.py
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

## Joint-exclusion schedule

For each of the 47 target alleles, `build_joint_novelty_schedule` removed
both that allele's own rows and every training row elsewhere sharing one of
the target's own test peptides, before fitting. This holds out allele and
peptide identity together rather than in isolation.

- **Retained allele count:** 46 for every target (only the target allele's
  own rows were removed at the allele level).
- **Deleted non-target training rows** (peptide overlap only): ranged
  1,320–70,740 per target, mean 30,085 — a target-varying amount, since some
  target peptides recur heavily elsewhere in the cohort and others barely at
  all.
- **Nearest retained pseudo-sequence distance** (normalized Hamming, 0–1):
  ranged 0.029–0.294, mean 0.108.

The shared all-or-fail preflight ran before any partition was scored; all 47
passed, including the deliberate peptide-leakage corruption test in the
track's test suite.

## Arms and settings

The five arms fixed by the foundation, unchanged: random, peptide-only MLP,
shuffled-pseudo-mapping MLP, per-allele nearest-PWM-source retrieval, and the
pseudo-sequence MLP. Each MLP encoding used hidden sizes 55 and 66, seeds 0
through 4 (10 fits per arm per partition), Adam at 0.001, mean-squared error
on continuous normalized affinity, a fixed allele-stratified 90/10
fit/validation split within the retained training rows, a 200-epoch cap, and
patience 20. The observed device was CPU with `torch.set_num_threads(1)`.

## Results

Intervals are 95% percentile intervals from 20,000 draws with two-level row
and allele resampling, seed 0, group `Allele`. The headline is macro
standardized AUC0.1. All 47 alleles were scorable for every arm; none were
skipped.

| Arm | Macro AUC0.1, 20,000 draws |
|---|---:|
| Random | 0.5003 [0.4980, 0.5034] |
| Peptide-only MLP | 0.5119 [0.5011, 0.5250] |
| Shuffled-mapping MLP | 0.5477 [0.5250, 0.5744] |
| Nearest PWM | 0.6144 [0.5843, 0.6475] |
| Pseudo-sequence MLP | 0.7040 [0.6761, 0.7329] |

## Paired comparisons

| Comparison | Difference, 20,000 draws |
|---|---:|
| Pseudo-sequence MLP minus peptide-only MLP | +0.1921 [+0.1660, +0.2187] |
| Pseudo-sequence MLP minus shuffled-mapping MLP | +0.1563 [+0.1244, +0.1890] |
| Pseudo-sequence MLP minus nearest PWM | +0.0897 [+0.0698, +0.1101] |

Both primary lower bounds (peptide-only and shuffled-mapping) exceed zero,
and the nearest-PWM lower bound does too. No boolean pass/fail criterion is
recorded in the artifact for this track — the plan treats the joint estimand
as descriptive rather than a single predeclared claim, per the constraint
below.

## Limits

Per the plan's global constraint, this is **not** a pure allele-transfer
result and **not** an isolated peptide-novelty effect. Removing every row
sharing a target's own test peptides changes training-set composition by a
target-varying amount (1,320–70,740 rows), confounded with both allele and
peptide identity at once. The per-target deletion accounting in
`diagnostics.targets` should be read before drawing any conclusion from the
scores above. This result bounds transfer under joint novelty for this one
frozen 47-target schedule; it does not decompose the effect into an
allele-only or peptide-only component (see the `loao_allele_only` track for
the allele-only decomposition on the same cohort).

The novelty reading behind the allele-only degradation curve this track builds on was tested
against a no-novelty control (MHCflurry 2.2.1, which has seen every cohort allele) under a
decision rule frozen before the control ran, and survived: the control is not distinguishable
from flat rather than confirmed flat, and a residual intrinsic contribution up to about 16% of
the reference slope's magnitude is not excluded. `observed`; see `novelty_control.md`.

## Verify

```bash
uv run pytest -q tests/test_pmhc_loao_allele_peptide.py tests/test_pmhc_transfer.py
uv run python -m compileall -q src scripts
```

Raw rows and row-level predictions remain local and ignored. Only the
aggregate artifact above is durable.
