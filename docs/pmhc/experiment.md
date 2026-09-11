# pMHC binding replication experiment

**Status.** Observed run completed on 2026-09-11. The fixed experiment scored all
seven arms on 112,128 out-of-fold examples across 47 alleles.

**Output.** `data/pmhc/results.json` contains aggregate intervals, 47 per-allele
point records, paired comparisons, negative-source slices, retrieval diagnostics,
and predeclared criteria. It contains no row-level predictions or embeddings.

**Reproduce.** The observed run used:

```bash
bash scripts/fetch_pmhc_data.sh
uv run python scripts/run_pmhc.py
```

The fetch step retrieves only the public training archive. The NetMHCpan
executable was not run or redistributed.

## Source and contract

The data are the binding-affinity partitions from the public NetMHCpan training
archive:

- URL: `https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz`
- retrieval date recorded in the contract: 2026-09-10
- archive SHA-256: `06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4`

The runner verified the archive and every extracted input before scoring:

| File | SHA-256 |
|---|---|
| `c000_ba` | `a5704007e127c8c0e2a3b0043c52ad2f92be5efd01bd2b6a0f7e5424d25d06ef` |
| `c001_ba` | `754c5fecf1c8387b48fc8cf77473fc14c9e89adb2694ac977e51e7028b0304f1` |
| `c002_ba` | `f1ff916e3bd4ed01352b7fa4c6a1852fd9294104f6b344c5dd2eba7961a57aef` |
| `c003_ba` | `9cde2dd51abf1cde03383c5f8120e88243c7b6f486f341a7566bb4ad6f60eb69` |
| `c004_ba` | `a2a28d2565fb8a3e6c76e1f7d2be5b1a3aca28acbabe13c3db4e2063c404f2e5` |
| `MHC_pseudo.dat` | `f46d95dee821db6c139d6cee6f6bf72328468c1d8f6e9741daf21562752ad39a` |

The verified source contained 208,093 binding-affinity rows across all species,
170,107 HLA-A/B/C rows, and 126,375 nine-residue HLA-A/B/C rows. The latter
comprised 30,869 positives and 95,506 negatives before the minimum-support
filter.

## Cohort and folds

The analysis retained nine-residue peptides for classical HLA-A, HLA-B, and
HLA-C alleles with at least 100 positive and 100 negative rows. Affinity strictly
greater than `0.426` defined a positive for classification metrics. Exact affinity
`0.01` identified artificial negatives; other non-positive rows were measured
non-binders.

| Quantity | Verified count |
|---|---:|
| Supplied BA folds | 5 |
| Eligible alleles | 47 |
| Rows | 112,128 |
| Positives | 28,538 |
| Negatives | 83,590 |
| Measured non-binders | 82,448 |
| Artificial negatives | 1,142 |

The five supplied folds were `c000_ba` through `c004_ba`. The runner verified
zero peptide overlap across folds before scoring, fit each arm on four folds, and
filled predictions only for the held-out fold.

## Arms and settings

The seven arms were fixed before the result was inspected:

1. random scores from seed 0;
2. logistic regression on peptide length and the 20 amino-acid counts;
3. per-allele nearest-positive normalized Levenshtein retrieval;
4. the same retrieval operator using frozen ESM-2 35M layer 10, residue-mean
   pooling, and cosine similarity;
5. a per-allele nine-position positive-versus-negative PWM with pseudocount one;
6. a pan-allele MLP using raw BLOSUM50 peptide and 34-residue pseudo-sequence
   features; and
7. the same MLP with allele one-hot input as a seen-allele control.

Each MLP encoding used hidden sizes 55 and 66, seeds 0 through 4, and all five
outer folds: 50 fits per encoding and 100 fits total. Training used Adam at
`0.001`, mean-squared error on continuous normalized affinity, a fixed
allele-stratified 90/10 fit/validation split, a 200-epoch cap, and patience 20.
The observed device was Apple MPS. The ESM cache was accepted only after its
checkpoint name, layer availability, and exact 22,055-peptide set matched.

## Results

Intervals are 95% percentile intervals from 20,000 draws with two-level row and
allele resampling, seed 0. The headline is macro standardized AUC0.1 by allele.
Pooled AUROC and AUPRC use the same bootstrap draws; macro AUROC is a point-only
diagnostic.

| Arm | Macro AUC0.1, 20,000 draws | Pooled AUROC, 20,000 draws | Macro AUROC | Pooled AUPRC, 20,000 draws | Predictive |
|---|---:|---:|---:|---:|---|
| Random | 0.4998 [0.4978, 0.5023] | 0.4929 [0.4880, 0.4979] | 0.4944 | 0.2519 [0.2189, 0.2839] | no |
| Length/composition | 0.5154 [0.5091, 0.5227] | 0.6312 [0.5933, 0.6640] | 0.6033 | 0.3555 [0.2849, 0.4221] | yes |
| Edit retrieval | 0.6088 [0.5964, 0.6231] | 0.7847 [0.7689, 0.7990] | 0.7754 | 0.5100 [0.4697, 0.5425] | yes |
| ESM-2 retrieval | 0.5660 [0.5542, 0.5807] | 0.6769 [0.6606, 0.6908] | 0.6642 | 0.4484 [0.4068, 0.4822] | yes |
| PWM | 0.7443 [0.7231, 0.7664] | 0.9233 [0.9141, 0.9309] | 0.9107 | 0.7855 [0.7503, 0.8157] | yes |
| Pseudo-sequence MLP | 0.8283 [0.8108, 0.8456] | 0.9493 [0.9423, 0.9549] | 0.9421 | 0.8735 [0.8516, 0.8904] | yes |
| One-hot MLP control | 0.8252 [0.8079, 0.8424] | 0.9475 [0.9403, 0.9533] | 0.9402 | 0.8704 [0.8472, 0.8884] | yes |

All 47 alleles were scorable for every arm, no allele was skipped, and no arm
was degenerate. Under the predeclared strict rule, all six non-random arms were
predictive because their macro-AUC0.1 lower bounds exceeded 0.5. Random was not.

## Paired comparisons and criteria

Each interval below is a paired macro-AUC0.1 difference over the same 47 alleles
and 20,000 draws.

| Comparison | Difference, 20,000 draws |
|---|---:|
| Length/composition minus random | +0.0156 [+0.0087, +0.0230] |
| Edit retrieval minus random | +0.1090 [+0.0967, +0.1227] |
| ESM-2 retrieval minus random | +0.0663 [+0.0542, +0.0804] |
| PWM minus random | +0.2445 [+0.2235, +0.2661] |
| Pseudo-sequence MLP minus random | +0.3285 [+0.3114, +0.3450] |
| One-hot MLP control minus random | +0.3254 [+0.3085, +0.3418] |
| Pseudo-sequence MLP minus PWM | +0.0841 [+0.0688, +0.1008] |

PWM was predictive, but it was not sufficient relative to the pseudo-sequence
MLP. The upper bound for `MLP - PWM` was 0.1008, above the predeclared 0.02
margin.

## Negative-source analysis

The measured-nonbinder slice combined all 28,538 positives with 82,448 measured
non-binders, for 110,986 rows. The artificial-negative slice combined the same
positives with 1,142 artificial negatives, for 29,680 rows. These are separate
fixed-cohort evaluations, not resampled negative-ratio curves.

| Arm | Positive + measured non-binder macro AUC0.1 | Positive + artificial negative macro AUC0.1 |
|---|---:|---:|
| Random | 0.4998 [0.4979, 0.5023] | 0.5113 [0.5090, 0.5333] |
| Length/composition | 0.5153 [0.5090, 0.5225] | 0.5441 [0.5422, 0.5861] |
| Edit retrieval | 0.6072 [0.5948, 0.6212] | 0.7052 [0.6901, 0.7442] |
| ESM-2 retrieval | 0.5651 [0.5533, 0.5795] | 0.6344 [0.6253, 0.6801] |
| PWM | 0.7434 [0.7219, 0.7657] | 0.8089 [0.7905, 0.8590] |
| Pseudo-sequence MLP | 0.8271 [0.8091, 0.8453] | 0.8968 [0.8776, 0.9333] |
| One-hot MLP control | 0.8242 [0.8064, 0.8422] | 0.8967 [0.8765, 0.9337] |

Both retrieval arms remained above the observed random point after artificial
negatives were removed: edit uplift was +0.1074 and ESM-2 uplift was +0.0653 on
the measured-nonbinder slice. On the artificial-negative slice, the descriptive
uplifts were larger at +0.1939 and +0.1231. No paired edit-minus-ESM interval was
estimated for either source slice.

Among negative rows, the nearest-positive edit distance was 5.272 residues on
average (median 5) for measured non-binders and 5.689 residues on average (median
6) for artificial negatives. Neither source contained an exact peptide duplicate
of a training-fold positive. The easier artificial-negative slice and its larger
distance show that the magnitude is construction-sensitive even though the
retrieval ordering persists without artificial negatives.

## TCR/pMHC retrieval contrast

The closed TCR points were read from their existing artifacts, not recalculated.
TCR edit retrieval was 0.5654 versus observed random 0.5009, an uplift of +0.0645;
TCR ESM-2 retrieval was 0.5358, an uplift of +0.0349. In this pMHC run, edit
retrieval was 0.6088 versus observed random 0.4998, an uplift of +0.1090; ESM-2
retrieval was 0.5660, an uplift of +0.0663.

These are within-task point contrasts shown side by side. They are not one paired
cross-task estimand, and no cross-task interval or causal interpretation is
licensed.

## Limits

This is a descriptive result for one public training release, its supplied
common-motif folds, the retained 47 seen alleles, and the normalized binding-
affinity endpoint. It is not a test of antigen presentation, immunogenicity,
unseen-allele transfer, publication novelty, or biological cause. Similar
pseudo-sequence and one-hot MLP points under seen-allele folds do not establish
that MHC sequence features are unnecessary; this design cannot test their value
for allele transfer.

## Verify

```bash
uv run pytest -q
uv run python -m compileall -q src scripts
git diff --check
git ls-files data/pmhc/raw data/pmhc/derived shared/pmhc
```

Raw rows, row-level predictions, and the ESM cache remain local and ignored. Only
aggregate results are durable.
