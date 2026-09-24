# LOAO pseudo-sequence-cluster transfer evaluation (#34)

**Status.** Observed run completed on 2026-09-12. The fixed experiment held out
11 pseudo-sequence clusters in turn, one holdout schedule covering all 47
target alleles, and scored the five compatible arms on each held-out cluster.

**Output.** `data/pmhc/loao_cluster_results.json` contains the aggregate
config, per-arm macro-AUC0.1 intervals, paired comparisons, cluster
diagnostics (exact membership, cut rule, per-cluster retention/class-support
counts, nearest-retained-pseudo-sequence distance), 47 per-allele point
records, and the predeclared criteria. It contains no row-level predictions.

**Reproduce.** The observed run used:

```bash
bash scripts/fetch_pmhc_data.sh
uv run python scripts/run_pmhc_loao_cluster.py
```

This branches from the committed LOAO foundation (`fa86a76`) and reuses its
partition, preflight, scoring, and evaluation primitives unchanged.

## Source and contract

Same public NetMHCpan training archive as the pMHC binding replication and
the LOAO foundation, verified against the same recorded hashes:

- URL: `https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz`
- archive SHA-256: `06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4`

The verified, eligible cohort (nine-residue peptides, classical HLA-A/B/C,
at least 100 positive and 100 negative rows per allele) was unchanged from
the pMHC binding replication: 47 alleles, 112,128 rows, 28,538 positives,
83,590 negatives.

## Cluster construction and group schedule

`build_pseudo_sequence_clusters` grouped the 47 frozen 34-residue pseudo-
sequences by complete-linkage agglomeration over normalized Hamming distance,
with sorted-allele tie-breaking, never using target labels or arm outcomes.

- **Cut rule:** smallest integer Hamming threshold that leaves at most one
  singleton cluster under complete linkage.
- **Realized cut:** Hamming distance 12.
- **Result:** 11 clusters, sizes `[1, 2, 2, 3, 3, 3, 5, 6, 6, 8, 8]` — one
  singleton (`HLA-C15:02`), matching the frozen cohort's expected partition.
- **Nearest-retained pseudo-sequence distance** across the 47 held-out
  alleles ranged 5–10 residues (mean 7.7) — every target had at least one
  same-cluster neighbour excluded alongside it, and at least one retained
  allele outside its cluster within that range.

Each of the 11 group-holdout partitions excluded one whole cluster from
training; per-partition retained-allele counts, train/test row counts, and
class support are recorded per cluster in `diagnostics.cluster_composition`
(e.g. the `HLA-A01:01`/`HLA-A80:01` cluster retained 45 alleles, 106,956
training rows, 27,851/79,105 train positive/negative, 5,172 test rows). The
shared all-or-fail preflight ran before any partition was scored; all 11
passed.

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
standardized AUC0.1.

| Arm | Macro AUC0.1, 20,000 draws |
|---|---:|
| Random | 0.4987 [0.4968, 0.5011] |
| Shuffled-mapping MLP | 0.5184 [0.5045, 0.5352] |
| Peptide-only MLP | 0.5613 [0.5505, 0.5730] |
| Nearest PWM | 0.5481 [0.5324, 0.5655] |
| Pseudo-sequence MLP | 0.6218 [0.6013, 0.6429] |

All 47 alleles were scorable for every arm; none were skipped.

## Paired comparisons and criteria

| Comparison | Difference, 20,000 draws |
|---|---:|
| Pseudo-sequence MLP minus peptide-only MLP | +0.0605 [+0.0436, +0.0778] |
| Pseudo-sequence MLP minus shuffled-mapping MLP | +0.1034 [+0.0827, +0.1253] |
| Pseudo-sequence MLP minus nearest PWM (secondary) | +0.0737 [+0.0538, +0.0941] |

`cluster_transfer_supported: true` — the primary criterion (both the
peptide-only and shuffled-mapping lower bounds exceed zero) held. The
nearest-PWM comparison is reported as a secondary, descriptive
sequence-neighbour baseline, not a criterion input.

## Limits

This is cluster-held-out transfer under one frozen pseudo-sequence grouping,
not a causal claim that sequence distance alone causes any performance
difference. Holding out a whole cluster changes training-set composition
(retained allele count and class support) at the same time as distance, so
the two are confounded by construction. The nearest-PWM comparison is a
sequence-neighbour baseline, not a non-transfer control. This result does not
establish a general transfer bound for arbitrary held-out alleles, only for
this one frozen 11-group partition of the 47-allele cohort.

The novelty reading behind the allele-only degradation curve this track relates to was tested
against MHCflurry 2.2.1 under a decision rule frozen before the control ran. Its models support
every cohort allele, but direct training-allele overlap was not verified. The rule returned
`novelty_effect`; this interpretation depends on the training-overlap assumption. The control
is not distinguishable from flat rather than confirmed flat, and a residual intrinsic
contribution up to about 16% of the reference slope's magnitude is not excluded. `observed`
for the rule outcome; see `novelty_control.md`.

## Verify

```bash
uv run pytest -q tests/test_pmhc_loao_cluster.py tests/test_pmhc_transfer.py
uv run python -m compileall -q src scripts
```

Raw rows and row-level predictions remain local and ignored. Only the
aggregate artifact above is durable.
