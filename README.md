# cognate — binding-prediction baselines and transfer evaluation

Two separate lines of work on immunology sequence pairs, sharing a metric implementation,
retrieval operators and evidence discipline — **not** an arm set.

| Line | Task | Does it transfer to unseen targets? |
|---|---|---|
| **TCR–epitope** | peptide (~9aa) + CDR3β (~12aa) → binds? | **No.** Nothing beat chance on unseen peptides |
| **pMHC** | peptide (9aa) + HLA pseudo-sequence (34 residues) → binds? | **Yes**, and it degrades monotonically with novelty |

**Status: both lines are closed.** No implementation work is open. The successor project is
[`mimicry`](https://github.com/freddysongg/mimicry), on TCR cross-reactivity evaluation
methodology. The reasoning for stopping here is recorded in
[`docs/next_candidates.md`](docs/next_candidates.md) — a literature gate killed two of four
candidate directions outright, and the pattern was the finding: pMHC binding is a mature field,
so auditing its evaluation methods mostly establishes that they are fine.

This was a **baseline build**, not a research contribution. It exists to get hands-on contact with
protein language models and retrieval baselines on real immunology data. Read every number with
its evidence label.

**All results, with evidence labels: [`docs/findings.md`](docs/findings.md)** — written for a
reader with no prior context. Session-by-session working is in
[`docs/session_log.md`](docs/session_log.md).
The headline figure: [`docs/t6a_distance_vs_support.png`](docs/t6a_distance_vs_support.png).

## TCR–epitope results, Phase 1

Phase 1 numbers, on the original 20-peptide IMMREP23 test set. **The evaluation later moved** to a
purpose-built 88-peptide VDJdb set (48 seen / 40 unseen) because 20 peptides could not separate the
arms; see [`docs/eval_set_construction.md`](docs/eval_set_construction.md) for how that set was
built and [`docs/findings.md`](docs/findings.md), which carries a partial-supersession banner.
The conclusion did not change: retrieval wins on seen peptides, nothing beats chance on unseen.

Macro AUC0.1 on the IMMREP23 test set (per-peptide partial AUC at FPR ≤ 0.1, McClish-standardised,
averaged over peptides). 95% CIs from a two-level bootstrap over peptides and rows.

| | seen peptides (13) | unseen peptides (7) |
|---|---|---|
| random predictor | 0.511 [0.493, 0.540] | 0.504 [0.484, 0.535] |
| **k-NN baseline** (edit distance) | **0.641 [0.569, 0.724]** | 0.500 [0.500, 0.500] *(degenerate)* |
| logistic head on ESM-2 | 0.571 [0.512, 0.647] | 0.502 [0.481, 0.535] |
| MLP head on ESM-2 | 0.571 [0.518, 0.644] | 0.509 [0.484, 0.547] |

The lookup baseline beats both trained heads on seen peptides (paired bootstrap,
**Δ = +0.070 [+0.012, +0.142], p = 0.016** over 13 peptides). Everything is at chance on unseen
peptides.

## Setup

```bash
uv sync                          # Python 3.11, arm64; torch.backends.mps.is_available() -> True
./scripts/fetch_data.sh          # IMMREP23 CSVs (MIT), ~15 MB
uv run pytest                    # 337 tests
```

## Reproducing every number

Four commands regenerate the headline table from a clean checkout:

```bash
./scripts/fetch_data.sh
uv run python scripts/build_embeddings.py     # ESM-2 8M + 35M, all layers   (~15 s, 354 MB)
uv run python scripts/build_t6a_data.py       # per-peptide points for the figure
uv run python scripts/run_all.py              # -> data/headline.json
```

`run_all.py` is the reproducibility contract: **two runs produce byte-identical
`data/headline.json`, and so does a fresh source-only checkout that has never held any generated
artifact** (`observed`, verified in a clean room). Every number in the table above and in
`findings.md` §3 comes from that file.

The remaining scripts reproduce the intermediate analyses and the figure:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_inventory.ipynb
uv run python scripts/run_baseline_knn.py         # the k-NN baseline, all slices
uv run python scripts/run_layer_sweep.py          # 20 logistic fits across layers  (~4 min)
uv run python scripts/run_head_report.py          # heads vs baseline, seen/unseen
uv run python scripts/plot_t6a.py                 # the figure
uv run python scripts/make_kaggle_submissions.py  # -> data/submissions/*.csv
```

## pMHC line

A second, separately-contracted line on the public NetMHCpan training archive: peptide (9aa) + HLA
pseudo-sequence (34 residues) → binds? Frozen cohort of 47 HLA-A/B/C alleles and 112,128 nine-mer
rows, with its own source contract and hashes in `data/pmhc/source_contract.json`.

It began as a descriptive replication and was extended by a three-track leave-one-allele-out
(LOAO) transfer study, then tested against a no-novelty control. Macro standardized AUC0.1 for the
pseudo-sequence MLP, 95% percentile intervals, 20,000 draws, two-level row-and-allele resampling:

| Holdout | Pseudo-sequence MLP | Random |
|---|---:|---:|
| allele-only | 0.7487 [0.7177, 0.7801] | 0.5003 |
| joint allele-and-peptide | 0.7040 [0.6761, 0.7329] | 0.5003 |
| pseudo-sequence-cluster | 0.6218 [0.6013, 0.6429] | 0.4987 |

These three rows are mutually comparable. The in-distribution figure of 0.8283 is deliberately
**not** in the table — it comes from cross-validation rather than a holdout, so prefixing it would
compare across designs at the first step.

### The no-novelty control

Every arm above was trained under holdout, so a genuine novelty effect and mere intrinsic
per-allele difficulty predict the same curve — the design could not tell them apart. MHCflurry 2.0
has already seen every allele in the cohort, so it carries no novelty penalty:

| Arm | Slope of per-allele AUC0.1 on distance | 95% CI | r² |
|---|---:|---|---:|
| Pseudo-sequence MLP (holdout-trained) | −1.0315 | [−1.3116, −0.7514] | 0.5366 |
| MHCflurry pan BA (has seen every allele) | −0.1674 | [−0.3957, +0.0609] | 0.0439 |

Outcome `novelty_effect` under a decision rule frozen *before* the control data existed —
verifiable with `git merge-base --is-ancestor 6d4a7d6 5a5d623`. Only the slopes are comparable;
MHCflurry's training data overlaps these test rows, so its absolute accuracy is partly
memorization. Full limits, including that a small intrinsic contribution up to ~16% of the
reference slope is not excluded, are in
[`docs/pmhc/novelty_control.md`](docs/pmhc/novelty_control.md).

The unplanned finding: the distance-to-performance relationship is **predictor-dependent**,
roughly six-fold on an identical allele set, axis and metric. That bears on published per-allele
performance predictors fitted for a single model.

### pMHC documents

| Path | Holds |
|---|---|
| `docs/pmhc/experiment.md` | The binding replication |
| `docs/pmhc/loao_allele_only.md` | Allele-only holdout track |
| `docs/pmhc/loao_allele_peptide.md` | Joint allele-and-peptide track. **No defensible boolean claim by design** — read its Limits |
| `docs/pmhc/loao_cluster.md` | Pseudo-sequence-cluster track |
| `docs/pmhc/novelty_control.md` | The control above, and what it does and does not show |
| `docs/pmhc/novelty_control_predeclaration.md` | The frozen decision rule. Committed before the data |

Reproduce:

```bash
uv run python scripts/run_pmhc_loao_allele.py          # ~15-23h
uv run python scripts/run_pmhc_loao_allele_peptide.py
uv run python scripts/run_pmhc_loao_cluster.py
uv run python scripts/export_pmhc_cohort.py           # cohort to CSV for isolated consumers
uv run --no-project --python 3.11 \
  --with "mhcflurry>=2.0,<3.0" --with "tensorflow>=2.16" --with "pandas>=2.2" \
  python scripts/predict_mhcflurry.py coverage        # then: predict
uv run python scripts/run_pmhc_novelty_control.py     # seconds
```

TensorFlow is deliberately absent from `pyproject.toml`. MHCflurry runs only under an isolated
interpreter and its sole interface to this project is a CSV on disk.

## Layout

| Path | What |
|---|---|
| `src/cognate/data.py` | IMMREP23 loaders |
| `src/cognate/metrics.py` | AUROC, AUPRC, macro AUC0.1; bootstrap CIs; paired comparison; degeneracy guard |
| `src/cognate/negatives.py` | Negative samplers — `matched` (default), `shuffle`, `hard`, `uniform` |
| `src/cognate/split.py` | Leakage-free train/val splitting on bipartite components |
| `src/cognate/baseline_knn.py` | The nearest-neighbour baseline |
| `src/cognate/embed.py` | ESM-2 embedding cache, all layers, keyed by sequence |
| `src/cognate/features.py` | Feature blocks `[p, t, \|p−t\|, p⊙t]` |
| `src/cognate/train_head.py` | Logistic and MLP heads |
| `scripts/run_all.py` | Regenerates every headline number deterministically |
| `docs/findings.md` | **Every result, with evidence labels** — the document to read |
| `docs/session_log.md` | Session-by-session working, including superseded material |
| `docs/negative_sampling.md` | Negative-sampling decision and its Session 5 correction |
| `tests/test_findings.py` | Pins every number quoted in `findings.md` to the raw data |
| `src/cognate/pmhc.py` | pMHC cohort loading, source-contract verification, eligibility gate (≥100 rows per class) |
| `src/cognate/pmhc_transfer.py` | Transfer partitions, the frozen five-arm scoring contract, pseudo-sequence clustering |
| `src/cognate/pmhc_control.py` | Novelty-control analysis — per-allele AUC0.1, OLS slope fit, the frozen decision rule |
| `scripts/predict_mhcflurry.py` | Isolated MHCflurry inference. Never imported by `src/cognate` |
| `scripts/run_pmhc_novelty_control.py` | Fits both slopes, applies the frozen rule, writes the artifact |

## Five things worth knowing before reading any number

1. **13 of 20 test peptides appear in training; 7 do not.** That split (69.4% / 30.6% of rows) is
   the dataset's most important property.
2. **Macro AUC0.1 has a floor of 9/19 ≈ 0.474, not 0**, and is *exactly* invariant to any monotone
   rescaling of scores within a peptide. A pooled AUROC on this data is a different quantity and is
   contaminated by how much training data each peptide had.
3. **17.9% of test positives appear verbatim in the training file.** Dropping those rows costs the
   baseline 0.051; denying the database the verbatim answer costs 0.009. The leakage makes the
   *evaluation set* easier, it does not make the method a lookup table.
4. **A peptide-held-out split is not TCR-clean.** 201 of 8,993 training CDR3b bind more than one
   peptide, so holding out peptides still leaks ~11% of validation TCRs. Split on connected
   components of the peptide–TCR graph instead.
5. **The negative sampler is a modelling decision that can invent a shortcut.** Uniform peptide
   sampling let peptide identity alone score 0.94 pooled AUROC while macro AUC0.1 stayed at exactly
   0.500 — invisible in the headline metric for two sessions. See `docs/negative_sampling.md`.

## Scope

Not attempted, deliberately, on the TCR line: comparison against published models, reproduction of
anyone's numbers, Kaggle submission, LoRA or full fine-tuning of ESM-2, the BLOSUM62 k-mer kernel
(T4 v2).

Not attempted on the pMHC line: any new architecture, fine-tuning or hyperparameter search, an
external dataset, an antigen-presentation endpoint, MHC Class II, the NetMHCpan executable, and any
post-result change to the cohort, clustering or bootstrap. These were forbidden by the study's own
brief rather than merely skipped.

One known wart: `data/pmhc/cohort_rows.csv` is a 2.9 MB derived intermediate in version control.
It is deterministic output of `scripts/export_pmhc_cohort.py` from an already-committed archive, so
it could be regenerated instead of tracked. `data/pmhc/mhcflurry_predictions.csv` is *not*
regenerable from this repo alone — it needs the isolated environment and downloaded model files,
neither of which is pinned here — so tracking that one is deliberate.
