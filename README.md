# TCR–epitope binding baselines on IMMREP23

Given a peptide string (~9 amino acids) and a CDR3b string (~12 amino acids), predict whether the
T-cell receptor binds the peptide. String-pair binary classification, scored the way the IMMREP23
challenge scores it.

This is a **baseline build**, not a research contribution. It exists to get hands-on contact with
protein language models and retrieval baselines on real immunology data. Nothing here is novel.

**All results, with evidence labels: [`docs/findings.md`](docs/findings.md)** — written for a
reader with no prior context. Session-by-session working is in
[`docs/session_log.md`](docs/session_log.md).
The headline figure: [`docs/t6a_distance_vs_support.png`](docs/t6a_distance_vs_support.png).

## Results in one table

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
uv run pytest                    # 123 tests
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

Not attempted, deliberately: comparison against published models, reproduction of anyone's numbers,
Kaggle submission, LoRA or full fine-tuning of ESM-2, the BLOSUM62 k-mer kernel (T4 v2).
