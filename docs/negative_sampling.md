# T3 — Negative sampling strategy

`VDJdb_paired_chain.csv` is positives-only: 11,312 rows, every one `Target == 1`.
Negatives are therefore a modelling decision, and the decision changes the numbers.

## Default for T5 onwards

**`matched`, `ratio=5.0`, `seed=0`.** This replaced `shuffle` in Session 5. See the correction
below — the original default carried a shortcut that let a model score 0.94 global AUROC without
learning anything about binding.

Two reasons, in order of weight:

1. **It matches how the test negatives were built, on both axes.** The IMMREP23 organisers built
   test negatives by swapping TCRs *among the peptides in the set* at Levenshtein distance > 3 from
   the true partner, five per positive — so every peptide keeps the same 5:1 ratio in both classes
   and the peptide marginal is identical for positives and negatives. `matched` reproduces that by
   drawing the negative peptide proportional to its positive frequency. `shuffle` reproduces only
   the distance half.
2. **`ratio=5.0` matches the test set's 1:5 positive:negative ratio.** The test set is 17.2%
   positive by construction. AUPRC has a baseline equal to the positive rate, so training at
   `ratio=1.0` and evaluating at 1:5 would make train and test AUPRC numbers incomparable and
   invite a misread. AUROC and AUC0.1 are ratio-invariant, but AUPRC is not, and it is one of the
   three metrics being reported.

`shuffle` and `hard` remain available as ablations.

## Correction (Session 5) — why `shuffle` was replaced

`shuffle` draws the negative peptide **uniformly** over the 808 training peptides, but the
positives are heavily skewed. The result is a peptide marginal that differs enormously between the
two classes:

| Peptide | share of positives | share of `shuffle` negatives | ratio |
|---|---:|---:|---:|
| GILGFVFTL | 16.07% | 0.10% | **159×** |
| RAKFKQLL | 9.42% | 0.10% | 99× |
| KLGGALQAK | 8.06% | 0.11% | 75× |

**Peptide identity alone then predicts the label at 0.9396 global AUROC** on a `shuffle` training
set. A model can score extremely well by memorising which peptides are frequent, never learning
anything about TCR-peptide compatibility. Measured, not argued.

Two things made this invisible for two sessions:

- **Macro AUC0.1 is exactly 0.5000 for the shortcut**, because the metric is computed per peptide
  and peptide identity is constant inside a group. The headline metric was immune, so nothing
  looked wrong until a head was trained and its *train* AUROC came back at 0.970 against a macro
  AUC0.1 of 0.621.
- The original justification for `shuffle` checked the **Levenshtein distance** profile, which is
  genuinely fine (0.13% of draws within distance 3), and stopped there. The distribution of *which*
  peptides get drawn was never checked.

`matched` cuts the shortcut from **0.9396 to 0.5671** global AUROC. The residual is structural and
not removable: a TCR that binds `GILGFVFTL` can never draw `GILGFVFTL` as its own negative, so the
most frequent peptides stay slightly under-represented in the negative class (1.6× rather than 1.0×).

Effect on a trained logistic head (35M embeddings, peptide-and-TCR-disjoint validation):

| negatives | train AUROC | train macro | val macro | test macro | test seen |
|---|---|---|---|---|---|
| `shuffle` | 0.970 | 0.621 | 0.509 | 0.528 | 0.532 |
| `matched` | 0.747 | 0.592 | 0.532 | **0.551** | **0.578** |

Train AUROC falls because the shortcut is gone; every generalisation number rises.

## The two strategies

| | `shuffle` | `matched` | `hard` |
|---|---|---|---|
| Rule | uniform random non-cognate peptide | non-cognate peptide drawn proportional to positive frequency | the *k* peptides nearest the true partner by Levenshtein distance |
| Peptide-identity-only AUROC | **0.9396** | **0.5671** | — |
| Seeded | yes | yes | no — deterministic, ties break alphabetically |
| Median distance to true peptide | 8 | 8 | 5 |
| Draws within distance 3 | 0.13% | 0.4% | 17.7% |
| Peptides used (of 808) | 808 | 808 | 741 |
| Wall clock, 56,560 negatives | 0.2 s | 0.5 s | 0.6 s |

What they pick for a TCR whose true partner is `GILGFVFTL`:

```
shuffle -> TMETIDWKV, LPPSYTNSF, RFPLTFGWCF, WLTYHGAIK, NTNSSPDDQIGYY
hard    -> GILEFVFTL, GILGLVFTL, ILGFVFTLT,  ALLLQLFTL, ALYGFVPVL
```

`hard` is picking single-substitution and frame-shifted variants of the true epitope.

## Why they should give different results — the two sentences

`shuffle` produces negatives that are trivially separable from positives on peptide identity
alone, so a model can score well by memorising which peptides appear in which role rather than
learning anything about TCR–peptide compatibility. `hard` removes that shortcut by making the
negative peptide nearly identical to the positive one, which forces the model onto the actual
binding signal — and, for the same reason, manufactures far more false negatives, because a TCR
that binds `GILGFVFTL` has a real chance of also binding `GILGLVFTL`.

## Caveats worth carrying forward

- **`hard` is deliberately anti-matched to this benchmark.** Its negatives sit in exactly the
  distance band the organisers *excluded* from the test set. A model trained on `hard` negatives is
  optimised for a decision boundary the IMMREP23 test set never probes, so expect it to score
  *worse* here while plausibly being the better model. Do not read a `hard` regression as evidence
  that hard negatives are a bad idea.
- **`hard` is weaker than its name suggests on this peptide pool.** Only 10.2% of its draws are
  within distance 1 of the true peptide; the median is 5. With 808 peptides, most simply have no
  close relative in the set, so `hard` degrades toward `shuffle` for the long tail.
- **`hard` concentrates.** Its top-5 peptides are 17.1% of all negatives and 67 peptides are never
  drawn at all, versus a near-uniform spread for `shuffle`. That skew is a second confound on top
  of the distance effect.
- **False negatives are unavoidable in both.** Cognate exclusion is keyed on `CDR3b` across the
  whole positives table, not just the row, because 201 of 8,993 training CDR3b bind more than one
  peptide (max 12). That removes *known* false negatives. It cannot remove unrecorded ones, and
  VDJdb is nowhere near a complete record of what any TCR binds.

## Interface

```python
from cognate.negatives import make_negatives, make_training_set

negatives = make_negatives(positives_df, strategy="matched", ratio=5.0, seed=0)
combined  = make_training_set(positives_df, strategy="matched", ratio=5.0, seed=0)
```

Returned rows carry the same columns as the input with `Target == 0`, and keep the **index of the
positive row they were derived from** so a negative is traceable without adding a provenance column
that could leak into features. A swapped peptide brings its HLA with it.

`uniform` is also registered — every distinct TCR against every peptide, labelled by membership in
the positive set. It is guarded at 2M pairs because the full cross product on this training set is
7.3M.

## Reproducibility

`make_negatives(train, s, ratio=5.0, seed=0)` returns a frame equal to itself across calls for both
`shuffle` and `hard` (`observed`, asserted in `tests/test_negatives.py` and re-checked against the
real 11,312-row training set). `hard` ignores the seed except for the fractional part of `ratio`.
