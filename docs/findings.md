# TCR–epitope binding: what we built and what we found

A record of a six-session baseline build on the IMMREP23 benchmark, written for someone who
was not there. Nothing here is novel; the point was to get real numbers on real immunology
data and understand what they mean.

Every claim carries an evidence label:

| Label | Means |
|---|---|
| `observed` | A check ran and its output was seen. Reproducible from this repo. |
| `failure-proven` | The check was also shown able to fail — red first, then green. |
| `analytic` | Follows from the definition of the metric, not from a measurement. |
| `claimed` | Read off code or a README; no run behind it. |

Session-by-session working is preserved in [`session_log.md`](session_log.md). This document
supersedes it for anything the two disagree on.

---

## 1. What the task is

**Two strings in, one bit out.**

Given a peptide (a short protein fragment, ~9 letters) and a CDR3b (the ~12-letter tip of a
T-cell receptor), predict whether they bind: 1 or 0. That is the whole task. It is string-pair
binary classification; no structure, no 3D, no chemistry beyond what the sequence implies.

### Glossary

| Term | Plain meaning |
|---|---|
| **Protein** | A chain of beads; 20 bead types (amino acids), each written as a letter. A protein is a string. |
| **Peptide / epitope** | A short protein fragment, 8–15 letters. The bit an immune cell reacts to. |
| **MHC / HLA** | A display shelf on a cell surface that holds peptide fragments up for inspection. HLA is the human name. |
| **T cell** | The immune cell that inspects those shelves. |
| **TCR** | T-cell receptor — the T cell's scanner. Also a string. |
| **CDR3** | The ~12-letter tip of the scanner that does the actual binding. `CDR3a`/`CDR3b` are the alpha and beta chains. Usually the only part modelled. |
| **Binding** | The label. Does this TCR recognise this peptide: 1 or 0. |
| **Seen vs unseen peptide** | Whether a test peptide appeared in training. The central generalisation question in this field. |
| **Cognate** | The peptide a given TCR actually binds. |
| **Repertoire** | The full set of TCRs a person carries (~10⁷ distinct). |

### The data

| | Rows | Peptides | Notes |
|---|---:|---:|---|
| `VDJdb_paired_chain.csv` (train) | 11,312 | 808 | **Positives only.** Every row is `Target == 1`. |
| `solutions.csv` (test) | 3,484 | 20 | 598 positive / 2,886 negative = **17.2% positive** |

Source: `github.com/justin-barton/IMMREP23`, MIT licensed. The test negatives were built by the
organisers by swapping each positive TCR onto peptides at Levenshtein distance > 3 from its true
partner, five negatives per positive.

**The single most important property: 13 of the 20 test peptides appear in training, 7 do not**
(69.4% / 30.6% of test rows). `observed`.

### The metric

**Macro AUC0.1**: partial ROC AUC truncated at 10% false-positive rate, McClish-standardised,
computed **independently per peptide** and then averaged over peptides. In scikit-learn terms,
`roc_auc_score(..., max_fpr=0.1)` per peptide group, then the mean.

The per-peptide grouping is the part that matters and the part people get wrong. A global AUC0.1
on the pooled test set is a different number and is not what this benchmark reports.

---

## 2. What was built

Four components, roughly 1,400 lines of Python plus 122 tests.

| Component | What it does |
|---|---|
| `metrics.py` | AUROC, AUPRC, macro AUC0.1. Two-level bootstrap CIs (resampling peptides *and* rows). A **paired** bootstrap for comparing two models on the same peptides. A **degeneracy guard** that flags score columns which cannot express a ranking. |
| `baseline_knn.py` | The lookup baseline. For a query TCR, score = highest normalised edit-distance similarity to any training positive **for that peptide**. No learning. |
| `embed.py` + `features.py` + `train_head.py` | ESM-2 (8M and 35M) embeddings for every unique sequence, all layers cached; logistic and MLP heads over `[p, t, \|p−t\|, p⊙t]` features. |
| `negatives.py` + `split.py` | Four negative samplers behind one interface; leakage-free train/validation splitting on connected components of the peptide–TCR graph. |

### Three engineering decisions that were not obvious

**Deduplicate before embedding.** The dataset has 10,369 distinct strings against 142,712 naive
rows — a 13.8× reduction. Embedding both ESM-2 models over every layer takes **17.6 seconds
total** on an M1 Pro (5.7 s for 8M, 11.9 s for 35M; 354 MB on disk). `observed`. Compute was
never the constraint on this project.

**`add_pooling_layer=False`.** The published ESM-2 checkpoints contain no pooler, so loading
`EsmModel` without this flag *randomly initialises* `pooler.dense` and `pooler_output` is noise.
Mean-pool over residues instead, dropping BOS and EOS. `observed` — the load report lists
`pooler.dense.weight MISSING` without the flag and does not with it.

**A peptide-held-out split is not TCR-clean.** 201 of 8,993 training CDR3b bind more than one
peptide, so holding out peptides still leaks a mean **10.9% of validation rows** (max 26.8%)
whose TCR also appears in training. Splitting on connected components of the bipartite
peptide–TCR graph gives 0 shared peptides and 0 shared CDR3b at a 20% split. `observed`.

---

## 3. What was found

> **Superseded in part by B3 (2026-09-07).** Everything in this section is correct *for the
> IMMREP23 test set*, which is what it describes. It is no longer the project's best estimate of
> the underlying effects: the same analyses rerun on 88 VDJdb peptides with every IMMREP23 training
> TCR removed hold ① ④ ⑤, weaken ② badly (r² 0.770 → 0.129), cut ③ to r² 0.052, and show the
> trained head landing 0.010 above a random predictor. Read this section with
> [`eval_set_construction.md` §8](eval_set_construction.md) and the verdicts in
> [`belief_list.md`](belief_list.md).


### 3a. Established — clears its bootstrap CI

**① The lookup baseline beats the ESM-2 head on seen peptides.**

| Model | seen peptides (13) | unseen peptides (7) |
|---|---|---|
| random predictor | 0.511 [0.493, 0.540] | 0.504 [0.484, 0.535] |
| **k-NN, edit distance** | **0.641 [0.569, 0.724]** | 0.500 [0.500, 0.500] *(degenerate)* |
| ESM-2 + logistic head | 0.571 [0.512, 0.647] | 0.502 [0.481, 0.535] |
| ESM-2 + MLP head | 0.571 [0.518, 0.644] | 0.509 [0.484, 0.547] |

Paired bootstrap on the same 13 peptides: **Δ = +0.070 [+0.012, +0.142], p = 0.016**. `observed`.

A ten-line nearest-neighbour lookup beats a protein language model with a trained head, on the
half of the problem where either works at all.

**② The baseline's performance is explained by distance to the training set.**
Per-peptide macro AUC0.1 against the median normalised edit distance from that peptide's binders
to the nearest training CDR3b: **r = −0.878, r² = 0.770**, CI [−0.965, −0.494]. Survives dropping
the extreme point (−0.844) and the top two by support (−0.851). `observed`.

This is the strongest and most robust result in the project.

**③ The head's performance is explained by training support, which does *not* explain the
baseline's.** Per-peptide score against log₁₀(training positives for that peptide):

| | r | r² | 95% CI | |
|---|---:|---:|---|---|
| k-NN | +0.109 | 0.012 | [−0.598, +0.735] | refuted |
| head | **+0.674** | 0.455 | [+0.355, +0.887] | robust (+0.57 to +0.70 under leave-one-out) |

The two models are limited by different things: the lookup by how far the query is from anything
it has seen, the head by how much data each peptide had. The two predictors are largely
independent (corr = −0.354, CI spans zero), so this is not one finding stated twice. `observed`.

**④ The baseline's database saturates.** Per-peptide **mean raw score over all rows** rises almost
perfectly with database size: **r = +0.990**, CI [+0.981, +0.996]. Restricted to *negatives* the
correlation is **+0.983** (point estimate only — no bootstrap CI was computed for that slice); for
positives it is +0.697. A max-over-database operator has no scale correction — with 1,818
candidates almost any query finds a close match. `observed`.

The consequence is about metrics rather than about the baseline: **any pooled metric on this data
is partly measuring training-set size.** That is the concrete mechanism behind the invariance
result in §3b — the level shift §3b proves macro AUC0.1 cannot see is this one.

**⑤ Both models are at chance on unseen peptides.** k-NN cannot even produce a ranking there (its
database is empty, so all 1,066 rows take one constant score); the head produces 1,019 distinct
scores and still lands at 0.502. `observed`.

### 3b. Analytic — true by definition, not by measurement

**Macro AUC0.1 is exactly invariant to any monotone within-peptide rescaling.** Because it is
computed per peptide and AUC depends only on ranking, z-scoring or rank-normalising scores inside
each peptide leaves every per-peptide value untouched — verified to `abs=1e-12`. The same
transform moves pooled AUROC by **+0.035**. `analytic`, and `failure-proven` in tests.

Consequence: **pooled metrics on this data are contaminated by how much training data each peptide
had; the macro metric is not.** This is why the two disagree throughout, and it is the reason the
Session 5 bug stayed hidden (§4).

**McClish-standardised AUC0.1 has a floor of 9/19 ≈ 0.474, not 0.** A model that ranks every
positive below the top 10% of negatives scores 0.474. The entire below-random range is 0.026 wide,
so the metric is nearly one-sided and "0.48" is catastrophic rather than slightly-below-chance.
`analytic`.

### 3c. Not established — point estimate only

| Claim | Point estimate | Why it does not stand |
|---|---|---|
| Head performance tracks distance to training set | r = −0.735 | Collapses to −0.341 when one peptide (`GILGFVFTL`) is removed. Leverage-driven. |
| Head inflates raw score level with support | r ≈ +0.51 | 95% CI [−0.12, +0.90] spans zero at n = 13. |
| Some ESM-2 layer is better than others | spread 0.028 across 20 fits | Below the ~0.05 noise floor, and validation-to-test Spearman is **−0.035** — validation selection has no predictive relationship with test performance. |

The widely-repeated expectation that a middle layer of a protein language model beats the last
layer holds *numerically* here (35M layer 9: 0.553 vs layer 12: 0.547) and is an order of
magnitude below noise. It is not evidence.

### 3d. One nuance worth keeping

**17.9% of test positives (107 of 598) appear verbatim as `(Peptide, CDR3b)` pairs in the training
file.** Two ablations separate what that means:

- Dropping those rows from evaluation costs the baseline **0.051 [0.005, 0.107]** — they were easy rows.
- Denying the database the verbatim answer while keeping the rows costs **0.009 [−0.026, 0.043]**,
  indistinguishable from zero — the second-nearest neighbour ranks them nearly as well.

So the leakage inflates the score by making the *evaluation set* easier, not by letting the method
do table lookup. Reporting "17.9% of the score is lookup" would have been wrong. `observed`.

---

## 4. The Session 5 bug — the most instructive thing here

### What it was

`VDJdb_paired_chain.csv` is positives-only, so negatives have to be generated. The first sampler,
`shuffle`, paired each TCR with a **uniformly random** non-cognate peptide.

But the positives are heavily skewed. The result was a peptide marginal that differed enormously
between the two classes:

| Peptide | share of positives | share of `shuffle` negatives | ratio |
|---|---:|---:|---:|
| `GILGFVFTL` | 16.07% | 0.10% | **159×** |
| `RAKFKQLL` | 9.42% | 0.10% | 99× |
| `KLGGALQAK` | 8.06% | 0.11% | 75× |

**Peptide identity alone predicts the label at 0.9396 pooled AUROC** on that training set. A model
could score extremely well by memorising which peptides are frequent, learning nothing whatsoever
about TCR–peptide compatibility. `observed`.

### Why the metric hid it

**Peptide identity alone scores exactly 0.5000 macro AUC0.1.** The metric is computed per peptide,
and peptide identity is constant inside a peptide group, so the shortcut is *invisible* to the
headline number — by the same invariance property (§3b) that makes macro AUC0.1 the right metric.

The property that made the metric correct is the property that hid the bug. Every reported macro
AUC0.1 was unaffected; the sampler was still wrong, and every trained head was wasting its capacity
on the shortcut.

### Why it survived two sessions

The original justification for `shuffle` checked the **Levenshtein distance** profile of the drawn
peptides — only 0.13% of draws fell within distance 3 of the true peptide, so it matched the
organisers' construction on that axis — and stopped there. **The distribution of *which* peptides
were drawn was never checked.** The check that was run passed; the check that mattered was never
written.

### How it was caught

By asking a question that had nothing to do with the headline metric: **can the head fit its own
training data?**

```
train pooled AUROC 0.970    train macro AUC0.1 0.621
```

A model that fits the pooled task nearly perfectly while barely beating chance on the per-peptide
task is not learning the task. That gap is a shortcut signature, and it is only visible if you look
at two metrics that disagree.

### The fix and its effect

`matched` draws the negative peptide proportional to its positive frequency, reproducing what the
organisers did (TCRs swapped *among* the peptides in the set, 5:1 preserved per peptide). The
shortcut falls from **0.9396 to 0.5671** pooled AUROC. The residual is structural and not
removable: a TCR that binds `GILGFVFTL` can never draw `GILGFVFTL` as its own negative.

| negatives | train AUROC | val macro | test macro | test seen |
|---|---|---|---|---|
| `shuffle` | 0.970 | 0.509 | 0.528 | 0.532 |
| `matched` | 0.747 | 0.532 | 0.551 | **0.578** |

Train AUROC falls because the shortcut is gone; every generalisation number rises.

And the gain was **not uniform**. It was concentrated entirely in the three peptides closest to the
training set — `GILGFVFTL` +0.337, `GLCTLVAML` +0.234, `NLVPMVATV` +0.119 — with all 17 others in
[−0.067, +0.037]. Removing the shortcut did not make the head broadly better; it let the head use
near-neighbour structure it had been ignoring, and only where that structure exists. `observed`.

### The standing procedure that came out of it

Any new sampler, dataset, or feature set is now probed **before** it is scored:

1. Peptide-features-only classifier → should be near 0.5 pooled AUROC
2. TCR-features-only classifier → should be near 0.5 pooled AUROC
3. Train-AUROC vs train-macro-AUC0.1 gap → a large gap is a shortcut signature

---

## 5. Limits

**The binding constraint is n = 13 and n = 20 peptides.** Not compute, not model size, not
features. Every correlation in §3 rests on 13 or 20 points, and leave-one-out already killed one of
them (head-vs-distance) as leverage-driven. Three findings clear their bootstrap CI; everything
else is a point estimate with an interval spanning zero, and is labelled that way.

**One dataset.** A single train/test split of a single benchmark, constructed by one group with one
negative-generation choice. Nothing here has been checked against a second evaluation set.

**One model family.** ESM-2 at 8M and 35M parameters, mean-pooled. 650M was not run. No
fine-tuning — the head sits on frozen embeddings. It is entirely possible that a fine-tuned ESM-2
beats the lookup, and this project provides no evidence either way.

**The embeddings are severely anisotropic.** Mean pairwise cosine similarity between CDR3b
embeddings is 0.94–0.98 with an effective rank of 20–33 against 320/480 nominal dimensions. The
head may be limited by the representation rather than by the head. `observed`.

**The metric implementation is self-verified only.** Macro AUC0.1 was implemented here from the
competition README's prose spec. It agrees with hand-computed values on a 10-row toy example and
those tests are `failure-proven`, but no independent implementation has ever scored these
predictions. External verification was attempted via the IMMREP23 Kaggle scorer on 2026-09-07 and
**could not be obtained**: every submission fails with `Scoring session had non-zero exit code` —
including Kaggle's own unmodified `sample_submission.csv`, which the README states must score 0.5.
Three formats were tried (the 18-column format per the README, a minimal `ID,Prediction` file, and
Kaggle's reference file itself) with identical results, after rules acceptance and with working
authentication. The scoring container for this closed 2023 competition is broken on Kaggle's side.
`observed`.

For the record, the internally computed predictions were, on the split Kaggle would have scored:

| submission | internal Private-split macro AUC0.1 |
|---|---|
| k-NN, edit distance | 0.5922 |
| ESM-2 + logistic head | 0.5474 |

**Nothing published exists to compare the unseen column against.** A deep-research pass
([outlook.md](outlook.md) Part C) retrieved **no unseen-epitope result reported in macro AUC0.1
anywhere in the literature**, so our unseen numbers are uncomparable to published work by
construction rather than by choice. Separately, **no published evaluation of ESM-C or ESM-3 on TCR
specificity, CDR3 representation, or short-peptide representation was retrieved** (outlook.md B4),
so "a newer protein language model would do better" is untested rather than refuted. Both are
absences of retrieved evidence, not demonstrated absences.

**The deduplication standard is a moving external convention, not a property of this repository.**
Every number here is computed after exact CDR3β removal, which was the weakest of the three
published 2025–2026 standards. See §11 instance 4 and [outlook.md](outlook.md) Part D.

**Two structural caveats on the test set:**
- 17.9% of test positives appear verbatim in training (§3d).
- 8 test rows labelled 0 are positives in the training file — unwinnable by anything that trusts
  the training labels, and a hard ceiling on the achievable score. Left in place deliberately.

**If you want one reason to doubt the headline:** the claim "a lookup beats a protein language
model" rests on a paired comparison over **13 peptides** with p = 0.016 — one peptide short of the
smallest sample most people would accept, on a benchmark where the lookup gets 17.9% of its
positives handed to it verbatim.

---

## 6. Reproducing this

```bash
uv sync && ./scripts/fetch_data.sh
uv run python scripts/build_embeddings.py
uv run python scripts/build_t6a_data.py
uv run python scripts/run_all.py        # writes data/headline.json
```

`run_all.py` regenerates every number in §3 deterministically; two runs produce byte-identical
output. Full command list and layout in [`README.md`](../README.md). Negative-sampling decisions
and the Session 5 correction in [`negative_sampling.md`](negative_sampling.md). Session-by-session
working, including material superseded by later sessions, in [`session_log.md`](session_log.md).

---

## 7. Phase D — two forks, and what learned representations bought

**Run 2026-09-07 on the frozen n = 88 evaluation set.** Two independent forks were built from one
shared foundation, in separate worktrees off commit `d7a6275`, with the comparison below
pre-registered before either ran. Their merge-base is exactly that commit: neither fork saw the
other's code. Results: [`fork1_results.json`](../data/fork1_results.json),
[`fork2_results.json`](../data/fork2_results.json),
[`knn_esm_cosine.json`](../data/knn_esm_cosine.json).

### 7a. The table

**Deduplication standard: exact CDR3β match against the IMMREP23 training set, and nothing
stricter.** Every number in this table is regime-dependent and none of them is a fixed constant.
Under the strictest published criterion (Liao et al., three CDR3β substitutions) the baseline is
**0.5302 [0.5175, 0.5486]**, not 0.5654. The full curve is in [outlook.md Part D](outlook.md) and
the CD-HIT row is in [cdhit_and_issues.md](cdhit_and_issues.md).

| Model | seen (48) | unseen (40) |
|---|---|---|
| random predictor | 0.5009 [0.498, 0.505] | 0.5015 |
| **k-NN, edit distance** | **0.5654 [0.546, 0.585]** *(exact-match regime; 0.5302 at sub 3)* | 0.5000 *(degenerate)* |
| logistic head, mean-pooled | 0.5107 [0.502, 0.523] | 0.4997 |
| MLP head | 0.5114 [0.504, 0.521] | 0.4991 |
| ESM-2 cosine k-NN, 35M layer 6 | 0.5434 [0.529, 0.559] | 0.5000 *(degenerate)* |
| ESM-2 cosine k-NN, 35M layer 10 | 0.5358 [0.523, 0.550] | 0.5000 *(degenerate)* |
| fork 1 · 1a, TCR metric learning | 0.5320 [0.519, 0.545] | 0.5000 *(degenerate)* |
| fork 1 · 1b, two-tower alignment | 0.5075 [0.502, 0.517] | 0.5018 |
| fork 2 · 2a, cross-attention | 0.5039 [0.499, 0.510] | 0.4985 |
| fork 2 · 2b, mean-pool control | 0.5093 [0.501, 0.522] | 0.5022 |

Every learned variant is below the baseline, and every paired CI excludes zero:

| Comparison | Δ | p |
|---|---|---|
| 1a − edit k-NN | −0.034 [−0.048, −0.023] | < 0.001 |
| 1b − edit k-NN | −0.057 [−0.077, −0.037] | < 0.001 |
| 2a − edit k-NN | −0.061 [−0.081, −0.043] | < 0.001 |
| 2b − edit k-NN | −0.056 [−0.075, −0.039] | < 0.001 |
| **2a − 2b (attention ablation)** | **−0.005 [−0.016, +0.005]** | **0.366** |
| 1a − 2b (best of each fork) | +0.022 [+0.012, +0.032] | < 0.001 |

### 7b. Phase 0 — removing the representation/algorithm confound

Phase 1 compared an edit-distance k-NN against an ESM-2 head, so algorithm and representation
differed at once. Holding the algorithm fixed and swapping only the similarity function settles
it: **ESM-2 cosine loses to edit distance at every model size and depth tested** — 35M and 8M,
last and middle layers, five configurations, all five paired CIs excluding zero.

The sharpest detail is a resolution argument that runs the wrong way. Edit distance produces
**122 distinct scores** across 73,440 seen rows; ESM-2 cosine produces **60,000–67,000**. The far
coarser measure ranks better, so this is not a tie-handling or degeneracy artifact.

Middle layers beat last layers at both model sizes, independently reproducing the Session 4 norm
collapse: measured mean L2 norm is **7.20** at layer 12 against **87.74** at layer 10.

### 7c. Fork 1 — the objective was not the bottleneck

The premise was that binary classification forces negatives the data does not contain, and that a
contrastive objective would dissolve the problem. It did dissolve it — no negative sampler appears
anywhere in Fork 1's training — and it bought nothing.

**1a is indistinguishable from not training at all.** An *untrained* random projection of the same
embeddings scores 0.5341; the trained projection scores 0.5320, against frozen cosine at the same
layer at 0.5358. A learning-curve sweep peaks at 0.5380 around epoch 4 and decays to 0.5257 by
epoch 40. Even an oracle selecting the best epoch by test score stays far below 0.5654 (the
exact-match-regime baseline; 0.5302 at sub 3). With the
backbone frozen, a projection can only reweight dimensions the pooled embedding already has; it
cannot recreate what pooling discarded.

**1b learns, and what it learns does not generalise.** Seen improves +0.023 over an untrained
model across 40 epochs while unseen stays at chance and validation loss rises monotonically the
whole time. Validation peptides are disjoint by construction and 48 of the 88 evaluation "seen"
peptides were trained on, so the gain is peptide-specific memorisation. The component-split
validation loss detects that and early stopping declines to bank it.

**1b is nonetheless the only construction here with a mechanism on unseen peptides.** It produces
~43,190 distinct scores where the k-NN produces exactly 1. That distinction is worth keeping: the
k-NN *structurally cannot* score an unseen peptide because its database is empty, whereas 1b can
and the answer it gives is chance. A missing mechanism and a measured absence of signal are
different findings.

### 7d. Fork 2 — attention recovers nothing that pooling destroyed

The premise was that mean-pooling ~15 residue vectors destroys exactly the contact information
binding depends on. The control was built first and reproduced its target: **2b scores 0.5093
against the 0.511 reference**, which is what makes the ablation readable at all.

**2a − 2b = −0.005 [−0.016, +0.005], p = 0.366.** Cross-attention makes no measurable difference.
The direction is consistent across all three seeds, but the magnitude sits well inside the paired
CI, so seed-consistency does not upgrade it to an effect.

One diagnostic detail: attention lowered validation *loss* at every seed while lowering validation
*macro AUC0.1*. The extra capacity fit the binary objective without improving the per-peptide
ordering the metric measures.

2b is not merely similar to the Phase 1 head — masked mean-pooling commutes with a linear
projection (`mean(Wx + b) == W·mean(x) + b`), so 2b **is** that head with a learned input
projection in front. That is why it lands on 0.511 rather than near it by luck.

### 7e. What this adds up to

Five independent approaches now sit between 0.504 and 0.544 while counting letter differences
gets 0.5654 in the exact-match regime, 0.5302 at sub 3. The pre-registered reading for this
outcome was written before any of them ran:
*consistent with the k-NN's advantage being intrinsic to the operator, not the representation.*

The stronger statement the evidence supports is about the representation rather than the
operator. Mean-pooled ESM-2 on short TCR sequences is a worse similarity measure than raw
sequence identity, and neither a learned metric over it nor attention beneath it recovers the
difference. Session 4's effective rank of 20–33 against 480 nominal dimensions predicted this.

**No method in this project has scored above chance on unseen peptides.** Every unseen CI in the
table above contains or abuts 0.5.

### 7f. What would make this wrong

- **One hyperparameter configuration per fork.** No sweep was run. The claim is that these
  configurations fail and that the frozen-backbone argument explains why the ceiling is low, not
  that contrastive learning or cross-attention cannot work here.
- **The backbone was frozen throughout.** LoRA or full fine-tuning is untested and is the single
  most likely thing to change the answer.
- **The component split leaves only 275 of 808 peptides in Fork 1's training arm.** That is a real
  constraint on what any objective could learn, imposed by the frozen splitter.
- **CDR3β only.** The alpha chain is present in the evaluation set and unused.
- Cross-fork comparisons in §7a use a group-level paired bootstrap over peptides, because only
  per-peptide AUC0.1 was persisted. It reproduces the frozen `compare_macro_auc01` point estimates
  exactly and its intervals to within 0.002 on every comparison where both were run.

## 8. The operator diagnostic — the 2×2's operator axis was measuring scope

Phase D closed with a 2×2 read as *representation × operator*:

| | retrieval | parametric head |
|---|---|---|
| edit distance | **0.5654** *(exact-match regime; 0.5302 at sub 3)* | *empty* |
| ESM-2 (35M L10) | 0.5358 | 0.5107 |

The literature check found the representation half already published — Nagano et al.,
*Cell Systems* 16(1):101165 (2025), highlight ①: *"Existing language models underperform sequence
alignment for predicting TCR specificity."* Their Fig. S6 ran the closest thing to our operator
axis and got **the opposite sign**: a linear SVC fitted on ESM-2 features beat ESM-2
nearest-neighbour. Ours lost. That disagreement, plus the fact that our operator effect was a
difference of two point estimates with no paired interval, is what this diagnostic exists to
resolve. `scripts/run_operator_diagnostic.py`, `data/operator_diagnostic.json`.

### 8a. The missing interval

**esm_retrieval − logistic_global = +0.0250 [+0.0116, +0.0411], p < 0.001, n = 48.** `observed`.

The interval clears zero. The operator effect, as the 2×2 defined it, is real — so the first
possible reading (that it was noise) is refused outright.

### 8b. It is not an operator effect

The 2×2's operator axis changed two things at once, and only one of them is the operator:

* **estimator** — a max-over-similarities rule versus a fitted linear model.
* **scope** — retrieval fits *one model per peptide* (`score_by_nearest_positive` builds a
  separate database for each). The head fits **one global model across all peptides**.

Nothing in the original design separated these. Two new arms do:

| arm | what it changes | seen macro AUC0.1 |
|---|---|---|
| `edit_retrieval` | — | 0.5654 [0.5461, 0.5849] *(exact-match regime)* |
| **`svc_per_peptide`** | per-peptide linear SVC on ESM-2 (SCEPTR-matched) | **0.5424 [0.5301, 0.5575]** |
| `esm_retrieval` | — | 0.5358 [0.5230, 0.5498] |
| `logistic_global` | — | 0.5107 [0.5021, 0.5228] |
| `svc_global` | SVC estimator, still global | 0.5078 [0.5002, 0.5179] |

The +0.0250 decomposes **exactly**:

| term | comparison | Δ | reading |
|---|---|---|---|
| scope | `svc_per_peptide − svc_global` | **+0.0346 [+0.0223, +0.0482]**, p < 0.001 | the whole effect |
| estimator | `svc_global − logistic_global` | −0.0029 [−0.0054, −0.0009], p = 0.006 | negligible |
| operator | `svc_per_peptide − esm_retrieval` | +0.0067 [−0.0006, +0.0142], p = 0.076 | **spans zero** |

−0.0067 + 0.0346 − 0.0029 = +0.0250. `observed`.

Per-peptide win counts agree with the intervals: `svc_per_peptide` beats `esm_retrieval` on
**26 of 48** peptides (a coin flip, as the spanning CI implies), beats `svc_global` on **37 of
48**, and loses to `edit_retrieval` on **40 of 48**.

**Given the same per-peptide scope retrieval always had, the parametric operator closes the entire
gap** and lands statistically indistinguishable from ESM-2 retrieval, with the point estimate on
SCEPTR's side of zero rather than ours. The operator axis of the 2×2 dissolves; what it was
measuring was scope.

Against the pre-registered outcomes: the row-1 condition (*SVC beats ESM-2 retrieval, CI clears
zero*) is **not** met — p = 0.076, the interval touches zero, so the narrow question "does a
parametric operator beat retrieval here" lands in **row 3, underpowered, no side picked**. The
diagnostic's actual question — *was the head the problem* — is answered by the scope row, and the
answer is yes. Seed spread is 0.0010 over three seeds, so this is not seed noise.

### 8c. What survives — rewritten at closeout

> **This section previously read "What survives, and is now stronger". That title is withdrawn.**
> It was written before Phase F measured the effect's dependence on the deduplication convention
> and before the red team ([redteam.md](redteam.md) §0b, [redteam_curve.md](redteam_curve.md) §2,
> §4). "Stronger" was never supported by anything the diagnostic ran; it described the *number of
> estimators agreeing*, which is not a measure of strength. The supported version follows.

**What survives.** In the exact-match deduplication regime, on the 48 seen peptides, edit-distance
retrieval outscores mean-pooled ESM-2 35M layer-10 retrieval by
**−0.0296 [−0.0415, −0.0188]**, p < 0.001. This run reproduced the frozen Phase D value to the
digit against `data/knn_esm_cosine.json`. `observed`.

It also outscores the per-peptide parametric arm: `svc_per_peptide − edit_retrieval` =
**−0.0230 [−0.0347, −0.0122]**, p < 0.001. So the ordering does not depend on which of the two
estimators is used. `observed`.

**Four qualifications, each of which the original section lacked.**

1. **Regime-dependent, and the dependence is the story.** −0.0296 is the exact-match number. Under
   Liao et al.'s three-substitution standard the same comparison on the same 48 peptides is
   **−0.0155 [−0.0329, −0.0012]**, p = 0.024 at the saved 1,000 draws and **0.0439** at 20,000
   synchronised draws. It still clears zero; it clears it barely. The full curve is
   [outlook.md](outlook.md) Part D.
2. **The attenuation between those two numbers is not established.** The magnitude falls 47.7%,
   but the paired interval on that attenuation is **[−0.1%, 98.2%], p = 0.0511** — indistinguishable
   from no attenuation and from complete attenuation alike
   ([cdhit_and_issues.md](cdhit_and_issues.md) Part 2). "Roughly halves" is a point estimate with
   no support and is withdrawn wherever it appears.
3. **"Under two operators" is retired as a description.** `svc_per_peptide` standardises features
   per dimension while `esm_retrieval` L2-normalises, so the contrast changes normalisation as
   well as estimator ([redteam.md](redteam.md) §0b). The arithmetic above stands; the word
   "operator" does not, and §8e's framing goes with it.
4. **Sensitive to peptide composition and to inference choices.** Leave-one-peptide-out at sub 3:
   11 of 48 omissions give an interval crossing zero, worst case dropping `RFPLTFGWCF` →
   −0.0113 [−0.0247, +0.0020], p = 0.108. Bonferroni across the five unique looks in the sweep puts
   the sub-3 p at 0.120 ([redteam_curve.md](redteam_curve.md) §4).

The `edit distance × parametric` cell stays empty and is not fillable by this route — an edit
distance has no feature vector to fit a linear model on. SCEPTR has the same hole for the same
reason: their SVC was fitted on PLM embeddings only, never on TCRdist or CDR3 Levenshtein.

### 8d. Probes and degeneracy

All three shortcut probes pass the two-sided |macro − 0.5| ≤ 0.05 gate:

| probe | macro AUC0.1 | deviation |
|---|---|---|
| `svc_global` peptide_only | 0.5001 | 0.0001 |
| `svc_global` tcr_only | 0.5000 | 0.0000 |
| `svc_per_peptide` shifted reference set | 0.5053 | 0.0053 |

The third is the new one: rotating each peptide onto a *different* peptide's binder set collapses
the per-peptide SVC to chance, which is what makes 0.5424 attributable to the reference set rather
than to anything structural in the eval rows.

`svc_per_peptide` is **degenerate on unseen peptides** (1 distinct score, 40 constant groups) for
the same structural reason retrieval is: no training binders means no model to fit. Its 0.500 is
arithmetic, not measurement. Note that the *global* arms are non-degenerate on unseen and still
score 0.4995–0.5107 — a mechanism that produces 43,000 distinct scores and lands at chance is a
different finding from no mechanism at all.

### 8e. What this changes

- The 2×2 should no longer be presented as representation × operator. Its operator axis was
  confounded with scope, and with scope controlled the operator term does not clear zero.
- The apparent disagreement with Nagano et al. Fig. S6 was our design, not their result. It is
  **dissolved rather than resolved in their favour**: their SVC advantage is scoped to k = 1–200
  reference TCRs against a fixed background, which this run does not occupy, so +0.0067 neither
  reproduces nor contradicts them (§8f).
- The project's remaining claim is the representation effect, which is a **replication** of a
  published finding on a broader evaluation (48 seen peptides, paired bootstrap) than the original
  (6 pMHCs, binomial test), now shown to be operator-independent.
- **No arm scores above chance on unseen peptides.** That is unchanged and unchallenged.

### 8f. What would make this wrong

- **`svc_per_peptide` uses the full training database as its positive set**, matching what
  retrieval queries, whereas the global arms use the component split's training arm. Scope and
  training-set size are therefore not fully separated; a per-peptide arm restricted to the split
  would separate them.
- **Background negatives come from other peptides' binders**, because the project has no unlabelled
  repertoire. SCEPTR sampled from one. This is the `matched`-negative assumption again.
- **One hyperparameter setting** (`C = 1.0`, 1000 background TCRs, balanced class weights). No
  sweep was run.
- **The +0.0067 is out of SCEPTR's regime, and therefore neither reproduces nor contradicts
  them.** Their SVC advantage is scoped explicitly to the low-data case — *"in the low data
  regime typical of most pMHCs, misalignment of pre-training to downstream tasks can only be
  partially remediated by training on reference TCRs"* — and their reference sets run k = 1–200
  TCRs per pMHC. `svc_per_peptide` fits on each peptide's **full** training database. Whether a
  parametric operator beats retrieval at k = 1–200 is a different question from whether it does
  at full support, and this run answers only the second. Agreement in sign is not a replication
  and disagreement would not have been a refutation.
- **The regime gap is narrower than that framing implies, and the framing should be read with
  these numbers.** Training support for the 48 seen peptides: min 1, median 25.5, max 1,818,
  quartiles [4, 25, 231]. **35 of 48 (73%) sit at or below SCEPTR's k = 200 ceiling**; 30 of 48
  are below 50 and 20 of 48 below 20. So most peptides entering the macro average are *inside*
  their sampled range by support count. What is still out of regime is the design, not the
  support: SCEPTR *controlled* k as an independent variable against a fixed background set,
  while this takes whatever support each peptide has and resamples the background per peptide.
  "Out of regime" is a statement about a controlled sweep versus an uncontrolled one, and it is
  weaker than the raw support numbers would let anyone assume. `observed`.
- Deviations from SCEPTR's setup are recorded under `deviations_from_sceptr` in
  `data/operator_diagnostic.json`. Their benchmark was 6 pMHCs by AUROC over reference sets of
  1–200 paired-chain TCRs; this is 48 peptides by macro AUC0.1 over full databases, CDR3β only.
  The comparison is directional, not a replication. Two deviations were missing from the list as
  first written and are added in §10a: SCEPTR used one shared background set where this
  resamples per peptide, and normalisation is not held fixed across the operator contrast.

## 9. Retired — closed, not deferred

Both entries below are **closed**. They are not parked, not blocked on capacity, and not
waiting for a better idea. Nothing in a future session should revive them on the grounds that
they were merely postponed.

### 9a. The empty 2×2 cell — a head on edit-distance-derived features

**Closed.** It existed to complete a *representation × operator* grid. §8 shows that grid's
operator axis was measuring scope, not operator: once a parametric arm is given the same
per-peptide scope retrieval has, the operator term is +0.0067 [−0.0006, +0.0142] and does not
clear zero. Filling the cell would estimate an interaction on an axis that has no established
main effect, so the number it produced would not answer any question that is still open.

The secondary reason stands on its own: an edit distance has no feature vector, so any such
head would score a *derived* representation — kernel embedding, distance-to-landmarks, k-NN
features — which is a third representation, not the edit distance. The cell as specified is not
constructible. SCEPTR has the same hole for the same reason; their SVC was fitted on PLM
embeddings only, never on TCRdist or CDR3 Levenshtein.

### 9b. The 2×2 as a novel contribution

**Closed.** Superseded twice over.

The representation half is a replication: Nagano et al., *Cell Systems* 16(1):101165 (2025),
highlight ① — *"Existing language models underperform sequence alignment for predicting TCR
specificity"* — holds nearest-neighbour fixed and swaps only the similarity function across six
representations including ESM-2, on VDJdb, with the same sign. Also stated at abstract level by
IMMREP22 (Meysman et al. 2023) and in TITAN's own abstract (Weber et al. 2021). Our baseline is
itself a published method: `baseline_knn.py:4` documents it as adapted from IMMREP23's TCRbase.

The operator half is not a finding but a **confound**, and is retained only as a methodology
note in §8: a per-peptide retrieval rule compared against a globally fitted head charges the
operator axis for a scope change. That note is the useful residue. It is not a contribution
about TCR binding.

What the project still has is a replication on a broader evaluation than the original — 48 seen
peptides with a paired bootstrap, against 6 pMHCs with a binomial test — now shown to hold under
two estimators rather than one. That is worth stating accurately and is not worth extending.

> **Closeout amendment.** "Two operators" is corrected to "two estimators": the second arm changes
> normalisation as well as estimator ([redteam.md](redteam.md) §0b), so the contrast was never a
> clean operator swap. And the replication is regime-dependent — it holds at exact-match dedup and
> at Liao's three-substitution standard, the latter at p = 0.024 (0.0439 at 20,000 draws), with
> 11 of 48 leave-one-peptide-out intervals crossing zero. See §8c as rewritten.

## 10. Erratum — `frozen_esm_cosine_seen_layer10`

> **RESOLVED PERMANENTLY at closeout, 2026-09-09. This erratum stands; the artifacts will not be
> regenerated or edited.** Machine-readable record: [`data/errata.json`](../data/errata.json).
> Pinned by `tests/test_reference_constants.py::test_committed_fork_artifacts_still_carry_the_erratum`,
> so the divergence cannot widen silently and 0.5364 cannot later be mistaken for a live number.
>
> **Why not regenerated.** Regenerating either JSON means re-running a fork driver, which trains
> models — forbidden at closeout, and it would also change the `results` block, which is correct.
> Hand-editing the scalar in place would leave a results artifact that is the output of no script,
> which is precisely the restatement failure mode §11 exists to document. The value is inert:
> nothing reads it, and no delta, interval or conclusion is computed from it. Both drivers carry
> the correct 0.5358 as of the closeout commit, and that is what the next run would write.
>
> Closes issues #10 and #18.

**Field:** `reference.frozen_esm_cosine_seen_layer10`
**Files:** `data/fork1_results.json:18`, `data/fork2_results.json:18`
**Reported:** 0.5364 · **Correct:** **0.5358**

Both fork result artifacts report the frozen ESM-2 cosine k-NN at 35M layer 10, seen slice, as
0.5364. The correct value is 0.5358 — `data/knn_esm_cosine.json`, key
`35M layer10 (headline)/seen`, which has held 0.5358 at every commit it has existed, and which
`data/operator_diagnostic.json` (`esm_retrieval/seen`) independently reproduced. Nothing has
ever computed 0.5364.

It was a hand-typed literal in `scripts/run_fork1.py` and `scripts/run_fork2.py`, entering at
`8c7a03d` / `16a050c`. The other three constants in the same block — `edit_knn_seen` 0.5654,
`logistic_head_seen` 0.5107, `random_seen` 0.5009 — match their sources exactly, which is what
kept the fourth invisible.

**Nothing downstream moves.** The `reference` block is context only; every fork delta, interval
and p-value in §7 is computed from score vectors, not from these constants. The error is 0.0006,
inside the CI [0.5230, 0.5498].

**The two JSONs were left uncorrected, deliberately.** Regenerating them means retraining six
models (two forks × three seeds), which is outside the audit's read-only scope; hand-editing
them would make them the output of no script and destroy the only thing a result artifact is
for. The drivers are fixed, so the next genuine run writes 0.5358 and this erratum retires with
it. Until then the artifacts are wrong in this one field and this section is the correction.

`tests/test_reference_constants.py` now parses the `reference` dict literal out of each driver
and asserts every constant equals its source artifact. `failure-proven`: run against
`git show HEAD:scripts/run_fork{1,2}.py` it reports
`{'frozen_esm_cosine_seen_layer10': (0.5364, 0.5358)}` for both, and passes after the fix.

### Corrections owed to the strategy handoff

The handoff is not a repository artifact, so these cannot be applied here and are recorded for
whoever writes the next one:

1. **2×2, ESM-2 retrieval cell: 0.5434 → 0.5358.** 0.5434 is `35M middle` (layer 6) in
   `data/knn_esm_cosine.json`; the head it was being compared against is layer 10. As written,
   the handoff's operator comparison crossed two layers and did not hold representation fixed.
   The layer-10 value is 0.5358.
2. **Cache environment variable: `TCRBENCH_CACHE_DIR` → `COGNATE_CACHE_DIR`**
   (`src/cognate/embed.py:36`). The former appears nowhere in the repository and is read by
   nothing.
3. **The 2×2 should not be described as representation × operator at all** (§8e).

### 10a. Erratum — `deviations_from_sceptr` was incomplete

> **RESOLVED PERMANENTLY at closeout, 2026-09-09, on the same terms as §10.** The committed
> `data/operator_diagnostic.json` keeps the five-entry list; the full seven are below and are the
> authoritative version. Regenerating the artifact would mean re-fitting 48 SVCs, and editing it
> by hand would make it the output of no script. Unlike §10 this one cannot be pinned by a test:
> the erratum is a claim of *completeness*, and no artifact holds the true list. Treat the list as
> `claimed`, always. §11 instance 3.

**Field:** `deviations_from_sceptr` · **File:** `data/operator_diagnostic.json`
**Reported:** 5 entries · **Correct:** **7**

The list is an assertion of completeness, and it was wrong. Two departures were missing:

6. **Background set.** SCEPTR used *one shared* 1000-TCR background set across every pMHC,
   stated in their methods §III.4 as being for consistency. `fit_per_peptide_svc` draws a fresh
   sample per peptide from a shared RNG, adding per-peptide noise SCEPTR deliberately removed.
   Seed spread is 0.0010, so the effect looks small, but it was undeclared.
7. **Normalisation.** `svc_per_peptide` standardises features per dimension; `esm_retrieval`
   L2-normalises. The operator contrast therefore carries a normalisation change as well as an
   operator change. Residual direction unknown. Internal to this project, not a SCEPTR
   departure — it belongs in the list because the list is what a reader checks the contrast
   against.

Both were found by reading the SCEPTR methods and the audit table in `docs/redteam.md` §0b,
not by any check. Nothing computed changes: the list is documentation, and no score, interval or
p-value reads it.

**The JSON was left uncorrected, same treatment and same reason as §10.** Regenerating it means
re-running the diagnostic; hand-editing makes it the output of no script. `deviations_from_sceptr`
in `scripts/run_operator_diagnostic.py` now carries all seven, so the next run writes them and
this erratum retires with it.

## 11. Methodology note — restatement drift

Three errors in this project share one mechanism, and it is not arithmetic. In each case a value
established in one place was **restated by hand** somewhere else, and nothing compared the copy
to the original. All three were found by accident or by a sweep looking for something else.

| # | restated value | copy | source | how found |
|---|---|---|---|---|
| 1 | ESM-2 retrieval cell of the 2×2 | 0.5434 (layer 6) | 0.5358 (layer 10) | noticed while setting up the operator diagnostic |
| 2 | `frozen_esm_cosine_seen_layer10` | 0.5364, in two fork drivers and their JSONs | 0.5358 in `knn_esm_cosine.json` | contract sweep for instances of #1 |
| 3 | `deviations_from_sceptr` | 5 entries | 7 | reading the SCEPTR methods for a different question |
| 4 | "the 48 seen peptides" | a fixed evaluation set | whatever the current external dedup standard says it is | Part A/D dedup recomputation |
| 5 | "frozen ESM-2 cosine at 0.544", in issue #6 | 0.544 (8M layer 3) | 0.5358 (35M layer 10) | issue audit at closeout |

**Instance 4** is recorded in full in [belief_list.md](belief_list.md) Phase F. **Instance 5** is
recorded in [cdhit_and_issues.md](cdhit_and_issues.md) §3c ② and was fixed in the issue body at
closeout; it is the same shape as #1 — a labelled row collapsed into an unlabelled target that
dropped the discriminator making it correct — and it is the first instance found in a tracker
rather than in code or a document.

#1 is the sharpest of the three because **the source was not ambiguous**. §7a lists both rows,
each labelled with its layer — `ESM-2 cosine k-NN, 35M layer 6` at 0.5434 and
`ESM-2 cosine k-NN, 35M layer 10` at 0.5358. The error happened in the act of collapsing two
labelled rows into one unlabelled 2×2 cell. Nothing about the source needed fixing; the copy
dropped the discriminator that made the source correct.

None was caught by a test, a CI, or a review of the number itself. Each copy was individually
plausible: #1 is a real number from the same artifact, #2 is wrong by 0.0006 and sits inside
its own interval, #3 is a list that looks complete because a list always does. Correctness of
the *source* is what everything here was set up to check; **agreement between a source and its
restatements was checked by nothing.**

This is the same shape as the MISATTRIBUTED verdict in `belief_list.md`. A pre-registered
overturn condition checks whether a number is right. It does not check whether the number is
the one the sentence is about, or whether a copy of it elsewhere still matches. Both failures
live in the gap between a value and its description.

### What is now covered, and what is not

`tests/test_reference_constants.py` closes the restated-scalar case. It parses the `reference`
dict literal out of each fork driver with `ast`, maps every constant to a named source artifact,
and fails on a mismatch or on a restated key with no source. `failure-proven` against
`git show HEAD:scripts/run_fork{1,2}.py`, which reports
`{'frozen_esm_cosine_seen_layer10': (0.5364, 0.5358)}` for both.

**It does not cover instances #3, #4 or #5, and no test here does.** #1 and #2 are restated *values*: there
is an authoritative number to compare against, so equality is checkable. #3 is an assertion of
*completeness* — a claim that a list of deviations contains every deviation. There is no
artifact holding the true list, because the true list is whatever a careful reading of another
paper and our own code turns up. A test could assert the list has seven entries; it could not
assert seven is right, and pinning the count would make the next omission harder to see rather
than easier.

**This gap is named and left open deliberately.** No test is being built for the completeness
class. The mitigation available is not automation: it is that a list asserting completeness
should be treated as `claimed` under the project's evidence scale no matter how carefully it was
assembled, and re-derived from source whenever it is load-bearing — which is how #3 was found.

**Instances 4 and 5 are uncatchable by an internal test, and for a sharper reason than #3.** #3 is
at least about objects this repository owns; a sufficiently patient reader could in principle
enumerate them. #4 and #5 are not. #4's authoritative answer — what "deduplicated" currently means
— lives in other groups' Methods sections and changed while this project was running. #5's
authoritative answer lived in a GitHub issue body, outside the repository the test suite can see.
**A test can only compare two things the repository holds.** When the authority is external,
consistency testing has nothing to compare against, and the only mitigation is the standing
literature check ([lessons.md](lessons.md)) — which is what caught #4, and which at closeout fired
*before* drafting rather than after.
