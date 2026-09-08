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

| Model | seen (48) | unseen (40) |
|---|---|---|
| random predictor | 0.5009 [0.498, 0.505] | 0.5015 |
| **k-NN, edit distance** | **0.5654 [0.546, 0.585]** | 0.5000 *(degenerate)* |
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
epoch 40. Even an oracle selecting the best epoch by test score stays far below 0.5654. With the
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
gets 0.5654. The pre-registered reading for this outcome was written before any of them ran:
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
