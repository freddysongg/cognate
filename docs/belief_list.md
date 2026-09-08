# Belief list

**Locked at n = 20 test peptides (13 seen / 7 unseen), 2026-09-06, end of Phase A.**

Five claims I would defend as written, each stated no more strongly than its interval allows.
Nothing on this list has a CI spanning zero. The point of locking it now is that Phase B changes
n to 50–100; once that happens it becomes very easy to write claims that fit the new data and
lose the record of what was believed beforehand. B3's held / weakened / reversed verdicts are
only meaningful against a record written before the new numbers existed.

Each claim carries a **What would overturn it** line — a statement checkable at B3 without
re-litigating what the claim meant.

> **B3 verdicts are in, and are appended to each claim below.** The claim wording above each
> table is the n = 20 record and has **not** been edited — that is the whole point of having
> locked it. Results at n = 88: `data/b3_results.json`, `docs/eval_set_construction.md` §8.
> Summary: **① held and strengthened, ② weakened, ③ held on its asserted half while the half
> it deliberately refused to assert died, ④ held exactly, ⑤ held unchanged.**

The numbering here is local to this document and does **not** line up with the ①–⑤ in §3a of
`findings.md`. Mapping: belief ① = findings §3a ①, belief ② = §3a ②, belief ③ = §3a ③ (narrowed),
belief ④ = §3b, belief ⑤ = §3a ④.

---

## ① The lookup baseline beats the ESM-2 head on seen peptides

> On the 13 IMMREP23 test peptides that also appear in training, a max-similarity edit-distance
> lookup scores higher macro AUC0.1 than ESM-2 35M embeddings with a trained logistic head:
> **+0.070**. I believe the sign. I do not believe the size, and I would not defend this as a
> general fact about lookup versus language models — only that on this benchmark, at this amount
> of training data, the pretrained representation bought nothing over plain string similarity.

| | |
|---|---|
| **Evidence** | Paired bootstrap over the same 13 seen peptides. k-NN 0.641, ESM-2 + logistic head 0.571. |
| **Interval** | **Δ = +0.070 [+0.012, +0.142], p = 0.016** |
| **Strongest objection** | n = 13 peptides. The lower bound is +0.012 — barely clear of zero. And the baseline gets 17.9% of its test positives handed to it verbatim from the training file; dropping those rows costs it 0.051, which is most of the margin. A critic could reasonably say the result is "leakage plus 13 points". |
| **Best answer to that objection** | The exact-match ablation was run both ways. Denying the *database* the verbatim answer while keeping the rows costs only 0.009 [−0.026, 0.043] — the second-nearest neighbour ranks those rows nearly as well, so the method does not depend on the leak. But the margin does shrink to 0.041 on the leak-free slice (0.591 vs 0.550), and that comparison was not run as a paired test. |
| **What would overturn it** | The paired Δ at 50–100 peptides has a CI spanning zero, or flips sign. **Weakened**, not reversed, if the sign holds but the leak-free slice (which is where the honest margin lives) goes to zero. |
| **B3 verdict (n = 48 seen)** | **HELD, and strengthened.** Δ = **+0.055 [+0.039, +0.074], p < 0.001** over 48 peptides, against +0.070 [+0.012, +0.142], p = 0.016 over 13. The interval moved away from zero and tightened roughly fourfold. The MLP arm agrees: +0.054 [+0.037, +0.073]. The overturn line's real worry — that the margin lives on the leak — is answered directly, because **this evaluation set is entirely leak-free**: every IMMREP23 training TCR was removed, so there is no verbatim-positive slice left to strip. The margin survived its removal. |
| **What the claim now understates** | It says the language model "bought nothing over plain string similarity". At n = 88 that is too kind to the head: 0.511 [0.502, 0.523] against a **random predictor at 0.501 [0.498, 0.505]**. The honest reading is no longer "worse than k-NN" but "barely distinguishable from chance". |

## ② The baseline is limited by distance to the training set

> How well the lookup baseline scores on a given peptide is largely determined by how far that
> peptide's binders sit from the nearest training TCR: **r = −0.878** over 13 peptides. I believe
> this as a description of the baseline's failure mode. I do not treat it as a discovery — a
> max-similarity scorer is made out of distance, so the correlation is close to definitional.
> What it buys is knowing which axis to predict along: distance to training, not peptide
> popularity, and not amount of training data.

| | |
|---|---|
| **Evidence** | Per-peptide macro AUC0.1 vs median normalised edit distance from that peptide's binders to the nearest training CDR3b. |
| **Interval** | **r = −0.878, r² = 0.770, CI [−0.965, −0.494]** |
| **Robustness** | Survives leave-one-out: −0.844 dropping the extreme point, −0.851 dropping the top two by support. |
| **Strongest objection** | 13 points, and the relationship is close to mechanically true. The interesting version of the claim would be that distance predicts performance for methods that are *not* built from distance, and that is exactly what did **not** hold for the head (r = −0.735, collapsing to −0.341 on leave-one-out). |
| **What would overturn it** | The correlation's CI spans zero at larger n, or it survives only because thinly-supported peptides carry it — check by refitting on peptides above a minimum TCR count and by weighting per-peptide points by their sample size. |
| **B3 verdict (n = 48 seen)** | **WEAKENED.** r = **−0.359 [−0.603, −0.024]**, against −0.878 [−0.965, −0.494]. The interval still excludes zero, but only just, and **r² falls from 0.770 to 0.129**. The word "largely" in the claim above does not survive: distance explains about an eighth of the variance in per-peptide score, not three quarters. |
| **B3 sensitivity, as pre-registered** | Not leverage — leave-one-out stays in −0.450 to −0.250. Support-weighted −0.391, fine. But the **well-supported refit fails**: on the 33 of 48 peptides with ≥ 100 positives, r = −0.411 **[−0.717, +0.010]**, spanning zero. One of the two checks the claim named for itself does not pass. Across all 88 peptides, r = −0.281 [−0.465, −0.054]. |

## ③ The head is limited by how much training data each peptide had

> The head's per-peptide score rises with how much training data that peptide had:
> **r = +0.674 [+0.355, +0.887]**. That is the whole claim, and it is the only half of it that
> clears an interval. The lookup baseline shows no such relationship (r = +0.109, CI spanning
> zero) — but that is a failure to detect one, not evidence there is none, so I do not assert
> "the two models are limited by different things" as a belief. I think it is the more likely
> reading of two intervals that barely overlap, and I would concede immediately that no paired
> test on the difference of correlations was ever run.

| | |
|---|---|
| **Evidence** | Per-peptide score vs log₁₀(training positives), 13 seen peptides. |
| **Interval** | **head r = +0.674, r² = 0.455, CI [+0.355, +0.887]**. The k-NN comparison figure, not asserted: r = +0.109, CI [−0.598, +0.735]. |
| **Robustness** | Head correlation holds at +0.571 and +0.701 under leave-one-out. The two predictors are largely independent (support vs distance r = −0.354, CI spans zero), so this is not ② restated. |
| **Strongest objection** | The head's correlation clears its CI, but the contrast between the two models was never tested directly. Asserting "different for the two models" is an eyeball comparison of two intervals that happen not to overlap much. |
| **What would overturn it** | The head's support correlation loses its interval at larger n, or survives only on thinly-supported peptides. The unasserted contrast becomes assertable — or dies — at B3, where n is finally large enough to run a paired test on the difference of correlations; the excuse for not running one was n = 13. |
| **B3 verdict (n = 48 seen)** | **HELD on the asserted half, much reduced in size.** r = **+0.229 [+0.043, +0.593]**, against +0.674 [+0.355, +0.887]; r² falls from 0.455 to 0.052. It clears zero by a margin of 0.043. Leave-one-out is stable (+0.166 to +0.380), and the well-supported refit **strengthens** it — r = +0.500 [+0.232, +0.698] on the 33 peptides with ≥ 100 positives — so the "survives only on thinly-supported peptides" failure mode is ruled out, not merely unobserved. |
| **B3 verdict on the contrast: it died** | The paired test this claim named for itself was run: **corr(support, head) − corr(support, k-NN) = +0.204 [−0.118, +0.707], p = 0.206**, spanning zero at n = 48. The k-NN arm on its own is +0.024 [−0.263, +0.304], still a null. **So "the two models are limited by different things" is not established, and refusing to assert it at n = 13 was correct.** Reading it off two barely-overlapping intervals — which is exactly what the eye wanted to do — would have produced a claim that a proper paired test does not support even after tripling the peptide count. |

## ④ Macro AUC0.1 is blind to anything that only moves scores within a peptide

> The metric cannot see a per-peptide level shift. Z-scoring or rank-normalising scores inside each
> peptide leaves it identical to 1e-12 while pooled AUROC moves +0.035. I believe this without an
> interval, because it follows from the definition rather than from a measurement. I believe the
> cost as strongly as the property: the same blindness that makes it the right headline metric here
> is what let a sampler shortcut worth 0.9396 pooled AUROC sit undetected for two sessions at
> exactly 0.5000 macro.

| | |
|---|---|
| **Evidence** | z-scoring or rank-normalising scores within each peptide leaves macro AUC0.1 identical to 1e-12 (0.641419 in all three cases) and moves pooled AUROC by +0.035. |
| **Interval** | None needed — this is `analytic`, not statistical. It follows from AUC being rank-based and the metric being computed per group. |
| **Strongest objection** | None to the claim itself. The objection is to over-reading it: it buys robustness to one confound at the price of blindness to it. Treating a 0.5000 macro as "no shortcut present" is exactly the inference that failed in Session 5. |
| **What would overturn it** | Nothing at larger n — this is analytic and n-independent. It carries a different risk: the metric implementation is `observed`, self-verified only, never scored by an independent implementation (Kaggle's scorer for this closed competition is broken, §5 of `findings.md`). If that code is wrong, this claim is about the wrong function. |
| **B3 verdict (n = 88)** | **HELD exactly, as predicted.** Macro AUC0.1 is **0.5356597979372727 under all three scorings** — raw, z-scored within peptide, rank-normalised within peptide — identical in every printed digit, not merely within 1e-12. Pooled AUROC moves 0.5319 → 0.5685 (z) and 0.5663 (rank), **+0.037**, almost exactly the +0.035 seen at n = 20. The named risk is untouched: the metric code is still self-verified only. |


## ⑤ The baseline's raw score level is made of database size, which is what makes ④ load-bearing

> How large a score the lookup baseline hands a peptide's rows is almost entirely a function of how
> many training TCRs that peptide had: **r = +0.990 [+0.981, +0.996]** for per-peptide mean raw
> score against log₁₀(training support). A max-over-database operator has no scale correction —
> with 1,818 candidates almost any query finds a close match, and with 6 almost none does. I
> believe this is the best-supported number in the project. Its consequence is not about the
> baseline, it is about metrics: **any pooled metric on this data is partly measuring training-set
> size.** ④ says macro AUC0.1 cannot see a per-peptide level shift; ⑤ says there is an enormous one
> here and names what it is made of. Neither is worth much without the other — ④ alone is an
> abstract property of a metric, ⑤ alone is a quirk of one scorer.

| | |
|---|---|
| **Evidence** | Per-peptide mean raw score vs log₁₀(training positives), 13 seen peptides, 5,000 peptide-level bootstrap resamples. Database sizes span 6 to 1,818 TCRs. |
| **Interval** | **r = +0.990, CI [+0.981, +0.996]** — the tightest interval on this list by a wide margin. |
| **Supporting, not asserted** | Restricted to *negatives* the correlation is +0.983 (no bootstrap CI computed); for positives it is +0.697. The positive−negative gap, which is what actually discriminates (gap vs AUC0.1 r = +0.932), shows no relationship with support at all (r = +0.001). That last number is a null at n = 13 and I do not assert it — but +0.990 on the level beside +0.001 on the gap is the shape you would expect if added data buys reach and not discrimination. |
| **Strongest objection** | It is close to arithmetic. A maximum over a larger sample is larger in expectation, so "bigger database → higher max similarity" is nearly true by construction, and r = +0.990 mostly measures how cleanly that arithmetic shows through. The claim is only interesting because of what it implies for pooled metrics, and that implication is ④'s, not this one's. |
| **What would overturn it** | Essentially nothing at larger n — this is the claim most likely to survive Phase B untouched, which is itself a reason to be suspicious of how much it teaches. **Weakened** if at n = 50–100 the relationship turns out non-monotone, or if peptides with mid-size databases break the log-linear fit that 13 well-spread points made look clean. |
| **B3 verdict (n = 48 seen)** | **HELD, essentially unchanged.** r = **+0.973 [+0.963, +0.982]**, against +0.990 [+0.981, +0.996]. Leave-one-out +0.972 to +0.976; well-supported refit +0.977 [+0.967, +0.987]; support-weighted +0.979. Mid-size databases did not break the log-linear fit. This played out exactly as written, including the part warning that surviving untouched is not the same as being informative — it remains the best-supported and least surprising thing in the project. |


---

## Deliberately excluded — CI spans zero or leverage-driven

| Candidate | Point estimate | Why excluded | B3 at n = 48 |
|---|---|---|---|
| The head's performance tracks distance to training | r = −0.735 | Collapses to −0.341 dropping one peptide. Leverage-driven. | **Exclusion vindicated.** r = −0.144 **[−0.359, +0.239]**, spanning zero. Had this been promoted at n = 13 on its point estimate, B3 would have reversed it. |
| The head inflates raw score level with support | r ≈ +0.51 | CI [−0.12, +0.90] spans zero at n = 13. | not re-run |
| A middle ESM-2 layer beats the last | spread 0.028 over 20 fits | Below the ~0.05 noise floor; validation-to-test Spearman is −0.035. | not re-run |
| Removing the sampler shortcut helps the head near the training set | Δ +0.337 / +0.234 / +0.119 on three peptides | Three points. Suggestive, not a result. Phase C tests it properly. | Phase C |
| Both models are at chance on unseen peptides | 0.500 / 0.502 | Not excluded for uncertainty — excluded because "indistinguishable from chance" is a failure to reject, not a finding. State it as an observation, not a belief. | Still correct to exclude — see the note below. |

---

## A note on the chance-level result

"Both models are at chance on unseen peptides" is the result most people would quote, and it is the
one you should be most careful with. The k-NN's 0.500 is **arithmetic, not measurement** — its
database is empty for unseen peptides, so all 1,066 rows take one constant score, and a constant
column scores exactly 0.5 by construction. The head's 0.502 *is* a measurement (1,019 distinct
scores). Those two numbers look identical and mean completely different things.

At B3 this stays off the list regardless of what the new n does to it. A wider unseen set could
produce a head score whose CI clears 0.5, and that would be a finding worth adding — but it would
be a *new* claim, not this one held.

**B3 result: it did not clear.** At 40 unseen peptides the head scores 0.500 [0.495, 0.506] and the
MLP 0.499 [0.495, 0.505] — 5.7× the unseen peptides of Phase 1, and still nothing. The k-NN's 0.500
is once again arithmetic: 1 distinct score over 43,218 rows, 40 constant groups, flagged
`DEGENERATE` by the guard. The distinction the note is about survived the change in n exactly as
stated, and the paired k-NN-minus-head comparisons on unseen peptides are +0.000 [−0.006, +0.005],
p = 0.994 and +0.001 [−0.005, +0.005], p = 0.900 — the two arms are indistinguishable because
neither is doing anything.

---

# Phase D additions

**Locked at n = 88 evaluation peptides (48 seen / 40 unseen), 2026-09-07, end of Phase D.**

Three further claims, written after Phase D's numbers existed but before any widening of the
evaluation set, before any hyperparameter sweep, and before the backbone is unfrozen. The overturn
lines below are the record against which the next phase gets judged. Same rule as above: none of
these has a CI spanning zero except where the claim is explicitly *about* an interval spanning
zero, which ⑦ is.

Supporting numbers: [`fork1_results.json`](../data/fork1_results.json),
[`fork2_results.json`](../data/fork2_results.json),
[`knn_esm_cosine.json`](../data/knn_esm_cosine.json), `findings.md` §7.

---

## ⑥ Learned representations over frozen ESM-2 do not beat edit distance on seen peptides

> Across five independent constructions — cosine over frozen embeddings, a contrastively trained
> TCR metric, a two-tower peptide↔TCR alignment, a residue cross-attention model, and its
> mean-pool control — every one scores below a max-similarity edit-distance lookup on the 48 seen
> peptides, and every paired interval excludes zero. I believe the sign and the ordering. I do not
> believe this generalises past a frozen backbone.

| | |
|---|---|
| **Evidence** | Paired bootstrap over the same 48 seen peptides, same database, same max-over-database operator, same metric, same evaluation set. Best learned variant is 1a at 0.5320; baseline is 0.5654. |
| **Interval** | 1a **−0.034 [−0.048, −0.023]**; 1b −0.057 [−0.077, −0.037]; 2a −0.061 [−0.081, −0.043]; 2b −0.056 [−0.075, −0.039]. All p < 0.001. |
| **Strongest objection** | One hyperparameter configuration per fork, one frozen backbone, one layer for the learned variants. A critic can say the ceiling was set by choices nobody swept, not by the representation. |
| **Best answer to that objection** | The Phase 0 arm requires no hyperparameters at all — cosine over frozen embeddings is parameter-free and loses at five model/layer configurations. And 1a's trained projection (0.5320) does not beat its own *untrained* random projection (0.5341), which bounds how much the training configuration could have been at fault. |
| **What would overturn it** | Any learned representation clearing 0.5654 with a paired CI excluding zero on these 48 peptides. **Weakened** if a hyperparameter sweep or an unfrozen backbone closes most of the gap without clearing it, since that would relocate the cause from the representation to the training budget. |

## ⑦ Cross-attention over residues adds nothing over mean-pooling them

> Attending across peptide and TCR residue positions, rather than averaging them first, does not
> change performance. The point estimate is slightly negative and the interval spans zero. This is
> a claim that a difference is absent, and it is only worth anything because the control was
> validated first.

| | |
|---|---|
| **Evidence** | 2a and 2b are the same architecture, optimiser, split, negatives and seeds; `use_attention` is the only difference. 2b reproduced its independent target, scoring 0.5093 against the Phase 1 head's 0.511. |
| **Interval** | **Δ = −0.005 [−0.016, +0.005], p = 0.366** over 48 peptides. Seed spread 0.0019 / 0.0025, both below CI width. |
| **Strongest objection** | Absence of evidence. A 1–2 head, 1–2 layer block at width 64 is small; a bigger attention stack, more layers, or a different pooling might find something. And a CI spanning zero cannot distinguish "no effect" from "underpowered". |
| **Best answer to that objection** | The interval is tight — ±0.016 against a baseline gap of 0.056 — so an effect large enough to matter for the headline question is excluded even though zero is not. The mechanism is independently predicted: Session 4 measured effective rank 20–33 against 480 nominal dimensions in the pooled embeddings, so there was little for attention to recover. Attention also lowered validation loss at every seed while lowering validation macro AUC0.1, which is what added capacity without added signal looks like. |
| **What would overturn it** | A larger attention stack, or attention over an unfrozen backbone, producing 2a − 2b with a CI excluding zero on these 48 peptides. **Weakened** if the interval merely tightens around a small positive value without clearing zero. Note the asymmetry: this claim is cheap to overturn and should be attacked first if Fork 2 is revisited. |

## ⑧ Nothing in this project has scored above chance on unseen peptides

> Across every method built — baseline, heads, cosine k-NN, both forks, both variants each — no
> construction scores above chance on the 40 peptides absent from training. The failures are not
> all the same kind, and the distinction is the useful part.

| | |
|---|---|
| **Evidence** | k-NN and every cosine variant are *structurally* unable to score: the per-peptide database is empty, every row takes `default_score`, and a constant column is exactly 0.5 by arithmetic. 1a inherits this because it is evaluated with the same operator. The heads, 1b, 2a and 2b all produce varied scores and land at chance anyway. |
| **Interval** | 1b **0.5018 [0.4974, 0.5070]**, the only variant with both a mechanism and a non-degenerate score column (~43,190 distinct values). 2b 0.5022 [0.5000, 0.5129]; 2a 0.4985 [0.4907, 0.4999]; logistic 0.4997; MLP 0.4991. |
| **Strongest objection** | 2a's interval sits fractionally below 0.5, which under a two-sided reading is as exploitable as being above it. Reporting the whole set as "at chance" glosses that. |
| **Best answer to that objection** | 2a's per-seed values are [0.4947, 0.4972, 0.5037] — one seed above 0.5 — and its seed spread (0.0090) matches its CI width (0.0092). By the project's own rule, seed variation is the dominant uncertainty there, so "at chance" is the defensible reading and "below chance" is not. |
| **What would overturn it** | Any method scoring unseen peptides with a CI whose lower bound clears 0.5 on these 40 peptides. The two-tower construction is the only current candidate with a mechanism, so it is where to look. **Partially overturned** — and worth reporting loudly — if a method clears chance on unseen while still losing on seen, since nothing in the project has separated those two axes yet. |

---

## What Phase D says about Phase C

Phase C asked whether a negative sampler changes the margin over a model sitting 0.010 above
chance. Phase D moves the answer: the best learned model anywhere in the project is 2b at 0.5093,
still 0.056 below the baseline, and Fork 1 reached its ceiling with **no negative sampler at all**.
The binding constraint is the representation, not the sampler. Keeping C deferred is the reading
the evidence supports; the condition that would revive it is ⑥ being weakened by an unfrozen
backbone, which would mean there is finally a model with room for a sampler to move.
