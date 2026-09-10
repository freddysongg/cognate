# Building a wider evaluation set (Phase B1)

Every correlation in [`findings.md`](findings.md) rests on **20 test peptides — 13 seen in
training, 7 not**. That is the binding constraint on the whole project; compute never was
(embedding the entire dataset takes 17.6 s). This document records how a replacement evaluation
set was built from VDJdb, what it contains, and what it cannot do.

**Output.** `data/vdjdb_eval.csv` — 88 peptides, 48 seen in IMMREP23 training and 40 not,
19,443 positives and 97,215 negatives over 18,760 distinct CDR3b. Per-peptide summary in
`data/vdjdb_eval_peptides.csv`; construction counters in `data/vdjdb_eval_stats.json`.

**Reproduce.** `scripts/fetch_data.sh && uv run python scripts/build_vdjdb_eval.py`.
Seeded throughout; two consecutive runs are byte-identical. `observed`.

---

## 1. Source and licence

VDJdb release **2026-06-03**, file `vdjdb.slim.txt`, from
`github.com/antigenomics/vdjdb-db`.

**The licence is AGPL-3.0, not CC BY-ND 4.0.** The Phase B work order recorded VDJdb as CC BY-ND
and flagged redistribution as an open question. That was wrong: both the repository `LICENSE` and
the `LICENSE` bundled inside the release archive are the GNU Affero General Public License v3.
`observed`. AGPL permits redistribution of modified and derived works, so shipping a filtered
subset as `data/vdjdb_eval.csv` is allowed — which a NoDerivatives licence would have forbidden,
since a filtered subset is exactly a derivative. Attribution and the citation below are still
required.

> Goncharov M, Bagaev D, Shcherbinin D, *et al.* **VDJdb in the pandemic era: a compendium of T
> cell receptors specific for SARS-CoV-2.** *Nature Methods* (2022).
> [doi:10.1038/s41592-022-01578-0](https://doi.org/10.1038/s41592-022-01578-0)

---

## 2. The problem that shaped everything: IMMREP23 *is* VDJdb

The IMMREP23 training file is named `VDJdb_paired_chain.csv`, and it is what every Phase 1 model
was trained on. So a VDJdb-derived evaluation set is contaminated by construction unless the
overlap is measured and removed.

Measuring it required fixing a string-format mismatch first. **VDJdb stores the full IMGT junction
(`CASSFSGNTGELFF`); IMMREP23 stores the stripped core (`ASSFSGNTGELF`).** Compared naively the two
sources share **zero** CDR3b — which is not a real disjointness, it is a units error, and taking it
at face value would have produced an evaluation set that looked leak-free and was not. After
stripping the leading `C` and trailing `F`/`W`:

| | count | share |
|---|---:|---:|
| IMMREP23 training pairs found verbatim in VDJdb | 7,680 / 9,305 | **82.5%** |
| IMMREP23 training CDR3b found in VDJdb | 7,626 / 8,993 | **84.8%** |

`observed`. IMMREP23's training set is essentially a subset of VDJdb. Any evaluation set built from
VDJdb without removing it would be scoring models on their own training data.

---

## 3. Construction

| Step | Rule | Rows remaining |
|---|---|---:|
| 1 | VDJdb slim table | 197,729 |
| 2 | `species = HomoSapiens`, `gene = TRB`, `mhc.class = MHCI` | 116,477 |
| 3 | CDR3 normalised to IMMREP23 convention; 20-letter alphabet only; peptide 7–15 aa, CDR3b 4–30 aa; dedup on (peptide, CDR3b) | 116,404 |
| 4 | **Drop every CDR3b that appears anywhere in IMMREP23 training** | 106,881 |
| 5 | Keep peptides with **≥ 50** surviving TCRs | 88 peptides |
| 6 | Cap positives at **500 per peptide**, seeded | 19,443 positives |
| 7 | Add **5 negatives per positive**, same peptide | 116,658 total |

**Step 4 is deliberately stricter than IMMREP23's own test set.** Dropping the TCR wherever it
appears — not just the exact (peptide, TCR) pair — costs only 1.7% of the remaining pairs and
removes TCR-level leakage entirely, including the "same TCR, different peptide" path that
[`findings.md` §2](findings.md) showed leaks 10.9% of rows under a peptide-held-out split. It also
eliminates the 17.9% verbatim-positive leak that inflated the Phase 1 k-NN baseline by 0.051. The
consequence: **k-NN should score lower here than on IMMREP23, and that is correct, not a
regression.**

> **Closeout amendment, 2026-09-09 — step 4 is exact-match removal, and that is the *weakest* of
> the published 2025–2026 deduplication standards, not the strictest.** The paragraph above
> compares step 4 against IMMREP23 and is right on that comparison. It is the wrong frame against
> the literature. Three published criteria, from [outlook.md](outlook.md) B2:
>
> | criterion | what it removes from this set |
> |---|---|
> | Drost et al., ePytope-TCR (2025): exact CDR3–epitope pair exclusion | step 4 is stronger — it drops the TCR everywhere, not just the pair |
> | Lu et al., *Nat Methods* (2026): CD-HIT >95% | a further **0.74%** of seen rows, all of it the indel class |
> | Liao et al. (2026): up to three CDR3β substitutions | a further **68.6%** of seen rows |
>
> **68.5% of this set's evaluation TCRs sit within three substitutions of an IMMREP23 training
> CDR3β, and 80.8% sit within Levenshtein 3.** Every number computed on this set is therefore
> reported at the exact-match standard, and the baseline moves from 0.5654 to 0.5302 at Liao's
> criterion. The sensitivity curve is [outlook.md](outlook.md) Part D; the CD-HIT row and its
> degeneracy are [cdhit_and_issues.md](cdhit_and_issues.md) Part 1. This is §11 instance 4: the
> evaluation set is itself a restatement of an external convention that moved after the set was
> frozen.

### Negatives

Each positive contributes 5 negatives **assigned to the same peptide**, with the TCR drawn from
the other peptides' pools subject to: every known cognate of that TCR in the full human TRB VDJdb
is more than 3 Levenshtein edits from the assigned peptide. That is the IMMREP23 organisers' own
rule, applied per peptide rather than globally.

Assigning negatives per peptide rather than uniformly is the fix for the Session 5 bug
([`findings.md` §4](findings.md)). The positive rate is **exactly 1/6 for all 88 peptides**, so
peptide identity alone scores **0.5000 pooled AUROC** on this set — the shortcut is not merely
absent, it is structurally impossible. `observed`. For comparison, the IMMREP23 test set scores
0.5143 on the same probe and the `shuffle` training sampler scored 0.9396.

Negatives are *assumed* non-binding, not experimentally verified. Same assumption IMMREP23 made.

---

## 4. What the set contains

| | |
|---|---|
| Peptides | **88** — 48 seen in IMMREP23 training, **40 unseen** (was 13 / 7) |
| Positives per peptide | min 52, median 159, max 500 (capped) |
| Distribution | 34 peptides < 100 positives, 29 at 100–299, 25 at ≥ 300 |
| Pre-cap support | 20 peptides hit the 500 cap; largest is `SLLMWITQV` at 29,698 TCRs |
| HLA | 19 alleles; **HLA-A\*02:01 on 30 of 88 peptides**, then B\*07:02 and A\*11:01 at 7 each |
| Antigen source | HomoSapiens 15, HIV-1 15, EBV 13, SARS-CoV-2 13, CMV 9 |

The per-peptide file carries `n_positive` and `n_tcr_before_cap` so **every analysis can report
per-peptide sample size alongside per-peptide AUC0.1.** This matters more here than at n = 20:
macro AUC0.1 weights a 52-TCR peptide identically to a 500-TCR one, and 34 of 88 peptides sit
under 100 positives. Any correlation found on this set has to be re-checked with those peptides
excluded and with points weighted by support before it is believed.

---

## 5. Limits

**The confidence-filtered analysis cannot be run at large n.** VDJdb's `vdjdb.score` rates evidence
quality 0–3. The work order asked for results at ≥ 0 and ≥ 2. At ≥ 2 the set collapses:

| Threshold | Peptides with ≥ 50 TCRs | Peptides with ≥ 10 TCRs |
|---|---:|---:|
| `vdjdb.score ≥ 0` | 88 | 88 |
| `vdjdb.score ≥ 2` | **6** | **11** |

61 of the 88 peptides have *zero* high-confidence TCRs; the median is 0. A ≥ 2 replication would
run at n ≈ 11 peptides — worse than the n = 13 that Phase B exists to escape. The `vdjdb_score`
column is in the output so the slice can be taken, but **a ≥ 2 analysis cannot answer the questions
this set was built for**, and reporting one as a robustness check would misrepresent it. The honest
statement is that this evaluation set is a *breadth* instrument, not a *quality* one: it buys 88
peptides at score ≥ 0, and no construction from VDJdb buys both.

**HLA-A\*02:01 dominates**, on 30 of 88 peptides — the standard bias of TCR databases toward the
most-studied allele. Any claim about generalisation across peptides is partly a claim about
generalisation within one HLA supertype. Stratifying by HLA leaves most alleles with too few
peptides to say anything.

**VDJdb is a union of studies, not a designed sample.** Peptide support reflects how much attention
an epitope received, not its biology, and the score ≥ 0 tier includes single-study records with no
independent confirmation.

**Metric code is still self-verified.** Every number this set will produce runs through
`src/cognate/metrics.py`, which has never been checked against an independent implementation —
Kaggle's scorer for the closed competition is broken ([`findings.md` §5](findings.md)). A wider
evaluation set does not fix that, and 88 peptides scored by possibly-wrong code is not better than
13 scored by it.

---

## 6. Acceptance checks

All `observed`, and pinned as regression tests in `tests/test_vdjdb_eval.py` (21 tests) so a
later change to the builder cannot quietly reintroduce what step 4 removed. The leakage
assertions are `failure-proven`: planting a single IMMREP23 training pair into
`data/vdjdb_eval.csv` turns 9 of the 21 red, including both leakage checks and both
peptide-marginal checks; restoring the artefact returns all 21 to green.

| Check | Result |
|---|---|
| Two consecutive builds byte-identical | pass |
| Eval (peptide, CDR3b) pairs also in IMMREP23 training | **0** |
| Eval CDR3b also in IMMREP23 training | **0** |
| Negative pairs that are known VDJdb binders | **0** |
| Negatives violating the > 3 edit-distance rule | **0** |
| Peptide-identity-only pooled AUROC | **0.5000** |
| Per-peptide positive rate range | 0.1667 – 0.1667 |

The last two are B2 probes run early, because they are also construction-validity checks. The rest
of B2 is below.

---

## 7. B2 — shortcut probes

`scripts/run_vdjdb_probes.py` → `data/vdjdb_probes.json`. Heads are logistic on ESM-2 35M layer 10,
fitted on IMMREP23 training with `matched` negatives at ratio 5, seed 0 — the headline config.

| Probe | pooled AUROC | macro AUC0.1 |
|---|---:|---:|
| peptide features only, on training | 0.5847 | 0.4995 |
| **peptide features only, on eval** | **0.5000** | **0.5000** |
| TCR features only, on training | 0.5000 | 0.5352 |
| **TCR features only, on eval** | **0.5004** | **0.5004** |
| TCR frequency, on eval (model-free) | 0.3889 | 0.4883 |
| CDR3b length, on eval (model-free) | 0.4968 | 0.4989 |
| full interaction model, on training | 0.7411 | 0.5851 |

**Verdict: PASS.** Largest macro deviation from chance among the eval-set probes is 0.0117,
inside the 0.05 tolerance. The gate is two-sided: a macro AUC0.1 *below* 0.5 is exactly as
exploitable as one above, since inverting the score recovers it.

**The peptide-only 0.5000 is arithmetic, not a measurement.** Every row of a peptide gets an
identical feature vector, so the model emits a constant column within each peptide, and a constant
column scores exactly 0.5 per group. The positive rate is also identical across peptides, which
pins pooled AUROC at 0.5 too. This probe could not have returned anything else on this set, and
reading it as evidence would repeat the error §3a ⑤ of `findings.md` warns about. **The
informative probe is TCR-only at 0.5004**, which had 12,894 distinct scores and every opportunity
to find something.

**The training set still carries a residual peptide shortcut.** Peptide features alone score
0.5847 pooled AUROC on the training data under `matched` — reduced from `shuffle`'s 0.9396 but not
gone. Macro AUC0.1 is 0.4995, blind to it as always. The full model's train-side gap (0.7411 pooled
vs 0.5851 macro, **+0.156**) is mostly this residual, not a new defect.

### One real asymmetry, documented rather than fixed

The TCR-frequency probe is the only non-trivial deviation. Negatives reuse TCRs drawn from other
peptides' pools, and a CDR3b appears 7.12 times on average in negative rows against 6.26 in
positive ones — so frequent TCRs are mildly identifiable as negatives. Most of that gap is
self-referential: a negative row's own draw counts toward its total, so roughly 1.0 of the 0.86
difference is the row counting itself.

It matters for pooled AUROC (0.3889, a deviation of 0.111) and barely for the headline metric
(0.4883). But 0.0117 is a larger deviation than it looks: the McClish floor is 9/19 ≈ 0.4737, so
the entire below-chance range is 0.0263 wide and this probe sits **44.5% of the way from chance to
the floor**.

It is left in place because **no model in this project can exploit it.** Every scorer here sees
sequences only; none receives an appearance count, and the artefact would only bite through a
correlation between draw frequency and sequence content, which the TCR-only probe at 0.5004 rules
out. The practical consequence is narrower: it is one more reason not to read pooled AUROC on this
set, alongside the database-saturation result in `findings.md` §3a ④.

### What this does not license

The full interaction model scores **0.5057 macro AUC0.1** on this evaluation set. That number is in
the probe output because the full model is the reference arm the probes are read against. It is
**not** the B3 result: no paired comparison against the k-NN baseline, no bootstrap CI, no
seen/unseen split, no per-peptide correlations. It is a preview, and it is a stark one against the
0.571 recorded on IMMREP23's 13 seen peptides — but this set removed every training TCR and 40 of
its 88 peptides were never trained on, so a drop is expected and its size is exactly what B3 has to
quantify.

---

## 8. B3 — the Phase 1 analyses rerun at n = 88

`scripts/run_b3.py` → `data/b3_results.json`, `data/b3_per_peptide.csv`. Same models, same headline
config (ESM-2 35M layer 10, `matched` negatives, ratio 5, seed 0), same metric code. The only thing
that changed is the evaluation set.

### Scores

Macro AUC0.1 with two-level bootstrap CIs. IMMREP23 figures from `data/headline.json` alongside.

| Model | seen, n = 48 | *(IMMREP23, n = 13)* | unseen, n = 40 | *(IMMREP23, n = 7)* |
|---|---|---|---|---|
| random predictor | 0.501 [0.498, 0.505] | *0.511* | 0.501 [0.497, 0.507] | *0.504* |
| **k-NN, edit distance** | **0.565 [0.546, 0.585]** *(exact-match regime; 0.5302 at sub 3)* | *0.641* | 0.500 [0.500, 0.500] `DEGENERATE` | *0.500* |
| ESM-2 + logistic head | 0.511 [0.502, 0.523] | *0.571* | 0.500 [0.495, 0.506] | *0.502* |
| ESM-2 + MLP head | 0.511 [0.504, 0.521] | *0.571* | 0.499 [0.495, 0.505] | *0.509* |

Both models drop hard, and the drop is expected: this set removed every IMMREP23 training TCR, so
the 17.9% verbatim-positive leak and the 10.9% same-TCR path are both gone. What the drop reveals
is how much of the Phase 1 result those paths were carrying. **The head lands 0.010 above a random
predictor.** The k-NN keeps a real but modest edge.

The k-NN's unseen 0.500 is arithmetic again, and the guard says so: 1 distinct score over 43,218
rows, 40 constant groups. It is not a measurement and must never be averaged with the head's.

### Paired comparisons, k-NN minus head

| Comparison | Δ macro AUC0.1 | p | peptides |
|---|---|---:|---:|
| k-NN − logistic, seen | **+0.055 [+0.039, +0.074]** | <0.001 | 48 |
| k-NN − MLP, seen | +0.054 [+0.037, +0.073] | <0.001 | 48 |
| k-NN − logistic, unseen | +0.000 [−0.006, +0.005] | 0.994 | 40 |
| k-NN − MLP, unseen | +0.001 [−0.005, +0.005] | 0.900 | 40 |

### Per-peptide correlations

Every row reports the pre-registered support sensitivity: leave-one-out range, a refit on the 33 of
48 peptides with ≥ 100 positives, and a support-weighted fit. `data/b3_per_peptide.csv` carries
`n_positive` beside every per-peptide score.

| Relationship | r at n = 48 | *(n = 13)* | leave-one-out | refit n ≥ 100 | weighted |
|---|---|---|---|---|---|
| k-NN AUC0.1 vs distance to training | −0.359 [−0.603, −0.024] | *−0.878* | −0.450 … −0.250 | **−0.411 [−0.717, +0.010]** | −0.391 |
| head AUC0.1 vs log₁₀ support | +0.229 [+0.043, +0.593] | *+0.674* | +0.166 … +0.380 | +0.500 [+0.232, +0.698] | +0.334 |
| k-NN AUC0.1 vs log₁₀ support | +0.024 [−0.263, +0.304] | *+0.109* | −0.064 … +0.095 | −0.118 [−0.496, +0.250] | −0.044 |
| k-NN mean raw score vs log₁₀ support | +0.973 [+0.963, +0.982] | *+0.990* | +0.972 … +0.976 | +0.977 [+0.967, +0.987] | +0.979 |
| head AUC0.1 vs distance to training | −0.144 [−0.359, +0.239] | *−0.735* | −0.191 … −0.013 | −0.024 [−0.375, +0.419] | −0.081 |

Across all 88 peptides, k-NN AUC0.1 vs distance is −0.281 [−0.465, −0.054].

### The difference of correlations, finally tested

The belief list refused at n = 13 to assert that the two models are limited by different things,
because no paired test on the difference of correlations had been run. It has now been run:

> corr(log support, head AUC0.1) − corr(log support, k-NN AUC0.1) =
> **+0.204 [−0.118, +0.707], p = 0.206** over 48 peptides.

**It spans zero.** Tripling the peptide count was not enough to establish the contrast, and the
eyeball reading of two barely-overlapping intervals would have asserted something a proper test
still does not support. That restraint is the single decision in Phase 1 that B3 most clearly
vindicates.

### Claim ④, re-verified

| Scoring | macro AUC0.1 | pooled AUROC |
|---|---|---|
| raw | **0.5356597979372727** | 0.531859 |
| z-scored within peptide | **0.5356597979372727** | 0.568511 |
| rank-normalised within peptide | **0.5356597979372727** | 0.566319 |

Identical in every printed digit; pooled moves +0.037, against +0.035 at n = 20.

### Verdicts

Recorded per claim in [`belief_list.md`](belief_list.md). In one line each:

| | Claim | Verdict |
|---|---|---|
| ① | lookup beats the head on seen peptides | **held, strengthened** — and the leak-free worry is answered, since this whole set is leak-free |
| ② | the baseline is limited by distance to training | **weakened** — sign survives, r² falls 0.770 → 0.129, and the well-supported refit spans zero |
| ③ | the head is limited by training support | **held**, r² 0.455 → 0.052; the contrast it declined to assert **died** under its own named test |
| ④ | macro AUC0.1 is blind to within-peptide level shifts | **held exactly** |
| ⑤ | the baseline's score level is made of database size | **held unchanged** |

Two claims out of five survive at anything like their recorded strength, and the two that were
written most cautiously — ③'s refusal to assert the contrast, and the decision to exclude
"head performance tracks distance" as leverage-driven — are the ones B3 shows were right. The
claims that overstated are the ones that used a strong word: "largely determined by" in ②.
