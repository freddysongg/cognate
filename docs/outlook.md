# Literature outlook and deduplication recheck

One pass, 2026-09-08. No new training runs, no new arms, no hyperparameter changes. Part A
re-scores three existing arms on a subset of the frozen evaluation set; the models are
unchanged, only which test rows are counted. Every external claim below cites a source
retrieved in this session.

---

## Part A — dedup recomputation

`scripts/run_dedup_recheck.py` → `data/dedup_recheck.json`,
`data/dedup_recheck_per_peptide.csv`. No frozen artifact was touched. Reads existing caches;
embeds nothing.

**Criterion.** Remove every eval row whose CDR3β is within Levenshtein ≤ 3 of *any* IMMREP23
training CDR3β. Note the deviation from Liao et al.: they allow **up to three amino-acid
substitutions**, which excludes indels. Levenshtein ≤ 3 is strictly more aggressive, so the
numbers below are an **upper bound on the damage**, not a like-for-like reproduction.

### How much it removes

| slice | rows | rows removed | distinct TCRs | TCRs removed |
|---|---|---|---|---|
| all | 116,658 | 94,170 (**80.7%**) | 18,760 | 15,164 (**80.8%**) |
| seen (48 peptides) | 73,440 | 59,435 (**80.9%**) | 18,566 | 15,006 (**80.8%**) |
| unseen (40 peptides) | 43,218 | 34,735 (**80.4%**) | 17,255 | 13,917 (**80.7%**) |

Nearest-training edit distance across the eval set: min 1, **median 2**, max 12. Our exact-match
dedup removed 100% of verbatim CDR3β; at radius 3 four fifths of what remains is still a near
duplicate of something in training.

### Surviving support

**All 48 seen peptides remain scorable. None falls below 5 surviving positives.** Surviving
positives per peptide: min 5, median 32.5, max 174. So the collapse below is not an artifact of
peptides becoming unstable — it is measured on the same 48 peptides throughout.

### Recomputed macro AUC0.1, seen slice

| arm | full (n=48) | deduped (n=48) | Δ |
|---|---|---|---|
| `edit_retrieval` | 0.5654 [0.5461, 0.5849] | **0.5246 [0.5127, 0.5432]** | −0.0407 |
| `esm_retrieval` | 0.5358 [0.5230, 0.5498] | **0.5108 [0.5027, 0.5260]** | −0.0250 |
| `svc_per_peptide` | 0.5424 [0.5301, 0.5575] | **0.5129 [0.5039, 0.5268]** | −0.0296 |

Subset ablation, deduped − full, per arm (different row sets, resampled independently):

| arm | Δ |
|---|---|
| `edit_retrieval` | **−0.0407 [−0.061, −0.017]**, p < 0.001 |
| `esm_retrieval` | −0.0250 [−0.040, −0.007], p = 0.004 |
| `svc_per_peptide` | −0.0296 [−0.043, −0.014], p < 0.001 |

### The decision this drives — superseded by Part D

Paired, same rows both arms, deduped seen slice:

| comparison | Δ | p | spans zero |
|---|---|---|---|
| `edit_retrieval − esm_retrieval` | **+0.0138 [−0.0031, +0.0299]** | 0.110 | **yes** |
| `svc_per_peptide − esm_retrieval` | +0.0021 [−0.0130, +0.0173] | 0.828 | yes |
| `svc_per_peptide − edit_retrieval` | −0.0117 [−0.0311, +0.0043] | 0.144 | yes |

**The representation effect does not survive.** At full evaluation it is
−0.0296 [−0.0415, −0.0188], p < 0.001 — the project's one surviving claim. Under Levenshtein-3
dedup the same comparison on the same 48 peptides is +0.0138 [−0.0031, +0.0299], p = 0.110. The
interval crosses zero and the point estimate does not carry the same interpretation, because on
the deduped subset **all three arms are statistically indistinguishable from one another**.

Each arm individually still clears the 0.5009 random floor by its lower bound (0.5127, 0.5027,
0.5039), so the collapse is toward near-chance, not to chance. But the *ordering* — which is what
the writeup's central comparison asserts — is gone.

> **Corrected by Part D.** The conclusion originally recorded here — *"it collapses"* — was
> criterion-dependent and is wrong as a statement about Liao et al.'s standard. Levenshtein ≤ 3
> is not their criterion. Under substitutions-only at radius 3, which is, the representation
> effect **survives**: −0.0155 [−0.0329, −0.0012], p = 0.024. The numbers in this section are
> correct for Levenshtein ≤ 3; the reading of them was not. See Part D.

Read against the pre-registered decision *at this radius*: 0.5654 falls to 0.5246 and the
ordering between arms is no longer resolvable. Whether that is the relevant radius is exactly
what Part D settles, and the answer is no.

---

## Part B — targeted lookups

**B1. Has any published work reported an alignment or edit-distance baseline on IMMREP23
evaluated with macro AUC0.1, split seen vs unseen?** **Partly — and closer than expected.**
IMMREP23's official metric *is* macro AUC0.1, "calculated independently for each peptide and then
averaged" ([Kaggle competition
page](https://www.kaggle.com/competitions/tcr-specificity-prediction-challenge/overview)), and
the workshop report includes TCRbase — a distance method — as its baseline while reporting seen
and unseen separately (Nielsen et al., *ImmunoInformatics* 16:100045, 2024). What that paper does
**not** do is publish a TCRbase macro-AUC0.1 number split by seen/unseen; TCRbase appears as the
line defining the poorest performance tier (group G3), not as a scored row. Liao et al.
(arXiv:2606.04994, 3 Jun 2026) evaluate eight models on IMMREP23 with "macro-averaged partial AUC
(pAUC 0.1)" grouped by "seen versus unseen epitopes", but their eight are ERGO, ERGO-II,
NetTCR2.0, NetTCR2.2, TITAN, PanPep, SCEPTR and EPACT — **no distance baseline among them**.
*Decision:* our 0.5654 (exact-match regime; 0.5302 at sub 3) is not a reproduction of a published
number; it remains a new data point.
But the metric, the benchmark and the seen/unseen split are all now standard, which makes it a
data point in a populated frame rather than an isolated one.

**B2. What is the prevailing deduplication standard in 2025–2026?** **It has moved past exact
match, and our exact-match dedup is now the weakest of the three published standards.** Three
papers, three thresholds: (i) **Lu et al., *Nature Methods* 23:248–259 (2026)** — CD-HIT, "to
exclude highly similar sequences (>95% similarity) between the training and test sets", applied
again between training and independent test sets; (ii) **Liao et al., arXiv:2606.04994 (2026)** —
"allowing up to three amino acid substitutions in the CDR3β region", which "eliminated between
40% and 70% of sequences in test datasets"; (iii) **Drost et al., ePytope-TCR, *Cell Genomics*
5:100946 (2025)** — exact pair separation only, "we made sure that no CDR3-epitope pair was
present in the database during the development of the predictors". Worth noting that CD-HIT at
>95% identity on a ~14-residue CDR3β is close to exact match, so the genuine spread is between
Drost/Lu (≈exact) and Liao (3 substitutions). *Decision:* exact-match dedup is defensible against
two of the three but not against Liao. Part A should be judged against the 3-substitution
standard, and our Levenshtein-3 implementation overshoots it.

**B3. Has SCEPTR's Fig. S6 comparison been replicated, extended, or contradicted since Jan
2025?** **No — nothing found.** SCEPTR (Nagano et al., *Cell Systems* 16(1):101165) has 32–35
citations per Google Scholar via PubMed and Cell listings, and Liao et al. (2026) evaluate SCEPTR
*as a model* on IMMREP23, but no retrieved work reproduces or contests the specific Fig. S6
design — linear SVC versus nearest-neighbour on matched PLM features with k swept 1–200 against a
fixed background. *Decision:* the operator thread is **neither closed nor reopened**.
`findings.md` §8f needs no revision; the +0.0067 remains uncomparable for the regime reason
already recorded there, and no external result now bears on it.

**B4. Has ESM-C or ESM-3 been evaluated on TCR specificity, CDR3 representation, or
short-peptide tasks?** **Not found in public sources.** ESM-C was released 4 Dec 2024 at 300M /
600M / 6B ([EvolutionaryScale
announcement](https://www.evolutionaryscale.ai/blog/esm-cambrian)). The most relevant benchmark,
Feng et al., *Briefings in Bioinformatics* 26(1):bbaf030 (2025), evaluates 19 TCR CDR3 embedding
models and concludes "handcrafted embeddings surpassed data-driven ones in modeling TCR-epitope
interactions" — but it predates ESM-C uptake and does not include it. No paper evaluating ESM-C
or ESM-3 on TCR specificity, CDR3 representation, or short-peptide representation was retrieved.
*Decision:* record as an open question in the limitations section. Not run, as instructed.

**B5. How many 2024–2026 TCR-epitope benchmark papers include a distance or alignment
baseline?** *(Corrected 2026-09-08 — see the note below on the first version of this entry.)*
Of the papers retrieved: **include a distance baseline** — IMMREP22 (Meysman et al.,
*ImmunoInformatics* 9:100024, 2023; TCRbase and tcrdist3 among the 23 models, with the finding
stated at abstract level), IMMREP23 (Nielsen et al., *ImmunoInformatics* 16:100045, 2024;
TCRbase as the designated baseline), and **TCRcube** — Culka et al., "Predicting specificity of
TCR-pMHC interactions using machine learning and biophysical models", *Cell Systems*, 13 Aug
2026, [PMC13502210](https://pmc.ncbi.nlm.nih.gov/articles/PMC13502210/), which states: *"The
IMMREP_2022 data splits were previously used to benchmark various ML methods and we report these
results as baselines for TCRcube models. These baselines include peptide-specific sequence
similarity models (**tcrdist3** and **TCRbase**), shallow supervised ML models (DiffRBM, SETE,
sonia, tcrex, TCRGRP), supervised neural networks (netTCR, TCRAI, pMTnet), and unsupervised
language…"* One qualifier from the paper's own Methods: *"the repository contains results of
various published methods. We use those unmodified previous results as a baseline"* — TCRcube
**inherits** IMMREP22's published distance-baseline numbers rather than re-running tcrdist3 and
TCRbase itself. It is still a distance baseline present in the comparison.

**Do not include one** — Lu et al., *Nature Methods* 23:248–259 (2026): 50 models, all learned
(`observed` — the full PMC text was grepped for TCRdist / TCRbase / TCRMatch / GLIPH / ClusTCR /
kNN / edit distance / Hamming, zero hits, and the word "baseline" absent from the article body
entirely); Drost et al., ePytope-TCR, *Cell Genomics* 5:100946 (2025): 18 general + 3 categorical
predictors, all learned models; Liao et al., arXiv:2606.04994 (2026): eight learned models
(ERGO, ERGO-II, NetTCR2.0, NetTCR2.2, TITAN, PanPep, SCEPTR, EPACT).

*Decision:* the true claim is narrower than the first version of this entry allowed. Both IMMREP
workshops and TCRcube include distance baselines; the three large 2025–2026 assessment papers do
not. So the defensible statement is **"the recent large-scale assessment papers omit distance
baselines, while the challenge/benchmark-reuse papers include them"** — not anything field-wide.

**Note on how this entry was wrong.** The first version recorded that TCRcube's Cell Systems
publication "could not be retrieved" and stopped there under the rule about unanswerable
questions. That was a misapplication of the rule. **Non-retrieval by a search tool is not
evidence of absence.** Three queries against one search backend returned only the GitHub
repository; the paper existed the whole time and was reachable directly via its PMC identifier.
The rule exists to stop *substituting a related answer for the one asked* — it does not license
converting "my tool did not surface it" into a reported negative. The correct move was to say
the search backend returned nothing and that this settles nothing, or to try another access
route, which is what resolved it. Logged here because it is the same shape as the §11 pattern in
findings.md: a label ("unanswerable from public sources") that did not match what was actually
measured ("not returned by these queries").

---

## Part C — deep research summary

One task, processor `ultra`, run `trun_91050b9898fa485e9c119205fce96e7e`. Reported, not acted on.

**Best reported unseen-epitope numbers, with metric and dedup.** The rigorous benchmarks cluster
near chance. TChard (Grazioli et al.) reports **AUROC ≈ 0.55** under a peptide-level hard split
(85/15 minimum holdout; no sequence-similarity threshold reported). Lu et al., *Nature Methods*
2026 reports best unseen **AUPRC 0.55** (ImRex), ATM-TCR 0.52, and **13 of 28 models at or below
0.5**, under CD-HIT >95% removal. The strongest positive claim is **CATCR-D: AUROC 0.891 ± 0.006**
on an epitope-held-out external fold — but the ± is not identified as a confidence interval, no
edit-distance or clustering control is reported, and the split is not independently reproduced.
MixTCRpred's leave-one-epitope-out **median AUC 0.59**, with 11 of the 16 cases above 0.8
involving a training epitope differing by a single residue under the same MHC. A 2026
structure-based GNN reached median ROC-AUC > 0.6 on one peptide-holdout cluster and not
significantly above 0.5 on another; removing one leaked peptide dropped NetTCR-2.2 to
**0.4953 ± 0.0554**. **No reliable unseen-epitope result using AUC0.1 was found.**

**Has anyone moved meaningfully above chance?** *In individual reported experiments, yes; as a
reproduced field-wide result, no.* The synthesis: "no method has yet demonstrated broadly
reproduced, confidence-interval-certified zero-shot generalization on a standardized public
benchmark with fully disclosed deduplication and negative controls." The positive cases (CATCR-D,
the favourable structure cluster, MixTCRpred's 16 transfers) consistently retain sequence, MHC or
structural proximity to training targets; the negative assessments are the ones that remove or
expose those shortcuts. This is a coherent pattern rather than a contradiction.

**What the field names as the bottleneck — and they broadly agree.** Four layers of one problem,
in decreasing emphasis. (1) **Labelled-data quantity and diversity** is the dominant named
constraint: ePytope-TCR attributes unseen-target failure to "the lack of diverse training data";
Lu et al. find performance rises with TCRs per epitope yet collapses on unseen epitopes;
MixTCRpred shows transfer needs a highly similar same-MHC training epitope. (2)
**Negative-sample construction** as a major confounder — IMMREP23's own leakage finding, TChard's
demonstration that random negatives create disjoint positive/negative CDR3 distributions enabling
memorisation, Lu et al. on external negatives introducing batch-like confounders. (3) **Task
formulation**: recognising known peptide-specific patterns is not the same problem as modelling
compatibility over arbitrary TCR/peptide/MHC. (4) **Representation is necessary but
insufficient** — the embedding benchmark found near-AUC-0.5 on epitope split across all methods,
and Lu et al. found richer-feature models still near AUPRC 0.5. The report states explicitly that
the evidence "does not support 'use a larger language model' as the primary solution." These do
not conflict: poor negatives make the available labels less informative, which compounds scarcity.

**Bearing on us.** Two items land directly. ClusTCR was strong on random and TCR splits and
**near chance on epitope split** — the same seen/unseen asymmetry our §3a ⑤ and belief ⑧ report
for the edit-distance k-NN, independently observed. And the report's evaluation-standard
recommendation #3 is "Exact-match removal is not enough. Report CD-HIT/MMseqs2 identity,
normalized Levenshtein, and preferably structure-aware similarity" — which is Part A's finding
arriving from the opposite direction.

---

## Documents that must change — ALL APPLIED at closeout, 2026-09-09

*Rewritten after Part D. The first version of this list was written on Part A's Levenshtein
result and overstated every item.*

| item | target | status |
|---|---|---|
| 1 | `findings.md` §7a dedup qualifier | **applied** — table now carries the regime and the sub-3 value |
| 2 | `findings.md` §8c quote the sub-3 interval | **applied** — §8c rewritten; "and is now stronger" withdrawn, and item 2's own instruction to "note the magnitude roughly halves" is **superseded**: the attenuation interval is [−0.1%, 98.2%], p = 0.0511, so "roughly halves" is a point estimate with no support ([cdhit_and_issues.md](cdhit_and_issues.md) Part 2) |
| 3 | `belief_list.md` ⑥ verdict | was already applied in Phase F; final verdict added at closeout |
| 4 | `eval_set_construction.md` record exact-match as the weakest standard | **applied** |
| 5 | `findings.md` §5 Limits: ESM-C/ESM-3, no published unseen AUC0.1 | **applied** |
| 6 | `findings.md` §11 fourth instance | **applied**, plus a fifth found in the issue audit |
| 7 | no change needed: `findings.md` §8f | unchanged, still correct |
| 8 | no change needed: §9a/§9b retirements | §9a unchanged; §9b amended only to correct "operators" → "estimators" |

1. **`docs/findings.md` §7a, results table (line ~359).** `k-NN, edit distance` **0.5654
   [0.546, 0.585]** is reported with no deduplication qualifier. It is an exact-match number. At
   the strictest published standard (3 substitutions) the same arm is 0.5302 [0.5175, 0.5486].
   The table needs its dedup standard stated.

2. **`docs/findings.md` §8c, "What survives, and is now stronger".** Cites
   −0.0296 [−0.0415, −0.0188] and −0.0230 [−0.0347, −0.0122], both exact-match. **The claim
   survives** at 3 substitutions but at −0.0155 [−0.0329, −0.0012], p = 0.024. The section should
   quote the sub-3 interval as the defensible one and note the magnitude roughly halves.

3. **`docs/belief_list.md` ⑥.** Verdict recorded in the Phase F section: HELD at every published
   standard, with the sensitivity table and the mandatory dedup qualifier. Its overturn condition
   anticipated a learned model rising, not the baseline falling.

4. **`docs/eval_set_construction.md`.** Documents exact-match removal as the leakage control.
   Should record that this is the weakest of the published 2025–2026 standards (B2), that 68.5%
   of eval TCRs sit within 3 substitutions of a training CDR3β and 80.8% within Levenshtein 3,
   and that the project's results are reported at the exact-match standard.

5. **`docs/findings.md` §5, "Limits".** Add: no published evaluation of ESM-C or ESM-3 on TCR
   specificity or CDR3 representation (B4); and no unseen-epitope result using AUC0.1 exists
   anywhere in the retrieved literature (Part C), so our unseen column is uncomparable to
   published work by construction.

6. **`docs/findings.md` §11.** Add the fourth instance of the restatement pattern, recorded in
   `belief_list.md` Phase F: the evaluation set a claim is computed on is itself a restatement of
   an external convention, and unlike the first three it cannot be covered by an internal
   consistency test.

7. **No change needed: `docs/findings.md` §8f.** B3 found nothing bearing on the operator
   comparison. It stands as written.

8. **No change needed: §9a/§9b retirements.** Part D does not revive either. §9b's reasoning —
   that the representation half is a replication of Nagano et al. — is unaffected by the dedup
   standard.

---

## Part D — substitutions-only, and the sensitivity curve

`scripts/run_dedup_sensitivity.py` → `data/dedup_sensitivity.json`,
`data/dedup_sensitivity_per_peptide.csv`. New files; no frozen artifact touched. Score vectors
are computed once and re-evaluated under each mask — nothing is refit per radius, because
deduplication changes which test rows count, not the models.

### How unequal lengths are handled

A substitution count is **undefined** between sequences of different length, so an eval CDR3β is
compared only against training CDR3β **of the same length**, and is removed at radius *r* only if
some same-length training sequence is within *r* substitutions. A sequence with no training
counterpart of its length survives every finite radius (22 eval rows).

This is the strict reading and the conservative one — it removes the fewest rows. The alternative,
padding the shorter sequence, scores a length-14/15 pair sharing a prefix as distance 1, which is
an indel counted as a substitution and reintroduces exactly what "substitutions only" excludes.

### The curve

Representation effect is `esm_retrieval − edit_retrieval`, paired bootstrap over the same 48
peptides. All 48 peptides remain scorable at every radius, and **no peptide falls below 5
surviving positives anywhere in the sweep**.

| regime | rows removed | TCRs removed | support min/med/max | `edit_retrieval` | `esm_retrieval` | `svc_per_peptide` | representation effect | p |
|---|---|---|---|---|---|---|---|---|
| full | 0.0% | 0.0% | 52 / 182 / 500 | 0.5654 [0.5461, 0.5849] | 0.5358 [0.5230, 0.5498] | 0.5424 [0.5301, 0.5575] | **−0.0296 [−0.0415, −0.0188]** | <0.001 |
| sub 0 | 0.0% | 0.0% | 52 / 182 / 500 | 0.5654 [0.5461, 0.5849] | 0.5358 [0.5230, 0.5498] | 0.5424 [0.5301, 0.5575] | −0.0296 [−0.0415, −0.0188] | <0.001 |
| sub 1 | 12.7% | 12.3% | 41 / 162 / 488 | 0.5494 [0.5338, 0.5687] | 0.5234 [0.5143, 0.5346] | 0.5310 [0.5211, 0.5444] | −0.0261 [−0.0394, −0.0142] | <0.001 |
| sub 2 | 40.3% | 39.8% | 23 / 112 / 398 | 0.5390 [0.5245, 0.5562] | 0.5164 [0.5099, 0.5254] | 0.5210 [0.5137, 0.5313] | −0.0226 [−0.0373, −0.0091] | <0.001 |
| **sub 3** (Liao) | **68.6%** | **68.5%** | 10 / 56 / 240 | **0.5302 [0.5175, 0.5486]** | 0.5147 [0.5075, 0.5251] | 0.5123 [0.5052, 0.5235] | **−0.0155 [−0.0329, −0.0012]** | **0.024** |
| lev 3 | 80.9% | 80.8% | 5 / 32 / 174 | 0.5246 [0.5127, 0.5432] | 0.5108 [0.5027, 0.5260] | 0.5129 [0.5039, 0.5268] | −0.0138 [−0.0300, +0.0027] | 0.110 |

**Radius 0 removes 0.0% of rows.** That is the sweep's own control: our exact-match dedup already
removed every verbatim CDR3β, so a same-length zero-substitution radius has nothing left to take.
The sweep reproduces the frozen full-evaluation numbers exactly at that point.

**Our substitutions-only implementation lands inside Liao's reported band.** They report the
criterion eliminating "between 40% and 70% of sequences in test datasets depending on datasets
and models". Ours removes 40.3% at radius 2 and 68.6% at radius 3 — the two ends of their range.
That is independent evidence the implementation matches theirs, which the Levenshtein version did
not (80.8%, above their band).

### What the curve shows

**The effect survives Liao's actual criterion.** At sub 3 it is −0.0155 [−0.0329, −0.0012],
p = 0.024. The interval clears zero. It dies between sub 3 and lev 3 — that is, it survives
removing 68.6% of test rows by substitution and fails only once indels are also counted, which
takes removal to 80.9%.

The decay is monotone and roughly halves the effect: −0.0296 → −0.0261 → −0.0226 → −0.0155 →
−0.0138. Every arm's absolute score falls with radius, `edit_retrieval` fastest (0.5654 → 0.5302
→ 0.5246), which is the expected direction — its score *is* similarity to the training set.

So the honest statement is **not** "the headline collapses" and **not** "the headline is
unaffected". It is: *the representation effect is real at every published deduplication standard
including the strictest one, its magnitude is roughly halved by the strictest, and it becomes
unresolvable one step beyond any standard anyone currently uses.* The p = 0.024 at sub 3 is a
weak result compared to the p < 0.001 at full evaluation, and the writeup should quote the
sub-3 interval rather than the full-evaluation one.

### Correction to the Part A report

Part A concluded the headline collapsed. That conclusion came from applying Levenshtein where the
cited standard is substitutions, and it is wrong about Liao's standard. The Levenshtein numbers
were correctly computed and correctly labelled as an upper bound at the time; the error was
treating an acknowledged upper bound as the answer rather than running the criterion actually
cited. This is the fourth instance of the §11 pattern and is logged in `belief_list.md`.
