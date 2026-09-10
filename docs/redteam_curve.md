# Red team — sensitivity curve and its claim

Audit: 2026-09-09. **No class A established; substantive class B findings below.** A = invalidates the central claim, B = weakens or narrows it, C = cosmetic. Missing evidence for a broader claim is classified B; no tested effect reversed sign.
Read-only apart from this report. Existing embeddings and retrieval rules only; no training, fitting, embedding generation, new arms, or changes to results. Δ = ESM retrieval − edit retrieval. Checks ran in memory.

## 1. Construction of the curve

**(C) Denominator:** the 22 unmatched-length rows cover the entire evaluation; only **17 (2 positives, 15 negatives)** enter the 48-seen-peptide headline.
Only IPSINVHHY has both classes among these rows (1 positive/1 negative): unmatched AUC0.1 **0.47368/0.47368**, versus rest **0.57106/0.55520** (edit/ESM).
The other 18 peptide subsets have undefined unmatched-only AUC (one class). The four unseen peptides—EIYKRWII, LPRRSGAAGA, LVVDFSQFSR, VMTTVLATL—have rest AUC **0.50000/0.50000**, and are outside the claim.
Rest-of-peptide AUCs for every affected seen peptide (edit/ESM):
- AVFDRKSDAK 0.50003/0.49526; ELAGIGILTV 0.52392/0.50940; GLCTLVAML 0.59469/0.54272.
- IMDQVPFSV 0.50473/0.50714; IPSINVHHY 0.57106/0.55520; KAFSPEVIPMF 0.70403/0.63741.
- LLLDRLNQL 0.50614/0.50189; NLNCCSVPV 0.51615/0.51687; NLVPMVATV 0.51731/0.50378.
- QYIKWPWYI 0.58434/0.53899; RAKFKQLL 0.51442/0.50351; RLRAEAQVK 0.50458/0.50316.
- RPPIFIRRL 0.61719/0.54878; TPQDLNTML 0.53913/0.52321; TPRVTGGGAM 0.51878/0.51718.
Their difficulty is not estimable reliably, but removing all 17 moves sub-3 Δ by only **−0.000000752**. This cannot explain the observed decay.
**Support control:** 2,000 subsets per nontrivial regime, seed 20260909, sampled without replacement from FULL within each peptide and label, matching each regime's exact positive **and negative** counts; identical selected rows for both arms. All 48 peptides retained. Brackets are the central 95% of random-subset effects, not peptide-bootstrap CIs.

| Matched regime | Random edit | Random ESM | Random mean Δ [subset range] | Observed dedup Δ |
|---|---:|---:|---:|---:|
| full / sub 0 (unchanged) | 0.56538 | 0.53577 | −0.02960 [unchanged] | −0.02960 |
| sub 1 | 0.56542 | 0.53581 | −0.02961 [−0.03139, −0.02784] | −0.02606 |
| sub 2 | 0.56565 | 0.53603 | −0.02962 [−0.03357, −0.02594] | −0.02257 |
| sub 3 | 0.56630 | 0.53676 | −0.02954 [−0.03702, −0.02253] | −0.01548 |
| lev 3 | 0.56717 | 0.53734 | −0.02983 [−0.04033, −0.01861] | −0.01383 |
**Support loss does not reproduce the decay.** At sub 3 the mean moves only +0.000063, versus observed +0.014124; Monte Carlo SE of the control mean is 0.000084. Small arm-level upward bias is visible, but largely cancels in Δ.
Training databases are identical: one 11,312-row training frame supplies all scores before any radius loop; only evaluation masks change ([script:137–197](/Users/freddy/Documents/repos/cognate/scripts/run_dedup_sensitivity.py:137)). Both retrieval arms reproduce all six saved macro scores to rounding.
**(B) Bootstrap description:** all regimes use seed 0, 1,000 draws, and paired peptide-plus-within-peptide-row resampling, not peptide-only resampling. Arms share draws within a regime; cross-radius draws are not synchronized, because changing row counts changes RNG consumption. No paired test of the decay itself is reported ([metrics:487](/Users/freddy/Documents/repos/cognate/src/cognate/metrics.py:487)).
The in-memory count-based pAUC calculation matched the repository's sklearn metric across all peptide/regime/arm cells to ≤1.12×10⁻¹⁶.

## 2. The asymmetry argument

**(B) Absolute decay is not evidence of edit-specific susceptibility.** Scores and distances above the requested 0.5009 reference are:

| Regime | Edit score (above reference) | ESM score (above reference) | SVC score (above reference) |
|---|---:|---:|---:|
| full / sub 0 | 0.5654 (+0.0645) | 0.5358 (+0.0349) | 0.5424 (+0.0415) |
| sub 1 | 0.5494 (+0.0485) | 0.5234 (+0.0225) | 0.5310 (+0.0301) |
| sub 2 | 0.5390 (+0.0381) | 0.5164 (+0.0155) | 0.5210 (+0.0201) |
| sub 3 | 0.5302 (+0.0293) | 0.5147 (+0.0138) | 0.5123 (+0.0114) |
| lev 3 | 0.5246 (+0.0237) | 0.5108 (+0.0099) | 0.5129 (+0.0120) |
Full→sub 3 losses are **edit 0.0352, ESM 0.0211, SVC 0.0301**. Edit falls faster absolutely, but retains **45.4%** of its excess, versus ESM **39.5%** and SVC **27.5%**: its proportional degradation is the smallest.
Reference distributions below are **10th/50th/90th percentiles**, row-weighted over surviving seen rows. Distance = minimum raw Levenshtein to a training positive **for the row's peptide**; cosine = maximum ESM cosine to that same database. “Positive” restricts evaluation labels to 1.

| Regime | Distance, all | Distance, positive | Cosine, all | Cosine, positive |
|---|---|---|---|---|
| full / sub 0 | 3/5/8 | 2/4/8 | .97972/.99290/.99625 | .98093/.99342/.99676 |
| sub 1 | 3/5/8 | 3/5/8 | .97941/.99276/.99612 | .98023/.99310/.99633 |
| sub 2 | 3/5/9 | 3/5/8 | .97889/.99246/.99590 | .97996/.99274/.99602 |
| sub 3 | 4/6/9 | 4/6/9 | .97762/.99202/.99560 | .97889/.99243/.99580 |
| lev 3 | 5/6/10 | 4/6/9 | .97692/.99161/.99535 | .97825/.99200/.99553 |

**(B) Both arms lose reference signal.** The cosine decline also holds among positives; it is not merely a change in the positive/negative mixture. This contradicts an exclusive edit-distance mechanism, but does not establish a different cause.
**(B) Compression is a live explanation.** A least-squares common contraction of all three full-score excesses gives sub-3 factor 0.4006 and predicted Δ −0.01186, versus observed −0.0155. It reproduces substantial narrowing, though unequal retained fractions rule out exact uniform compression; this is descriptive compatibility, not a fitted causal or inferential test.
Across **any** training peptide, raw-Levenshtein distance percentiles are full/sub0 1/2/4, sub1 2/3/4, sub2 2/3/5, sub3 3/4/5, lev3 4/4/6. 0.5009 is a recorded random-arm score, not a mathematical floor: standardized chance is 0.5 and the metric can fall to 0.47368. Sources: [saved curve](/Users/freddy/Documents/repos/cognate/data/dedup_sensitivity.json), [metric](/Users/freddy/Documents/repos/cognate/src/cognate/metrics.py:46); SVC values were read, never refit.

## 3. Validity of the comparison to Liao

**(B) The paper does not establish the implementation equivalence claimed in the notes.** These are verbatim excerpts, with locations:
- Figure 1a caption: “model-specific data-leakage control”. [Liao et al., v1](https://arxiv.org/html/2606.04994v1).
- Benchmark Design and Results, TetTCR-SeqHD overlap paragraph: “any training dataset used by the eight models tested”. [Liao et al., v1](https://arxiv.org/html/2606.04994v1).
- Benchmark preprocessing paragraph: “duplicate or nearly identical TCR/antigenic-epitope pairs”; the removal range varies “depending on datasets and models”. [Liao et al., v1](https://arxiv.org/html/2606.04994v1).

**Own-model versus union reference remains ambiguous** despite explicit model-specific controls; selecting either would be inference. The paper names pairs and CDR3β substitutions but does not specify whether exclusion requires peptide equality or how unequal lengths are handled.
**(B) The band argument is void as validation:** our 68.6% against fixed IMMREP23 training and their model/dataset-dependent 40–70% do not establish matching predicates or references ([our filter](/Users/freddy/Documents/repos/cognate/scripts/run_dedup_sensitivity.py:63)).
Sources: [Liao primary paper](https://arxiv.org/html/2606.04994v1), [claim being audited](/Users/freddy/Documents/repos/cognate/docs/outlook.md:310). Search results were read from stdout; no auxiliary output file was created.

## 4. The p=0.024

**(B) Selection/multiplicity is unresolved.** The saved sweep contains six labelled regimes but **five distinct evaluations**: full equals sub 0 exactly. The earlier Levenshtein recheck is documented before the substitution sweep; the complete number of exploratory looks and a prospective endpoint choice cannot be recovered from these artifacts ([sweep](/Users/freddy/Documents/repos/cognate/scripts/run_dedup_sensitivity.py:168), [earlier recheck](/Users/freddy/Documents/repos/cognate/scripts/run_dedup_recheck.py:1)).
No multiplicity adjustment is implemented. If an endpoint was selected from the sweep, familywise adjustment is relevant; counting six independent tests would be wrong because one is identical and the others overlap. Bonferroni nevertheless remains a valid conservative bound under dependence.
A prospectively specified conjunction that **all** specified standards pass is an intersection-union claim and does not automatically require sixfold correction. No such prospective specification is established here.

| Family/correction | Adjusted reported p | Interval for sub-3 Δ |
|---|---:|---|
| Five unique looks, Bonferroni | 0.120 | 99%: [−0.03730, +0.00405] |
| Six labelled looks, conservative Bonferroni | 0.144 | 99.1667%: [−0.03798, +0.00447] |

Intervals use 20,000 fresh draws of the same two-level paired bootstrap, 20 batches of 1,000 with seeds 20260909–20260928. Applying Holm to the **original saved** sweep p-values instead gives sub-3 adjusted p **0.048** for either family; correction choice cannot be hidden.
**(B) Monte Carlo fragility:** seed-0/1,000-draw reproduction gives **[−0.032872, −0.001180], p=0.024** exactly. The 20,000-draw estimate gives **[−0.031702, −0.000641], p=0.0405**. Thus 0.024 is a noisy bootstrap-tail estimate; using the latter p gives Bonferroni 0.2025/0.243, not 0.120/0.144.
**Bootstrap unit confirmed:** peptides are drawn with replacement, then TCR rows within each drawn peptide; both arms use the same row draw. It is neither peptide-only nor a pooled-TCR bootstrap ([implementation](/Users/freddy/Documents/repos/cognate/src/cognate/metrics.py:540)).
**(B) Leave-one-peptide-out:** all 48 omissions, each with the original 1,000 draws/seed 0. **11/48 intervals cross zero**; all point differences remain negative, ranging −0.017169 to −0.011312.
Worst upper endpoint: drop **RFPLTFGWCF**, leaving 47 peptides: **Δ −0.011312, 95% CI [−0.024739, +0.002042], p=0.108**. A 5,000-draw check, seed 20260909, gives **[−0.025224, +0.001815], p=0.0964**.
Plainly: one peptide's removal can erase nominal statistical significance; it does **not** reverse the measured ordering. The worst omission is selected from 48 sensitivity checks, not an independently selected hypothesis test.
**(B) p=0.024 tests the sub-3 gap, not the full→sub-3 change or a “halving” hypothesis.** The observed magnitude falls 47.7%; no uncertainty interval for that attenuation is supplied by the original sweep.

## 5. Scope of the written claim

**(B) “Every published deduplication standard” exceeds what was evaluated.** [B2](/Users/freddy/Documents/repos/cognate/docs/outlook.md:105) enumerates:

| Published criterion in B2 | Actually represented? |
|---|---|
| Drost et al., ePytope-TCR (2025): exact CDR3–epitope pair exclusion | Our exact CDR3β exclusion against IMMREP23 also excludes exact pairs against that reference, but is stronger/different; their exact protocol is not reproduced. |
| Lu et al., Nature Methods (2026): CD-HIT >95% similarity | **No.** No CD-HIT clustering, mask, or equivalence check appears in the sweep. The original paper explicitly uses CD-HIT ([Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12791011/)). |
| Liao et al. (2026): up to three CDR3β substitutions | Only a same-length Hamming-radius proxy against IMMREP23, with the reference/pair ambiguities in §3. |

Being approximately exact-match at a typical short sequence length does not demonstrate that CD-HIT produces an identical retained set. The absence of a reversal at sub 0–3 says nothing conclusive about an uncomputed clustering mask.
**(B) “No dedup” is also inaccurate:** full already excludes exact training CDR3β matches; it means no **additional** near-duplicate filtering ([construction:88](/Users/freddy/Documents/repos/cognate/scripts/build_vdjdb_eval.py:88)).
**Narrowest supported claim, one sentence:** On this VDJdb-derived evaluation's 48 IMMREP23-seen peptides, using CDR3β only, fixed IMMREP23 positive databases and macro standardized AUC0.1, normalized-Levenshtein nearest-positive retrieval exceeds mean-pooled ESM-2 35M layer-10 cosine retrieval under the implemented masks, with Δ changing from −0.02960 after exact-match exclusion to −0.01548 after same-length substitution-radius-3 exclusion.
That is one representation/configuration, one retrieval operator, one metric, one constructed benchmark and seen peptides only—not a result across ESM models, unseen epitopes, paired chains, or every published filtering protocol. The endpoint remains nominally significant but is sensitive to peptide composition and inference choices (§4).

## 6. The strongest attack

**(B) Reviewer paragraph:** The manuscript promotes a narrowly measured retrieval difference into a claim about every published deduplication standard without implementing one of those standards, CD-HIT, or establishing that its substitute for Liao's protocol excludes the same objects against the same reference. Its supposedly robust endpoint has a p-value that moves from 0.024 to 0.0405 merely by resolving the bootstrap more accurately, and loses nominal significance when a single peptide is removed. Both retrieval arms lose reference similarity, while edit distance retains more of its above-chance performance than ESM, so the asserted edit-specific mechanism is not demonstrated. The headline combines an incompletely matched literature comparison, fragile endpoint inference, and an untested causal explanation; the observed narrowing establishes none of those broader propositions.
**Can existing data answer this? Partly.** They reproduce the ordering, rule out unmatched-length rows and random support loss as explanations of the decay, and directly quantify peptide sensitivity and shared signal loss. The current artifacts do not establish CD-HIT equivalence, resolve Liao's underspecified predicate/reference, establish prospective endpoint selection, or identify the cause of narrowing.
**Stopped after §6. No fixes proposed.**
