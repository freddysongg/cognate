# Cognate — final state

**Status: CLOSED, 2026-09-09.** Not paused. No further work is planned, nothing here is
publishable, and the learning goal was met. This is the top-level document; it assumes you
remember nothing and links everything else rather than restating it.

---

## 1. What the project asked

A **T-cell receptor (TCR)** recognises a short protein fragment called an **epitope** (used
interchangeably here with **peptide**, ~9 amino acids) displayed on a cell surface, and the part
of the receptor that does most of the recognising is a variable loop called **CDR3β** (~12 amino
acids in the convention used here); predicting which TCR binds which epitope is an open problem.
The project asked one question: **does a protein language model's learned representation of a
CDR3β beat simply counting letter differences between it and CDR3βs already known to bind that
epitope?** The language model was **ESM-2** (a transformer trained on protein sequences, used
frozen at 8M and 35M parameters); the letter-counting comparison was normalised **Levenshtein
distance** (the number of single-character edits between two strings, scaled by length); and the
scoring rule for both was **retrieval** — score a candidate by its highest similarity to any known
binder of that epitope. The answer is **no, the language model loses**, which was already
published, so the project's real output is the set of methodological failures it walked into on
the way to confirming it.

### Glossary

| term | meaning here |
|---|---|
| TCR | T-cell receptor. Has α and β chains; only the β chain's CDR3 loop is used in this project |
| CDR3β | The hypervariable loop of the TCR β chain. IMMREP core convention (flanking C and F/W stripped): 4–23 residues, median 12 |
| epitope / peptide | The short fragment the TCR recognises, ~9 residues |
| MHC / HLA | The molecule that displays the epitope. Recorded in the data, not modelled here |
| IMMREP23 | A 2023 community benchmark for TCR–epitope prediction. Supplies the fixed *training* set of known binders |
| VDJdb | A public database of TCR–epitope pairs. Supplies the *evaluation* set built for this project |
| ESM-2 | Meta's protein language model. Used frozen — never fine-tuned — at layer 10 of the 35M model for every headline number |
| retrieval / k-NN | Score a test TCR by its maximum similarity to any known binder of that peptide. No learning |
| macro AUC0.1 | The metric. See §3 |
| seen / unseen | Whether the peptide also appears in the training set. 48 seen, 40 unseen |
| deduplication | Removing evaluation rows too similar to training rows. **The convention is set outside this project and it moved during it** — this turned out to matter more than anything else |

---

## 2. What was built

**Evaluation set** — [`docs/eval_set_construction.md`](eval_set_construction.md),
`data/vdjdb_eval.csv`. Derived from VDJdb release 2026-06-03. **88 peptides (48 seen / 40 unseen),
116,658 rows, 19,443 positives, 5 negatives per positive drawn from the same peptide.** Every TCR
appearing anywhere in IMMREP23 training was removed — stricter than removing only the exact
(peptide, TCR) pair. Frozen: sha256 verified identical across both fork worktrees and all four
relevant commits ([`data/frozen_contract_hashes.json`](../data/frozen_contract_hashes.json)).

**Metric** — `src/cognate/metrics.py`. **Macro AUC0.1**: partial ROC AUC truncated at 10% false
positive rate, McClish-standardised so random = 0.5 and perfect = 1.0, computed independently per
peptide and averaged over peptides. Intervals are a two-level paired bootstrap: peptides with
replacement, then rows within peptide, with both arms sharing the row draw. Implemented from the
IMMREP23 README's prose spec; **never independently verified** — the Kaggle scorer for that closed
competition is broken and rejects its own reference file (§5 of [findings.md](findings.md)).

**Arms.** All nine, seen-slice macro AUC0.1, **exact-match deduplication regime**. Every one of
these numbers is regime-dependent; see §4.

| arm | what it is | representation | operator | seen | unseen |
|---|---|---|---|---|---|
| `edit_retrieval` | the baseline | normalised Levenshtein | retrieval, per-peptide | **0.5654** [0.546, 0.585] | 0.5000 *(degenerate)* |
| `esm_retrieval` | the headline comparison | ESM-2 35M L10, mean-pooled, L2 | retrieval, per-peptide | **0.5358** [0.523, 0.550] | 0.5000 *(degenerate)* |
| ESM-2 35M layer 6 | depth control | same, layer 6 | retrieval | 0.5434 [0.529, 0.559] | 0.5000 *(degenerate)* |
| `svc_per_peptide` | SCEPTR-matched parametric arm | ESM-2 35M L10, standardised | linear SVC, per-peptide | 0.5424 [0.5301, 0.5575] | 0.500 *(degenerate)* |
| `logistic_global` | Phase 1 head | ESM-2 35M L10, standardised | logistic, global | 0.5107 [0.5021, 0.5228] | 0.4997 |
| `svc_global` | estimator control | ESM-2 35M L10, standardised | linear SVC, global | 0.5078 [0.5002, 0.5179] | — |
| fork 1 · `1a` | contrastive TCR metric | trained projection on frozen ESM-2 | retrieval | 0.5320 [0.5195, 0.5455] | 0.5000 *(degenerate)* |
| fork 1 · `1b` | two-tower peptide↔TCR | trained projection, both towers | cosine | 0.5075 [0.5023, 0.5169] | 0.5018 [0.4974, 0.5070] |
| fork 2 · `2a` | residue cross-attention | per-residue ESM-2 | attention + MLP | 0.5039 [0.4989, 0.5101] | 0.4985 [0.4907, 0.4999] |
| fork 2 · `2b` | mean-pool control | per-residue ESM-2 | masked mean + MLP | 0.5093 [0.5008, 0.5219] | 0.5022 [0.5000, 0.5129] |
| random | floor | — | — | 0.5009 [0.498, 0.505] | 0.5015 |

Two forks were built in isolated worktrees off one shared commit so their results could be
compared; the frozen contract that makes that comparison valid is
[ADR 0001](ADR/0001-frozen-contract.md).

---

## 3. What was found

Each finding carries its evidence, its interval, and the scope it does **not** extend past.
Numbered independently of any prior document's numbering.

**1. Edit distance beats mean-pooled ESM-2 under an identical retrieval rule.**
Δ = `esm_retrieval − edit_retrieval` = **−0.0296 [−0.0415, −0.0188]**, p < 0.001, paired over 48
peptides. `observed`, reproduced independently three times (`knn_esm_cosine.json`,
`operator_diagnostic.json`, `cdhit_and_attenuation.json`).
*Scope:* exact-match dedup, CDR3β only, one evaluation set, one model family, frozen backbone, seen
peptides only. **Already published** — Nagano et al., *Cell Systems* 16(1):101165 (2025),
highlight ①. This is a replication on a broader evaluation, not a discovery.

**2. The result holds at every depth and size tested, and coarser is better.**
Five configurations (35M and 8M, last and middle layers), all five paired CIs excluding zero. Edit
distance produces **122 distinct scores** across 73,440 rows where ESM-2 cosine produces
60,000–67,000, so this is not a tie-handling artifact. Middle layers beat last layers at both
sizes, matching a measured norm collapse (mean L2 norm 7.20 at layer 12 vs 87.74 at layer 10).
*Scope:* ESM-2 only. ESM-C and ESM-3 were never tested and no published evaluation of them on this
task was found.

**3. The gap depends on the deduplication convention, and this is the project's most useful
measurement.** Δ over the sweep: −0.0296 (exact) → −0.0261 (1 substitution) → −0.0226 (2) →
**−0.0155 [−0.0329, −0.0012], p = 0.024** (3 substitutions, Liao et al.'s standard) → −0.0138
[−0.0300, +0.0027], p = 0.110 (Levenshtein ≤ 3, beyond any published standard). At 20,000
synchronised draws the sub-3 p is **0.0439**.
*Scope:* it clears zero at the strictest **published** criterion and fails one step beyond it.
It is sensitive to peptide composition — **11 of 48** leave-one-peptide-out intervals cross zero —
and does not survive Bonferroni across the sweep's five unique looks (p = 0.120).
[outlook.md Part D](outlook.md), [redteam_curve.md](redteam_curve.md) §4.

**4. The size of that narrowing is not established.** The magnitude falls 47.7% from full to
sub 3. Paired interval on the attenuation itself, 20,000 synchronised draws:
**[−0.1%, 98.2%], p = 0.0511**. Indistinguishable from *no* attenuation and from *complete*
attenuation alike. Also computed at sub 1 (12.0% [−0.5%, 30.8%]) and sub 2 (23.8% [−6.4%, 61.8%]);
all three intervals contain zero. [cdhit_and_issues.md](cdhit_and_issues.md) Part 2.

**5. CD-HIT >95% cannot discriminate at CDR3β length.** Running the actual program (CD-HIT 4.8.1,
`cd-hit-2d -c 0.95 -n 5`) against the fixed IMMREP23 reference **retains 99.26%** of seen rows.
All 139 removals are at exactly **100.00%** identity and **not one is a substitution**: CD-HIT
divides identities by the shorter sequence's length, so one substitution only reaches 95% at
length ≥ 20, and our median CDR3β is 12. The entire 0.74% is the indel class.
*Scope:* this is a measurement of the criterion at this sequence length, not a claim about CD-HIT
or about Lu et al.'s retained set. [cdhit_and_issues.md](cdhit_and_issues.md) Part 1.

**6. NULL — cross-attention over residues adds nothing over mean-pooling them.**
`2a − 2b` = **−0.005 [−0.016, +0.005], p = 0.366**. The control was built first and hit an
independent target (2b = 0.5093 against the Phase 1 head's 0.511), which is what makes the null
readable. Seed spread 0.0019, below CI width.
*Scope:* a 1–2 head, 1–2 layer block at width 64 over a frozen backbone. The interval is tight
(±0.016 against a baseline gap of 0.056), so an effect large enough to matter is excluded — but a
CI spanning zero cannot separate "no effect" from "underpowered". **The cleanest result in the
project.**

**7. NULL — a contrastively trained projection is indistinguishable from not training.**
1a scores 0.5320; an *untrained random* projection of the same embeddings scores **0.5341**;
frozen cosine at the same layer is 0.5358. A learning-curve sweep peaks at 0.5380 around epoch 4
and decays to 0.5257 by epoch 40.
*Scope:* frozen backbone. With the backbone frozen a projection can only reweight dimensions the
pooled embedding already has.

**8. NULL — nothing in the project scored above chance on unseen peptides.**
Every arm, every regime. 1b is the only construction with a *mechanism* on unseen peptides
(~43,190 distinct scores where retrieval produces exactly 1) and it lands at
**0.5018 [0.4974, 0.5070]**. Retrieval arms are *structurally* unable to score unseen peptides —
the per-peptide database is empty, every row takes the same default, and a constant column is
exactly 0.5 by arithmetic, not by measurement.
*Scope:* none needed. Unchallenged by anything in the record.

**9. The 2×2's "operator" axis was measuring scope, not operator.**
The +0.0250 gap between `esm_retrieval` and `logistic_global` decomposes exactly: scope
(`svc_per_peptide − svc_global`) = **+0.0346 [+0.0223, +0.0482]**, estimator
(`svc_global − logistic_global`) = −0.0029, operator (`svc_per_peptide − esm_retrieval`) =
**+0.0067 [−0.0006, +0.0142], p = 0.076 — spans zero.** Given the same per-peptide scope, the
parametric arm closes the whole gap. [findings.md](findings.md) §8.

**10. The frozen contract held.** sha256 of `metrics.py`, `split.py`, `features.py` and
`vdjdb_eval.csv` is identical at the shared foundation commit, both fork tips, `main`, and all
three worktrees on disk. **PASS.** `observed`,
[`data/frozen_contract_hashes.json`](../data/frozen_contract_hashes.json).
*Limit:* the embedding caches are not in git; all worktrees resolve to one shared directory, so
their identity holds by shared path, and their hashes attest to current contents only — nothing
retroactively proves the bytes each fork read at run time.

---

## 4. What was retired

Each of these was believed at some point and is **withdrawn**, not merely qualified.

| retired claim | why | where |
|---|---|---|
| **The operator axis** | The 2×2 was read as *representation × operator*. Its operator axis changed scope at the same time; with scope controlled the operator term spans zero (p = 0.076). Worse, the contrast also changes normalisation — `svc_per_peptide` standardises, `esm_retrieval` L2-normalises — so it was never a clean operator swap even after §8b. | [findings.md](findings.md) §8b, §8e; [redteam.md](redteam.md) §0b |
| **The edit-specific mechanism** | The story was that the gap narrows under deduplication because edit distance is peculiarly dependent on near-duplicates. Both arms lose reference signal, and edit distance retains the **largest** proportion of its above-chance excess (45.4% vs ESM 39.5% vs SVC 27.5%). A uniform-compression account fits nearly as well and is untested. The cause of the narrowing is **unidentified**. | [redteam_curve.md](redteam_curve.md) §2 |
| **"Roughly halves"** | A 47.7% point estimate with no interval. The interval is [−0.1%, 98.2%], p = 0.0511. Withdrawn everywhere. | [cdhit_and_issues.md](cdhit_and_issues.md) Part 2 |
| **"Every published deduplication standard"** | Three standards were named; only two had been run. The third, CD-HIT >95%, was run at closeout and **cannot express a substitution at CDR3β length** — it retains 99.26%. It is satisfied and contributes nothing. Honest form: *held at the two standards that can discriminate at this sequence length, and at a third that cannot.* | [cdhit_and_issues.md](cdhit_and_issues.md) Part 1; [belief_list.md](belief_list.md) final verdicts |
| **The 2×2 as a contribution** | The representation half is a replication of Nagano et al. (2025), also stated at abstract level by IMMREP22 and by TITAN. The operator half is a confound. What remains is a replication on a broader evaluation, retained as a methodology note. | [findings.md](findings.md) §9b |
| **"What survives, and is now stronger"** | §8c's former title. "Stronger" described the number of estimators agreeing, which is not a measure of strength. Rewritten. | [findings.md](findings.md) §8c |

---

## 5. Why the project stopped

Not because the result was negative — a clean replication plus four well-controlled nulls is a
fine outcome. It stopped because **the last remaining candidate for an original contribution did
not survive a literature check**, and that check is the honest reason.

The candidate was finding 5: that a 2026 *Nature Methods* benchmark used CD-HIT >95% as its
train/test leakage control on CDR3-length sequences, where the criterion is degenerate. A one-pass
literature check ([cdhit_litcheck.md](cdhit_litcheck.md)) established three things, in order of
how much each cost:

1. **The consequence is derivable from documented behaviour.** CD-HIT's `-c` help text states, in
   its first line, that identity is "number of identical amino acids in alignment divided by the
   full length of the shorter sequence". Everything follows from that. It is not documented *as a
   short-sequence warning* — CD-HIT's stated limitations are low thresholds and very long
   sequences, and neither founding paper (Li & Godzik 2006; Fu et al. 2012) mentions short
   sequences at all — but derivable-and-unstated is a much weaker claim than undocumented.
2. **The antibody field has argued around it since 2023.** Saputri et al., *mSystems* 8(6), 2023:
   *"Typical pseudo sequences are long enough to provide sufficient alignment coverage at rigorous
   sequence identity thresholds … which is not the case for CDR3 sequence alignments, in
   general."* Their fix is structural — concatenate CDR1+CDR2+CDR3 into a longer pseudo-sequence
   before clustering at all. Same problem shape, stated in coverage terms rather than substitution
   counts, in the neighbouring subfield, three years earlier.
3. **It is already compensated for inside our own subfield.** T-SCAPE, *Science Advances* (2025),
   Methods: CD-HIT at *"-c 0.8, to ensure less than two amino acids mismatches are tolerated for
   9-nucleotide oligomer fragment"*. That is our arithmetic — translating an identity threshold
   into a residue-mismatch budget at a known sequence length — performed in public, in print, a
   year earlier.

**No discovery claim survives.** What is left is a measurement on one benchmark: the criterion
retains 99.26%, and 100% of what it removes is the indel class rather than the substitution class.
That is worth having in the repository and is not worth a paper. Continuing would mean either
re-running arms that already answered their question or making a novelty claim the literature does
not support, and neither is a reason to keep going.

---

## 6. Everything else

| document | what it holds |
|---|---|
| [`findings.md`](findings.md) | The full technical record: what was built, all findings, the Session 5 bug, limits, the operator diagnostic, retirements, errata, the restatement pattern |
| [`lessons.md`](lessons.md) | **The part with value outside this project.** Five failure patterns and four things that worked |
| [`belief_list.md`](belief_list.md) | Eight pre-registered claims with overturn conditions, and final verdicts on all of them |
| [`eval_set_construction.md`](eval_set_construction.md) | How the n=88 evaluation set was built, and its limits |
| [`outlook.md`](outlook.md) | Deduplication recomputation, the sensitivity curve, targeted literature lookups, deep-research summary |
| [`redteam.md`](redteam.md) | Red team pass 1 — shared contract sweep, SCEPTR source check |
| [`redteam_curve.md`](redteam_curve.md) | Red team pass 2 — the sensitivity curve and its claim. The strongest attack on the project is §6 |
| [`cdhit_and_issues.md`](cdhit_and_issues.md) | CD-HIT regime, attenuation intervals, full issue audit |
| [`cdhit_litcheck.md`](cdhit_litcheck.md) | The literature check that closed the project |
| [`negative_sampling.md`](negative_sampling.md) | Negative construction and its confounds |
| [`session_log.md`](session_log.md) | Chronological working record |
| [`ADR/0001-frozen-contract.md`](ADR/0001-frozen-contract.md) | The four frozen files and the procedure for changing them |

**Reproducing anything:** `findings.md` §6. A fresh clone needs `bash scripts/fetch_data.sh`
before the test suite passes — the raw VDJdb and IMMREP23 dumps are gitignored bulk data.
