# CD-HIT regime, attenuation interval, and an audit of every open issue

Run 2026-09-09. Read-only on results: no training, no new arms, nothing refit, no frozen
artifact overwritten. `data/dedup_sensitivity.json` is read and never written. New artifacts
only: `data/cdhit_and_attenuation.json`, `data/cdhit_regime_per_peptide.csv`, driver
`scripts/run_cdhit_and_attenuation.py`.

Classification as before: **(A)** invalidates a reported number · **(B)** weakens an
interpretation · **(C)** cosmetic.

**Findings: 0 class (A). 3 class (B). 2 class (C).** No reported number was wrong. The two
substantive results both narrow claims that are currently stated without qualification.

---

## Part 1 — CD-HIT >95% as a sixth regime

### 1a. Retained fraction, first

**CD-HIT >95% retains 99.26% of the seen evaluation rows.** It removes **541 of 73,440** rows
and clusters away **139 of the 18,760** distinct evaluation CDR3β.

**The regime is uninformative at CDR3β length and should be reported that way, not as a standard
the result passed.** That is the finding. It is not a failure of the run, and it is not evidence
for the claim it was supposed to test.

The arithmetic: CD-HIT computes global identity as identical residues divided by the length of
the **shorter** sequence. At length *L*, one substitution gives (*L*−1)/*L*, which reaches 0.95
only at *L* ≥ 20. Our CDR3β run 4–23 residues with median **12** under the IMMREP core
convention, and only **24** of 18,760 distinct evaluation sequences reach length 20. So at this
sequence length ">95% identity" means *identical, or identical after a deletion* — nothing else
can qualify.

That is not a prediction. Every one of the 139 removals is at exactly **100.00%** identity, read
from CD-HIT's own `.clstr` cluster assignments. Not one is a substitution. Since exact CDR3β
matches were already removed at evaluation-set construction
(`scripts/build_vdjdb_eval.py:88`), the entire 0.74% is the **indel class** — an evaluation
CDR3β that is a gapped subsequence of a longer training CDR3β:

```
eval SGTSGSSYNEQF      train ASSSGTSGSGSYNEQF   100.00%   len 12/16
eval ASSDGQVANEKLF     train ASSDGQVATNEKLF     100.00%   len 13/14
eval ASSLGASSYNEQF     train ASSLAGASSYNEQF     100.00%   len 13/14
```

This is exactly the class the substitution-radius regimes exclude by design, so CD-HIT is not
nested inside them: **65 of the 139 survive sub 1**, while **137 of the 139** are also removed by
sub 3. The regime is far weaker than sub 1 in row count and largely a subset of sub 3 in
membership.

### 1b. The sweep row

Same table format, same machinery, same seed 0 / 1,000 draws as the saved sweep.

| regime | rows removed | TCRs removed | support min/med/max | `edit_retrieval` | `esm_retrieval` | `svc_per_peptide` | representation effect | p |
|---|---|---|---|---|---|---|---|---|
| full / sub 0 | 0.00% | 0.00% | 52 / 182 / 500 | 0.5654 | 0.5358 | 0.5424 | −0.0296 [−0.0415, −0.0188] | <0.001 |
| **cd-hit >95%** | **0.74%** | **0.74%** | 52 / 181.5 / 500 | **0.5659** | **0.5356** | **0.5421** | **−0.0303 [−0.0436, −0.0193]** | **<0.001** |
| cd-hit >95%, `-s2 0.0` | 3.07% | 3.04% | 50 / 177.5 / 499 | 0.5649 | 0.5346 | 0.5405 | −0.0303 [−0.0452, −0.0183] | <0.001 |
| sub 1 | 12.7% | 12.3% | 41 / 162 / 488 | 0.5494 | 0.5234 | 0.5310 | −0.0261 [−0.0394, −0.0142] | <0.001 |
| sub 2 | 40.3% | 39.8% | 23 / 112 / 398 | 0.5390 | 0.5164 | 0.5210 | −0.0226 [−0.0373, −0.0091] | <0.001 |
| sub 3 (Liao) | 68.6% | 68.5% | 10 / 56 / 240 | 0.5302 | 0.5147 | 0.5123 | −0.0155 [−0.0329, −0.0012] | 0.024 |
| lev 3 | 80.9% | 80.8% | 5 / 32 / 174 | 0.5246 | 0.5108 | 0.5129 | −0.0138 [−0.0300, +0.0027] | 0.110 |

All 48 peptides remain scorable; the minimum surviving support is unchanged from full.

The one CD-HIT choice the paper does not pin is the length asymmetry. `cd-hit-2d -s2 1.0`
(default) absorbs an evaluation sequence only into a training sequence at least as long.
Relaxing to `-s2 0.0` triples the removal to 3.07% and leaves the effect at −0.0303. All 571 of
those removals are also at exactly 100.00% identity, in both length directions.

### 1c. Support-matched random control

Same protocol as §1 of `redteam_curve.md`: 2,000 subsets, seed 20260909, sampled without
replacement from FULL within each peptide and label, matching this regime's exact positive
**and** negative counts, identical selected rows for both arms, all 48 peptides retained.

| regime | random edit | random ESM | random mean Δ [central 95%] | observed dedup Δ | MC SE |
|---|---:|---:|---:|---:|---:|
| cd-hit >95% | 0.56538 | 0.53578 | −0.02960 [−0.02995, −0.02924] | **−0.03030** | 0.000004 |
| cd-hit >95%, `-s2 0.0` | 0.56541 | 0.53578 | −0.02963 [−0.03044, −0.02883] | **−0.03030** | 0.000009 |

The observed value sits just outside the control band for the default regime, by **0.0007**. Two
things about that, both worth stating plainly. It is real: the band is tight, and 0.0007 is
about 175 Monte Carlo SEs. It is also interpretively empty: the shift runs toward a *wider* gap,
the opposite direction from every substitution regime, on 0.74% of rows, and it does not change
any conclusion. Under `-s2 0.0` the observed value falls inside the band.

### 1d. What Lu et al. actually apply CD-HIT to — verified against the Methods

Lu, Wang, Xu, Xie, Yang, Xu, Suo, "Assessment of computational methods in predicting TCR–epitope
binding recognition", *Nature Methods* 23(1):248–259, doi 10.1038/s41592-025-02910-0.
[PMC12791011](https://pmc.ncbi.nlm.nih.gov/articles/PMC12791011/). Verbatim:

> "To prevent data leakage, we used CD-HIT to exclude highly similar sequences (>95% similarity)
> between the training and test sets. Specifically, after integrating the positive samples with
> the generated negative samples for each data group, CD-HIT was applied to eliminate these
> highly similar TCR sequences, ensuring robust and unbiased evaluation of the models."

> "In model retraining, we also used CD-HIT to exclude TCR sequences with greater than 95%
> similarity between the training and test sets, and between the training set and the
> independent test sets."

**What matches.** CD-HIT is applied to **TCR sequences** — not to TCR–epitope pairs, and not
conditioned on peptide. That is how our regime applies it and how the substitution-radius
regimes apply their reference, so the predicate is of the same kind. It is applied **between
training and test sets**, which is what the fixed IMMREP23 reference gives us. The threshold is
identical.

**What does not match. Do not read this as a demonstrated equivalence.**

1. **Removal direction.** Lu cluster a *merged* set, so which member of a cluster survives
   depends on CD-HIT's length-descending sort and input order. We remove only from the
   evaluation side against a fixed reference. Ours is the only well-defined choice when the
   reference must stay identical across all regimes, but it is not their procedure.
2. **Program variant and parameters are unspecified in the paper.** No word size, no `-s`/`-s2`,
   no statement of `cd-hit` versus `cd-hit-est`. The `-s2` sensitivity in §1b exists because the
   paper does not pin it.
3. **Sequence representation and length window.** Lu retained "TCR sequences ranging from 10 to
   18 amino acids". Our CDR3β uses the IMMREP core convention (flanking C and F/W stripped),
   4–23. This does not change the conclusion, because one substitution fails to reach 95% at
   every length in either window below 20.
4. **They apply it after merging positives with generated negatives per data group.** Our
   evaluation set's negatives are constructed differently, so "the set CD-HIT sees" is not the
   same object even where the predicate is.

**Narrowest supportable statement:** this is CD-HIT at Lu et al.'s threshold against our fixed
reference, and the retained fraction shows the threshold cannot discriminate at CDR3β length. It
is not a demonstration that Lu's retained set and ours agree.

### 1e. Exactly what was run

Not a proxy and not a reimplementation. **CD-HIT 4.8.1**, built from source in this session:

```
git clone --depth 1 https://github.com/weizhongli/cdhit    # 4f6720f573d3d9d4c835793a05f09e098003bfe9
make -C cdhit openmp=no
cd-hit-2d -i <train CDR3β> -i2 <eval CDR3β> -o <kept> -c 0.95 -n 5 -l 4 -d 0 -M 0
```

`cd-hit-2d` is the two-database form of the same program, sharing `cdhit-common.c++` with
`cd-hit`. It is the right tool here: clustering the union makes which member survives depend on
input order, and the sweep needs a mask on the evaluation side against a fixed reference.

**`-l 4` rather than the stock `-l 10` is a required deviation and is recorded in the artifact.**
CD-HIT's default silently discards every sequence of length ≤10, which is 2,081 of our 18,760
distinct evaluation CDR3β; left at the default it would have looked like a 2,081-sequence
removal. `-l 4` is the lowest value CD-HIT accepts at word length 5
(`cdhit-common.c++:428`). Two 4-mers (`AGGC`, `ASSY`) still fall below the floor; neither has an
exact training match, so both are retained by hand, and the script records them under
`too_short_to_cluster`. `-n 5` is the documented default word size for `-c 0.95`.

The script raises `FileNotFoundError` if the binary is absent rather than falling back to a
proxy.

### 1f. Class

**(B)** — `docs/belief_list.md` Phase F reads "**HELD at every published standard**". After this
run the claim is literally checkable against all three rather than two, and the third contributes
almost nothing: it removes 0.74% of rows and cannot express a substitution. Counting it as
support overstates the coverage. The sentence should say the CD-HIT criterion is degenerate at
this sequence length, not that the result passed it.

**(C)** — `docs/outlook.md:113` says "CD-HIT at >95% identity on a ~14-residue CDR3β is close to
exact match". Right in direction, incomplete in mechanism (it is exact match *plus subsequence
containment*, and containment is 100% of what it removes) and wrong on the length (median 12
under our convention, not ~14).

---

## Part 2 — a paired interval for the attenuation itself

### 2a. How the draws were synchronised

`redteam_curve.md` §1 recorded why no paired test of the decay existed:

> Arms share draws within a regime; cross-radius draws are not synchronized, because changing
> row counts changes RNG consumption.

**Resolved by drawing rows once, from the full row set, and letting each regime's mask select
from the drawn rows.** Deduplication is a deterministic function of the row, so a replicate that
resamples evaluation rows induces every regime's subset simultaneously. Peptides are drawn with
replacement first, then row positions within each drawn peptide; both arms share the row draw
exactly as `compare_macro_auc01` does within a regime. `Δ_full − Δ_regime` is therefore paired
inside each replicate.

**The cost, stated rather than hidden.** A regime's replicate row count becomes binomial around
its fixed count instead of being held at it. The *marginal* per-regime intervals under this
scheme are therefore not identical to the saved sweep's, and both are reported in §2c. The
pairing is what the attenuation needs; the saved sweep's marginals remain the ones to quote for
the gap itself.

20,000 draws, seed 20260909. `Δ_full` was negative in all 20,000 replicates, so the proportion is
well defined throughout. Peptide skip rate from a resampled subset losing a class: 0 at full,
sub 1 and sub 2; 1.04 × 10⁻⁶ at sub 3.

### 2b. The intervals

| change | absolute, Δ_full − Δ_regime | as a proportion of Δ_full | p |
|---|---|---|---|
| full → sub 1 | −0.00354 [−0.00792, **+0.00015**] | 12.0% [−0.5%, 30.8%] | 0.0598 |
| full → sub 2 | −0.00704 [−0.01604, **+0.00193**] | 23.8% [−6.4%, 61.8%] | 0.1199 |
| **full → sub 3** | **−0.01412 [−0.02774, +0.00004]** | **47.7% [−0.1%, 98.2%]** | **0.0511** |

**Every interval contains zero.** The 47.7% point estimate is unchanged and correct. What is new
is that it is indistinguishable both from *no attenuation* and from *complete attenuation*: at
sub 3 the 95% interval on the proportion runs from −0.1% to 98.2%.

The direction is consistent and the point estimates are monotone across the curve, so this is not
evidence *against* attenuation. It is the absence of evidence for it at this sample size, which
is a different thing and is the honest reading.

### 2c. Marginal deltas, this scheme against the saved sweep

| regime | this scheme, 20,000 nested draws | saved sweep, 1,000 draws, seed 0 |
|---|---|---|
| full | −0.02960 [−0.04286, −0.01848], p 0.0000 | −0.0296 [−0.0415, −0.0188], p <0.001 |
| sub 1 | −0.02606 [−0.04009, −0.01402], p 0.0000 | −0.0261 [−0.0394, −0.0142], p <0.001 |
| sub 2 | −0.02257 [−0.03891, −0.00832], p 0.0008 | −0.0226 [−0.0373, −0.0091], p <0.001 |
| sub 3 | −0.01548 [−0.03230, −0.00041], p **0.0439** | −0.0155 [−0.0329, −0.0012], p **0.024** |

Point estimates reproduce the sweep exactly. Intervals are slightly wider, the expected
consequence of the binomial row count. The sub-3 p moves from 0.024 to **0.0439**, independently
consistent with §4's finding that resolving the bootstrap properly moves it to 0.0405. It still
clears 0.05 under both, barely.

### 2d. Monte Carlo fragility, again, and worse in this direction

A 20-draw version of the identical computation gave sub 3 as **47.7% [21.3%, 94.6%], p = 0.0000**.
The percentile tails of a 20-draw sample carry no information, and the under-resolved answer looks
*more* significant, not less — the same trap as §4's 0.024 versus 0.0405, in the same direction.
This argues for a standing minimum draw count on any interval that enters a claim.

### 2e. Class

**(B)** — the attenuation is the quantity the writeup actually asserts, and it is not
established.

- `docs/belief_list.md` Phase F: "its margin is **roughly halved** by that criterion".
- `docs/outlook.md` Part D: "The decay is monotone and **roughly halves** the effect".
- `docs/outlook.md` Documents-that-must-change item 2, which instructs `findings.md` §8c to
  "note the magnitude roughly halves".

All three are true as point estimates and unsupported as inferences. Each needs the interval
attached or the claim dropped.

**Does not affect** the sub-3 gap itself. −0.0155 with a CI clearing zero is what p = 0.024 (or
0.0439) tests, and that is unchanged. **Does not resolve** §4's selection and multiplicity
question: Bonferroni across five unique looks already puts the sub-3 p at 0.120.

### 2f. The check behind the numbers

The bootstrap never calls sklearn — 7.7M partial AUCs would not finish. Its count-based McClish
partial AUC is verified equal to `sklearn.roc_auc_score(..., max_fpr=0.1)` on **every** peptide,
regime and arm cell, maximum absolute difference **2.22 × 10⁻¹⁶**, in the script's `check()`,
which raises rather than warns. `observed`.

---

## Part 3 — issue audit

21 issues open, all created 2026-09-07, all 2 days old, none ever closed, zero comments before
today. No pull requests have ever been opened on this repository. Two issues were created today
by this run.

Every issue below now carries an audit comment with its cross-links. **Nothing was closed.**

### 3a. New issues

| # | title | covers |
|---|---|---|
| [#22](https://github.com/freddysongg/cognate/issues/22) | dedup sensitivity sweep — CD-HIT >95% regime, and why it cannot discriminate at CDR3β length | Part 1 |
| [#23](https://github.com/freddysongg/cognate/issues/23) | dedup sensitivity sweep — paired interval for the attenuation, which does not clear zero | Part 2 |

Both quote the motivating red-team section, record what was run, give the result, and name the
claim they affect. Both link `main` rather than a commit, because the work is uncommitted in the
working tree at the time of filing — see the closing note.

### 3b. Audit of every open issue

State column: **done** = every acceptance item has a named artifact; **partial** = at least one
acceptance item is unmet; **standing** = a rule, not a deliverable.

| # | title | age | state | justification (specific) | cross-link that was missing |
|---|---|---|---|---|---|
| 1 | Phase 0: shared foundation | 2d | **done** | Body says "Status: complete". `d90f955`, `d7a6275`. The one unchecked box, "Create epics, labels, and child issues", is satisfied by #2–#21 existing with `epic` / `phase-0` / `fork-1` / `fork-2` / `frozen-contract` labels | `data/knn_esm_cosine.json`, `scripts/verify_residue_cache.py`, `findings.md` §7b |
| 2 | Fork 1: contrastive / metric learning | 2d | **done** | `8c7a03d`, merge `be0d1c9`. `data/fork1_results.json`: 1a 0.5320 [0.5195, 0.5455], 1b 0.5075, probes `PASS` at 0.0004, seeds `[0,1,2]` | commits, `findings.md` §7c/§7f, `belief_list.md` ⑥ |
| 3 | Fork 2: cross-attention over residues | 2d | **done** | `16a050c`, merge `d772293`. `data/fork2_results.json`: 2a 0.5039, 2b 0.5093, ablation −0.005 [−0.016, +0.005] p 0.366 | commits, `findings.md` §7d/§7f, `belief_list.md` ⑦ |
| 4 | PxK batch sampler | 2d | **done** | `contrastive.py:114` `pxk_batches`; `test_contrastive.py:90`, `:105`, `:116` | `8c7a03d`, the three tests |
| 5 | Projection head + InfoNCE loss | 2d | **done** | `contrastive.py:49`, `:73`. The stated acceptance test exists: `test_contrastive.py:121` `test_loss_decreases_on_a_memorizable_batch` | `8c7a03d`, the named test |
| 6 | 1a training loop + k-NN eval | 2d | **done**, body number wrong | `scripts/run_fork1.py`, `fork1_results.json` `1a`, `findings.md` §7c. 0.5320; paired −0.0338 [−0.0478, −0.0226] | `8c7a03d`; see §3c ② |
| 7 | 1b two-tower + cosine scoring | 2d | **done** | `fork1_results.json` `1b`: unseen 0.5018 [0.4974, 0.5070], non-degenerate, ~43,190 distinct scores. Reported prominently in `findings.md` §7c | `8c7a03d`, `findings.md` §7c |
| 8 | Shortcut probes on the learned representation | 2d | **done** | `fork1_results.json` `probes`: peptide_only 0.5000, tcr_only 0.4996, worst deviation 0.0004, verdict `PASS` | `8c7a03d` |
| 9 | Multi-seed runs (fork 1) | 2d | **done** | `config.seeds [0,1,2]`; per-seed seen 0.5316/0.5308/0.5337, spread 0.0029, `seed_spread_exceeds_ci_width: false` | `8c7a03d` |
| 10 | Emit fork1_results.json + CSV | 2d | **done**, artifact carries a known-stale value | `data/fork1_results.json`, `data/fork1_per_peptide.csv` in the Section 6 schema. But `reference.frozen_esm_cosine_seen_layer10` still reads 0.5364 against 0.5358 | `8c7a03d`, `findings.md` §10, `tests/test_reference_constants.py`; see §3d ③ |
| 11 | Fork 1 writeup | 2d | **done** | `findings.md` §7c (built/scored) and §7f (what would make it wrong); `a42c5f8` | `a42c5f8`, `belief_list.md` ⑥ |
| 12 | Variable-length per-residue batching | 2d | **done** | `crossattn.py:48`, `:72`, `:91`. Acceptance test exists: `test_crossattn.py:72` and `:105`; the two 4-mers named in the body are covered by `:128` | `16a050c`, the three tests |
| 13 | Cross-attention block | 2d | **done** | `crossattn.py:102` `CrossAttentionBlock`, `:139` `ResidueBindingModel`; `test_crossattn.py:57`, `:65`, `:143`, `:158` | `16a050c`, the four tests |
| 14 | 2b mean-pool control | 2d | **done** | `fork2_results.json` `2b` = 0.5093 [0.5008, 0.5219] against the 0.511 target, layer 10. Both variants reported together | `16a050c`, `findings.md` §7d |
| 15 | 2a training loop + component split | 2d | **done** | `scripts/run_fork2.py`, `src/cognate/split.py`, `fork2_results.json` `2a` = 0.5039 | `16a050c`, `findings.md` §7d |
| 16 | Shortcut probes (fork 2) | 2d | **done**, body number wrong | `fork2_results.json` `probes`: 0.5000 / 0.5004, verdict `PASS`. But the body's expected +0.156 gap measured 0.1183 (2a) and 0.1305 (2b) | `16a050c`; see §3c ④ |
| 17 | Multi-seed runs (fork 2) | 2d | **done** | `config.seeds [0,1,2]`; 2a per-seed 0.5043/0.5028/0.5047, spread 0.0019 | `16a050c` |
| 18 | Emit fork2_results.json + CSV | 2d | **done**, artifact carries a known-stale value | Section 6 schema plus an `attention_ablation` block. Same 0.5364 defect as #10 | `16a050c`, `findings.md` §10; see §3d ③ |
| 19 | Fork 2 writeup | 2d | **done** | `findings.md` §7d and §7f; `a42c5f8` | `a42c5f8`, `belief_list.md` ⑦ |
| 20 | Frozen-contract change procedure | 2d | **standing** | A rule, not a deliverable; no commit closes it. Held during Phase D: both forks record `frozen_contract_changes: []` and share merge-base `d7a6275` | `d7a6275`, both results JSONs |
| 21 | Comparison session | 2d | **partial** | Done: `findings.md` §7a table, §7 through §10, `belief_list.md` ⑥⑦⑧, merges `be0d1c9` / `d772293`, all at `a42c5f8`. **Not done: "Hash-verify both worktrees used identical frozen artifacts."** `grep -rn "sha256\|hashlib\|md5"` over `scripts/`, `src/`, `docs/` returns nothing | `a42c5f8`, both merge commits |

**Twenty of twenty-one issues are complete or standing. One, #21, has a genuinely unmet
acceptance item.**

### 3c. Issues referencing a claim the red team narrowed or retired

These need **rewording, not closing**. The four claims named in the brief — the operator axis,
the edit-specific mechanism, "every published deduplication standard", the p=0.024 headline —
are almost entirely absent from the issue tracker. `grep -i "operator\|dedup\|0.024\|substitut\|
cd-hit\|published standard"` over all 21 bodies returns two hits, both benign uses of "operator"
in #2 meaning "the k-NN rule", which the red team did not narrow.

**That absence is itself the finding: the red-team-narrowed claims have no tracking issue at
all.** They live only in `findings.md`, `outlook.md` and `belief_list.md`. Carried into §3d.

What the issues *do* carry is the number those claims re-score:

**① `0.5654` as a fixed target, in #2, #3 and #6.** Phase F established it is the
exact-match-deduplication value, and the same arm scores 0.5302 [0.5175, 0.5486] under Liao's
three-substitution standard. `outlook.md` Documents-that-must-change item 1 already says the
table needs its dedup qualifier; the same applies to the issue bodies.

> #2: "> **Number to beat: 0.565** (edit-distance k-NN, seen peptides). Head baseline is 0.511 against chance at 0.501."
>
> #3: "> **Number to beat: 0.565.** Internal control 2b should reproduce ~0.511."
>
> #6: "Reuse score_by_nearest_positive with a swapped similarity_fn. Beat 0.565; also report the delta against frozen ESM-2 cosine at 0.544."

**② `0.544` in #6 is the wrong frozen-cosine reference, and it is a fifth instance of the §11
restatement pattern — the first found in an issue body.** Fork 1 runs 35M layer 10, whose frozen
cosine is **0.5358**. `0.544` is the **8M layer 3** row of #1's own table (35M layer 6 is 0.543).
`findings.md` §7c used the right number: "against frozen cosine at the same layer at 0.5358".
Same shape as §11 instance #1 — a labelled row collapsed into an unlabelled target that dropped
the discriminator making it correct.

**③ `~0.511` as 2b's reproduction target, in #3 and #14.** Also an exact-match number. Lower
stakes because it is an internal reproduction target rather than a claim, but it takes the same
qualifier if either body is rewritten.

> #14: "Must reproduce ~0.511 using layer 10. Report 2a and 2b together or neither."

**④ `+0.156` as the expected train-side pooled/macro gap, in #3 and #16.** Not a red-team
narrowing, but the same defect class: the number in the body is not the number that was measured.

> #16: "Expect an inherited train-side pooled/macro gap around +0.156 under matched negatives. Not a new bug."

Measured: **0.1183** (2a), **0.1305** (2b), `data/fork2_results.json` `probes.train_pooled_macro_gap`.

### 3d. Should exist as an issue and does not

Ordered by consequence. None of these has a tracking issue today.

**① `deviations_from_sceptr` completeness — §11 instance #3, deliberately left open.**
`findings.md` §11 "What is now covered, and what is not" states the gap and the decision not to
build a test for it: *"This gap is named and left open deliberately."* A named open gap with a
standing mitigation ("treat a completeness assertion as `claimed` and re-derive it whenever it is
load-bearing") is exactly what an issue is for. Right now the decision lives only in prose that
nothing points at.

**② §11 instance #4 — the evaluation set is itself a restatement of an external convention.**
`belief_list.md` Phase F records it and `outlook.md` Documents-that-must-change item 6 instructs
`findings.md` §11 to add it. §11 still lists three instances. Untracked and undone.

**③ `fork1_results.json` / `fork2_results.json` still carry 0.5364.** `scripts/run_fork1.py:336`
and `scripts/run_fork2.py:339` are fixed in the working tree; the JSONs are not regenerated,
because regenerating them means re-running training. `findings.md` §10 records the erratum and
`tests/test_reference_constants.py` guards the drivers going forward — but nothing tracks the
committed artifacts still holding the wrong value, and nothing decides whether they should be
regenerated or annotated. This is the most concrete of the list.

**④ `outlook.md` "Documents that must change", items 1, 2, 4, 5, 6 — a written work list with no
tracker.** Verified against the current files:

| item | target | done? |
|---|---|---|
| 1 | `findings.md` §7a table needs its dedup qualifier | **no** — the table still reads `k-NN, edit distance 0.5654 [0.546, 0.585]` unqualified |
| 2 | `findings.md` §8c should quote the sub-3 interval | **no** — §8c still cites −0.0296 and −0.0230 only |
| 3 | `belief_list.md` ⑥ verdict | **yes** — Phase F section |
| 4 | `eval_set_construction.md` should record exact-match as the weakest standard | **no** — the file mentions no published standard, and neither 68.5% nor 80.8% appears in it |
| 5 | `findings.md` §5 Limits: ESM-C/ESM-3 gap, no published unseen AUC0.1 | **no** — §5 mentions neither |
| 6 | `findings.md` §11 fourth instance | **no** — §11 still lists three |

Item 2 now needs Part 2's interval as well, not just the sub-3 gap.

**⑤ `redteam_curve.md` §3 — Liao's predicate and reference are underspecified.** Own-model versus
union reference is unresolved, and the 68.6%-inside-their-40–70%-band argument was declared void
as validation. Nothing tracks the open question.

**⑥ `redteam_curve.md` §4 — selection and multiplicity.** No prospective endpoint specification is
recoverable; no multiplicity adjustment is implemented; Bonferroni across five unique looks puts
sub-3 at p = 0.120, Holm on the saved p-values at 0.048. Untracked.

**⑦ `redteam_curve.md` §4 — leave-one-peptide-out fragility.** 11 of 48 omissions cross zero;
dropping `RFPLTFGWCF` gives −0.011312 [−0.024739, +0.002042], p = 0.108. A known, quantified
fragility in the headline endpoint with no tracker.

**⑧ `redteam_curve.md` §2 — the cause of the narrowing is unidentified.** Both arms lose reference
signal, the edit arm retains the *largest* proportion of its excess (45.4% vs ESM 39.5% vs SVC
27.5%), and uniform compression is a live but untested explanation. The edit-specific mechanism
asserted in `outlook.md` Part D is not demonstrated. Untracked.

**⑨ `redteam.md` §0b — normalisation and negative-construction confounds.** `svc_per_peptide −
esm_retrieval` (+0.0067) carries a normalisation change as well as an operator change, and
cross-arm deltas in §7a and §8b span four different negative schemes. Recorded in `findings.md`
§8f prose; no tracker.

**⑩ #21's hash-verification item.** Listed above as an unmet acceptance item; it is arguably its
own issue since it is the only unfinished engineering task in the Phase 0 tree.

### 3e. Close recommendations — ALL EXECUTED at closeout, 2026-09-09

> **Status update.** This section was written as recommendations with nothing closed. At closeout
> every recommendation below was carried out and the tracker is now **empty: 0 open issues, 30
> closed.** #6 and #16 were reworded before closing; #10 and #18 closed on the permanent-erratum
> resolution (`data/errata.json`); #20 became [ADR 0001](ADR/0001-frozen-contract.md); #21's unmet
> hash verification was run and **PASSED** (`data/frozen_contract_hashes.json`). The seven gaps in
> §3d below were each filed as an issue (#24–#30) and immediately closed as **not planned**, so a
> deliberately-abandoned gap is visibly different from an untracked one. See
> [final_state.md](final_state.md).

The recommendations as originally written:

**Recommend closing, all evidence in §3b:** #1, #2, #3, #4, #5, #7, #8, #9, #11, #12, #13, #14,
#15, #17, #19. Fifteen issues, every acceptance item satisfied by a named commit, artifact, test
or `findings.md` section.

**Recommend reword-then-close:** #6 (fix the 0.544 reference first — §3c ②) and #16 (fix the
+0.156 expectation first — §3c ④). Both are functionally done; closing them as written would
freeze a wrong number into the record, which is exactly the §11 failure mode.

**Recommend blocking on a decision that has no issue yet:** #10 and #18. Both satisfied their
acceptance criterion — the JSON and CSV exist in the Section 6 schema — but the artifacts they
emit still carry `frozen_esm_cosine_seen_layer10: 0.5364` against an authoritative 0.5358.
Closing them now files the stale value away as finished. They should close once §3d ③ is decided
one way or the other: regenerate the artifacts, or annotate them in place.

**Recommend keeping open:** #21 — one acceptance item is genuinely unmet.

**Recommend converting, not closing:** #20. It is a standing rule with no terminal state, and the
two-worktree instruction in it is dormant now that both branches are merged into `main`. It
belongs in `CLAUDE.md` or a short `docs/ADR/` entry where it is read before work starts.

---

## Files

| path | status |
|---|---|
| `scripts/run_cdhit_and_attenuation.py` | new |
| `data/cdhit_and_attenuation.json` | new |
| `data/cdhit_regime_per_peptide.csv` | new |
| `docs/cdhit_and_issues.md` | new, this file |
| `data/dedup_sensitivity.json` | **read only, unchanged** |
| `data/fork1_results.json`, `data/fork2_results.json` | **read only, unchanged** |

The work is uncommitted in the working tree on `main` at the time #22 and #23 were filed, so both
issues link the branch rather than a commit. Committing is not done without an explicit
instruction; once a commit exists, the two issue bodies want its SHA added.
