# Lessons

The transferable part. Written at closeout, 2026-09-09, and stated so that none of it depends on
caring about T-cell receptors.

Every project-specific claim below links its evidence. The evidence scale used throughout:
`claimed` (I said so, including a careful reading of code), `observed` (a check ran and I saw the
output), `failure-proven` (the check was also shown able to fail).

---

## 1. The restatement pattern — five instances, and two of them are uncatchable

**The pattern.** A value established in one place is **restated by hand** somewhere else, and
nothing ever compares the copy to the original. Correctness of the *source* is what every process
here was set up to check; **agreement between a source and its restatements was checked by
nothing.**

| # | the restatement | copy | source | how it was found | catchable by a test? |
|---|---|---|---|---|---|
| 1 | ESM-2 cell of the 2×2 | 0.5434 (layer 6) | 0.5358 (layer 10) | noticed by accident while setting up a different diagnostic | **yes** — same class as #2 |
| 2 | `frozen_esm_cosine_seen_layer10` in two fork drivers | 0.5364 | 0.5358 in `knn_esm_cosine.json` | a deliberate sweep looking for more instances of #1 | **yes** — `tests/test_reference_constants.py`, `failure-proven` |
| 3 | `deviations_from_sceptr` | 5 entries | 7 | reading the source paper for an unrelated question | **no** — see below |
| 4 | "the 48 seen peptides" | a fixed evaluation set | whatever the current external dedup convention says | recomputing the dedup radius | **no** — see below |
| 5 | "frozen ESM-2 cosine at 0.544" in GitHub issue #6 | 0.544 (8M layer 3) | 0.5358 (35M layer 10) | the closeout issue audit | **no** — see below |

**Note how each was found.** One by accident, one by a sweep deliberately hunting the pattern, one
by reading a source for a different reason, one by recomputation, one by an audit. **Zero were
found by a test, a CI run, or a review of the number itself** — every one of them was individually
plausible. #2 is wrong by 0.0006 and sits inside its own confidence interval. #5 is a real number
from the project's own table. A list asserting completeness looks complete because a list always
does.

### What a test can and cannot cover

`tests/test_reference_constants.py` closes the **restated-scalar** class. It parses the `reference`
dict literal out of each driver with `ast`, maps every constant to a named source artifact, and
fails on a mismatch or on a restated key with no source. It is `failure-proven` against
`git show HEAD:scripts/run_fork{1,2}.py`, which reported
`{'frozen_esm_cosine_seen_layer10': (0.5364, 0.5358)}` for both.

**#3 is uncatchable because it is a completeness assertion.** #1 and #2 are restated *values*:
there is an authoritative number, so equality is checkable. #3 claims a list of deviations from
another paper's method contains *every* deviation. No artifact holds the true list, because the
true list is whatever a careful reading of an external paper turns up. A test could assert the
list has seven entries; it could not assert that seven is right, and pinning the count would make
the next omission harder to see rather than easier.

**#4 and #5 are uncatchable for a sharper reason: the authority is outside the repository.** #3 is
at least *about* objects this repository owns. #4's authoritative answer — what "deduplicated"
currently means — lives in other groups' Methods sections and **changed while the project was
running**. #5's authoritative answer lived in a GitHub issue body, outside anything the test suite
can see.

> **A test can only compare two things the repository holds.** When the authoritative answer is
> external, internal consistency testing has nothing to compare against. This is not a gap to be
> closed with more tests, and pretending otherwise produces a green suite over a wrong claim.

The only mitigation that works on #4 and #5 is §5's standing literature check, plus the discipline
of treating any completeness assertion as `claimed` no matter how carefully it was assembled, and
re-deriving it from source whenever it becomes load-bearing. That is exactly how #3 was found.

---

## 2. The under-resolution trap

**A percentile interval estimated from too few bootstrap draws has meaningless tails, and the
number it reports is unstable in both directions.** Four occurrences.

| # | quantity | low resolution | properly resolved | verdict changed? |
|---|---|---|---|---|
| 1 | sub-3 representation effect, p | **0.024** at 1,000 draws, seed 0 | **0.0405** at 20,000 draws | no, but Bonferroni moves 0.120 → 0.2025 |
| 2 | full→sub-3 attenuation | **47.7% [21.3%, 94.6%], p = 0.0000** at 20 draws | **47.7% [−0.1%, 98.2%], p = 0.0511** at 20,000 | **yes** — significant to not |
| 3 | sub-3 effect, p, under a *different* pairing scheme | 0.024 at 1,000 draws | **0.0439** at 20,000 synchronised draws | no — independent corroboration of #1 |
| 4 | worst leave-one-peptide-out omission, p | 0.108 at 1,000 draws | **0.0964** at 5,000 draws | no — and it moved the **other way** |

**Do not read this as "under-resolution inflates significance".** #4 moved toward significance,
not away. The failure is *noise*, not bias: with few draws the tail of the sampling distribution
is estimated from a handful of order statistics and lands wherever it lands. What creates the
appearance of a directional bias is **which unstable number gets reported** — and the one that
looks publishable is the one that gets kept.

#2 is the sharpest case and deserves a second look. A 20-draw estimate of a 95% interval is
computing the 2.5th percentile of twenty numbers. It reported **p = 0.0000** because zero of
twenty replicates crossed zero, which is exactly what twenty draws will do to a quantity whose
true two-sided p is ~0.05. Nothing about the output announced that it was under-resolved.

**Mitigation, stated as a rule rather than a caution.**

1. **Any interval or p-value that enters a claim gets ≥ 20,000 draws.** 1,000 is enough for a
   point estimate and a rough width; it is not enough for a tail.
2. **Report the draw count next to every interval.** An interval without its resolution is not
   interpretable, and #1 through #4 are all invisible unless the count is printed.
3. **When an interval is near a decision boundary, re-run it at higher resolution before quoting
   it** — not after someone challenges it.
4. **A p-value of exactly 0.0000 from a bootstrap is a resolution report, not a result.** It means
   no replicate crossed; it does not mean the probability is small. State it as `< 1/n_draws`.

---

## 3. MISATTRIBUTED, as a verdict category

**Definitions.**

- **REFUTED** — the claim's own stated overturn condition was tested and the claim failed.
- **MISATTRIBUTED** — the claim is about a **different axis than its wording asserts**. It was not
  shown false. It was shown to be a comparison of one thing wearing the label of another.

**Why pre-registration alone does not catch it.** A pre-registered overturn condition tests
whether a claim is **true**. It does not test whether the claim is **about what it says it is
about**.

The worked example: claim ① said a lookup beats a language-model head, and read the gap as what
"the pretrained representation bought". Its overturn line said the paired difference would span
zero or flip at 50–100 peptides. It did neither — **+0.055 [+0.039, +0.074]** at n = 48, passing
its own test twice while tightening fourfold. The number was right the whole time. The attribution
was not: the two arms differed in operator and in *scope* as well as representation, and
decomposition showed the scope term was the entire effect
(**+0.0346 [+0.0223, +0.0482]** for scope; **+0.0067 [−0.0006, +0.0142], p = 0.076** for operator,
spanning zero). [findings.md](findings.md) §8b.

**No amount of narrowing the interval would have caught it, because the interval was around the
right number for the wrong quantity.** Tightening a confidence interval is orthogonal to whether
the estimand is the one in the title.

**The fix that generalises:** every pre-registered claim carries an **attribution line** alongside
its overturn line — *what axis is this comparison actually varying, and what else varies with it?*
Then extend that once more, because claim ⑥ needed a third line: **what convention defines the
evaluation set this interval is computed on, and who controls that convention?**

---

## 4. Report sensitivity curves, not point estimates, whenever a cleaning convention is upstream

**The rule.** If an effect size depends on a data-cleaning choice — deduplication radius,
similarity threshold, filtering criterion, outlier rule — **report it as a curve over that choice,
not as a number at one setting.** The number at one setting is a claim about the convention as much
as about the data, and conventions move.

**Why it matters here.** The project's central result at one deduplication setting:

| dedup criterion | rows removed | Δ | p |
|---|---|---|---|
| exact CDR3β match | 0% | −0.0296 [−0.0415, −0.0188] | <0.001 |
| CD-HIT >95% | 0.74% | −0.0303 [−0.0436, −0.0193] | <0.001 |
| 1 substitution | 12.7% | −0.0261 [−0.0394, −0.0142] | <0.001 |
| 2 substitutions | 40.3% | −0.0226 [−0.0373, −0.0091] | <0.001 |
| 3 substitutions (strictest published) | 68.6% | **−0.0155 [−0.0329, −0.0012]** | **0.024** |
| Levenshtein ≤ 3 (beyond any standard) | 80.9% | −0.0138 [−0.0300, +0.0027] | 0.110 |

Quoting only the first row is defensible and quoting only the last row is defensible, and they
support opposite readings of the same data. **The curve is the honest object, and it is also the
more informative one** — the monotone decay is a finding in its own right, and the fact that
significance dies one step past the strictest published standard is exactly the sort of thing a
reader needs and a point estimate cannot convey.

**Three things the curve made possible that a point estimate would have hidden.**

1. It exposed that "the 48 seen peptides" is not a fixed object — restatement instance #4.
2. It let a red team ask about the *change* rather than the level, which is what produced the
   finding that the change is not established at all ([cdhit_and_issues.md](cdhit_and_issues.md)
   Part 2).
3. It made a null regime legible: CD-HIT >95% sits on the curve at 0.74%, and seeing it there is
   what shows it cannot discriminate.

**Two cautions.** Multiple settings are multiple looks — a curve invites a multiplicity problem
and you must say which endpoint was prospective (this project could not, and Bonferroni across
five unique looks moves the sub-3 p to 0.120). And a curve needs a **support-matched control** at
each point, or you cannot separate "the effect decayed" from "you removed most of the data": here,
2,000 random subsets per regime matching each regime's exact positive and negative counts showed
support loss reproduces essentially none of the decay (+0.000063 against an observed +0.014124 at
sub 3).

---

## 5. The standing literature check

**The rule.** Before claiming a finding is novel, check whether the field already knows it — and
check *before* drafting, not after.

Four instances, and what each one moved:

| # | when | what it checked | what it moved |
|---|---|---|---|
| 1 | after Phase D | is the representation result published? | **Demoted the headline from finding to replication.** Nagano et al. 2025 highlight ① states it. Also at abstract level in IMMREP22 and TITAN. |
| 2 | after the 2×2 | did anyone run the operator comparison? | Found SCEPTR Fig. S6 with the **opposite sign**, which is what motivated the operator diagnostic — which then dissolved our own operator axis. |
| 3 | after Part A | what is the current dedup standard? | **Found the standard had moved past exact match**, which is the origin of the whole sensitivity curve and of restatement instance #4. |
| 4 | **before drafting** | is CD-HIT's short-sequence behaviour known? | **Ended the project.** Derivable from documented behaviour; argued in the antibody field since 2023; already compensated for in-subfield in 2025. No discovery claim survived. |

**Instance 4 is the one that paid.** The first three fired *after* work was done and each one
demoted something already built. The fourth fired *before* a draft existed and cost one pass of
searching. Same information, a fraction of the price. **The check is cheapest at the moment you
first suspect you have something, which is exactly when you least want to run it.**

### The sub-rule that instance 4 needed

**Non-retrieval is not evidence of absence.** An earlier lookup recorded a paper as "could not be
retrieved" and reported a negative; three queries against one search backend had returned only a
GitHub repository, and the paper was reachable the whole time via its PMC identifier. The rule
against substituting a related answer for the one asked does **not** license converting "my tool
did not surface it" into a reported negative.

The correct forms are: *"these queries returned nothing, which settles nothing"*, or try another
access route. Same shape as the restatement pattern — a label ("unanswerable from public sources")
that did not match what was actually measured ("not returned by these queries").

In practice this meant, at closeout: reading CD-HIT's shipped user guide from the built source
rather than a summary of it, reading Li & Godzik 2006 and Fu et al. 2012 directly to confirm two
stated absences, and using a full-text search API rather than a web search when the question was
*"which papers do X"*.

---

## 6. What actually worked

Four things, all cheap, all of which produced results that survived every later challenge.

**1. Pre-registration with overturn conditions.** Eight claims locked before the data that would
test them existed, each with a *"what would overturn it"* line. Six of the eight came through the
full record unchanged. The two that moved (① and ⑥) moved along an axis nobody had pre-registered —
which is itself the finding in §3, and is only visible *because* the original wording was locked
and never edited to fit. **Writing the claim down beforehand was worth more when it failed than
when it passed.**

**2. Build the control first, and give it an independent target.** Fork 2's mean-pool control (2b)
was built before the cross-attention arm and had to reproduce ~0.511 — a number from an earlier,
unrelated run. It scored 0.5093. Only then was 2a built. This is what makes
**2a − 2b = −0.005 [−0.016, +0.005], p = 0.366** a readable null instead of an ambiguous one: the
control demonstrably worked, so "no difference" means no difference rather than a broken harness.
A null result without a validated control is not a result.

**3. Test the tests.** Every check that mattered was `failure-proven` — shown to go red on
deliberately corrupted input before being trusted green. `test_reference_constants.py` carries a
positive control that feeds it the exact literal that drifted. The count-based partial-AUC used for
the 20,000-draw bootstrap was verified equal to `sklearn.roc_auc_score(..., max_fpr=0.1)` on every
peptide, regime and arm cell (max |diff| **2.22 × 10⁻¹⁶**) before any interval computed with it was
quoted. **A green check never shown able to fail is `observed`, not `failure-proven`, and the
difference is the whole point of the scale.**

**4. Support-matched random controls.** Whenever a filter removes data and an effect changes, the
first question is whether removing *that much* data at random would do the same. 2,000 subsets per
regime, matching each regime's exact positive and negative counts per peptide, identical rows
across arms. At sub 3 the random control moves the effect by **+0.000063** against an observed
**+0.014124** — so support loss explains essentially none of the decay. This is the control that
lets a decay be attributed to *what* was removed rather than *how much*, and it costs one loop.

---

## 7. The one-line versions

- A test can only compare two things the repository holds. When the authority is external, no test
  helps.
- A bootstrap p-value of 0.0000 is a resolution report, not a result.
- Pre-registration tests whether a claim is true, never whether it is about what it says.
- If a cleaning convention sits upstream of your effect size, the effect size is a curve.
- The literature check is cheapest before you draft, and that is when you least want to run it.
- A null needs a control that was shown to work first, or it is not a null.
