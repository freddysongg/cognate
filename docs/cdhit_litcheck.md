# Literature check — CD-HIT behaviour on short sequences

Gate before drafting. 2026-09-09. Nothing was run in the repository; no experiment is proposed
here. Sources are primary where a primary source exists.

**Evidence rule applied throughout (outlook.md §B5).** Where a source was read directly, the
answer is `observed` against that source. Where only a search backend was consulted and returned
nothing, the report says *these queries returned nothing*, which settles nothing about existence.
Non-retrieval is never reported as a negative finding.

**Headline: our finding is a measurement, not a discovery.** Q1 and Q2 come back negative — the
degeneracy is not documented and no one appears to have written it up as a critique. Q3 comes
back positive and is the answer that narrows us most: the antibody subfield has stated the same
problem in a different currency and routes around it. Q4 and Q5 place Lu et al. in a small
pattern of under-reported usage.

---

## Q1. Is the arithmetic documented?

**No. Not in any CD-HIT primary source I read, and not implicitly by the word-size guidance
either.** The `-l 10` default is the closest thing to an acknowledgement, and it acknowledges the
adjacent fact, silently.

### What the sources actually say

**CD-HIT wiki, "1. Algorithm" → Algorithm limitations** ([GitHub wiki](https://github.com/weizhongli/cdhit/wiki/1.-Algorithm)),
and identically in the user guide shipped with the source I built (`doc/cdhit-user-guide.wiki`,
CD-HIT 4.8.1 @ `4f6720f`). The complete limitations section, verbatim:

> "A limitation of short word filter is that it can not be used below certain clustering
> thresholds. For proteins: word size 5 is for thresholds 0.7 ~ 1.0; word size 4 is for thresholds
> 0.6 ~ 0.7; word size 3 is for thresholds 0.5 ~ 0.6; word size 2 is for thresholds 0.4 ~ 0.5 (also
> see psi-cd-hit). […] Because of the algorithm, cd-hit may not be used for clustering proteins at
> <40% identity. Cd-hit-est cannot cluster very long sequences either (e.g. genome sized
> sequences)."

Two limitations are documented: **low identity thresholds** and **very long sequences**. Short
sequences are not mentioned. `observed` — I grepped the shipped guide for `short seq`, `too
short`, `small sequence`, `peptide`; the only hits are the program descriptions ("cd-hit Cluster
peptide sequences"), the `-l` option line, and an ASCII figure captioned "Short sequence" that
illustrates the alignment-coverage options.

**Li & Godzik, *Bioinformatics* 22(13):1658–1659 (2006)** — read directly. No discussion of short
sequences, minimum sequence length, `-l`/`throw_away_sequences`, or the identity denominator. The
only length-related caveat is about introns in `cd-hit-est`.

**Fu, Niu, Zhu, Wu & Li, *Bioinformatics* 28(23):3150–3152 (2012)** — read directly. No discussion
of minimum sequence length, short-sequence limitations, peptide-specific caveats, `-l`, the
identity denominator, or word-size selection. The paper is about parallelisation.

**There is no FAQ page.** The wiki index lists Home, 1. Algorithm, 2. Installation, 3. User's
Guide, 4. Web server, 5. Use cases, 6. CD_HIT_OTU_MiSeq, References. No FAQ, no troubleshooting,
no limitations page beyond the section quoted above.

### The one thing that *is* documented — and it is the ingredient, not the consequence

The identity **denominator** is documented prominently, in the `-c` help text and the guide:

> "this is the default cd-hit's 'global sequence identity' calculated as: number of identical
> amino acids in alignment divided by the full length of the shorter sequence"

So the arithmetic is fully derivable from documented behaviour. What is absent is anyone stating
the consequence: that at length *L* a single substitution gives (*L*−1)/*L*, so no substitution can
fall below 0.95 until *L* ≥ 20, and the criterion collapses to exact-match-plus-containment on
CDR3-length input. Derivable is not the same as documented, and it is also not the same as hidden.

### Do `-n` guidance and `-l 10` constitute implicit acknowledgement?

**`-n` (word size): no, and this matters.** The guidance is indexed on **identity threshold**, not
on sequence length — "word size 5 is for thresholds 0.7 ~ 1.0". It addresses whether the short-word
*filter* stays valid as the threshold drops. That is an orthogonal axis to what happens to a
threshold as sequences get short. Reading the word-size table as a short-sequence warning would be
reading a different variable.

**`-l 10` (throw-away length): partly, and in the worst possible way.** The default does encode
"sequences of 10 residues or fewer are outside this tool's intended domain", which is adjacent to
our point. Three qualifications:

1. It is a **length floor**, not a threshold-degeneracy warning. It says nothing about how `-c`
   behaves at length 12, which is where our median CDR3β sits and where the degeneracy is total.
2. The guide gives it **no rationale** — it appears only as `-l length of throw_away_sequences,
   default 10` in the option list, four times, unexplained.
3. It **discards silently**. There is no warning on stdout. The community documentation that exists
   is a user asking why sequences went missing
   ([Biostars 152941](https://www.biostars.org/p/152941/), [289336](https://www.biostars.org/p/289336/)) —
   which is what an undocumented floor produces. In our own run it would have removed 2,081 of
   18,760 distinct evaluation CDR3β without comment.

**Consequence for our contribution:** the finding is **not** "an undocumented tool behaviour". The
denominator that produces it is documented in the first line of the `-c` help text. The finding is
that a consequence derivable from that documented definition was not derived by a 2026 *Nature
Methods* benchmark, and that CD-HIT's own documentation does not derive it for the reader either.
That is a weaker and more accurate claim than "undocumented behaviour".

---

## Q2. Has anyone written this up as a critique?

**Not found.** I did not locate a paper, preprint, blog post or methods commentary arguing that
identity-threshold clustering is inappropriate for sub-30-residue sequences as its thesis.
Searches across Europe PMC full text and general web returned nothing of that shape. **Per the §B5
rule this settles nothing about existence** — it says these queries did not surface one.

What I did find, and what it is not:

**① The closest thing is practitioner compensation, not critique — and it is in our own subfield.**
T-SCAPE, Kim et al., *Science Advances* (2025), [PMC12680054](https://europepmc.org/articles/PMC12680054),
Methods §"Sequence identity filtering using CD-HIT":

> "The combined set of training and test 9-nucleotide oligomers was clustered using CD-HIT with a
> sequence identity threshold of 0.8 (**-c 0.8, to ensure less than two amino acids mismatches are
> tolerated for 9-nucleotide oligomer fragment**) and word size of 2 (-n 2)."

This is exactly our arithmetic, performed by someone else, in public, in 2025. They translated the
identity threshold into a **residue-mismatch budget at their sequence length** and picked `-c` to
hit the budget they wanted. They also set `-n 2` where the documented table prescribes `-n 5` at
0.8 — an undiscussed departure that only makes sense for very short input. They state the
reasoning for their own choice; they do not generalise it into a warning about the threshold.

**② The Linclust paper documents the denominator's consequence, framed as a definitional
difference.** Steinegger & Söding, *Nature Communications* (2018),
[PMC6026198](https://pmc.ncbi.nlm.nih.gov/articles/PMC6026198/): CD-HIT's identity is defined over
the shorter sequence, "and therefore sequence coverage of the shorter sequence must be at least as
large as the sequence identity threshold." True, relevant, and not a short-sequence critique.

**③ IEDB Cluster2 is not the critique, though it looked like it would be.** Dhanda, Vaughan,
Schulten, Grifoni, Weiskopf, Sidney, Peters & Sette, "Development of a novel clustering tool for
linear peptide sequences", *Immunology* (2018), doi 10.1111/imm.12984, builds a clustering tool
aimed at 8–25-residue peptides — the right length regime. Read directly: their justification is
about cluster connectivity and consensus, not threshold degeneracy —

> "none of these tools provide a complete connectivity of the peptides within a cluster and do not
> generate a clear consensus sequence representing each cluster"

I am recording this as **checked and negative** rather than omitting it, because it is the paper a
reader would expect to carry the argument.

**Consequence for our contribution:** no prior art was found claiming the general point, so the
general claim is not obviously pre-empted — but ① means the *specific* reasoning is demonstrably
already in circulation in TCR/peptide ML, which forecloses framing ours as a novel observation
about CD-HIT.

---

## Q3. Is it known in the adjacent field?

**Yes — the antibody repertoire field has stated the same problem in a different currency
(alignment coverage and statistical significance rather than substitution counts) and routes
around it by lengthening the sequence before clustering.** This is the answer that narrows us
most.

**The explicit statement.** Saputri, Ismanto, Nugraha, Xu, Horiguchi, Sakakibara & Standley,
"Deciphering the antigen specificities of antibodies by clustering their complementarity
determining region sequences", *mSystems* 8(6), 2023, doi 10.1128/msystems.00722-23
([PMC10734444](https://pmc.ncbi.nlm.nih.gov/articles/PMC10734444/)):

> "Typical pseudo sequences are long enough to provide sufficient alignment coverage at rigorous
> sequence identity thresholds, thereby providing pairwise scores that are statistically
> significant, **which is not the case for CDR3 sequence alignments, in general**."

Their response is structural: they cluster a **concatenated pseudo-sequence (CDRH1 + CDRH2 +
CDRH3)** with MMseqs2 at 80% identity and 90% coverage, rather than clustering CDR3 alone —
lengthening the input until an identity threshold becomes meaningful.

**Read this precisely.** They name *alignment coverage and statistical significance*, not the
substitution arithmetic. It is the same problem shape and not the same statement, and I am not
going to claim they identified our arithmetic.

**The field's default is distance, not identity threshold.** The 2025 antibody clustering
benchmark, [PMC12148228](https://europepmc.org/articles/PMC12148228), compares clustering
approaches on CDRH3 using **normalised Levenshtein distance** and clonotyping —

> "The Levenshtein distance calculates the minimum number of substitutions, insertions and
> deletions to align two sequences […] The normalized Levenshtein distance was derived from the
> absolute Levenshtein distance"

— with no identity-threshold clustering in the comparison. The standard repertoire tooling that
surfaced in searches (clonotyping by V/J plus CDR3 Hamming distance, clusTCR, tcrdist, immunarch,
CellaRepertorium) is distance-based on CDR3 throughout. I did not find a paper in this field
stating *"we chose edit distance over an identity threshold because thresholds degenerate at CDR3
length"*; the choice appears to be convention rather than an argued position.

**Consequence for our contribution:** this is the sharpest and narrowest version of the story. A
2026 TCR benchmark applied an identity threshold directly to CDR3-length sequences, in a
neighbouring subfield where the standard practice is edit distance on CDR3 and where the
inadequacy of identity thresholds on CDR3 has been stated in print since 2023. Our result is the
quantification of what that costs on one benchmark, not the identification of the problem.

---

## Q4. What exactly did Lu et al. do?

Lu, Wang, Xu, Xie, Yang, Xu & Suo, "Assessment of computational methods in predicting TCR–epitope
binding recognition", *Nature Methods* 23(1):248–259, doi 10.1038/s41592-025-02910-0
([PMC12791011](https://pmc.ncbi.nlm.nih.gov/articles/PMC12791011/)). Full text re-read directly.

### The invocation

| item | what the paper says |
|---|---|
| threshold | **>95% similarity**, stated three times |
| applied to | **TCR sequences** — not pairs, not peptide-conditional |
| which sets | training vs test; in retraining, also **training vs independent test** |
| when | after merging positives with generated negatives, per data group |
| program variant | **not stated**. Cited as ref 17 = **Li & Godzik 2006**, i.e. protein `cd-hit`, not `cd-hit-est` |
| word size `-n` | **absent** |
| `-l` | **absent** |
| version | **absent** |
| command line | **absent**, including Methods, Code Availability and supplementary |

Verbatim, Methods:

> "To prevent data leakage, we used CD-HIT to exclude highly similar sequences (>95% similarity)
> between the training and test sets. Specifically, after integrating the positive samples with the
> generated negative samples for each data group, CD-HIT was applied to eliminate these highly
> similar TCR sequences, ensuring robust and unbiased evaluation of the models."

> "In model retraining, we also used CD-HIT to exclude TCR sequences with greater than 95%
> similarity between the training and test sets, and between the training set and the independent
> test sets."

### Do they report a retained or removed fraction?

**Not for the main train/test filter — that number is absent.** But **there is one before/after
pair around a CD-HIT step**, in the cross-reactivity analysis, and it was not in the passages
quoted in our earlier work:

> "We identified 11,667 cross-reactive TCR–epitope entries (cross-reactive data from MIRA dataset
> were not included due to an unusually high ratio of cross-reactive TCRs). **After applying CD-HIT
> to eliminate sequences with high similarity in both test and independent test sets, 11,083
> unique cross-reactive entries were added** in this evaluation."

**11,083 / 11,667 = 95.0% retained, 5.0% removed.**

Two qualifications, both of which cut against reading too much into it. It is a **different
application** of the filter — cross-reactive entries against the test and independent test sets,
not the train/test leakage control — so it is not directly comparable to our 99.26% on evaluation
rows. And the sentence conflates the CD-HIT step with deduplication ("11,083 **unique**"), so 5.0%
is an **upper bound** on CD-HIT's contribution; the true removal may be smaller.

Taken with that care, it is still the only CD-HIT before/after number in the paper, it sits in the
same regime as ours, and **it passes without comment**. The paper does not remark that a leakage
control retained 95% of what it was pointed at.

### Do they justify CD-HIT or the 95% threshold?

**No.** Read directly: there is no sentence anywhere explaining why CD-HIT rather than another
tool, or why 95% rather than another number. The action is stated; the rationale is not.

**Consequence for our contribution:** the equivalence gap we recorded is confirmed and slightly
widened — not only are the parameters unstated, the tool and threshold are unjustified. But the
95.0% figure means the claim "they saw the same thing and did not flag it" is now **partly
supportable** rather than speculative, for a neighbouring application of the same filter, with the
deduplication ambiguity attached.

---

## Q5. Does anyone else in TCR use it?

**Yes. A small pattern, four papers, with parameter reporting that is worse than Lu et al.'s in
two of the three others.** Europe PMC full-text search for `"CD-HIT" AND "CDR3" AND ("T cell
receptor" OR TCR)` returns **56 hits** total, so the population is small but not a singleton.

| paper | year | what is clustered | threshold | parameters reported |
|---|---|---|---|---|
| Lu et al., *Nat Methods* ([PMC12791011](https://pmc.ncbi.nlm.nih.gov/articles/PMC12791011/)) | 2026 | TCR sequences, train vs test and train vs independent test | **>95%** | none |
| T-SCAPE, *Sci Adv* ([PMC12680054](https://europepmc.org/articles/PMC12680054)) | 2025 | peptide 9-mer fragments; applied across pMHC and TCR CDR3β datasets | **0.8** | **`-c 0.8 -n 2`**, with explicit mismatch-count reasoning |
| "TCR clustering by contrastive learning on antigen specificity" ([PMC11317525](https://europepmc.org/articles/PMC11317525)) | 2024 | **CDR3α and CDR3β directly** | **90%** | tool only |
| DeepAIR, *Sci Adv* ([PMC10411891](https://europepmc.org/articles/PMC10411891)) | 2023 | AIR sequences, two-database comparison | **not stated** | `CD-HIT-2D`, no threshold |

Verbatim, the CDR3-direct case (PMC11317525):

> "The final dataset was clustered by 90% similarity on both CDR3α and CDR3β similarity using
> CD-HIT and split into 70% training set and 30% validation and test set (15% each). Similar TCRs,
> as determined by CD-HIT, were assigned to the same fold."

Verbatim, DeepAIR:

> "In this study, we compared sequences and calculated their similarity with the CD-HIT-2D
> algorithm […] CD-HIT-2D is developed to compare two protein datasets and identifies the sequences
> in dataset-2 that are similar to dataset-1 at a certain threshold"

— "at a certain threshold", and the threshold is never given. Note also that DeepAIR independently
chose **`cd-hit-2d`**, the two-database mode, for exactly the eval-against-fixed-reference shape we
used; that is corroboration for our tool choice rather than a finding about theirs.

One further hit, PMC9455901 "Rapid Assessment of T-Cell Receptor Specificity of the Immune
Repertoire" (2021), was returned by the query but its full text was not retrievable through the
Europe PMC API route I used, so **it is uncharacterised**, not negative.

**Consequence for our contribution:** four papers is a pattern rather than an anecdote, but a thin
one, and the pattern is not simply "people use CD-HIT on CDR3". It is that **three of four do not
report enough to reproduce the filter**, and the one that does had to reason in residue-mismatch
counts and depart from the documented word-size table to make the threshold mean what it wanted.
The reproducibility gap is the stronger and better-evidenced pattern; the degeneracy is the reason
it matters.

---

## What our contribution is, given these answers

**Our contribution is a measurement, not a discovery: at CDR3β length the >95% identity criterion
that Lu et al. (2026) rely on for train/test leakage control removes 0.74% of our evaluation rows
and removes none of them on substitution grounds — a consequence fully derivable from CD-HIT's
documented shorter-sequence identity denominator, never stated in CD-HIT's own documentation or
either founding paper, never quantified or justified by Lu et al., already compensated for by
translating the threshold into a mismatch budget in at least one 2025 TCR/peptide paper, and
already argued around in the antibody subfield since 2023 by lengthening CDR3 into a concatenated
pseudo-sequence before clustering at all.**

What this rules out saying:

- **Not** "an undocumented tool behaviour" — the denominator that causes it is the first line of
  the `-c` help text (Q1).
- **Not** "nobody in the field knows" — T-SCAPE performed the same arithmetic in print in 2025
  (Q2 ①), and the antibody field stated the coverage form of the problem in 2023 (Q3).
- **Not** "they saw a 99% retention and hid it" — they report no retained fraction for the filter
  in question. The nearest thing is 95.0% on a different application of it, with a deduplication
  ambiguity attached, passing without comment (Q4).
- **Not** a general claim about identity-threshold clustering, which we did not test and which no
  retrieved source states as a thesis (Q2).

What survives as ours: the quantification on this benchmark, the demonstration that 100% of what
the criterion removes is the indel class rather than the substitution class, and the observation
that a leakage control can be reported as satisfied while being incapable of expressing the
difference it is nominally controlling for.
