# Next-project candidates — scoping only

**Status:** none of the five candidates below has been built or started. The screen was
performed on 2026-09-09 after reading [`lessons.md`](lessons.md) and
[`final_state.md`](final_state.md).

**This screen predates the pMHC work and does not account for it.** After it was written, the
pMHC binding replication and a three-track leave-one-allele-out (LOAO) transfer study were both
built and merged to `main`. Their outputs change two premises the screen rests on — that the
repository's modelling line was finished, and that its domain vocabulary was a pure cost. Read
[What the pMHC and LOAO studies changed](#what-the-pmhc-and-loao-studies-changed) before using
the ranking below. The five screened candidates themselves are unaffected: no literature gate
was re-run, and none of their questions touches pMHC binding.

## What the pMHC and LOAO studies changed

Recorded 2026-09-16, after all five pull requests merged to `main` at `ddaa8f7`. Every number
here is read from the committed artifacts, not restated from memory:
[`loao_allele_only.md`](pmhc/loao_allele_only.md),
[`loao_allele_peptide.md`](pmhc/loao_allele_peptide.md),
[`loao_cluster.md`](pmhc/loao_cluster.md) and their
`data/pmhc/loao_*_results.json` aggregates.

**Transfer to unseen alleles works, and degrades monotonically with novelty.** Macro
standardized AUC0.1 for the pseudo-sequence MLP, 95% percentile intervals from 20,000 draws with
two-level row and allele resampling, over a frozen cohort of 47 HLA-A/B/C alleles and 112,128
nine-mer rows:

| Holdout | Pseudo-sequence MLP | Random |
|---|---:|---:|
| allele-only | 0.7487 [0.7177, 0.7801] | 0.5003 |
| joint allele-and-peptide | 0.7040 [0.6761, 0.7329] | 0.5003 |
| pseudo-sequence-cluster | 0.6218 [0.6013, 0.6429] | 0.4987 |

The pseudo-sequence MLP beat both the peptide-only and shuffled-mapping ablations at every
level, every paired lower bound above zero. `observed`.

The seen-allele reference point is **0.8283 [0.8108, 0.8456]**, from the earlier binding
replication ([`experiment.md`](pmhc/experiment.md), same arm, same metric). It is deliberately
kept out of the table above because it comes from a different design — in-distribution
cross-validation over the same cohort, not a holdout — so reading 0.8283 → 0.7487 → 0.7040 →
0.6218 as one degradation curve compares across designs at the first step. The three holdout
rows are mutually comparable; the fourth number is context.

**This inverts the repository's earlier finding, on a different problem.** The TCR–epitope line
closed with nothing ever scoring above chance on unseen peptides, where a ten-line
edit-distance k-NN (0.5654) beat every learned representation. The pMHC line produces real
transfer. What the two share is the macro AUC0.1 metric implementation, the retrieval operators,
and the same evidence discipline — pre-declared criteria, ablation arms, resolved paired
intervals. They do **not** share an arm set: the TCR work compared retrieval, an ESM-2 head,
contrastive and cross-attention variants, while the pMHC tracks compare random, peptide-only,
shuffled-mapping, nearest-PWM and the pseudo-sequence MLP. So this is two problems answered
under one methodology, not one experiment run twice. `observed` for both underlying results; the
*comparison* has had no literature gate, so whether it is a novel contribution or a restatement
of what the pMHC field already knows is **unscreened** — see below.

**Consequences for this screen:**

1. **The premise that the modelling line was finished is false.** The screen was written to
   choose a successor to a closed project. Two studies have shipped since, and the sharpest open
   question in the repository is now inside them rather than outside.
2. **The terminology tax has been partly paid.** Candidate 3 below rates its domain-knowledge
   cost "High" partly because "Cognate exposed" that tax. The pMHC half of that vocabulary —
   HLA nomenclature, pseudo-sequences, binding-affinity thresholds, allele holdout — has since
   been worked through end to end. That does not lower candidate 3's cost, which is about TCR
   assay and CD-HIT vocabulary specifically, but it does mean a pMHC-adjacent question would now
   start from a paid-down base rather than from zero.
3. **LOAO evaluation is standard practice in the pMHC field.** Pan-specific predictors are
   validated by allele holdout — the original NetMHCpan (2007) trained on all data for a locus
   except the molecule under test, for every molecule in its dataset. The shipped study is
   therefore a well-executed replication rather than a new result, and should not be described as
   more than that. Gated 2026-09-16; `observed`, see the study-derived section below.

### Study-derived candidates — gated 2026-09-16

A literature gate was run on all four on 2026-09-16 (searches: NetMHCpan leave-one-allele-out
validation; TCR-versus-pMHC generalization contrast; performance-versus-pseudo-sequence-distance;
allele-novelty/peptide-novelty split design; allele-clustered holdout). Two are **dead** — the
question is answered in print by better-resourced work. One survives as a repair, not a
contribution. One was never screened and does not need to be. The shipped LOAO study was scoped
to its three transfer designs; anything below that proceeds needs a fresh study brief before code.

- **Decompose the joint allele-and-peptide confound. — SURVIVES as a repair.** The joint-novelty
  track is the one shipped result whose numbers carry no defensible claim: it records no boolean
  criterion by design, and removing every row sharing a target's own test peptides deletes a
  target-varying 1,320–70,740 training rows, so allele and peptide novelty move together. The
  *design* is not novel — the field already uses hard splits and distance splits that enforce a
  controlled median train/test peptide distance by percentile interval, and already reports that
  pan-specific models generalize poorly to unseen alleles and peptides. `observed`. That is the
  point in its favour: the fix is off the shelf. Vary peptide-novelty percentile while holding
  training-row count matched, instead of deleting a target-varying row count. Closes a hole in
  delivered work; will not publish.
- **Finer cluster granularity. — DEAD.** Pre-empted twice over.
  [Toward equitable MHC binding predictions](https://pmc.ncbi.nlm.nih.gov/articles/PMC11874272/)
  (PNAS 2025) ran the exact distance-versus-support decomposition this candidate proposes: 125
  test alleles, 10 successive training sets each, every step excluding the *next most
  sequence-similar* allele by BLOSUM62 subject to at least 10 further binding records removed —
  1,250 trained model variants. It then shipped MHCPerf, a network predicting per-allele PPV
  from pseudo-sequence plus training data, in which distance-to-nearest-training-allele features
  dominate. Separately,
  [geometric deep learning for MHC-I](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659464/) already
  runs clustered allele holdout (PAM30 hierarchical clustering per gene, most-distant clusters to
  ~10% of data) and reports sequence-based AUC 0.91 on a random split collapsing on the clustered
  split, with a structure-based EGNN beating it by 8–11% AUC there. `observed`.
- **The cross-problem contrast as the contribution. — DEAD.** Both halves are published.
  Allele holdout is the founding validation of the field: the original
  [NetMHCpan](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0000796) (2007)
  trained on all data for a locus except the molecule under test and repeated it for every
  molecule. The inversion itself is stated as settled background in the 2024–2026 TCR reviews —
  pMHC class I binding prediction succeeded, TCR generalization to unseen epitopes did not —
  along with its explanations (TCR docking-mode diversity, negative-sampling artifacts, motif
  shortcut learning that collapses with peptide distance). `observed`. This also retires the
  `claimed` note in item 3 above: LOAO being standard practice is now `observed`, and the shipped
  study is a well-executed replication.
  One narrow residue: [arXiv:2606.04994](https://arxiv.org/html/2606.04994v1) benchmarks eight
  TCR models **on Macro AUC0.1**, this repository's own metric, finds near-random performance on
  unseen epitopes, and runs **no sequence-similarity baseline**. The repository's edit-distance
  k-NN at 0.5654 beating every learned arm is the arm that benchmark omits. That is a paragraph,
  not a project.
- **Antigen-presentation endpoint, MHC Class II, external-dataset validation. — OUT OF SCOPE, no
  gate run.** Named in the LOAO brief as explicitly excluded. `claimed`: each is a new project in
  a dense field with entrenched incumbents (NetMHCpan-4.1 EL, MHCflurry 2.0 AP, the NetMHCIIpan
  line) that a sub-600M fixed-inference constraint cannot match head to head.

### Build-on-published candidates — spiked 2026-09-16, NOT briefed

Gated for whether an *existing* published result leaves room for an added arm, a transferred
technique, or an evaluation the original did not run. This is a spike, not a screen: no smallest
experiment, cost, or week-one kill has been written for any of them.

- **Synthetic allele or peptide generation as an evaluation device. — WEAKEST, mostly closed.**
  Peptide-side generation is occupied: conditional peptide generation for MHC binding exists, and
  [superHLA](https://www.biorxiv.org/content/10.1101/2025.07.10.664084.full.pdf) designs 9-mer
  class I superbinders in silico and validated 100 of them by biochemical assay. Allele-side
  generation was not retrieved as an evaluation device, but it carries a fatal validity problem:
  a synthetic pseudo-sequence has no measurable ground truth, so nothing can be scored on it. It
  is usable only as a counterfactual probe — in-silico saturation mutagenesis over pseudo-sequence
  positions — which is interpretability, not evaluation, and
  [Do Pseudosequences Matter in Neoantigen Prediction?](https://www.biorxiv.org/content/10.64898/2025.12.09.693250v1.full)
  (Dec 2025) has already run random-pseudo-sequence baselines, 5–100 residue length sweeps, and
  ESM-2 and annotation-graph encodings against curated pseudo-sequences. `observed`.
- **Conformal / selective prediction under allele shift. — OPEN, strongest technique transfer.**
  The technique is proven on the adjacent problem:
  [Calibrated Abstention under Epitope Shift](https://arxiv.org/pdf/2604.13254) applies temperature
  scaling plus a conformal abstention rule with finite-sample coverage guarantees to TCR–pMHC
  under an epitope-held-out protocol, cutting error from 18.7% to 10.9% at 80% coverage. On the
  pMHC side, uncertainty quantification exists
  ([Cell Systems 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6715517/), epistemic uncertainty
  flagging sequences far from training examples) but predates the pan-allele shift framing and is
  not conditioned on allele distance. A bounded search did not retrieve conformal risk control for
  pMHC class I conditioned on distance-to-nearest-training-allele. `observed`, and a
  non-retrieval is never evidence that no paper exists. Fit to this repository is unusually good:
  the three shipped holdout designs are already a shift-severity ladder, and the normalized
  Hamming helper is already extracted.
- **Predictor-agnostic performance prediction. — OPEN, evidence now points against
  agnosticism.** MHCPerf predicts *MHCGlobe's* PPV and has not been externally validated
  against a different predictor. IEDB does per-allele method selection, but from *observed*
  benchmark performance, so it cannot cover alleles with no benchmark data — precisely the
  alleles MHCPerf targets. Whether the distance-to-training-data → performance relationship is
  **predictor-agnostic** was not retrieved as an asked question. `observed`.
  **Novelty-control findings recorded 2026-09-16, bearing on but not closing this question.**
  The [novelty-control study](pmhc/novelty_control.md) asked a different question —
  whether the shipped LOAO degradation gradient is a novelty effect or intrinsic allele
  difficulty, given that every arm in the shipped study was trained under holdout and so cannot
  distinguish the two — and added MHCflurry 2.2.1 as a control under a decision rule frozen
  before the control ran. MHCflurry supports predictions for every cohort allele; overlap with
  its training alleles was inferred, not directly verified. The control's fitted slope was
  -0.1674 (95% CI [-0.3957, +0.0609]) against the
  shipped reference slope -1.0315 (95% CI [-1.3116, -0.7514]); the control's CI spans zero and
  the reference falls outside it, so the frozen rule declares the shipped gradient a novelty
  effect under that rule. `observed` for the fitted slopes and rule outcome; direct training
  exposure remains `claimed`. The same measured slopes bear directly on
  *this* candidate's question, and point the other way: on the identical distance axis, one
  predictor's fitted slope is roughly six times the other's magnitude, so in this cohort the
  distance-to-performance relationship reads as predictor-*dependent* — the opposite of
  agnostic. `claimed`. This candidate remains open; see
  [`novelty_control.md`](pmhc/novelty_control.md) for the full comparison, its caveats, and
  what the result does not show.

## How to read the literature gate

`observed` means the stated search was run and its output was inspected; the saved results are
`/tmp/tcr-reporting-audit.json`, `/tmp/virobench-litcheck.json`,
`/tmp/retrieval-dedup-litcheck.json`, `/tmp/agent-eval-litcheck.json`, and
`/tmp/rag-reporting-litcheck.json`. The 2026-09-16 gate above (the study-derived and
build-on-published candidates) was run and its output inspected the same way, but that output
was not saved to disk, so those `observed` labels rest on in-session inspection rather than a
committed search transcript. `claimed` marks a scope, cost, or feasibility estimate. A
non-retrieval is never evidence that no paper exists. Thus “not already answered” below means
that this bounded, current literature gate did not locate a paper answering the narrower question
as defined; it is not a universal novelty claim.

All live candidates use public material only, require no training, and either use no model or
use only fixed inference models comfortably below the 600M limit. Rank reflects fit to the stated
constraints and the sharpness of the week-one kill, not a recommendation.

## 1. Reporting completeness of public RAG evaluations

**Question.** In a pre-specified sample of public 2024–2026 RAG papers that report a retrieval
evaluation, what proportion report enough to reconstruct the *evaluated retrieval collection*?
The unit is one paper. “Reconstruct” has a frozen rubric: corpus snapshot or immutable locator,
document extraction, chunking and overlap, duplicate-removal rule, retriever version and index
settings, query split, relevance or answer-label source, and any judge prompt/model/version.
This is a reporting audit, not a claim that the systems themselves reproduce.

**Literature gate — viable, `observed` but bounded.** The search located RAG evaluation reviews
and reproducibility guidance, including a 2026 systematic review of 12 RAG-evaluation studies
([review](https://link.springer.com/article/10.1007/s42979-026-05134-x)), a
reproducible evaluation library ([BERGEN](https://aclanthology.org/2024.findings-emnlp.449.pdf)),
and a single-system reproduction that identifies unreleased subsets, index construction, and
prompt details as limiting factors ([MetaRAG reproduction](https://arxiv.org/html/2604.19899v1)).
It did not return a cross-paper, field-bounded audit measuring whether these exact construction
fields are reported. The closest literature therefore motivates the question but does not answer
it as posed.

**Smallest experiment that could say something real, `claimed`.** Freeze a sampling frame before
collection: 50 primary papers from a named index and date range, stratified by open versus closed
generator and by benchmark versus custom corpus. Report a Wilson interval for the completeness
proportion, not a p-value. At the conservative 50% rate, 50 papers give an approximately
±14-percentage-point 95% interval; that is a descriptive result about this frame, not all RAG
research.

**Cheap week-one kill.** Screen 12 randomly selected papers; double-code four. Seed the rubric
with one synthetic complete method description and one description missing a required locator or
parameter. Stop if fewer than 10 papers have accessible methods/supplements, fewer than eight
actually contain a retrieval evaluation, either seeded case is misclassified, or the two coders
cannot resolve more than one disagreement under the written rubric. Those failures mean that the
sampling frame or rubric cannot support the intended measurement.

**Comparison contract.** What varies in the primary estimate is only each paper’s rubric result.
The corpus of papers, rubric version, and field-extraction rule are fixed before coding. “Named
dataset” versus “immutable dataset locator” is a reporting convention, so completeness is reported
as a curve from lenient to strict locator requirements, never as a single post-hoc rate. Any
stratified comparison is descriptive unless its estimand, adjustment, and a 20,000-draw resolved
interval are fixed beforehand.

**Cost.** `claimed`: 30–45 person-hours; no paid data and negligible compute.

**Lessons exercised.** External authorities cannot be covered by repository tests; label the
quantity honestly (reported reconstruction inputs, not “reproducibility”); test the rubric with
planted cases; report sensitivity to the reporting convention; do not manufacture inferential
claims from a small audit.

**Domain-knowledge cost.** Low to medium. It needs working knowledge of corpus snapshots,
chunking, qrels, and LLM judging, but those are in the proposed expertise area rather than a new
scientific vocabulary.

## 2. Retriever-rank stability over a near-duplicate-removal curve

**Question.** On two fixed, small public retrieval collections, do the rankings and paired
effect sizes of fixed retrievers change over a documented near-duplicate-removal threshold curve
more than under relevance- and support-matched random removal? This is a question about the
evaluation collection, not about a universally superior retriever.

**Literature gate — viable, `observed` but bounded.** Near-duplicates are already known to bias
learning-to-rank evaluation ([Froebe et al., 2020](https://dl.acm.org/doi/abs/10.1145/3397271.3401212)),
and BEIR is an established heterogeneous zero-shot retrieval benchmark
([Thakur et al., 2021](https://arxiv.org/abs/2104.08663)). The search did not locate the narrower
combination: fixed-model rank stability reported over a semantic-deduplication threshold curve
with random removals matched on removal count and qrel status. The result is a bounded gap, not a
claim that no related experiment exists.

**Smallest experiment that could say something real, `claimed`.** Before any scores are seen,
freeze two collections below a stated corpus-size cap, BM25 plus two fixed sub-600M embedding
retrievers, nDCG@10, a duplicate detector, its threshold grid, and a canonical-document rule for
each duplicate cluster. At every threshold, map qrels to the retained canonical document and run a
random-removal control matched on removed-document count, document-length bin, and qrel status.
Report the threshold curve, per-query paired differences, and rank ordering. Any interval or
decision-boundary p-value uses at least 20,000 paired query resamples.

**Cheap week-one kill.** Run a detector-only pilot on one frozen collection, with planted exact
duplicates, planted near-duplicates, and planted topical nonduplicates. Stop if it fails a planted
case; if every pre-specified threshold removes under 1% of eligible documents; if qrels cannot be
mapped without changing the query population; or if the candidate-pair volume makes manual
calibration infeasible. A secondary prospective kill: if both collections retain identical model
order at every threshold and matched control, stop rather than inflate a clean null into a larger
benchmark.

**Comparison contract.** The duplicate threshold is the only intended varying axis. Corpus
snapshot, queries, qrels, model weights, scoring metric, retained-document choice, and the
support-matching procedure are fixed. Because duplicate definition is upstream preprocessing, the
curve—not an endpoint—would be the result. Random controls distinguish changing *which*
documents are removed from merely shrinking the corpus.

**Cost.** `claimed`: 25–40 person-hours; public collections; CPU/MPS inference for small fixed
encoders should be hours to roughly a day per completed grid on the M1 Pro. No training.

**Lessons exercised.** The core design directly applies the sensitivity-curve rule, a
support-matched control, planted test failures, resolved paired estimates, and an explicit
attribution line.

**Domain-knowledge cost.** Low to medium. It requires qrels, nDCG, retrieval indexing, and
duplicate calibration; it avoids the biological terminology tax.

## 3. How large a TCR CD-HIT reporting survey must be to say something real?

**Question.** Among a frozen 2021–2026 full-text sampling frame of primary TCR studies that use
CD-HIT on CDR3 or adaptive-immune-receptor sequences, what proportion report enough information
to reconstruct their stated filtering decision? The minimum rubric is tool/mode, identity
threshold, word size or stated default, sequence field and normalization, reference direction,
and whether clustering controls a split, a database comparison, or something else. This does not
claim that a paper, dataset, or biological conclusion is irreproducible.

**Literature gate — viable but high-friction, `observed` but bounded.** Cognate’s direct Q5
screen found four close papers, three without enough parameters to reconstruct the filter
([`cdhit_litcheck.md`](cdhit_litcheck.md)); the current search additionally found a systematic
assessment of public TCR-repertoire data availability
([Zhang et al., 2022](https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2022.918792/full))
and method papers showing that UMI/deduplication choices matter
([RUMINA](https://academic.oup.com/bioinformatics/article/42/3/btag097/8496272)). Neither is a
cross-paper audit of reported CD-HIT decision parameters. The small four-paper pattern is not a
prevalence result.

**Smallest survey that could say something real, `claimed`.** A 12-paper pilot establishes a
usable frame and rubric. A 50-paper final sample estimates an unknown proportion to about
±14 percentage points at 95% in the conservative case; 100 papers would narrow this to about
±10 points but is not a credible two-week part-time commitment. The 50-paper survey is therefore
the first defensible scope, with a Wilson interval and no field-wide causal claim.

**Cheap week-one kill.** Stop after a stratified 12-paper pilot if fewer than 10 have accessible
full methods/supplements, fewer than eight actually apply CD-HIT to the specified sequence type,
the frozen rubric cannot classify two seeded complete/incomplete method descriptions correctly,
or double-coding exposes an unresolvable interpretation rule. This kills the survey before the
manual work expands.

**Comparison contract.** The primary quantity varies only by a paper’s reporting status, with a
fixed paper frame and rubric. The strictness of “reconstructible”—for example, whether a named
software default counts without a version—is an upstream convention. Report completeness across
ordered rubric variants; do not headline one convention. Any date or venue strata remain
descriptive unless separately pre-specified and estimated at proper resolution.

**Cost.** `claimed`: 35–55 person-hours, public papers only, negligible compute. It is near the
two-week boundary because adjudication—not coding—is the work.

**Lessons exercised.** This is the direct Q5 follow-up: converts an anecdotal four-paper result
into a bounded prevalence estimate; separates the label from what is measured; uses a convention
curve; and tests the audit mechanism before trusting it.

**Domain-knowledge cost.** High. TCR assay, CDR3, sequence identity, CD-HIT modes, UMI handling,
and split purpose all need an explicit vocabulary. The study avoids new biological inference, but
it does not escape the terminology tax that Cognate exposed.

## Dead at the literature gate

## 4. “Nucleotide foundation models degrade under phylogenetic and temporal shift”

**Status: dead.** The exact seed thesis is directly answered by
[ViroBench](https://arxiv.org/abs/2605.25388): it evaluates 66 nucleotide foundation models over
18 viral-genomics scenarios and reports degradation under genus-disjoint phylogenetic and temporal
splits. The source also makes its public data/code available. A sub-600M, M1-only rerun would be a
partial replication or an implementation check, not a new answer to the stated question.

**Cheapest kill and cost.** `observed`: one literature check, completed before any download or
model run. `claimed`: the direct replication would exceed the useful scope because the published
comparison spans 66 models; shrinking it would sacrifice the thesis rather than isolate a new
estimand.

**Comparison contract and lesson.** ViroBench already varies split type while holding its
benchmark protocol fixed; the proposed candidate would have repeated that axis. The standing
literature check fired at step zero, exactly as Cognate’s fourth literature lesson requires.

**Domain-knowledge cost.** High: viral taxonomy, phylogenetic splitting, temporal metadata, and
nucleotide-model evaluation would recreate the domain-learning burden without buying a novel
question.

## 5. Generic audit of LLM-agent benchmark contamination or reliability

**Status: dead in its generic form.** This is occupied by both benchmark-specific and cross-
benchmark work. [The SWE-Bench Illusion](https://arxiv.org/html/2506.12286v3) tests evidence
consistent with benchmark-specific memorization and compares external/recent task sets;
[SWE-rebench](https://arxiv.org/abs/2505.20411) constructs a continuing fresh-task pipeline; and
[Automated Benchmark Auditing for AI Agents and Large Language Models](https://arxiv.org/html/2605.26079v2)
already defines an executable audit protocol across benchmark families. A generic project would
restate an established problem or make untestable claims about hidden training data.

**Cheapest kill and cost.** `observed`: the literature gate above killed the broad question before
task collection. `claimed`: a viable descendant would need a sharply bounded, observable target
not covered by those audits—for example a single frozen benchmark and one defined construction
defect—but that is a different candidate, not a reason to rescue the generic one.

**Comparison contract and lesson.** “Contaminated” mixes source overlap, semantic similarity,
benchmark curation, and private model-training exposure. Without fixing one observable axis, the
label cannot match the measurement. This is a direct repeat of Cognate’s misattribution failure.

**Domain-knowledge cost.** Medium. The vocabulary is familiar, but the apparent accessibility is
misleading: the central hidden-training-data premise is not measurable from public benchmark
artifacts.

## Literature sources used

- [ViroBench](https://arxiv.org/abs/2605.25388), 2026.
- [Automated Benchmark Auditing for AI Agents and Large Language Models](https://arxiv.org/html/2605.26079v2), 2026; [The SWE-Bench Illusion](https://arxiv.org/html/2506.12286v3), 2025; [SWE-rebench](https://arxiv.org/abs/2505.20411), 2025.
- [Sampling Bias Due to Near-Duplicates in Learning to Rank](https://dl.acm.org/doi/abs/10.1145/3397271.3401212), 2020; [BEIR](https://arxiv.org/abs/2104.08663), 2021.
- [Evaluating Retrieval Augmented Generation](https://link.springer.com/article/10.1007/s42979-026-05134-x), 2026; [BERGEN](https://aclanthology.org/2024.findings-emnlp.449.pdf), 2024; [MetaRAG reproduction](https://arxiv.org/html/2604.19899v1), 2026.
- [Data Availability of Open T-Cell Receptor Repertoire Data](https://www.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2022.918792/full), 2022; [RUMINA](https://academic.oup.com/bioinformatics/article/42/3/btag097/8496272), 2026; and Cognate’s prior [CD-HIT literature check](cdhit_litcheck.md).
