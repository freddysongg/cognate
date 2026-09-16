# LOAO HLA-transfer study programme

## Problem

Determine whether a 34-residue MHC pseudo-sequence improves pMHC binding
prediction when the target HLA allele has no binding rows in training. The
answer must separate direct allele transfer, joint peptide-and-allele novelty,
and transfer over greater pseudo-sequence distance.

## North star

Produce one limited, credible answer about MHC-sequence transfer without
reopening the completed seen-allele replication or adapting the design to
intermediate results.

## Chosen design

Run three independent, predeclared tracks from one shared LOAO foundation.
Each track has its own worktree, results artifact, report, tests, and GitHub
epic. The tracks are different estimands, not three attempts to establish the
same claim.

| Track | Epic | Test partition | Claim boundary |
|---|---|---|---|
| Allele-only LOAO | #32 | all rows for one target allele are test rows; all its rows are absent from training | direct transfer to a new allele, with training peptide overlap reported |
| Allele-and-peptide LOAO | #33 | target-allele rows are test rows; the target allele and every test peptide are absent from training | joint novelty of allele and peptide, not pure allele transfer |
| Pseudo-sequence-cluster LOAO | #34 | all rows for a predeclared pseudo-sequence group are test rows; all group alleles are absent from training | cluster-held-out transfer under the frozen grouping rule; it does not isolate distance from training-set composition |

The common foundation must be committed before the three worktrees are
created. It owns source validation, all-or-fail candidate preflight,
transfer-partition guards, a target-only validation split, shared evaluation,
and result-schema validation. The worktrees branch from that foundation and
contain only their track-specific partitioner, diagnostics, outputs, tests,
and report.

## Shared frozen contract

- Retain the verified NetMHCpan binding-affinity source, nine-residue HLA-A/B/C
  filtering, normalized-affinity target, and source-hash checks from the
  completed pMHC replication.
- Use the existing 47-allele cohort as the primary cohort. Do not expand,
  shrink, or retune it after inspecting results. If a track cannot score an
  allele, report the reason before headline aggregation.
- Train MLP arms on continuous normalized affinity and use `affinity > 0.426`
  only for classification metrics.
- Use a fixed 90/10 validation split stratified by classification target only,
  seed 0. This replaces the seen-allele `Allele × Target` validation strata,
  which are invalid after target-specific peptide exclusions.
- Keep macro standardized AUC0.1 by held-out allele as headline metric,
  20,000 two-level bootstrap draws, seed 0, paired comparisons, and
  aggregate-only durable outputs.
- Retain raw source rows, row-level predictions, and embedding caches locally
  and ignored. Do not run or redistribute NetMHCpan.

## Fixed arms and comparisons

Every track uses the same transfer-compatible arms:

1. fixed random scores;
2. pooled peptide-only MLP using raw BLOSUM50 peptide rows, hidden sizes 55
   and 66, seeds 0–4, continuous-affinity MSE, and no allele feature;
3. shuffled-mapping MLP with the same peptide-plus-pseudo-sequence architecture
   as the correct-mapping MLP, but a deterministic cyclic derangement of sorted
   unique pseudo-sequences across allele names;
4. nearest-pseudo-sequence PWM transfer, fitted on the retained allele with
   lowest normalized Hamming distance across the 34 pseudo-sequence positions;
5. correct-mapping pseudo-sequence MLP, with the target allele's correctly
   mapped pseudo-sequence but no target-allele binding rows.

The seen-allele one-hot MLP is explicitly excluded: it has no valid feature
for an allele absent from training. Per-allele nearest retained
pseudo-sequence distance is a required diagnostic in all tracks. The
allele-only track also reports peptide overlap; the joint-novelty track makes
that overlap zero by construction.

The nearest-PWM arm is a sequence-neighbour baseline, not a non-transfer
control. A correct-mapping result is declared only when the paired 95% macro
AUC0.1 intervals for correct-mapping MLP minus pooled peptide-only MLP and
correct-mapping MLP minus shuffled-mapping MLP both have lower bounds above
zero in the allele-only track. Correct-mapping MLP minus nearest-PWM is
secondary. The joint-novelty and cluster tracks are separate estimands; no
pooled cross-track effect, causal biological explanation, or result-driven
revision of the allele-only conclusion is allowed.

## Track-specific frozen rules

For every target, the train partition contains only non-target alleles. The
nearest-PWM source is selected after every track-specific exclusion from
retained alleles with at least one positive and one negative training row. It
uses normalized Hamming distance across the 34 pseudo-sequence positions and
breaks ties by lexicographically smallest allele name. The shuffled-mapping
control uses the fixed cyclic shift by one over sorted unique pseudo-sequences;
the runner stops before scoring if it cannot construct a derangement with a
different pseudo-sequence for every mapped allele.

The allele-and-peptide track additionally removes from training every row
whose peptide occurs in the target allele's test rows. It reports target-level
training-row deletion, positive/negative counts, and retained allele count.
Its conclusion is conditional on this joint exclusion; it does not isolate a
peptide-novelty effect from altered training-set composition.

The cluster track uses complete-linkage hierarchical clustering on normalized
Hamming distance across the 34 pseudo-sequence positions. Its cut is the
smallest integer Hamming threshold that leaves at most one singleton in the
frozen 47-allele pseudo-sequence set; that rule selects `12 / 34` before any
binding score is inspected. Ties in cluster construction are resolved by
sorted allele name. It reports the resulting membership, training rows, and
nearest retained pseudo-sequence distance. It is named cluster-held-out
transfer and does not claim that distance alone caused any difference.

## Preconditions and safeguards

- The completed pMHC working tree is uncommitted. An explicit user-authorized
  baseline commit is required before worktree creation; no worktree may be
  based on an incomplete historical commit.
- Before a track scores any arm, its preflight must establish that all 47 target
  alleles have both test classes, a global two-class training partition, a
  valid shuffled mapping, and an eligible nearest-PWM source. Failure for even
  one target stops the entire track; no target may be silently skipped and the
  paired macro denominator is fixed at 47.
- Partition tests must fail on deliberate target-allele leakage. The
  allele-and-peptide track must also fail on deliberate peptide leakage; the
  cluster track must fail on group-membership leakage.
- Pseudo-sequence availability and exact pseudo-sequence mapping must be
  verified for each scored target allele.
- The cluster rule must be deterministic, documented, and frozen before any
  score or outcome is inspected; it may use pseudo-sequences but not labels,
  arm scores, or result-driven thresholds.
- All later tracks run from their fixed contracts even if earlier results are
  weak, strong, slow, or inconclusive.

## Scope exclusions

- No new architecture, fine-tuning, hyperparameter search, external dataset,
  antigen-presentation endpoint, MHC class II data, or NetMHCpan executable.
- No use of a one-hot unseen-allele fallback.
- No post-result cohort changes, cluster redefinition, alternate bootstrap
  configuration, or reinterpretation of the three tracks as one estimand.
- No commit, worktree creation, or implementation in this brief phase.

## Remaining operational decision

Decide the common-foundation branch and commit boundary after the current
completed pMHC changes are explicitly committed. This is a repository-state
decision, not a change to any track's scientific contract.

## Review resolution

| Review objection | State | Resolution |
|---|---|---|
| Cluster track had no frozen rule | accepted | Complete-linkage normalized-Hamming clustering now uses a deterministic no-more-than-one-singleton cut rule, which selects `12 / 34` before scoring. |
| Transfer PWM was underspecified | accepted | The distance metric, timing, class eligibility, and lexicographic tie break are fixed. |
| Peptide-only predictor was unspecified | accepted | Its BLOSUM50 input, MLP architecture, target, seeds, and loss are fixed. |
| Success rule was outcome-adjustable | accepted | Both correct-minus-peptide-only and correct-minus-shuffled paired lower bounds must exceed zero. |
| Joint exclusion broke inherited validation strata | accepted | All tracks use fixed target-only validation strata and an all-or-fail 47-target preflight. |
| Correct MLP could win through capacity rather than mapping | accepted | A same-architecture deterministic shuffled-mapping MLP is now required. |
| Joint exclusions vary by target | accepted-risk | The track reports deletion diagnostics and claims only the stated compound intervention. |
| Cluster removals change data composition | accepted-risk | The track is named cluster-held-out transfer and makes no distance-causal claim. |
| Exact peptide overlap invalidates allele-only LOAO | refuted | The track claims allele transfer with overlap reported; #33 is the distinct zero-overlap estimand. |
| 47 observed alleles cannot test unseen alleles | refuted | Each target is unseen to its model; no claim extends to poorly characterized alleles. |
| Bootstrap proves population biology | refuted | Intervals are conditional uncertainty for the frozen source and fitting recipe only. |

## Evidence

True allele-held-out evaluation has long been used to test pan-allele binding
prediction, with comparison to peptide-only and closest-HLA baselines:
https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0000796

A recent pan-MHC evaluation withholds every row for an allele in its validation
fold and reports sparse pseudosequence coverage as a practical transfer limit:
https://www.pnas.org/doi/10.1073/pnas.2405106122
