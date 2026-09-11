# pMHC binding replication postmortem

**Status: CLOSED.** The fixed seven-arm experiment, source-sensitivity analysis,
TCR/pMHC contrast, and aggregate record are complete. The project stops here
without adding another dataset or model architecture.

## 1. Which arms were predictive?

All six non-random arms met the predeclared rule that the 95% macro-AUC0.1
interval lower bound exceed 0.5. The observed points and lower bounds were:
length/composition 0.5154 (lower 0.5091), edit retrieval 0.6088 (0.5964), ESM-2
retrieval 0.5660 (0.5542), PWM 0.7443 (0.7231), pseudo-sequence MLP 0.8283
(0.8108), and one-hot MLP control 0.8252 (0.8079). Random was not predictive:
0.4998 [0.4978, 0.5023].

## 2. Was PWM sufficient relative to the pseudo-sequence MLP?

No. PWM was predictive, satisfying the first condition, but the paired
pseudo-sequence-MLP-minus-PWM difference was +0.0841 [+0.0688, +0.1008]. Its
upper bound exceeded the predeclared 0.02 margin, so the sufficiency criterion
failed.

## 3. Did the retrieval contrast survive removal of artificial negatives?

Yes, descriptively. On positives plus measured non-binders, edit retrieval was
0.6072 [0.5948, 0.6212] and ESM-2 retrieval was 0.5651 [0.5533, 0.5795], versus
the observed random point of 0.4998. Their point uplifts were +0.1074 and +0.0653,
and the same edit-above-ESM ordering remained after artificial negatives were
removed. No paired edit-minus-ESM interval was estimated, so this is an ordering
of observed points rather than a tested representation difference.

## 4. What was construction-sensitive?

Magnitude was sensitive to negative source. The artificial-negative slice gave
edit and ESM-2 macro-AUC0.1 points of 0.7052 and 0.6344, compared with 0.6072 and
0.5651 on measured non-binders. Artificial negatives were also farther from the
nearest training-fold positive by edit distance: mean 5.689 and median 6 versus
mean 5.272 and median 5 for measured non-binders. Exact-duplicate counts were
zero for both sources. The length/composition control also rose from 0.5153 to
0.5441 on the artificial-negative slice, reinforcing that this slice is easier
under several representations rather than isolating a biological mechanism.

## 5. Does any follow-up deserve a new project?

No follow-up is required to close this replication. If a transferable MHC-
sequence claim is desired, a predeclared leave-one-allele-out study would deserve
its own project: the current seen-allele folds produced nearly equal
pseudo-sequence and one-hot MLP points, 0.8283 and 0.8252, but cannot test unseen-
allele transfer. This result does not justify expanding the present project into
cross-attention, fine-tuning, or a second dataset.

## Stop decision

The core descriptive result held after removal of artificial negatives, PWM did
not meet the stated sufficiency rule, and negative-source construction materially
changed effect size. Those answers satisfy the stop condition. They do not
identify why TCR and pMHC retrieval differ biologically, and they do not support
a publication or novelty claim.
