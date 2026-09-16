# pMHC binding replication — shaped brief

## Problem

The closed Cognate project found that neither edit-distance retrieval nor frozen
ESM-2 generalized above chance to unseen peptides in TCR–peptide prediction. The
next project should measure the upstream peptide–MHC class I binding problem with
the same retrieval operators and metric implementation, then document how their
behavior differs under explicit, non-equivalent data contracts.

The deliverable is mechanistic understanding, not a new model or a publication
claim. The study must stay small enough to reach a real result before process and
documentation become the work.

## North star

Produce one owned comparison showing how the same retrieval operators behave on
TCR–peptide prediction and peptide–MHC binding, and identify which simple
allele-specific models meet a predeclared sufficiency rule on the latter. This is descriptive evidence;
it does not identify a biological cause for the cross-task gap.

## Chosen approach

Run a minimal direct contrast on one public release and its supplied five-fold
partition. Reuse Cognate's edit-distance retrieval, ESM-2 cosine retrieval,
metrics, and bootstrap machinery without changing their operators. Add only the
two mechanisms needed to interpret the result: a per-allele position-weight
matrix (PWM) and a small allele-aware multilayer perceptron (MLP) modeled on the
published NetMHCpan-4.1 architecture.

This shaped brief supersedes the current draft at `docs/pmhc/brief.md` when it is
promoted to the durable project record. The presentation endpoint, second IEDB
dataset, MHCflurry comparator, cross-attention arm, and seven-document layout in
that draft are rejected scope and must not reappear in planning.

The NetMHCpan executable is not part of the experiment. Its academic license
prohibits redistribution, modification, commercialization, and publication of
benchmark results without prior written consent. The model in this project is
an independent implementation from the published method description.

## Experimental contract

### Endpoint

The primary endpoint is **peptide–MHC class I binding affinity**: whether a
nine-residue peptide physically binds a human MHC-I display molecule. The study
is restricted to human HLA-A, HLA-B, and HLA-C alleles. Models that learn affinity
use the release's continuous normalized target. Classification metrics label a
row positive only when its target is greater than `0.426`, the published rounded
500 nM threshold; this reproduces the release's stated positive count. Antigen
presentation—whether a cell actually displays the peptide—is a different endpoint
affected by processing and expression; it is an interpretation caveat and not a
completion requirement.

### Data

Use only nine-residue peptides from the binding-affinity (`c000_ba` through
`c004_ba`) partitions in the NetMHCpan-4.1 training release. Preserve the supplied
fold assignment and restrict rows to HLA-A/B/C. The locally inspected archive
contains:

- 208,093 binding-affinity rows across all species;
- 170,107 HLA-A/B/C rows across 109 alleles;
- 42,001 positive HLA-A/B/C rows at target greater than `0.426`;
- 10,895 HLA-A/B/C artificial negatives with target value `0.01`.

The nine-residue analysis population contains 30,869 positives and 95,506
negatives across 109 alleles. The headline macro cohort is fixed before scoring
to alleles with at least 100 positive and 100 negative out-of-fold rows: 47
alleles and 112,128 rows in the inspected release.

The source archive and derived row-level data remain local and out of git. The
repository records the source URL, retrieval date, SHA-256, filters, row counts,
and a reproducible fetch/build command.

The current IEDB MHC-ligand export was inspected but is excluded. It contains
5,789,555 records in a 9.21 GB CSV with a 112-column two-row header. Adding a
second, later-curation evaluation set would turn replication into dataset
construction and materially expand the project.

### Arms

1. **Random scores** — floor and metric degeneracy control.
2. **Length and composition** — logistic regression on peptide length and amino
   acid counts, ignoring the allele; detects shortcut signal in the negative
   construction.
3. **Edit-distance retrieval** — per-allele nearest-positive normalized
   Levenshtein similarity through `src/cognate/baseline_knn.py`.
4. **ESM-2 cosine retrieval** — the same nearest-positive operator with frozen
   ESM-2 35M, layer 10, mean-pooled peptide embeddings from
   `src/cognate/embed.py`; representation is the only axis changed from arm 3.
5. **Per-allele PWM** — a nine-position log-odds matrix trained from that fold's
   positive rows against its non-positive rows for the same allele. Each
   position/amino-acid cell receives a pseudocount of one; a peptide score is the
   sum of its nine position-wise log odds.
6. **Pan-allele MLP** — concatenate raw BLOSUM50 score vectors for the nine
   peptide residues and 34-residue MHC pseudo-sequence, standardize each input
   feature on the training fold, and use one ReLU hidden layer with 55 or 66 units
   plus a sigmoid scalar output. Train with Adam at learning rate `0.001`,
   mean-squared-error loss on continuous normalized affinity, at most 200 epochs,
   and patience-20 early stopping on a fixed allele-stratified 10% split of the
   training folds. Use seeds `0` through `4` per hidden size in each of five folds:
   50 trained networks total, averaged for each fold's prediction.
   Run the same architecture once with allele one-hot input instead of the
   pseudo-sequence; under seen-allele folds this compares structured residue
   encoding and cross-allele parameter sharing against an unstructured stable
   allele identifier. It does not test whether the pseudo-sequence contains extra
   information or transfers to unseen alleles.

Arms 3 and 4 are the cross-task contrast. Arm 5 tests whether a simple
allele-specific positional motif meets the sufficiency rule below. Arm 6 tests
whether a conventional allele-aware model fits the binding data and compares the
inductive bias of pseudo-sequence versus one-hot allele encoding on seen alleles.
None of these comparisons
identifies the biological cause of the TCR/pMHC gap. Arms 1 and 2 are controls,
not feature work.

### Evaluation

- Headline metric: per-allele macro standardized partial AUROC through false
  positive rate 0.1, using `src/cognate/metrics.py`.
- Diagnostics reported for every arm: pooled AUROC, per-allele macro AUROC, and
  pooled AUPRC. Per-allele AUPRC is not reported.
- Intervals and paired differences use 20,000 fixed bootstrap draws, with the draw
  count printed beside each interval.
- Comparisons use out-of-fold predictions from the supplied folds. Exact peptide
  sets were observed to be disjoint across all five HLA-A/B/C folds, so no custom
  deduplication or exact-match exclusion is added. That disjointness becomes a
  frozen contract check. The closed TCR contract remains unchanged; pMHC receives
  its own contract and frozen hashes.
- AUROC is not presented as a decoy-ratio curve because class prevalence does not
  define AUROC. AUPRC may be shown against pre-registered negative ratios when the
  source rows support them without replacement. AUC sensitivity instead reports
  the negative-construction convention and peptide-overlap rule explicitly.
- Before results exist, write a shared-contract table stating what varies and is
  held fixed across the two tasks. The pMHC fold is not described as equivalent
  to the TCR unseen-peptide split.
- Report metrics separately for measured non-binders (`target != 0.01`) and the
  artificial target-`0.01` negatives. Also report nearest-positive distance and
  exact-duplicate counts by negative source. The cross-task conclusion is
  dataset-specific if it disappears when artificial negatives are removed.
- The cross-task contrast is within-task uplift over random, shown side by side;
  the two macro AUC0.1 values are not treated as one matched estimand because TCR
  groups by peptide and pMHC groups by allele.
- An arm is **predictive** only when the lower bound of the 95% interval for
  headline macro AUC0.1 minus `0.5` is greater than zero. The PWM is **sufficient relative to the MLP** only
  when it is predictive and the upper bound of the paired 95% interval for
  `MLP - PWM` is below the predeclared non-inferiority margin of `0.02`. Otherwise
  the brief reports the measurements without calling the PWM sufficient.

## Literature gate outcome

- **Observed:** the NetMHCpan-4.1 partitions were produced with a Hobohm1-based
  common-motif algorithm using motif length eight.
- **Observed:** binding-affinity data were supplemented with 100 random UniProt
  negatives per MHC at target `0.01`; eluted-ligand data used length-wise random
  negatives at five times the most abundant positive length.
- **Observed:** the official NetMHCpan-4.1 and 4.2 pre-download license texts are
  byte-identical. No acceptance or software request was made.
- **Observed:** the IEDB export count and schema above came from a complete local
  CSV parse; every parsed record had 112 fields.
- **Observed:** the peer-reviewed 2019 VDJdb update reports 61,049 specificity
  records, 42,211 unique TCR sequences, and 212 epitopes. A 2025 primary preprint
  more directly estimates that ten epitopes account for about half of its MHC-I
  entries. The latter remains preprint evidence.

## Scope

### In scope

- A reproducible local adapter from the published binding-affinity partitions to
  the existing Cognate baseline and metric contracts.
- The six arms above on the same rows and folds.
- A control-first gate: corrupt each new contract check and observe it fail before
  trusting its passing state.
- One matched contrast table for the edit-distance and ESM-2 retrieval arms.
- Four durable records at most: `docs/pmhc/brief.md`, one combined experiment
  record, the accumulating `docs/contrast.md`, and `docs/pmhc/postmortem.md`.

### Non-goals

- Antigen presentation, immunogenicity, neoantigen ranking, or MHC class II.
- Running or redistributing the NetMHCpan executable.
- A second dataset, architecture search, cross-attention, ESM-2 fine-tuning, or a
  larger language model.
- Beating a published method, claiming novelty, or treating non-retrieval as
  evidence of absence.
- Rewriting closed TCR documents.

## Success and stop conditions

The replication succeeds when all six arms are scored through one implementation,
the two retrieval arms are juxtaposed with the closed TCR results under an
explicit shared-contract table, and the PWM/MLP controls show which simple
allele-specific models are sufficient on this release. Reaching a famous round
AUC is not a completion gate, and the project does not claim to have identified
the biological cause of the gap.

Stop after the contrast and postmortem. A weak MLP result triggers diagnosis of
the published contract, not architecture expansion. A surviving publication or
novelty question is recorded as follow-up work outside this project.

## Open questions

- None that gate implementation. A future leave-one-allele-out study would be
  required to claim transferable MHC sequence structure; it is explicitly outside
  this project's scope.

## Primary sources

- NetMHCpan-4.1 paper and supplementary material:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7319546/
- NetMHCpan-4.0 affinity target transformation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5679736/
- NetMHCpan-4.1 service, training release, and license entry point:
  https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
- IEDB MHC-ligand export:
  https://www.iedb.org/downloader.php?file_name=doc%2Fmhc_ligand_full_single_file.zip
- VDJdb 2019 database update:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6943061/
- Primary VDJdb validation preprint:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12132471/
