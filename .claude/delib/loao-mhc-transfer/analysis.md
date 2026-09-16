# Analysis — Can MHC pseudo-sequence features improve binding prediction for HLA alleles not seen in training, beyond peptide-only, PWM, and non-transfer controls?

## Flow gaps

- The present pMHC path is organized around five supplied binding-affinity folds; an allele-held-out partition is explicitly unverified. A LOAO result needs that partitioning seam before the existing pMHC scoring module can answer the transfer question. _(edge: **A leave-one-allele-out partitioner** → **pMHC scoring module (`src/cognate/pmhc.py`)**)_
- The connection from the pMHC runner to the pMHC scoring module is unverified. Until it is made explicit, a LOAO runner could bypass the existing cohort and scoring guards rather than reusing them. _(edge: **pMHC runner (`scripts/run_pmhc.py`)** → **pMHC scoring module (`src/cognate/pmhc.py`)**)_

## Rewrite candidates

- Treat the pMHC scoring module as the load-bearing boundary: it currently owns fold parsing, source-contract validation, cohort construction, and scoring arms. Extracting the partition/evaluation contract around it would make an allele-held-out experiment easier to audit without altering the established scoring arms. _(node: **pMHC scoring module (`src/cognate/pmhc.py`)**)_
- Make the runner-to-scorer entry point a first-class boundary before introducing LOAO-specific output. The existing results artifact is written by the runner, while its scoring-module call edge was not captured. _(edge: **pMHC runner (`scripts/run_pmhc.py`)** → **pMHC scoring module (`src/cognate/pmhc.py`)**; node: **pMHC results artifact (`data/pmhc/results.json`)**)_

## Optimization

- Reuse the shared retrieval module for any non-transfer control instead of reimplementing peptide retrieval for LOAO; both the pMHC scoring module and the two TCR runners already call it. _(node: **Shared retrieval module (`src/cognate/baseline_knn.py`)**)_
- Reuse the metrics module’s group-macro, bootstrap, diagnostic, and paired-comparison surface for LOAO reporting if its integration is confirmed; this keeps the new comparison aligned with the current measurement vocabulary. _(node: **Metrics module (`src/cognate/metrics.py`)**)_

## Rejected-already

- None recorded for the LOAO question. The knowledge capture has no `session_matches`, so no session-sourced direction is eligible to mark as previously rejected; the prior TCR material should not be promoted into an MHC-transfer decision. _(node: **pMHC postmortem (`docs/pmhc/postmortem.md`)**; session: `.claude/delib/loao-mhc-transfer/harvest-knowledge.md` — `session_matches: []`)_

## Open questions

- Can the NetMHCpan training archive and its 47-allele cohort form valid held-out-allele training/test partitions under the current source contract? The archive lifecycle is unknown and the map contains no LOAO partition edge. _(node: **NetMHCpan training archive**; node: **pMHC source contract (`data/pmhc/source_contract.json`)**; edge: **A leave-one-allele-out partitioner** → **pMHC scoring module (`src/cognate/pmhc.py`)**)_
- Are pseudo-sequences present for every candidate held-out allele and how should missing pseudo-sequences be handled? The file supplies sequences for requested alleles, but its lifecycle is unknown. _(node: **MHC pseudo-sequence file (`MHC_pseudo.dat`)**)_
- Which existing scoring arms, if any, depend on the embedding module during pMHC evaluation? The scoring-module-to-embedding-module edge is unverified. _(edge: **pMHC scoring module (`src/cognate/pmhc.py`)** → **Embedding module (`src/cognate/embed.py`)**)_
- Does the current runner already route pMHC scores through the shared metrics module, and which per-allele aggregation stays valid when each outer test unit is an entire allele? That integration edge is unverified. _(edge: **pMHC scoring module (`src/cognate/pmhc.py`)** → **Metrics module (`src/cognate/metrics.py`)**; node: **pMHC runner (`scripts/run_pmhc.py`)**)_
