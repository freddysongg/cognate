# TCR and pMHC retrieval contrast

| Axis | Closed TCR task | pMHC task |
|---|---|---|
| Endpoint | TCR–peptide binding label | normalized pMHC affinity; `>0.426` only for classification |
| Query | CDR3β | nine-residue peptide |
| Retrieval group | peptide | allele |
| Retrieval operator | maximum similarity to a positive | same |
| Edit representation | normalized Levenshtein | same |
| ESM representation | ESM-2 35M, layer 10, residue mean pool, cosine | same |
| Partition | closed seen-peptide slice | supplied common-motif BA folds with disjoint peptide sets |
| Headline metric | macro standardized AUC0.1 by peptide | same implementation, grouped by allele |
| Cross-task quantity | within-task uplift over observed random | same; shown side by side, never paired across tasks |

This comparison is descriptive and dataset-specific. The two headline values are not one matched estimand, and a difference does not identify a biological cause.

## Observed point contrast

| Task | Retrieval arm | Observed random | Arm macro AUC0.1 | Point uplift over observed random |
|---|---|---:|---:|---:|
| Closed TCR | Edit retrieval | 0.5009 | 0.5654 | +0.0645 |
| Closed TCR | ESM-2 35M layer 10 retrieval | 0.5009 | 0.5358 | +0.0349 |
| pMHC | Edit retrieval | 0.4998 | 0.6088 | +0.1090 |
| pMHC | ESM-2 35M layer 10 retrieval | 0.4998 | 0.5660 | +0.0663 |

For pMHC positives plus measured non-binders, excluding artificial negatives,
the edit and ESM-2 point uplifts remained +0.1074 and +0.0653. The artificial-
negative slice produced larger uplifts of +0.1939 and +0.1231, so effect magnitude
was construction-sensitive. These are arithmetic differences within each task or
slice; no cross-task or edit-minus-ESM paired interval was estimated.
