# Next-project candidates — scoping only

`docs/lessons.md` and `docs/final_state.md` are absent in this checkout. This screen therefore uses the self-contained lessons in the request. No build, download, model run, or dataset annotation was started.

## Screen used for every candidate

- **Novelty first:** the literature check below was run before proposing an experiment. “Not answered” means that this screen did not find a study answering the *bounded* question, not that no related paper exists.
- **Hardware and time:** every live candidate uses public material, no training, and at most small retrieval/embedding models (under 600M parameters) on the M1 Pro. A first result is scoped to two weeks of part-time work or less.
- **Measurement:** the unit, comparison, fixed variables, split definition, and preprocessing curve are specified before execution. A result cannot be headlined from a coarse bootstrap: use at least 20,000 paired resamples for any interval or test that is reported.
- **Kill before build-out:** every live candidate has a week-one condition that stops the project, not merely changes the story.

The ranks below reflect fit to the constraints and the sharpness of the kill condition; they are not a recommendation.

## 1. Reporting sufficiency in UMI-based TCR repertoire studies

**Question.** Among a pre-specified, stratified sample of public TCR-repertoire studies that say they use UMI-based deduplication, what proportion report enough information to reconstruct their *declared deduplication configuration*? The unit is a paper, not a patient, library, clone, or citation. “Enough” would be a frozen checklist: UMI layout and length, extraction rule, error-correction/clustering method and distance, minimum family support, consensus rule, read/quality filters, clonotype-collapsing rule, and software version/parameters.

**Why it is not already answered.** Existing work establishes that TCR assays and UMI deduplication choices change quantification, and compares methods or diversity metrics. For example, SEQTR reports method-dependent repertoire differences and describes a particular UMI consensus procedure; the 2025 diversity study evaluates metric robustness across protocols. The literature screen found no cross-paper audit that measures whether the published record contains the parameters needed to reconstruct a study’s own deduplication configuration. That is a narrower claim than “the papers are irreproducible,” which cannot be inferred when raw data, software, or environment are unavailable. [SEQTR](https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(23)00078-4), [TCR diversity-method evaluation](https://pubmed.ncbi.nlm.nih.gov/40369611/), [UMI-nea comparison](https://academic.oup.com/bioinformatics/article/41/9/btaf514/8256683)

**Smallest survey that could say something real.** Use a fixed sampling frame (for example, 2021–2025, PubMed-indexed primary bulk TCR repertoire studies mentioning UMI or molecular barcode). A 50-paper pilot estimates an unknown proportion only to roughly ±14 percentage points at 95% confidence in the conservative case; a 100-paper survey reaches roughly ±10 points. The viable two-week scope is **50 papers** and a descriptive estimate with its Wilson interval. It should not claim field-wide prevalence beyond that frame. Reproduction attempts belong only to the subset with public inputs and are reported separately.

**Week-one kill condition.** Screen a stratified 12-paper pilot, double-code four of them, and plant two synthetic method paragraphs—one complete and one missing a required parameter—before unblinding. Kill the survey if fewer than 10 of 12 papers have accessible full methods/supplements, fewer than 8 are actually UMI-deduplicated TCR studies, the planted incomplete paragraph passes, or the two coders disagree on more than one of the four double-coded papers after applying the frozen rubric. Those failures mean the sample frame or rubric cannot support the intended claim.

**Comparison contract.** The primary comparison is between reporting-complete and reporting-incomplete papers under one rubric. Year, journal, assay type, and study aim are descriptive strata, not causal explanations. Nothing varies inside the primary estimate except whether each required field is reported. If “complete” depends on a debatable rule (for example, whether a named pipeline without a command line counts), report the completeness rate as a curve over that rule, rather than selecting one convention after inspection.

**Cost.** Public full text and supplements; about 25–40 person-hours for a 50-paper sample, including double-coding and adjudication. CPU cost is negligible.

**Lessons exercised.** Directly tests label-to-measure alignment (“reproducible” versus “reported enough to reconstruct”), planted bad rows, pre-registered overturn conditions, and a preprocessing-convention curve. It avoids manufacturing a p-value: the result is a resolved interval on a survey proportion.

**Domain-knowledge cost.** **High.** The audit needs a small controlled vocabulary for TCR assay design, UMI handling, clonotypes, and sequencing pipelines. The scope controls this by auditing reporting, not interpreting biology; it still carries the Cognate terminology tax.

## 2. Does deduplication change retrieval-model rankings, beyond the amount of corpus removed?

**Question.** On a fixed small subset of public BEIR tasks, do rankings of fixed retrieval systems change over a documented near-duplicate-removal threshold curve, more than under a support-matched random-removal control? This is about benchmark construction, not a claim that one retriever is intrinsically better.

**Why it is not already answered.** Near-duplicates are known to create sampling bias in learning-to-rank, and BEIR is an established heterogeneous zero-shot evaluation suite. But the literature screen did not find the bounded experiment: a public BEIR-style evaluation that reports **model-rank stability as a curve over semantic-deduplication thresholds while matching the number and query-relevance support of removed documents with random controls**. Existing benchmark construction can remove overlaps, as PL-MTEB does, without answering whether the removal convention itself changes comparative conclusions. [Near-duplicate sampling bias](https://dl.acm.org/doi/abs/10.1145/3397271.3401212), [BEIR](https://arxiv.org/abs/2104.08663), [PL-MTEB split-level deduplication](https://arxiv.org/pdf/2405.10138v2)

**Smallest experiment that could say something real.** Freeze three manageable BEIR datasets before inspecting outcomes (for example, SciFact, NFCorpus, and a third selected only by a stated corpus-size cap), three fixed retrievers (BM25 plus two public embedding models below 600M), and NDCG@10. At each pre-specified deduplication threshold, remove duplicate clusters while preserving all qrel handling rules; run a random-removal control matched on removed-document count, document length bin, and whether a document has a relevance judgment. Report the full threshold curve, model ranks, and paired-query 20,000-resample intervals for score differences and rank stability.

**Week-one kill condition.** Run only the data plumbing and one retriever on the frozen datasets. Kill the project if deduplication removes fewer than 1% of eligible documents at every pre-specified threshold, qrels cannot be preserved without changing the evaluation population, or planted duplicate/near-duplicate documents are not detected while planted nonduplicates are removed. A second, substantive kill is pre-registered: if all three models retain the identical ordering across every threshold and every matched control in the first two datasets, stop rather than expand a null ranking result into a larger benchmark paper.

**Comparison contract.** What varies is the duplicate definition/threshold. What is held fixed is the corpus snapshot, query set, qrels, retriever weights, evaluation metric, and removal count/support through the control. The threshold is a preprocessing convention, so no single threshold gets a headline; the curve is the result.

**Cost.** Public BEIR data and small inference-only models; no training. Expected M1 CPU runtime is hours to roughly a day per complete grid, plus 20–30 person-hours of implementation and validation. No paid data is required.

**Lessons exercised.** Tests the “unseen/seen” and preprocessing-confound failures directly, uses support-matched random controls, validates the detector with planted bad rows, and prevents under-resolved significance by fixing the resampling budget before any comparison.

**Domain-knowledge cost.** **Low to medium.** Requires standard IR concepts (qrels, NDCG, sparse/dense retrieval, duplicate clustering), all close to the proposed domain. It avoids the biological vocabulary burden.

## 3. Observable provenance and near-duplicate structure in one public agent benchmark

**Question.** For one frozen public agent benchmark with task source metadata, how often do evaluation tasks form same-source, same-repository, or semantic near-duplicate clusters—and how does that rate change over a declared similarity threshold? The claim is about **observable benchmark provenance and within-benchmark task structure**, not whether any proprietary model trained on a task.

**Why it is not already answered.** Broad LLM benchmark contamination and agent-evaluation contamination are already active literatures, so a generic “are agent benchmarks contaminated?” project is dead (see candidate 5). This bounded audit is not the same question: it makes no unverifiable training-data assertion and asks whether the benchmark’s own label of task novelty aligns with publicly observable provenance and semantic structure. The search found contamination surveys and a 2026 agent-focused paper, but no exact public, threshold-curve audit for a single frozen benchmark with a matched random-pair baseline. [NLP Evaluation in Trouble](https://arxiv.org/abs/2310.18018), [Reliability, Contamination, and Evolution in LLM Agents](https://www.techrxiv.org/doi/10.36227/techrxiv.177222530.04005985), [Test of Time](https://arxiv.org/html/2509.00072v3)

**Smallest experiment that could say something real.** Choose exactly one benchmark only after confirming that its task records expose source URLs/repository revisions and its license permits analysis. For a 200-task stratified sample, construct pairwise candidate clusters using source identifiers, normalized task text, and a small local embedding model; manually adjudicate only pairs above a pre-registered retrieval cutoff plus an equal-size sampled below-cutoff control. Estimate each observable overlap rate with Wilson intervals and plot it across semantic-similarity thresholds. The random-pair control gives the background rate that a source or topic distribution alone would produce.

**Week-one kill condition.** Kill before manual annotation if fewer than 80% of sampled tasks have resolvable public source/revision metadata; if the task license blocks the planned analysis; or if a seeded set of exact, paraphrased, and unrelated task pairs fails the detector calibration. Also kill if the candidate-pair generator produces an unreviewable volume at the frozen cutoff, because arbitrary narrowing after looking would reintroduce the exact convention dependence this study is meant to expose.

**Comparison contract.** What varies is the semantic-clustering threshold. What is fixed is the benchmark version, task sample, metadata-resolution rule, embedding model, and manual annotation rubric. Same-source provenance, repository overlap, text similarity, and suspected model exposure are separate axes; no aggregate “contaminated” label is permitted. The project reports a curve over threshold, never a single duplicate rate.

**Cost.** Public benchmark metadata, Git history already referenced by tasks, and a small local embedding model; no agent runs or model training. About 25–45 person-hours, dominated by audit design and adjudication. CPU cost is small.

**Lessons exercised.** Directly attacks scope-confounded axes and “unseen” labels, includes planted controls, separates observable facts from external unobservable training exposure, and treats clustering as a preprocessing curve.

**Domain-knowledge cost.** **Medium.** Familiarity with agent benchmark schemas, repository revisions, and semantic-similarity calibration is needed, but it stays inside retrieval/agent evaluation rather than importing an unfamiliar scientific field.

## Dead at literature screen

### 4. ViroBench-style claim: nucleotide foundation models degrade under phylogenetic and temporal shift

**Status: dead.** This is already answered by the named 2026 work. ViroBench reports a large viral-genomics benchmark, genus-disjoint and temporal splits, 66 nucleotide foundation models, and the same headline conclusion: degradation under phylogenetic and temporal shifts. Re-running a smaller subset under the hardware limit would at best be replication or a benchmark audit, not a new project answering the seed question. [ViroBench](https://arxiv.org/abs/2605.25388)

**Why not revive it by shrinking.** The paper’s scope is 18 scenarios and 66 models; a sub-600M, M1-only reproduction would be underpowered for a competing headline and would risk repeating Cognate’s false-significance failure. The available Lite setting could make a debugging exercise feasible, but not the proposed result.

**Domain-knowledge cost.** **High.** Viral taxonomy, phylogenetic splitting, sequence-model tokenization, and biological task validity would recreate the terminology tax. Its kill condition has already fired at step zero: the core question is published.

### 5. Generic “LLM-agent benchmark contamination” survey

**Status: dead.** As posed, it is too broad and already occupied. The contamination literature includes systematic benchmark analyses and a 2026 work explicitly framed around reliability, contamination, and evolution in LLM agents. A broad audit would either restate known concerns or drift into claims about hidden training corpora that cannot be tested internally. [Data-contamination literature index](https://github.com/lyy1994/awesome-data-contamination), [Reliability, Contamination, and Evolution in LLM Agents](https://www.techrxiv.org/doi/10.36227/techrxiv.177222530.04005985), [Reliability Gap in Benchmark Auditing](https://arxiv.org/html/2606.03305v2)

**What survives from it.** Only the bounded, observable version in candidate 3: provenance and near-duplicate structure in one versioned benchmark, with no assertion of model-training membership.

**Domain-knowledge cost.** **Medium.** The domain is accessible, but its hidden-corpus premise makes the original form untestable. The cheap kill condition is therefore conceptual, not computational: it lacks an observable target and a defensible comparison contract.

## Literature checks run for this screen

- TCR UMI reporting/reproducibility, UMI methods, and TCR diversity-method evaluation.
- ViroBench and nucleotide foundation models under phylogenetic/temporal shift.
- Retrieval near-duplicate bias, BEIR, split-level deduplication, and ranking effects.
- LLM and agent benchmark contamination, temporal contamination, and semantic near-duplicate splitting.

This was a bounded literature screen, not an exhaustive systematic review. Its role was to kill direct rediscoveries before any build. The sources are cited inline where they support each novelty decision.
