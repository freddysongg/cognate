# Edge Map — What is the smallest pMHC replication that reuses Cognate's evaluation stack to explain the contrast with TCR-peptide prediction, without drifting toward novelty?

## Scope

What is the smallest pMHC replication that reuses Cognate's evaluation stack to explain the contrast with TCR-peptide prediction, without drifting toward novelty?

## Components / nodes

- **pMHC class I presentation replication** — Draft replication scope that reuses the TCR baseline, embeddings, metrics, and interval machinery after a literature gate. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:31-32`, `modules[docs/pmhc/brief.md].purpose`)_
- **TCR–epitope evaluation** — Existing peptide-plus-CDR3b binary-classification evaluation whose unseen-peptide results are recorded as prior-only. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:2-9`, `vault_matches[matched_keyword=cognate]`)_
- **VDJdb 88-peptide evaluation set** — Purpose-built dataset used for Phase B evaluation. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:7`, `vault_matches[matched_keyword=cognate].edges[0]`)_
- **IMMREP23 20-peptide evaluation set** — Earlier evaluation set from which the evaluation moved away. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:7`, `vault_matches[matched_keyword=cognate].edges[0]`)_
- **Shared embedding cache** — Shared Cognate embedding caches selected through `COGNATE_CACHE_DIR` or the repository shared directory. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:4`; `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:86-101`)_
- **docs/final_state.md** — Closed-state record for the TCR project, its metric contract, frozen boundaries, results, and evidence. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:23-24`, `modules[docs/final_state.md]`)_
- **docs/lessons.md** — Transferable methodological lessons from the closed TCR project. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:25-26`, `modules[docs/lessons.md]`)_
- **docs/belief_list.md** — Locked TCR claims, overturn conditions, evidence, and B3 outcomes. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:27-28`, `modules[docs/belief_list.md]`)_
- **docs/redteam_curve.md** — Read-only audit of deduplication sensitivity, bootstrap inference, and claim boundaries. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:29-30`, `modules[docs/redteam_curve.md]`)_
- **docs/pmhc/brief.md** — Draft scope document for the pMHC class I presentation replication. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:31-32`, `modules[docs/pmhc/brief.md]`)_
- **src/cognate/baseline_knn.py** — Per-group nearest-positive retrieval with edit or ESM-2 cosine similarity and exact-match controls. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:33-34`, `modules[src/cognate/baseline_knn.py]`)_
- **src/cognate/embed.py** — ESM-2 mean-pooled and per-residue embedding generation, lookup, and cache handling. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:35-36`, `modules[src/cognate/embed.py]`)_
- **src/cognate/metrics.py** — AUROC, AUPRC, macro groupwise AUC0.1, diagnostics, bootstrap intervals, and paired comparisons. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:37-38`, `modules[src/cognate/metrics.py]`)_
- **tests/test_baseline_knn.py** — Tests retrieval, similarity choices, exact-match behavior, and macro-versus-pooled scoring. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:39-40`, `modules[tests/test_baseline_knn.py]`)_
- **tests/test_embed.py** — Tests embedding lookup, pooling, batching, serialization, and residue caches. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:41-42`, `modules[tests/test_embed.py]`)_
- **tests/test_metrics.py** — Tests point metrics, groupwise AUC0.1, diagnostics, reports, and bootstrap intervals. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:43-44`, `modules[tests/test_metrics.py]`)_
- **tests/test_comparison.py** — Tests paired macro AUC0.1 bootstrap comparisons. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:45-46`, `modules[tests/test_comparison.py]`)_
- **tests/test_frozen_contract_hashes.py** — Verifies hashes for frozen metric, split, feature, and evaluation-contract files. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:47-48`, `modules[tests/test_frozen_contract_hashes.py]`)_
- **docs/eval_set_construction.md §8** — Repository component identified by the vault index as the location of current evaluation results. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:5-8`, `vault_matches[matched_keyword=cognate].components/edges`)_
- **docs/findings.md** — Phase 1 reference carrying a partial-supersession banner. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:5-8`, `vault_matches[matched_keyword=cognate].components/edges`)_
- **collections.abc** — Standard-library dependency imported by the baseline, embedding, and metrics modules. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:199-201,223-225,244-246`, `import_edges`)_
- **dataclasses** — Standard-library dependency imported by the baseline, embedding, and metrics modules. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:202-204,226-228,247-249`, `import_edges`)_
- **numpy** — Third-party dependency imported by the baseline, embedding, and metrics modules. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:205-207,232-234,253-255`, `import_edges`)_
- **pandas** — Third-party dependency imported by the baseline and metrics modules. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:208-210,256-258`, `import_edges`)_
- **rapidfuzz** — Third-party dependency imported by the baseline module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:211-213`, `import_edges`)_
- **os** — Standard-library dependency imported by the embedding module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:217-219`, `import_edges`)_
- **time** — Standard-library dependency imported by the embedding module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:220-222`, `import_edges`)_
- **pathlib** — Standard-library dependency imported by the embedding module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:229-231`, `import_edges`)_
- **torch** — Third-party dependency imported by the embedding module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:235-237`, `import_edges`)_
- **transformers** — Third-party dependency imported by the embedding module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:238-240`, `import_edges`)_
- **warnings** — Standard-library dependency imported by the metrics module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:241-243`, `import_edges`)_
- **typing** — Standard-library dependency imported by the metrics module. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:250-252`, `import_edges`)_
- **scikit-learn** — Third-party dependency imported by the metrics module and used for AUROC, AUPRC, and standardized partial AUC. _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:118-129,259-261`, `key_symbols[auroc/auprc/auc01]`, `import_edges`)_

## Edges / data flow

| Source | Edge type | Target | Cite |
|---|---|---|---|
| pMHC class I presentation replication | calls | src/cognate/baseline_knn.py | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:31-34`, pMHC brief purpose says the replication reuses the TCR baseline |
| pMHC class I presentation replication | calls | src/cognate/embed.py | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:31-36`, pMHC brief purpose says the replication reuses embeddings |
| pMHC class I presentation replication | calls | src/cognate/metrics.py | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:31-38,130-193`, pMHC brief purpose says the replication reuses metrics and interval machinery; metrics symbols implement intervals and comparisons |
| TCR–epitope evaluation | emits-consumes | VDJdb 88-peptide evaluation set | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:7`, evaluation moved onto the purpose-built VDJdb set in Phase B |
| src/cognate/baseline_knn.py | imports | collections.abc | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:199-201`, `import_edges[0]` |
| src/cognate/baseline_knn.py | imports | dataclasses | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:202-204`, `import_edges[1]` |
| src/cognate/baseline_knn.py | imports | numpy | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:205-207`, `import_edges[2]` |
| src/cognate/baseline_knn.py | imports | pandas | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:208-210`, `import_edges[3]` |
| src/cognate/baseline_knn.py | imports | rapidfuzz | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:211-213`, `import_edges[4]` |
| src/cognate/baseline_knn.py | imports | src/cognate/embed.py | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:214-216`, `import_edges[5]` (`cognate.embed`) |
| src/cognate/embed.py | emits-consumes | Shared embedding cache | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:35-36,74-117`, embedding module purpose plus cache generation, lookup, save, and load symbols |
| src/cognate/embed.py | imports | os | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:217-219`, `import_edges[6]` |
| src/cognate/embed.py | imports | time | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:220-222`, `import_edges[7]` |
| src/cognate/embed.py | imports | collections.abc | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:223-225`, `import_edges[8]` |
| src/cognate/embed.py | imports | dataclasses | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:226-228`, `import_edges[9]` |
| src/cognate/embed.py | imports | pathlib | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:229-231`, `import_edges[10]` |
| src/cognate/embed.py | imports | numpy | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:232-234`, `import_edges[11]` |
| src/cognate/embed.py | imports | torch | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:235-237`, `import_edges[12]` |
| src/cognate/embed.py | imports | transformers | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:238-240`, `import_edges[13]` |
| src/cognate/metrics.py | imports | warnings | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:241-243`, `import_edges[14]` |
| src/cognate/metrics.py | imports | collections.abc | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:244-246`, `import_edges[15]` |
| src/cognate/metrics.py | imports | dataclasses | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:247-249`, `import_edges[16]` |
| src/cognate/metrics.py | imports | typing | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:250-252`, `import_edges[17]` |
| src/cognate/metrics.py | imports | numpy | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:253-255`, `import_edges[18]` |
| src/cognate/metrics.py | imports | pandas | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:256-258`, `import_edges[19]` |
| src/cognate/metrics.py | imports | scikit-learn | `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:259-261`, `import_edges[20]` |

## Status

- **pMHC class I presentation replication** — designed _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:31-32`, described as draft scope)_
- **TCR–epitope evaluation** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:23-24`, closed-state record; `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:7-9`, completed Phase B evaluation status)_
- **VDJdb 88-peptide evaluation set** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:7`, purpose-built and used in Phase B)_
- **IMMREP23 20-peptide evaluation set** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:7`, prior evaluation set)_
- **Shared embedding cache** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:4`, shared caches in `cognate-shared`)_
- **docs/final_state.md** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:23-24`, harvested repository module)_
- **docs/lessons.md** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:25-26`, harvested repository module)_
- **docs/belief_list.md** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:27-28`, harvested repository module)_
- **docs/redteam_curve.md** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:29-30`, harvested repository module)_
- **docs/pmhc/brief.md** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:31-32`, harvested repository module)_
- **src/cognate/baseline_knn.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:33-34,50-73`, harvested module and symbols)_
- **src/cognate/embed.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:35-36,74-117`, harvested module and symbols)_
- **src/cognate/metrics.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:37-38,118-197`, harvested module and symbols)_
- **tests/test_baseline_knn.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:39-40`, harvested repository module)_
- **tests/test_embed.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:41-42`, harvested repository module)_
- **tests/test_metrics.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:43-44`, harvested repository module)_
- **tests/test_comparison.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:45-46`, harvested repository module)_
- **tests/test_frozen_contract_hashes.py** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-repo-cognate.md:47-48`, harvested repository module)_
- **docs/eval_set_construction.md §8** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:5-8`, indexed repository component)_
- **docs/findings.md** — built _(src: `/Users/freddy/Documents/repos/cognate/.claude/delib/pmhc-next-phase/harvest-knowledge.md:5-8`, indexed repository component)_
- **collections.abc** — unknown _(src: no fact beyond import reference)_
- **dataclasses** — unknown _(src: no fact beyond import reference)_
- **numpy** — unknown _(src: no fact beyond import reference)_
- **pandas** — unknown _(src: no fact beyond import reference)_
- **rapidfuzz** — unknown _(src: no fact beyond import reference)_
- **os** — unknown _(src: no fact beyond import reference)_
- **time** — unknown _(src: no fact beyond import reference)_
- **pathlib** — unknown _(src: no fact beyond import reference)_
- **torch** — unknown _(src: no fact beyond import reference)_
- **transformers** — unknown _(src: no fact beyond import reference)_
- **warnings** — unknown _(src: no fact beyond import reference)_
- **typing** — unknown _(src: no fact beyond import reference)_
- **scikit-learn** — unknown _(src: no fact beyond import reference)_

## Open edges (unverified)

- pMHC class I presentation replication → literature source or gate artifact; the brief says execution follows a literature gate, but neither capture identifies the gate's source or output.
- pMHC class I presentation replication → pMHC dataset and target labels; neither capture names a pMHC dataset, and the knowledge harvest reports `pMHC` and `peptide-MHC` as unmatched keywords.
- pMHC class I presentation replication → binding or presentation endpoint; the knowledge harvest reports `binding versus presentation` as unmatched.
- pMHC class I presentation replication → NetMHCpan or another reference comparator; the knowledge harvest reports `NetMHCpan` as unmatched.
- pMHC class I presentation replication → split and grouping contract; the captures verify an existing frozen TCR split contract but do not verify a pMHC split or grouping connection.
- Test modules → source modules; the harvest records test purposes but no test-to-source import or call edges.
