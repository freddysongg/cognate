# Phase 3 artifact and document scout

## Routing index

- **Exact TCR source lookups:** `data/knn_esm_cosine.json` and `data/b3_results.json`
- **Pinned pMHC provenance and counts:** `data/pmhc/source_contract.json`
- **Result artifact contract:** required top-level keys and the boundary between specified and unspecified nesting
- **Document conventions:** experiment record, contrast append, and evidence-bounded postmortem
- **Placement and ignore rules:** durable aggregate outputs versus ignored raw, row-level, and embedding files
- **Implementation hazards:** source-key spelling, observed-random uplift, slice denominators, and non-causal wording

## Scope

Read-only inspection was performed on the requested artifacts, the Phase 3 specification, the shaped pMHC brief, representative existing experiment/closeout documents, the fetch script, `.gitignore`, and git tracking/ignore state. `docs/pmhc/src/cognate/data.py` was neither read nor touched.

Evidence labels below follow the repository convention: artifact/file inspection and commands actually run are `observed`; proposed Phase 3 shapes not fixed by an existing contract are explicitly called `proposed` or `unspecified`.

## 1. Exact closed-TCR source lookups

Phase 3 names three exact reads. Do not substitute a numerically similar key or use the theoretical chance value.

| Quantity | Exact file and JSON path | Point | 95% interval | Rows / groups | Status |
|---|---|---:|---:|---:|---|
| TCR edit retrieval | `data/knn_esm_cosine.json` → `scores["edit/seen"]["macro_auc01"]` | 0.5654 | [0.5461, 0.5849] | 73,440 / 48 peptides | non-degenerate |
| TCR ESM-2 retrieval | `data/knn_esm_cosine.json` → `scores["35M layer10 (headline)/seen"]["macro_auc01"]` | 0.5358 | [0.5230, 0.5498] | 73,440 / 48 peptides | non-degenerate |
| TCR observed random | `data/b3_results.json` → `scores["random/seen"]["macro_auc01"]` | 0.5009 | [0.4980, 0.5048] | 73,440 / 48 peptides | non-degenerate |

`observed`: the two retrieval paths are at `data/knn_esm_cosine.json:33-49` and `data/knn_esm_cosine.json:113-129`; the random path is at `data/b3_results.json:38-54`.

The closed-TCR point uplifts over its **observed** random arm are therefore:

| TCR arm | Required calculation | Point uplift |
|---|---|---:|
| edit retrieval | `0.5654 - 0.5009` | **+0.0645** |
| ESM-2 retrieval | `0.5358 - 0.5009` | **+0.0349** |

These are within-task point differences used for the side-by-side contrast. They are not a paired cross-task estimand and must not be given a fabricated cross-task interval. The existing edit-versus-ESM paired result in `data/knn_esm_cosine.json` is a different quantity: `comparisons["35M layer10 (headline)_minus_edit/seen"]` = -0.0296 [-0.0415, -0.0188].

### Source-key hazards

- `"35M layer10 (headline)/seen"` contains spaces, parentheses, a slash, and lowercase `layer`; it must be copied exactly.
- The ESM key belongs to `data/knn_esm_cosine.json`, not `data/b3_results.json`.
- The random key belongs to `data/b3_results.json`, not `data/knn_esm_cosine.json` (which has no random arm).
- `data/b3_results.json` also contains `scores["k-NN/seen"]` = 0.5654, but Phase 3 explicitly requires the edit value from `data/knn_esm_cosine.json`.
- Use `.macro_auc01.point` for the contrast. `.auroc.point` and `.auprc.point` are different metrics.
- Use 0.5009 as the TCR floor. Subtracting 0.5 would change the reported edit and ESM uplifts to +0.0654 and +0.0358, violating the contract.

## 2. Input artifact schemas

### `data/knn_esm_cosine.json`

Top-level keys, in file order:

```text
config
scores
degeneracy
comparisons
```

Relevant shapes:

```text
config:
  seed: number
  configs: array of
    model: string
    layer: number
    label: string

scores[arm/slice]:
  macro_auc01: {point: number, lo: number, hi: number}
  auroc:       {point: number, lo: number, hi: number}
  auprc:       {point: number, lo: number, hi: number}
  n_rows: number
  n_peptides: number
  is_degenerate: boolean

degeneracy[arm/slice]:
  is_degenerate: boolean
  distinct_scores: number
  constant_groups: number

comparisons[arm_minus_edit/slice]:
  point: number
  lo: number
  hi: number
  p: number
  n_peptides: number
```

The `scores` key set is the Cartesian product of these labels and slices:

```text
edit
35M last
35M layer10 (headline)
35M middle
8M last
8M middle

seen
unseen
```

### `data/b3_results.json`

Top-level keys, in file order:

```text
config
composition
scores
degeneracy
comparisons
correlations
correlation_difference
invariance
```

Relevant shapes:

```text
config:
  model: string
  layer: number
  strategy: string
  ratio: number
  seed: number
  well_supported_min_positives: number

composition:
  n_peptides: number
  n_seen_peptides: number
  n_unseen_peptides: number
  n_rows: number

scores[arm/slice]:
  macro_auc01: {point: number, lo: number, hi: number}
  auroc:       {point: number, lo: number, hi: number}
  auprc:       {point: number, lo: number, hi: number}
  n_rows: number
  n_positive: number
  n_peptides: number
  is_degenerate: boolean
```

`scores` has arms `random`, `k-NN`, `logistic`, and `MLP`, each with slices `all`, `seen`, and `unseen`. The remaining blocks follow the same interval convention (`point`, `lo`, `hi`) but are not Phase 3 TCR inputs.

### `data/pmhc/source_contract.json`

Exact schema and values:

```text
source_url: string
  https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz
retrieval_date: string
  2026-09-10
archive_sha256: string
  06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4
file_sha256: object
  c000_ba: a5704007e127c8c0e2a3b0043c52ad2f92be5efd01bd2b6a0f7e5424d25d06ef
  c001_ba: 754c5fecf1c8387b48fc8cf77473fc14c9e89adb2694ac977e51e7028b0304f1
  c002_ba: f1ff916e3bd4ed01352b7fa4c6a1852fd9294104f6b344c5dd2eba7961a57aef
  c003_ba: 9cde2dd51abf1cde03383c5f8120e88243c7b6f486f341a7566bb4ad6f60eb69
  c004_ba: a2a28d2565fb8a3e6c76e1f7d2be5b1a3aca28acbabe13c3db4e2063c404f2e5
  MHC_pseudo.dat: f46d95dee821db6c139d6cee6f6bf72328468c1d8f6e9741daf21562752ad39a
source_counts: object
  ba_rows_all_species: 208093
  hla_abc_rows: 170107
  hla_abc_positive: 42001
  hla_abc_artificial_negative: 10895
  nine_mer_hla_rows: 126375
  nine_mer_hla_positive: 30869
  nine_mer_hla_negative: 95506
headline_counts: object
  alleles: 47
  rows: 112128
  positive: 28538
  negative: 83590
  measured_nonbinder: 82448
  artificial_negative: 1142
```

Observed arithmetic invariants worth asserting or printing before scoring:

```text
30869 + 95506 = 126375
28538 + 83590 = 112128
82448 + 1142 = 83590
```

The fixed negative-source evaluation slices implied by the brief/spec are:

```text
positive + measured non-binder: 28538 + 82448 = 110986 rows
positive + artificial negative: 28538 + 1142 = 29680 rows
```

These slice totals are `observed` arithmetic from the pinned headline counts, not evidence that the future runner selected the rows correctly. The run must verify them against loaded rows.

The fetch script duplicates the URL and all six hashes and places the verified archive at `data/pmhc/raw/NetMHCpan_train.tar.gz` and extracted files at `data/pmhc/raw/NetMHCpan_train/`. That duplication means the runner should treat `source_contract.json` as the durable reporting source and should verify rather than silently restate values.

## 3. Required Phase 3 result artifact

`data/pmhc/results.json` is a durable aggregate artifact. The Phase 3 spec fixes **only** these top-level keys:

```text
contract
config
scores
per_allele
comparisons
negative_sources
retrieval_diagnostics
criteria
```

No additional top-level keys are allowed. `observed` from the Phase 3 specification.

### What is fixed below the top level

- `scores` must emit seven rows/arms: random, length/composition, edit retrieval, ESM-2 retrieval, PWM, pseudo-sequence MLP, and one-hot MLP control.
- Every arm must report headline macro standardized AUC0.1 plus pooled AUROC, macro AUROC point diagnostic, and pooled AUPRC. The existing `evaluate` artifact convention for bootstrapped metrics is `{point, lo, hi}`.
- `per_allele` contains aggregate per-allele points inside the JSON; no separate Phase 3 per-allele CSV is requested.
- `comparisons` contains every non-random arm minus random, plus pseudo-sequence MLP minus PWM, using paired macro-AUC0.1 comparisons.
- `negative_sources` contains the two fixed slices separately: positives plus measured non-binders, and positives plus artificial negatives.
- `retrieval_diagnostics` reports nearest-positive edit distance and exact-duplicate counts by negative source.
- `criteria` records predictive verdicts and the PWM-sufficiency verdict.
- `contract` and `config` must retain enough provenance/settings to write the experiment document: source/hash, fold/cohort counts, fixed arm settings, 20,000 draws, and seed 0.

### What is not yet specified

The plan does **not** fix exact nested key spellings for the seven score arms, the per-allele record layout, negative-source slice names, retrieval-diagnostic fields, or criteria fields. Those must be chosen once and pinned in Phase 3 tests before the real run. Do not present a guessed nested schema as an inherited contract.

A minimal naming set consistent with current prose is `random`, `length_composition`, `edit_retrieval`, `esm_retrieval`, `pwm`, `mlp_pseudo_sequence`, and `mlp_one_hot`; this is `proposed`, not observed. Existing artifacts mix display-oriented names (`k-NN`, `MLP`, `35M layer10 (headline)`) with code-oriented names, so accidental reliance on legacy spelling would be brittle.

### Criteria semantics

- An arm is predictive only when its headline macro-AUC0.1 95% interval has `lo > 0.5`.
- PWM is sufficient relative to the pseudo-sequence MLP only when both conditions hold: PWM is predictive, and the paired `mlp_pseudo_sequence - pwm` interval has `hi < 0.02`.
- Comparisons against random use paired observed scores, not comparison against the scalar 0.5.
- Evaluation uses `n_boot = 20000` and `seed = 0`, grouped by allele.

## 4. Existing `docs/contrast.md` contract

The current file has one heading, a nine-row pre-result table, and one interpretation-boundary paragraph. Phase 3 must append observed results without changing any of those lines.

The fixed rows are:

```text
Endpoint
Query
Retrieval group
Retrieval operator
Edit representation
ESM representation
Partition
Headline metric
Cross-task quantity
```

The last row already defines the result calculation: within-task uplift over observed random, shown side by side and never paired across tasks. The closing sentence already fixes the interpretation: descriptive and dataset-specific; the headline values are not one matched estimand; their difference does not identify a biological cause.

The appended results should therefore contain, at minimum, one row each for edit and ESM-2 with:

```text
task
arm
observed-random point
arm macro-AUC0.1 point
point uplift over observed random
```

TCR values come from Section 1 above; pMHC values must be read from the observed `data/pmhc/results.json`. Do not add a cross-task delta CI or p-value.

## 5. Experiment-document conventions

Representative experiment records inspected: `docs/eval_set_construction.md`, `docs/cdhit_and_issues.md`, `docs/session_log.md`, and `docs/findings.md`.

Conventions Phase 3 should preserve:

1. **Lead with the artifact and reproducibility boundary.** Existing records put `**Output.**` and `**Reproduce.**` near the top, name exact paths/commands, give fixed counts, and state whether reruns were observed.
2. **State provenance before results.** Source/release, license boundary, hashes, filters, and cohort construction precede score interpretation.
3. **Use compact tables for construction and results.** Counts appear as auditable stage totals; scores appear as `point [lo, hi]`, with group count and comparison direction made explicit.
4. **Print the bootstrap resolution.** Phase 3 requires 20,000 draws beside intervals. Existing lessons explicitly reject under-resolved intervals used for claims.
5. **Separate measurement from arithmetic/degeneracy.** Existing documents say when 0.500 is structural rather than measured. Phase 3 should likewise call out constant-score or invalid conditions, not narrate them as null evidence.
6. **Attach evidence labels to factual claims.** A completed run and inspected JSON supports `observed`; a code reading alone remains `claimed`. Do not call results `failure-proven` unless the relevant control was first observed red and then green.
7. **Put scope beside the claim.** Existing closeout prose follows major findings with a `Scope:` or `Limit:` sentence rather than collecting every caveat at the end.
8. **Distinguish endpoint and construction.** pMHC affinity is not antigen presentation or immunogenicity. Negative-source sensitivity is a dataset-construction result, not a biological mechanism.
9. **Name what the result does not license.** Avoid publication, novelty, causal, unseen-allele-transfer, or biological-cause claims.
10. **End with exact regeneration and verification.** The experiment record should include `bash scripts/fetch_pmhc_data.sh` followed by `uv run python scripts/run_pmhc.py`, plus the fixed validation commands if space permits.

A concise `docs/pmhc/experiment.md` structure consistent with those conventions is:

```text
# pMHC binding replication experiment
status/run date and one-sentence outcome
**Output.** artifact path and headline cohort
**Reproduce.** exact commands

## Source and contract
URL, retrieval date, archive/file hashes, verified fold/cohort counts

## Cohort and folds
filters, positive threshold, five supplied folds, zero peptide overlap, negative-source counts

## Arms and settings
seven score rows, exact retrieval/PWM/MLP settings

## Results
seven-row score table with 20,000-draw intervals and diagnostics

## Paired comparisons and criteria
non-random-minus-random rows, MLP-minus-PWM, predictive/sufficiency verdicts

## Negative-source and retrieval diagnostics
two fixed slices, nearest-positive distance, exact duplicates

## Limits
descriptive/non-causal, affinity-only, supplied-fold/seen-allele boundary

## Reproduce and verify
commands and artifact placement
```

## 6. Postmortem conventions

Representative closeout documents inspected: `docs/final_state.md`, `docs/lessons.md`, and the verdict/retirement sections of `docs/eval_set_construction.md` and `docs/cdhit_and_issues.md`.

The useful convention is an evidence-bounded decision record, not a second experiment narration:

- Open with a status and stop decision (`CLOSED` only if the Phase 3 stop condition is actually met).
- Answer each predeclared question directly with a verdict and the exact observed numbers that support it.
- Use strong verdict words only when the contract supports them: predictive only under `lo > 0.5`; sufficient only under both predeclared PWM conditions; `NULL` only with an interval that excludes a meaningful effect, not merely because a point is near zero.
- Distinguish **held**, **weakened**, **not established**, and **withdrawn**. Existing postmortem prose does not turn a spanning-zero interval into equivalence.
- Put construction sensitivity in its own finding: whether retrieval behavior survives excluding artificial negatives and how nearest-positive/exact-duplicate diagnostics move.
- State what stopped the project and why. A weak MLP result triggers contract diagnosis, not architecture expansion.
- Recommend follow-up only if the observed record leaves a question that merits a separate project. The already-named leave-one-allele-out transfer study is out of scope here and should not be smuggled into the current run.

The Phase 3 spec requires the postmortem to answer exactly five questions:

1. Which arms were predictive?
2. Was PWM sufficient relative to the pseudo-sequence MLP?
3. Did the retrieval contrast survive removal of artificial negatives?
4. What was construction-sensitive?
5. Does any follow-up deserve a new project?

A short section per question, followed by one stop-decision paragraph, matches the repository's strongest closeout writing and avoids duplicating `experiment.md`.

## 7. Output placement and ignore state

Observed `.gitignore` rules:

| Path class | Rule | Intended status |
|---|---|---|
| pMHC raw archive/extraction | `data/pmhc/raw/` | ignored, local, regenerable |
| pMHC row-level derived output | `data/pmhc/derived/` | ignored, local, regenerable |
| pMHC ESM cache | `/shared/` (therefore `shared/pmhc/esm2_35M_peptides.npz`) | ignored, local, large/regenerable |
| aggregate source contract | `data/pmhc/source_contract.json` | not ignored, durable |
| aggregate results | `data/pmhc/results.json` | not ignored, durable |
| experiment record | `docs/pmhc/experiment.md` | not ignored, durable |
| contrast | `docs/contrast.md` | not ignored, durable |
| postmortem | `docs/pmhc/postmortem.md` | not ignored, durable |

`git check-ignore -v` matched only the three local/regenerable classes above. It did not match the aggregate result or documents. `git ls-files data/pmhc docs/pmhc docs/contrast.md` returned no paths because the whole Phase 1/2 work is currently untracked; that is current worktree state, not a reason to ignore the durable outputs.

At inspection time:

```text
data/pmhc/source_contract.json exists
docs/pmhc/brief.md exists
docs/contrast.md exists
shared/pmhc does not yet exist
data/pmhc/results.json does not yet exist
docs/pmhc/experiment.md does not yet exist
docs/pmhc/postmortem.md does not yet exist
```

The final tracking guard required by the Phase 3 spec is:

```bash
git ls-files data/pmhc/raw data/pmhc/derived shared/pmhc
```

Its expected output is empty. This guard does not check `data/pmhc/results.json`; that file is intentionally durable.

## 8. Phase 3 artifact checklist

- Read all TCR numbers through the exact paths in Section 1.
- Compute TCR and pMHC uplift against each task's observed random point.
- Keep the fixed pre-result contrast table and interpretation boundary byte-for-byte; append results only.
- Print and record five folds, 47 alleles, 112,128 rows, 28,538 positives, 83,590 negatives, and zero cross-fold peptide overlap before scoring.
- Verify the two headline negative-source counts (82,448 measured non-binders and 1,142 artificial negatives) and their slice totals against loaded rows.
- Emit exactly eight top-level result keys and all seven score rows.
- Keep row-level predictions out of the JSON and out of git; `per_allele` is aggregate points only.
- Report 20,000-draw intervals and paired comparison direction explicitly.
- Derive criteria verdicts mechanically from saved intervals.
- Write the experiment record from the saved JSON, not from transient in-memory values.
- Make the postmortem answer only the five predeclared questions and the stop decision.
- Preserve descriptive/non-causal language throughout.
