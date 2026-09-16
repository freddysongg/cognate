# Phase 1: Freeze the data and comparison contract

## Target files

- `.gitignore`
- `docs/pmhc/brief.md`
- `docs/contrast.md`
- `data/pmhc/source_contract.json`
- `scripts/fetch_pmhc_data.sh`
- `src/cognate/pmhc.py`
- `tests/test_pmhc.py`

## Off-limits

- `src/cognate/metrics.py`
- `src/cognate/baseline_knn.py`
- `src/cognate/embed.py`
- `data/vdjdb_eval.csv`
- closed TCR documents
- `docs/next_candidates.md`
- `docs/pmhc/src/cognate/data.py`

## Boundaries

- Do not edit the frozen TCR contract: `src/cognate/metrics.py`, `src/cognate/baseline_knn.py`, `src/cognate/embed.py`, `data/vdjdb_eval.csv`, or closed TCR documents.
- Reuse `baseline_knn.py`, `embed.py`, `metrics.py`, and `train_head.py`; do not wrap them in new abstraction layers.
- Do not run or redistribute the NetMHCpan executable. Fetch only the public training archive.
- Score the fixed 47-allele, 112,128-row cohort before looking at results.
- Report six model families and seven rows: random, length/composition, edit retrieval, ESM-2 retrieval, PWM, pseudo-sequence MLP, and one-hot MLP control.
- The two MLP encodings each use hidden sizes 55 and 66, seeds 0–4, and five folds: 50 fits per encoding, 100 total.
- Keep raw rows, derived row-level predictions, and embeddings out of git.
- Keep only four durable documents: `docs/pmhc/brief.md`, `docs/pmhc/experiment.md`, `docs/contrast.md`, and `docs/pmhc/postmortem.md`.
- Preserve existing user changes in `docs/next_candidates.md` and `docs/pmhc/src/cognate/data.py`; do not import, move, or edit the latter.
- Each phase ends with a fresh `reviewer-py` pass. Reviews must report `<N> candidates, <K> refuted, <S> survived (<b> blocker, <m> major, <n> minor)` and resolve all blockers and majors before continuing.

## Minimal File Set

| Path | Action |
|---|---|
| `.gitignore` | ignore `data/pmhc/raw/` and `data/pmhc/derived/` |
| `docs/pmhc/brief.md` | replace with the reviewed brief |
| `docs/contrast.md` | create the shared-contract table before scoring; append results later |
| `data/pmhc/source_contract.json` | pin source URL, hashes, and counts |
| `scripts/fetch_pmhc_data.sh` | non-overwriting fetch and extraction |
| `src/cognate/pmhc.py` | all pMHC-only data, PWM, BLOSUM, MLP, and fold-scoring logic |
| `tests/test_pmhc.py` | contract corruptions and focused numerical tests |
| `scripts/run_pmhc.py` | one fixed experiment runner |
| `data/pmhc/results.json` | aggregate results and per-allele points; no row-level predictions |
| `docs/pmhc/experiment.md` | method, provenance, results, and criteria |
| `docs/pmhc/postmortem.md` | final evidence-bounded closeout |

No separate baseline, PWM, MLP, evaluation, renderer, or per-allele CSV modules are needed. They would each have one caller.

## Instructions

- [ ] Promote the reviewed brief, then verify it exactly:

```bash
cmp .claude/delib/pmhc-next-phase/brief.md docs/pmhc/brief.md
```

- [ ] Create `docs/contrast.md` before model code. Its shared-contract table must state:

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

End the file with: "This comparison is descriptive and dataset-specific. The two headline values are not one matched estimand, and a difference does not identify a biological cause." Do not add an empty results section.

- [ ] Add these ignore rules:

```gitignore
# pMHC source rows and regenerable row-level outputs
data/pmhc/raw/
data/pmhc/derived/
```

`/shared/` already ignores the pMHC embedding cache.

- [ ] Create `data/pmhc/source_contract.json` with the inspected source:

```json
{
  "source_url": "https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz",
  "retrieval_date": "2026-09-10",
  "archive_sha256": "06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4",
  "file_sha256": {
    "c000_ba": "a5704007e127c8c0e2a3b0043c52ad2f92be5efd01bd2b6a0f7e5424d25d06ef",
    "c001_ba": "754c5fecf1c8387b48fc8cf77473fc14c9e89adb2694ac977e51e7028b0304f1",
    "c002_ba": "f1ff916e3bd4ed01352b7fa4c6a1852fd9294104f6b344c5dd2eba7961a57aef",
    "c003_ba": "9cde2dd51abf1cde03383c5f8120e88243c7b6f486f341a7566bb4ad6f60eb69",
    "c004_ba": "a2a28d2565fb8a3e6c76e1f7d2be5b1a3aca28acbabe13c3db4e2063c404f2e5",
    "MHC_pseudo.dat": "f46d95dee821db6c139d6cee6f6bf72328468c1d8f6e9741daf21562752ad39a"
  },
  "source_counts": {
    "ba_rows_all_species": 208093,
    "hla_abc_rows": 170107,
    "hla_abc_positive": 42001,
    "hla_abc_artificial_negative": 10895,
    "nine_mer_hla_rows": 126375,
    "nine_mer_hla_positive": 30869,
    "nine_mer_hla_negative": 95506
  },
  "headline_counts": {
    "alleles": 47,
    "rows": 112128,
    "positive": 28538,
    "negative": 83590,
    "measured_nonbinder": 82448,
    "artificial_negative": 1142
  }
}
```

- [ ] Make `scripts/fetch_pmhc_data.sh` refuse destructive behavior:

  - Reuse an existing archive only after its SHA-256 passes.
  - Download to a new `.part` path, verify it, then rename it only when the final archive is absent.
  - Reuse an existing extraction only after all six file hashes pass.
  - Never delete, truncate, or overwrite an existing path.

- [ ] Add the data path to `src/cognate/pmhc.py`:

```python
@dataclass(frozen=True)
class PmhcDataset:
    all_nine_mer_hla_rows: pd.DataFrame
    rows: pd.DataFrame
    eligible_alleles: tuple[str, ...]

def load_source_contract(path: Path) -> dict[str, object]: ...
def verify_source(source_dir: Path, archive_path: Path, contract: Mapping[str, object]) -> None: ...
def load_pmhc_dataset(source_dir: Path, *, minimum_class_rows: int = 100) -> PmhcDataset: ...
def iter_pmhc_folds(dataset: PmhcDataset) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]: ...
def load_pseudo_sequences(path: Path, alleles: Sequence[str]) -> dict[str, str]: ...
```

The loader must:

- require exactly `c000_ba` through `c004_ba`;
- parse three whitespace-separated columns as `Peptide`, `Affinity`, and `Allele`;
- require targets in `[0,1]` and the standard 20-amino-acid alphabet;
- retain only HLA-A/B/C nine-mers;
- set binary `Target = Affinity > 0.426`;
- label positives, measured non-binders, and exact-`0.01` artificial negatives;
- choose alleles with at least 100 positives and 100 negatives across all folds;
- require both classes for every retained allele in each test fold and its training complement;
- reject exact peptide overlap across folds;
- normalize `HLA-A02:01` to pseudo key `HLA-A0201` and require 34 residues.

- [ ] In `tests/test_pmhc.py`, write one parametrized `test_contract_corruption_is_rejected` with named cases for archive hash, extracted-file hash, fold set, malformed row, invalid affinity, invalid residue, fold overlap, minimum class support, missing class in a test fold, missing class in its training complement, source counts, headline counts, missing pseudo-sequence, and short pseudo-sequence. Run each case against a callable non-rejecting implementation first, observe its own red failure, then add the guard and rerun green.

- [ ] Run and review:

```bash
uv run pytest tests/test_pmhc.py -q
```

The corrupt-fixture cases observed red then green are `failure-proven`; the valid parse is `observed`. Run the Phase 1 `reviewer-py` gate and resolve blockers/majors.

## Validation

```bash
uv run pytest tests/test_pmhc.py -q
```
