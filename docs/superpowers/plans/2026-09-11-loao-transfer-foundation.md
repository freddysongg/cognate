# LOAO Transfer Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared, testable pMHC leave-one-allele-out foundation used by all three transfer tracks.

**Architecture:** Keep the completed seen-allele code untouched except for intentional reusable primitives. Add `pmhc_transfer.py` as the boundary for transfer partitions, preflight validation, pseudo-sequence mapping, transfer-compatible scorers, and aggregate schema checks. Track runners consume this boundary after branching from the foundation commit.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn, PyTorch, pytest.

**Spec:** `.claude/delib/loao-mhc-transfer/brief.md`

## Global Constraints

- Start only after PR #35 merges and branch from updated `main`.
- Retain the verified source, 47-allele cohort, target threshold, 20,000 draws, seed 0, and aggregate-only artifacts.
- Never run or redistribute NetMHCpan; raw rows, row predictions, and caches stay ignored.
- Use no `any`, mutate no function argument, and leave the seen-allele one-hot arm out of transfer scoring.
- Do not stage, commit, push, or create a PR without fresh explicit user authorization.

---

### Task 1: Transfer partition contract

**Files:**
- Create: `src/cognate/pmhc_transfer.py`
- Create: `tests/test_pmhc_transfer.py`

**Interfaces:**
- Consumes: `PmhcDataset.rows`, `load_pseudo_sequences`, and the frozen eligible-allele tuple.
- Produces: `TransferPartition`, `build_allele_partition`, and `build_joint_novelty_partition`.

- [ ] **Step 1: Write failing partition tests.**

```python
def test_joint_partition_removes_target_rows_and_test_peptides() -> None:
    partition = build_joint_novelty_partition(rows, "HLA-A02:01")
    assert "HLA-A02:01" not in set(partition.train["Allele"])
    assert set(partition.test["Peptide"]).isdisjoint(partition.train["Peptide"])
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `uv run pytest -q tests/test_pmhc_transfer.py::test_joint_partition_removes_target_rows_and_test_peptides`

Expected: FAIL because the transfer module does not exist.

- [ ] **Step 3: Implement immutable partition construction.**

```python
@dataclass(frozen=True)
class TransferPartition:
    label: str
    held_out_alleles: tuple[str, ...]
    train: pd.DataFrame
    test: pd.DataFrame

def build_transfer_partition(
    rows: pd.DataFrame,
    held_out_alleles: Sequence[str],
    *,
    exclude_test_peptides: bool,
) -> TransferPartition:
    ...
```

Require nonempty train/test frames, both target classes in test, no held-out allele in train, and zero peptide overlap when requested.

- [ ] **Step 4: Run focused tests.**

Run: `uv run pytest -q tests/test_pmhc_transfer.py`

Expected: PASS.

### Task 2: Preflight, pseudo-sequence mapping, and transfer validation split

**Files:**
- Modify: `src/cognate/pmhc_transfer.py`
- Modify: `tests/test_pmhc_transfer.py`

**Interfaces:**
- Consumes: `TransferPartition`, the 47 eligible alleles, and exact 34-residue pseudo-sequences.
- Produces: `split_transfer_fit_validation`, `build_shuffled_pseudo_mapping`, `select_nearest_pwm_source`, and `preflight_partitions`.

- [ ] **Step 1: Add failing tests for each rejection path.**

```python
def test_preflight_rejects_a_partition_without_a_pwm_source() -> None:
    with pytest.raises(ValueError, match="eligible PWM source"):
        preflight_partitions(partitions, pseudo_sequences)

def test_transfer_validation_split_is_target_stratified() -> None:
    fit_indices, validation_indices = split_transfer_fit_validation(train)
    assert set(train.iloc[fit_indices]["Target"]) == {False, True}
```

- [ ] **Step 2: Run the tests red.**

Run: `uv run pytest -q tests/test_pmhc_transfer.py -k 'preflight or validation or mapping or pwm'`

Expected: FAIL for missing symbols.

- [ ] **Step 3: Implement the fixed rules.**

```python
def select_nearest_pwm_source(
    train: pd.DataFrame,
    target_allele: str,
    pseudo_sequences: Mapping[str, str],
) -> str:
    ...
```

Use normalized 34-position Hamming distance after track exclusions, require one positive and one negative source row, and break ties by allele name. Build a cyclic derangement over sorted unique pseudo-sequences and reject a non-derangement. Split 90/10 by `Target` only with seed 0. Preflight every target before any arm is scored; reject the whole track instead of skipping a target.

- [ ] **Step 4: Run focused tests green.**

Run: `uv run pytest -q tests/test_pmhc_transfer.py`

Expected: PASS, including deliberate leakage and mapping corruption cases.

### Task 3: Fixed transfer-compatible scorers and result contract

**Files:**
- Modify: `src/cognate/pmhc_transfer.py`
- Modify: `tests/test_pmhc_transfer.py`

**Interfaces:**
- Consumes: a valid `TransferPartition`, pseudo-sequence mappings, and the current MLP/PWM primitives.
- Produces: scores named `random`, `peptide_only_mlp`, `shuffled_mapping_mlp`, `nearest_pwm`, and `pseudo_sequence_mlp`.

- [ ] **Step 1: Add score-identity tests.**

```python
def test_transfer_scores_share_test_row_order() -> None:
    scores = score_transfer_partition(partition, pseudo_sequences, device="cpu")
    assert set(scores) == {
        "random", "peptide_only_mlp", "shuffled_mapping_mlp",
        "nearest_pwm", "pseudo_sequence_mlp",
    }
    assert all(values.shape == (len(partition.test),) for values in scores.values())
```

- [ ] **Step 2: Run the test red.**

Run: `uv run pytest -q tests/test_pmhc_transfer.py::test_transfer_scores_share_test_row_order`

Expected: FAIL because transfer scoring is absent.

- [ ] **Step 3: Implement minimal fixed scorers.**

Reuse BLOSUM50 peptide encoding, the existing continuous-affinity MLP ensemble settings, and PWM pseudocount one. The peptide-only scorer receives no allele feature. The shuffled and correct MLPs have identical architecture and differ only in their pseudo-sequence mapping. Validate finite, exactly-once output arrays before evaluation.

- [ ] **Step 4: Run the foundation verification gate.**

Run: `uv run pytest -q tests/test_pmhc_transfer.py && uv run python -m compileall -q src scripts && git diff --check`

Expected: PASS. Pause for explicit commit authorization.
