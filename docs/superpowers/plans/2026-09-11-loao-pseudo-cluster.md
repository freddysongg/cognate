# LOAO Pseudo-Sequence-Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute #34 as the frozen pseudo-sequence-cluster-held-out pMHC transfer evaluation.

**Architecture:** Branch from the committed LOAO foundation. This worktree adds deterministic cluster construction and group-holdout scheduling, then uses shared transfer scoring and evaluation without changing the common arms or criteria.

**Tech Stack:** Python 3.11, NumPy, pandas, pytest.

**Spec:** `.claude/delib/loao-mhc-transfer/brief.md`

## Global Constraints

- Start only from the committed LOAO foundation after PR #35 reaches `main`.
- Cluster only the frozen 34-residue pseudo-sequences; never use targets or arm outcomes.
- Use complete-linkage normalized Hamming clustering and the no-more-than-one-singleton cut rule, which selects 12/34 for this cohort.
- Do not claim that distance alone causes any difference.
- Do not stage, commit, push, or create a PR without fresh explicit user authorization.

---

### Task 1: Frozen cluster construction and group schedule

**Files:**
- Create: `tests/test_pmhc_loao_cluster.py`
- Modify: `src/cognate/pmhc_transfer.py`

**Interfaces:**
- Consumes: the 47 exact pseudo-sequences and `build_transfer_partition`.
- Produces: `build_pseudo_sequence_clusters` and `build_cluster_schedule`.

- [ ] **Step 1: Write failing deterministic-cluster tests.**

```python
def test_cluster_rule_is_label_independent_and_has_one_singleton() -> None:
    clusters = build_pseudo_sequence_clusters(pseudo_sequences)
    assert sum(len(cluster) == 1 for cluster in clusters) == 1
    assert sorted(map(len, clusters)) == [1, 2, 2, 3, 3, 3, 5, 6, 6, 8, 8]
```

- [ ] **Step 2: Run the test red.**

Run: `uv run pytest -q tests/test_pmhc_loao_cluster.py`

Expected: FAIL because cluster construction is absent.

- [ ] **Step 3: Implement exact grouping.**

```python
def build_pseudo_sequence_clusters(
    pseudo_sequences: Mapping[str, str],
) -> tuple[tuple[str, ...], ...]:
    ...
```

Iterate integer Hamming cuts from zero upward, use complete linkage and sorted-allele tie-breaking, and select the first cut with at most one singleton. Reject a pseudo-sequence set that is not exactly the frozen 47 mapping.

- [ ] **Step 4: Run focused tests green.**

Run: `uv run pytest -q tests/test_pmhc_loao_cluster.py`

Expected: PASS, including group-leakage corruption.

### Task 2: Cluster-held-out runner and report

**Files:**
- Create: `scripts/run_pmhc_loao_cluster.py`
- Create: `data/pmhc/loao_cluster_results.json`
- Create: `docs/pmhc/loao_cluster.md`
- Modify: `tests/test_pmhc_loao_cluster.py`

**Interfaces:**
- Consumes: the 11-group schedule and shared transfer scorers.
- Produces: aggregate group-held-out results plus group and per-allele diagnostics.

- [ ] **Step 1: Write a mocked runner contract test.**

```python
def test_cluster_runner_records_membership_and_per_allele_scores(tmp_path: Path) -> None:
    artifact = run_cluster_holdout(source_dir, tmp_path / "result.json", device="cpu")
    assert len(artifact["diagnostics"]["clusters"]) == 11
    assert len(artifact["per_allele"]) == 47
```

- [ ] **Step 2: Run it red.**

Run: `uv run pytest -q tests/test_pmhc_loao_cluster.py::test_cluster_runner_records_membership_and_per_allele_scores`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement the runner and interpretation boundary.**

Use shared all-or-fail preflight and all five arms. Record exact membership, cut rule, retained rows, class support, nearest retained distance, paired comparisons, and the result criterion. State that this is cluster-held-out transfer under a frozen composition, not a causal distance analysis.

- [ ] **Step 4: Verify and run the fixed experiment once.**

Run: `uv run pytest -q tests/test_pmhc_loao_cluster.py && uv run python scripts/run_pmhc_loao_cluster.py`

Expected: tests pass and aggregate-only output is written. Pause for explicit commit authorization.
