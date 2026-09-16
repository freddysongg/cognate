# LOAO Allele-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute #32 as a 47-target, allele-only pMHC transfer evaluation.

**Architecture:** Branch from the committed LOAO foundation. This worktree supplies only the allele-only partition schedule, runner, aggregate artifact, report, and track-specific tests; shared scoring and guards remain in `pmhc_transfer.py`.

**Tech Stack:** Python 3.11, NumPy, pandas, pytest.

**Spec:** `.claude/delib/loao-mhc-transfer/brief.md`

## Global Constraints

- Start only from the committed LOAO foundation after PR #35 reaches `main`.
- Use all 47 target alleles and stop the whole track if preflight fails for one.
- Keep raw/row-level outputs ignored; only aggregate results are durable.
- Do not stage, commit, push, or create a PR without fresh explicit user authorization.

---

### Task 1: Allele-only schedule and diagnostics

**Files:**
- Create: `tests/test_pmhc_loao_allele.py`
- Modify: `src/cognate/pmhc_transfer.py`

**Interfaces:**
- Consumes: `build_allele_partition` and shared preflight functions.
- Produces: `build_allele_only_schedule(rows, alleles)` with one partition per eligible allele.

- [ ] **Step 1: Write the failing schedule test.**

```python
def test_allele_only_schedule_has_47_zero_target_leakage_partitions() -> None:
    schedule = build_allele_only_schedule(rows, eligible_alleles)
    assert len(schedule) == 47
    assert all(set(partition.test["Allele"]) == set(partition.held_out_alleles) for partition in schedule)
```

- [ ] **Step 2: Run it red.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele.py`

Expected: FAIL for missing schedule builder.

- [ ] **Step 3: Implement and test diagnostics.**

Return peptide-overlap count and fraction for every target, plus nearest retained pseudo-sequence distance and selected PWM source. Do not drop overlapping peptides in this track.

- [ ] **Step 4: Run focused tests green.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele.py`

Expected: PASS.

### Task 2: Allele-only runner, artifact, and report

**Files:**
- Create: `scripts/run_pmhc_loao_allele.py`
- Create: `data/pmhc/loao_allele_only_results.json`
- Create: `docs/pmhc/loao_allele_only.md`
- Modify: `tests/test_pmhc_loao_allele.py`

**Interfaces:**
- Consumes: the 47-partition schedule and shared transfer scorers.
- Produces: one aggregate-only artifact with paired comparisons and target diagnostics.

- [ ] **Step 1: Write a mocked end-to-end failing test.**

```python
def test_runner_writes_all_required_allele_only_sections(tmp_path: Path) -> None:
    artifact = run_allele_only(source_dir, tmp_path / "result.json", device="cpu")
    assert set(artifact) >= {"config", "scores", "comparisons", "per_allele", "diagnostics"}
```

- [ ] **Step 2: Run it red.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele.py::test_runner_writes_all_required_allele_only_sections`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement fixed evaluation and interpretation fields.**

Use macro AUC0.1 by target allele, 20,000 paired bootstrap draws, and explicit comparisons for correct MLP minus peptide-only MLP, shuffled MLP, and nearest PWM. Save the primary criterion as true only if the first two lower bounds exceed zero.

- [ ] **Step 4: Verify and run the fixed experiment once.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele.py && uv run python scripts/run_pmhc_loao_allele.py`

Expected: tests pass and the aggregate artifact plus report record one fixed run. Pause for explicit commit authorization.
