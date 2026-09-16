# LOAO Allele-and-Peptide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute #33 as a 47-target joint allele-and-peptide novelty evaluation.

**Architecture:** Branch from the committed LOAO foundation. This worktree defines only the joint-exclusion schedule, deletion diagnostics, runner, aggregate artifact, report, and tests. It reuses every shared arm and evaluation primitive unchanged.

**Tech Stack:** Python 3.11, NumPy, pandas, pytest.

**Spec:** `.claude/delib/loao-mhc-transfer/brief.md`

## Global Constraints

- Start only from the committed LOAO foundation after PR #35 reaches `main`.
- Retain all 47 target alleles or fail the complete track during preflight.
- Do not call this a pure allele-transfer or pure peptide-novelty result.
- Do not stage, commit, push, or create a PR without fresh explicit user authorization.

---

### Task 1: Joint-exclusion schedule

**Files:**
- Create: `tests/test_pmhc_loao_allele_peptide.py`
- Modify: `src/cognate/pmhc_transfer.py`

**Interfaces:**
- Consumes: `build_joint_novelty_partition` and shared preflight.
- Produces: `build_joint_novelty_schedule(rows, alleles)` with deletion diagnostics.

- [ ] **Step 1: Write failing no-overlap and accounting tests.**

```python
def test_joint_schedule_has_zero_target_and_peptide_overlap() -> None:
    schedule = build_joint_novelty_schedule(rows, eligible_alleles)
    for partition in schedule:
        assert set(partition.test["Allele"]).isdisjoint(partition.train["Allele"])
        assert set(partition.test["Peptide"]).isdisjoint(partition.train["Peptide"])
```

- [ ] **Step 2: Run them red.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele_peptide.py`

Expected: FAIL for missing schedule builder.

- [ ] **Step 3: Implement schedule and diagnostics.**

For every target record removed non-target training rows, retained positive/negative counts, retained allele count, selected PWM source, and nearest retained pseudo-sequence distance. Run the shared all-or-fail preflight before creating scores.

- [ ] **Step 4: Run focused tests green.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele_peptide.py`

Expected: PASS, including deliberate peptide-leakage corruption.

### Task 2: Joint-novelty runner and report

**Files:**
- Create: `scripts/run_pmhc_loao_allele_peptide.py`
- Create: `data/pmhc/loao_allele_peptide_results.json`
- Create: `docs/pmhc/loao_allele_peptide.md`
- Modify: `tests/test_pmhc_loao_allele_peptide.py`

**Interfaces:**
- Consumes: the joint-novelty schedule and shared transfer scorers.
- Produces: a labelled compound-intervention result artifact.

- [ ] **Step 1: Write a mocked end-to-end artifact test.**

```python
def test_joint_runner_records_retention_for_each_target(tmp_path: Path) -> None:
    artifact = run_joint_novelty(source_dir, tmp_path / "result.json", device="cpu")
    assert len(artifact["diagnostics"]["targets"]) == 47
```

- [ ] **Step 2: Run it red.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele_peptide.py::test_joint_runner_records_retention_for_each_target`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement the runner without changing shared settings.**

Keep the five fixed arms, target-only validation, metric, bootstrap, score guards, and paired comparisons identical to #32. Label the report and artifact as joint allele-and-peptide novelty; include deletion diagnostics before any interpretation.

- [ ] **Step 4: Verify and run the fixed experiment once.**

Run: `uv run pytest -q tests/test_pmhc_loao_allele_peptide.py && uv run python scripts/run_pmhc_loao_allele_peptide.py`

Expected: tests pass and aggregate-only output is written. Pause for explicit commit authorization.
