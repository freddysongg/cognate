# pMHC Novelty Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the shipped pMHC transfer gradient is a novelty effect or intrinsic per-allele difficulty, by adding one predictor that has already seen every allele in the cohort.

**Architecture:** MHCflurry 2.0 runs in a fully isolated environment and writes a CSV; nothing about it enters this project's dependency set. A new pure-analysis module joins that CSV to the already-committed per-allele AUC0.1 and distance diagnostics, fits one pre-declared linear relation per predictor, and compares slopes. The frozen 47-allele cohort, the five-name scoring contract, and every shipped artifact are untouched.

**Tech Stack:** Python 3.11, uv, pandas, numpy, scipy, pytest. MHCflurry 2.0 + TensorFlow, isolated via `uv run --no-project --with`.

**Spec:** `.claude/delib/pmhc-allele-performance-rule/brief.md` (revision 2)

## Global Constraints

- **Never modify the frozen contract:** `src/cognate/metrics.py`, `src/cognate/split.py`, `src/cognate/features.py`, `data/vdjdb_eval.csv`. `tests/test_frozen_contract_hashes.py` enforces this. Call `auc01` from `metrics.py`; never edit it.
- **Never modify** `TRANSFER_SCORE_NAMES`, `TRANSFER_REFERENCES`, or `validate_transfer_result` in `src/cognate/pmhc_transfer.py`. MHCflurry is not a sixth arm.
- **Never modify** `pyproject.toml`. TensorFlow must not enter the project environment; the shipped results were produced without it.
- **Never modify** any existing `data/pmhc/loao_*_results.json` or `data/pmhc/source_contract.json`.
- **Cohort is frozen at 47 alleles** — `load_pmhc_dataset(source_dir)` defaults, no `minimum_class_rows` override anywhere.
- **Only slopes are comparable across predictors.** MHCflurry's training data overlaps these test rows, so its absolute AUC0.1 is partly memorization. No function, artifact field, or document table may compare absolute AUC0.1 between MHCflurry and our arms.
- **Python:** explicit return types on every function, `import type`-equivalent (`from __future__ import annotations`) where the codebase does it, no `Any`, literal unions instead of magic strings, no inline comments.
- **Local CI, must pass before every commit:** `uv run pytest -q` then `uv run python -m compileall -q src scripts`. Baseline is 321 passing tests.
- **Reference values, `observed` 2026-09-16** from `data/pmhc/loao_allele_only_results.json`: `pseudo_sequence_mlp` slope of AUC0.1 on nearest-retained normalized Hamming distance is **−1.0315**, SE 0.1429, 95% CI [−1.3116, −0.7514], intercept 0.8597, r² 0.537, n=47, over 8 unique distance values spanning 0.0294–0.2941.

---

## File Structure

- **Create** `docs/pmhc/novelty_control_predeclaration.md` — the frozen pre-declaration. Committed alone, before any code that touches MHCflurry.
- **Create** `src/cognate/pmhc_control.py` — pure analysis. Loads committed artifacts, scores per-allele AUC0.1 from a predictions frame, fits slopes, classifies the outcome. No TensorFlow, no torch, no network access.
- **Create** `scripts/predict_mhcflurry.py` — standalone, run under an isolated interpreter. Reads cohort rows, writes `data/pmhc/mhcflurry_predictions.csv`. Never imported by `src/cognate`.
- **Create** `scripts/run_pmhc_novelty_control.py` — reads the CSV plus committed artifacts, writes `data/pmhc/novelty_control_results.json`.
- **Create** `tests/test_pmhc_control.py` — tests for the analysis module.
- **Create** `docs/pmhc/novelty_control.md` — the writeup.
- **Modify** `docs/pmhc/loao_allele_only.md`, `loao_allele_peptide.md`, `loao_cluster.md` — only in Task 7, and only if the outcome requires a correction.

---

### Task 1: Freeze the pre-declaration

This task exists to be its own commit. Its entire purpose is that `git log` shows it landing before any MHCflurry code. Do not combine it with Task 2 or later.

**Files:**
- Create: `docs/pmhc/novelty_control_predeclaration.md`
- Create: `tests/test_pmhc_control.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PREDECLARATION_PATH` constant used by Task 6's artifact; the three outcome literals `"novelty_effect"`, `"intrinsic_difficulty"`, `"mixed"` used by Tasks 5 and 6.

- [ ] **Step 1: Write the pre-declaration document**

Create `docs/pmhc/novelty_control_predeclaration.md`:

```markdown
# Pre-declaration — pMHC novelty control

Frozen 2026-09-16, before any MHCflurry code exists in this repository. Committed alone so
that `git log` shows this commit strictly precedes the commit that runs the control. This is
the integrity device for the whole study; if the order is not visible in history, the result
is not trustworthy and should be discarded.

## Question

The shipped LOAO study reports per-allele AUC0.1 falling with pseudo-sequence distance to the
nearest retained allele. Every arm in that study was trained under holdout, so a novelty
effect and intrinsic per-allele difficulty predict the same curve. MHCflurry 2.0 has already
seen these alleles, so it carries no novelty penalty. Its gradient discriminates the two.

## Functional form, frozen

Ordinary least squares of per-allele standardized AUC0.1 on nearest-retained normalized
Hamming distance. One fit per predictor. Linear, untransformed, unweighted, no interaction
terms, no polynomial expansion. Chosen because the shipped relation is close to linear
(r = -0.733, r-squared 0.537) and because the distance axis carries only 8 unique values,
which will not support a more flexible form.

## Covariate split, frozen

- **Deployment-available**, may enter a predictive rule: nearest-retained pseudo-sequence
  distance, retained-neighbour support.
- **Null-test-only**, may enter confound tests and never a deployable rule: target `n_rows`,
  target `n_positive`.

## Reference slope, frozen

`pseudo_sequence_mlp`, from `data/pmhc/loao_allele_only_results.json`: slope -1.0315,
standard error 0.1429, 95% confidence interval [-1.3116, -0.7514], n = 47.

`nearest_pwm` is excluded from this comparison. `select_nearest_pwm_source` picks its source
by the same Hamming metric used as the x-axis, and its source distance equals the x-axis for
47 of 47 alleles, so its slope (-1.0114) is tautological and carries no evidence.

## Decision rule, frozen

Let `C` be MHCflurry's fitted slope with 95% confidence interval `[C_lo, C_hi]`, and let
`R = -1.0315` be the reference slope point estimate.

| Condition | Outcome literal | Reading |
|---|---|---|
| `C_lo <= 0 <= C_hi` and `R < C_lo` | `novelty_effect` | control is flat and distinguishable from the reference; the shipped gradient is novelty |
| `C_lo <= R <= C_hi` | `intrinsic_difficulty` | control is as steep as the reference; the shipped gradient is confounded with allele difficulty |
| neither | `mixed` | both contribute; report the decomposition and bound the shipped interpretation |

Evaluated in that order. The interval is the tolerance; no separate epsilon is used, because
an interval-based rule cannot be tuned after seeing the scatter the way a hand-picked
threshold can.

## Exclusions, frozen

Any cohort allele absent from MHCflurry's curated training alleles is not a no-novelty
control for that allele. Such alleles are identified before the regression is fitted, are
excluded from the primary fit, and are reported with their count and names. A secondary fit
including them is reported alongside, labelled as such.

## What would make this study invalid

- This commit not preceding the control run in `git log`.
- Any change to the functional form, covariate split, reference slope, or decision rule after
  MHCflurry predictions exist on disk.
- Comparing MHCflurry's absolute AUC0.1 to any of our arms. Its training data overlaps these
  test rows; only slopes are comparable.
```

- [ ] **Step 2: Write the failing test that pins the pre-declaration's load-bearing values**

Create `tests/test_pmhc_control.py`:

```python
"""Tests for the pMHC novelty control analysis."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDECLARATION = ROOT / "docs" / "pmhc" / "novelty_control_predeclaration.md"


def test_predeclaration_pins_the_reference_slope() -> None:
    text = PREDECLARATION.read_text(encoding="utf-8")
    assert "-1.0315" in text
    assert "0.1429" in text
    assert "[-1.3116, -0.7514]" in text


def test_predeclaration_names_all_three_outcome_literals() -> None:
    text = PREDECLARATION.read_text(encoding="utf-8")
    for literal in ("novelty_effect", "intrinsic_difficulty", "mixed"):
        assert literal in text


def test_predeclaration_forbids_absolute_accuracy_comparison() -> None:
    text = PREDECLARATION.read_text(encoding="utf-8")
    assert "only slopes are comparable" in text.lower()
```

- [ ] **Step 3: Run the tests to verify they pass against the document just written**

Run: `uv run pytest tests/test_pmhc_control.py -v`
Expected: 3 passed.

- [ ] **Step 4: Run local CI**

Run: `uv run pytest -q`
Expected: 324 passed (321 baseline plus 3 new).

Run: `uv run python -m compileall -q src scripts`
Expected: no output, exit 0.

- [ ] **Step 5: Commit, alone**

```bash
git add docs/pmhc/novelty_control_predeclaration.md tests/test_pmhc_control.py
git commit -m "add frozen pre-declaration for pmhc novelty control"
```

Do not stage anything else in this commit. Verify with `git show --stat HEAD` that exactly two files changed.

---

### Task 2: Identify which cohort alleles MHCflurry has seen

Must complete before any regression is fitted. An allele MHCflurry has not trained on is not a no-novelty control for that allele.

**Files:**
- Create: `scripts/predict_mhcflurry.py` (coverage subcommand only in this task)
- Modify: `tests/test_pmhc_control.py`

**Interfaces:**
- Consumes: `load_pmhc_dataset` from `src/cognate/pmhc.py`, `data/pmhc/source_contract.json`.
- Produces: `data/pmhc/mhcflurry_allele_coverage.json`, shape
  `{"supported": [str, ...], "unsupported": [str, ...], "mhcflurry_version": str}`.

- [ ] **Step 1: Write the failing test for the coverage artifact's shape**

Add to `tests/test_pmhc_control.py`:

```python
COVERAGE_PATH = ROOT / "data" / "pmhc" / "mhcflurry_allele_coverage.json"


def test_coverage_artifact_partitions_the_frozen_cohort() -> None:
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    assert set(coverage) == {"supported", "unsupported", "mhcflurry_version"}
    supported = set(coverage["supported"])
    unsupported = set(coverage["unsupported"])
    assert not supported & unsupported
    assert len(supported | unsupported) == 47
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pmhc_control.py::test_coverage_artifact_partitions_the_frozen_cohort -v`
Expected: FAIL with `FileNotFoundError` on `mhcflurry_allele_coverage.json`.

- [ ] **Step 3: Write the coverage script**

Create `scripts/predict_mhcflurry.py`:

```python
"""MHCflurry 2.0 inference for the pMHC novelty control.

Runs under an isolated interpreter, never under the project environment:

    uv run --no-project --python 3.11 \
        --with "mhcflurry>=2.0,<3.0" --with "tensorflow>=2.16" --with "pandas>=2.2" \
        python scripts/predict_mhcflurry.py coverage

TensorFlow is deliberately absent from pyproject.toml. The shipped LOAO results were
produced without it, and this script's only interface to the rest of the project is a CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT_ALLELES_PATH = ROOT / "data" / "pmhc" / "cohort_alleles.json"
COVERAGE_PATH = ROOT / "data" / "pmhc" / "mhcflurry_allele_coverage.json"
PREDICTIONS_PATH = ROOT / "data" / "pmhc" / "mhcflurry_predictions.csv"
COHORT_ROWS_PATH = ROOT / "data" / "pmhc" / "cohort_rows.csv"


def write_coverage() -> dict[str, object]:
    from mhcflurry import Class1AffinityPredictor, __version__

    cohort_alleles = json.loads(COHORT_ALLELES_PATH.read_text(encoding="utf-8"))
    predictor = Class1AffinityPredictor.load()
    known = set(predictor.supported_alleles)
    supported = sorted(allele for allele in cohort_alleles if allele in known)
    unsupported = sorted(allele for allele in cohort_alleles if allele not in known)
    coverage = {
        "supported": supported,
        "unsupported": unsupported,
        "mhcflurry_version": __version__,
    }
    COVERAGE_PATH.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return coverage


def write_predictions() -> None:
    raise NotImplementedError("implemented in task 6")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"coverage", "predict"}:
        raise SystemExit("usage: predict_mhcflurry.py {coverage|predict}")
    if sys.argv[1] == "coverage":
        coverage = write_coverage()
        print(
            f"supported {len(coverage['supported'])} "
            f"unsupported {len(coverage['unsupported'])}",
            flush=True,
        )
        return
    write_predictions()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the cohort-allele exporter and run it in the project environment**

Create `scripts/export_pmhc_cohort.py`:

```python
"""Export the frozen 47-allele cohort to CSV and JSON for isolated consumers."""

from __future__ import annotations

import json
from pathlib import Path

from cognate.pmhc import load_pmhc_dataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train"
COHORT_ALLELES_PATH = ROOT / "data" / "pmhc" / "cohort_alleles.json"
COHORT_ROWS_PATH = ROOT / "data" / "pmhc" / "cohort_rows.csv"


def main() -> None:
    dataset = load_pmhc_dataset(SOURCE_DIR)
    alleles = list(dataset.eligible_alleles)
    if len(alleles) != 47:
        raise AssertionError(f"expected the frozen 47-allele cohort, got {len(alleles)}")
    COHORT_ALLELES_PATH.write_text(
        json.dumps(alleles, indent=2) + "\n", encoding="utf-8"
    )
    dataset.rows[["Allele", "Peptide", "Target"]].to_csv(COHORT_ROWS_PATH, index=False)
    print(f"exported {len(alleles)} alleles and {len(dataset.rows)} rows", flush=True)


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/export_pmhc_cohort.py`
Expected: `exported 47 alleles and 112128 rows`

- [ ] **Step 5: Run the coverage step in the isolated environment**

```bash
uv run --no-project --python 3.11 \
  --with "mhcflurry>=2.0,<3.0" --with "tensorflow>=2.16" --with "pandas>=2.2" \
  python scripts/predict_mhcflurry.py coverage
```

Expected: a line reporting supported and unsupported counts summing to 47. If MHCflurry's model files are not yet present it will raise; run `mhcflurry-downloads fetch models_class1_pan` under the same isolated invocation first.

**If `unsupported` is non-empty, stop and report the names before continuing.** The brief requires those alleles be identified before the regression, not after.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_pmhc_control.py -v`
Expected: 4 passed.

- [ ] **Step 7: Run local CI and commit**

Run: `uv run pytest -q` then `uv run python -m compileall -q src scripts`

```bash
git add scripts/predict_mhcflurry.py scripts/export_pmhc_cohort.py \
  tests/test_pmhc_control.py data/pmhc/cohort_alleles.json \
  data/pmhc/cohort_rows.csv data/pmhc/mhcflurry_allele_coverage.json
git commit -m "export frozen cohort, record mhcflurry allele coverage"
```

---

### Task 3: Per-allele AUC0.1 from a predictions frame

Pure function, no MHCflurry, no network. This is the unit that turns MHCflurry's raw scores into the same quantity the shipped artifacts carry.

**Files:**
- Create: `src/cognate/pmhc_control.py`
- Modify: `tests/test_pmhc_control.py`

**Interfaces:**
- Consumes: `auc01` from `src/cognate/metrics.py` (frozen, call only).
- Produces: `score_per_allele_auc01(predictions: pd.DataFrame) -> dict[str, float]`, keyed by allele. Input frame requires columns `Allele`, `Target` (bool), `Score` (float).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pmhc_control.py`:

```python
import pandas as pd
import pytest

from cognate.pmhc_control import score_per_allele_auc01


def test_score_per_allele_auc01_is_perfect_when_scores_rank_targets_first() -> None:
    predictions = pd.DataFrame(
        {
            "Allele": ["HLA-A01:01"] * 4 + ["HLA-B07:02"] * 4,
            "Target": [True, True, False, False] * 2,
            "Score": [0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1],
        }
    )
    scored = score_per_allele_auc01(predictions)
    assert set(scored) == {"HLA-A01:01", "HLA-B07:02"}
    assert scored["HLA-A01:01"] == pytest.approx(1.0)


def test_score_per_allele_auc01_rejects_an_allele_missing_a_class() -> None:
    predictions = pd.DataFrame(
        {
            "Allele": ["HLA-A01:01"] * 3,
            "Target": [True, True, True],
            "Score": [0.9, 0.8, 0.7],
        }
    )
    with pytest.raises(ValueError, match="single-class allele"):
        score_per_allele_auc01(predictions)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pmhc_control.py -k score_per_allele -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cognate.pmhc_control'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/cognate/pmhc_control.py`:

```python
"""Novelty-control analysis for the pMHC transfer gradient.

Pure analysis over committed artifacts and one predictions frame. Imports nothing from
TensorFlow or torch, reaches no network, and never writes to a shipped artifact.
"""

from __future__ import annotations

import pandas as pd

from cognate.metrics import auc01

_PREDICTION_COLUMNS = frozenset({"Allele", "Target", "Score"})


def score_per_allele_auc01(predictions: pd.DataFrame) -> dict[str, float]:
    """Standardized AUC0.1 per allele over a predictions frame."""
    missing = _PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions frame missing columns: {sorted(missing)}")
    scored: dict[str, float] = {}
    for allele, group in predictions.groupby("Allele", sort=True):
        labels = group["Target"].to_numpy(dtype=bool)
        if labels.all() or not labels.any():
            raise ValueError(f"single-class allele cannot be scored: {allele}")
        scored[str(allele)] = auc01(labels, group["Score"].to_numpy(dtype=float))
    return scored
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pmhc_control.py -k score_per_allele -v`
Expected: 2 passed.

- [ ] **Step 5: Run local CI and commit**

Run: `uv run pytest -q` then `uv run python -m compileall -q src scripts`

```bash
git add src/cognate/pmhc_control.py tests/test_pmhc_control.py
git commit -m "add per-allele auc01 scoring for the novelty control"
```

---

### Task 4: Slope fitting and the structural slope-only guarantee

The test in Step 1 is the enforcement mechanism for the brief's "slopes only" constraint. It fails if anyone reimplements the classifier in terms of absolute AUC0.1.

**Files:**
- Modify: `src/cognate/pmhc_control.py`
- Modify: `tests/test_pmhc_control.py`

**Interfaces:**
- Consumes: `score_per_allele_auc01` from Task 3.
- Produces:
  - `PINNED_REFERENCE_SLOPE: float` = -1.0315, the single definition of the pre-declared
    reference slope. Task 5's and Task 6's consumers import it rather than restating the literal,
    so the study's most load-bearing number cannot drift between files.
  - `DistanceSlope` frozen dataclass with fields `slope: float`, `stderr: float`, `ci_lo: float`, `ci_hi: float`, `intercept: float`, `r_squared: float`, `n: int`.
  - `fit_distance_slope(distances: Sequence[float], scores: Sequence[float]) -> DistanceSlope`
  - `ControlOutcome = Literal["novelty_effect", "intrinsic_difficulty", "mixed"]`
  - `classify_control_outcome(reference_slope: float, control: DistanceSlope) -> ControlOutcome`

- [ ] **Step 1: Write the failing tests, including the slope-only property test**

Add to `tests/test_pmhc_control.py`:

```python
from cognate.pmhc_control import (
    PINNED_REFERENCE_SLOPE,
    DistanceSlope,
    classify_control_outcome,
    fit_distance_slope,
)

REFERENCE_SLOPE = PINNED_REFERENCE_SLOPE


def test_fit_distance_slope_recovers_a_known_line() -> None:
    distances = [0.1, 0.2, 0.3, 0.4]
    scores = [0.8, 0.6, 0.4, 0.2]
    fitted = fit_distance_slope(distances, scores)
    assert fitted.slope == pytest.approx(-2.0)
    assert fitted.intercept == pytest.approx(1.0)
    assert fitted.n == 4


def test_classify_returns_novelty_effect_for_a_flat_control() -> None:
    flat = DistanceSlope(
        slope=0.01, stderr=0.05, ci_lo=-0.09, ci_hi=0.11,
        intercept=0.9, r_squared=0.0, n=47,
    )
    assert classify_control_outcome(REFERENCE_SLOPE, flat) == "novelty_effect"


def test_classify_returns_intrinsic_difficulty_when_control_matches_reference() -> None:
    steep = DistanceSlope(
        slope=-1.00, stderr=0.15, ci_lo=-1.30, ci_hi=-0.70,
        intercept=0.85, r_squared=0.5, n=47,
    )
    assert classify_control_outcome(REFERENCE_SLOPE, steep) == "intrinsic_difficulty"


def test_classify_returns_mixed_when_control_is_steep_but_shallower() -> None:
    partial = DistanceSlope(
        slope=-0.45, stderr=0.08, ci_lo=-0.61, ci_hi=-0.29,
        intercept=0.88, r_squared=0.3, n=47,
    )
    assert classify_control_outcome(REFERENCE_SLOPE, partial) == "mixed"


def test_outcome_is_invariant_to_a_constant_offset_in_control_scores() -> None:
    """The slopes-only constraint, enforced structurally.

    MHCflurry's training data overlaps these test rows, so its absolute AUC0.1 is partly
    memorization and is not comparable to ours. Shifting every control score by a constant
    must not change the outcome. This fails if the classifier is ever rewritten to consume
    absolute performance.
    """
    distances = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    scores = [0.80, 0.76, 0.71, 0.67, 0.62, 0.58]
    shifted = [score + 0.15 for score in scores]
    baseline = classify_control_outcome(REFERENCE_SLOPE, fit_distance_slope(distances, scores))
    offset = classify_control_outcome(REFERENCE_SLOPE, fit_distance_slope(distances, shifted))
    assert baseline == offset
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_pmhc_control.py -k "slope or classify or offset" -v`
Expected: FAIL with `ImportError: cannot import name 'DistanceSlope'`.

- [ ] **Step 3: Write the implementation**

Add to `src/cognate/pmhc_control.py`, and extend the import block at the top to include
`from collections.abc import Sequence`, `from dataclasses import dataclass`, `from typing import
Literal`, and `import numpy as np` — numpy is first used in this task, so it belongs here rather
than in Task 3:

```python
CONFIDENCE_MULTIPLIER = 1.96
PINNED_REFERENCE_SLOPE = -1.0315

ControlOutcome = Literal["novelty_effect", "intrinsic_difficulty", "mixed"]


@dataclass(frozen=True)
class DistanceSlope:
    """An OLS fit of per-allele AUC0.1 on pseudo-sequence distance."""

    slope: float
    stderr: float
    ci_lo: float
    ci_hi: float
    intercept: float
    r_squared: float
    n: int


def fit_distance_slope(
    distances: Sequence[float], scores: Sequence[float]
) -> DistanceSlope:
    """Fit the pre-declared linear relation of AUC0.1 on distance."""
    x = np.asarray(distances, dtype=float)
    y = np.asarray(scores, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"length mismatch: {x.shape} distances, {y.shape} scores")
    if x.size < 3:
        raise ValueError(f"need at least three alleles to fit a slope, got {x.size}")
    if np.unique(x).size < 2:
        raise ValueError("distance axis is constant; no slope is identifiable")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - y.mean()) ** 2))
    degrees_of_freedom = x.size - 2
    stderr = float(
        np.sqrt(residual_sum / degrees_of_freedom / np.sum((x - x.mean()) ** 2))
    )
    return DistanceSlope(
        slope=float(slope),
        stderr=stderr,
        ci_lo=float(slope) - CONFIDENCE_MULTIPLIER * stderr,
        ci_hi=float(slope) + CONFIDENCE_MULTIPLIER * stderr,
        intercept=float(intercept),
        r_squared=0.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum,
        n=int(x.size),
    )


def classify_control_outcome(
    reference_slope: float, control: DistanceSlope
) -> ControlOutcome:
    """Apply the frozen decision rule from the pre-declaration, in its stated order."""
    is_flat = control.ci_lo <= 0.0 <= control.ci_hi
    if is_flat and reference_slope < control.ci_lo:
        return "novelty_effect"
    if control.ci_lo <= reference_slope <= control.ci_hi:
        return "intrinsic_difficulty"
    return "mixed"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pmhc_control.py -v`
Expected: all pass.

- [ ] **Step 5: Prove the offset test can fail**

Temporarily change `classify_control_outcome`'s first line to
`is_flat = control.intercept > 0.85` and rerun
`uv run pytest tests/test_pmhc_control.py -k offset -v`. Confirm it FAILS, then revert the
line. This makes the slopes-only guard `failure-proven` rather than merely `observed`.

- [ ] **Step 6: Run local CI and commit**

Run: `uv run pytest -q` then `uv run python -m compileall -q src scripts`

```bash
git add src/cognate/pmhc_control.py tests/test_pmhc_control.py
git commit -m "add distance slope fit, frozen decision rule, slopes-only guard"
```

---

### Task 5: Load the committed reference artifact

**Files:**
- Modify: `src/cognate/pmhc_control.py`
- Modify: `tests/test_pmhc_control.py`

**Interfaces:**
- Produces: `load_reference_table(results_path: Path, arm: str) -> pd.DataFrame` with columns
  `Allele`, `Distance`, `Auc01`, `NRows`, `NPositive`, one row per allele, sorted by `Allele`.

- [ ] **Step 1: Write the failing test against the real committed artifact**

Add to `tests/test_pmhc_control.py`:

```python
from cognate.pmhc_control import load_reference_table

ALLELE_ONLY_RESULTS = ROOT / "data" / "pmhc" / "loao_allele_only_results.json"


def test_load_reference_table_reproduces_the_pinned_reference_slope() -> None:
    table = load_reference_table(ALLELE_ONLY_RESULTS, "pseudo_sequence_mlp")
    assert len(table) == 47
    assert list(table.columns) == ["Allele", "Distance", "Auc01", "NRows", "NPositive"]
    fitted = fit_distance_slope(table["Distance"], table["Auc01"])
    assert fitted.slope == pytest.approx(REFERENCE_SLOPE, abs=5e-4)
    assert fitted.stderr == pytest.approx(0.1429, abs=5e-4)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pmhc_control.py -k load_reference -v`
Expected: FAIL with `ImportError: cannot import name 'load_reference_table'`.

- [ ] **Step 3: Write the implementation**

Add to `src/cognate/pmhc_control.py`, extending the import block with `import json` and
`from pathlib import Path`:

```python
def load_reference_table(results_path: Path, arm: str) -> pd.DataFrame:
    """Per-allele distance and AUC0.1 for one arm of a committed LOAO artifact."""
    artifact = json.loads(results_path.read_text(encoding="utf-8"))
    per_allele = artifact["per_allele"]
    diagnostics = artifact["diagnostics"]
    records = []
    for allele in sorted(per_allele):
        arms = per_allele[allele]["arms"]
        if arm not in arms:
            raise ValueError(f"arm {arm!r} absent for {allele}; have {sorted(arms)}")
        records.append(
            {
                "Allele": allele,
                "Distance": diagnostics[allele]["nearest_retained_pseudo_distance"],
                "Auc01": arms[arm]["auc01"],
                "NRows": per_allele[allele]["n_rows"],
                "NPositive": per_allele[allele]["n_positive"],
            }
        )
    return pd.DataFrame.from_records(records)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pmhc_control.py -k load_reference -v`
Expected: PASS. The recovered slope matching −1.0315 to four decimals confirms the analysis path reproduces the pinned reference.

- [ ] **Step 5: Run local CI and commit**

```bash
git add src/cognate/pmhc_control.py tests/test_pmhc_control.py
git commit -m "load per-allele reference table from the committed loao artifact"
```

---

### Task 6: Run MHCflurry and write the result artifact

**Files:**
- Modify: `scripts/predict_mhcflurry.py` (replace the `write_predictions` guard body)
- Create: `scripts/run_pmhc_novelty_control.py`
- Modify: `tests/test_pmhc_control.py`

**Interfaces:**
- Consumes: everything from Tasks 3–5, plus `data/pmhc/mhcflurry_allele_coverage.json`.
- Produces: `data/pmhc/novelty_control_results.json` with top-level keys exactly
  `{"config", "coverage", "reference", "control", "outcome", "support_nulls"}`.

- [ ] **Step 1: Replace the `write_predictions` body in `scripts/predict_mhcflurry.py`**

```python
def write_predictions() -> None:
    import numpy as np
    import pandas as pd
    from mhcflurry import Class1AffinityPredictor

    rows = pd.read_csv(COHORT_ROWS_PATH)
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    supported = set(coverage["supported"])
    scored = rows.loc[rows["Allele"].isin(supported)].reset_index(drop=True)
    predictor = Class1AffinityPredictor.load()
    predicted = predictor.predict(
        peptides=scored["Peptide"].tolist(),
        alleles=scored["Allele"].tolist(),
    )
    if len(predicted) != len(scored):
        raise AssertionError(
            f"expected {len(scored)} predictions, got {len(predicted)}"
        )
    scored["Score"] = -np.asarray(predicted, dtype=float)
    scored[["Allele", "Peptide", "Target", "Score"]].to_csv(
        PREDICTIONS_PATH, index=False
    )
    print(f"wrote {len(scored)} predictions to {PREDICTIONS_PATH}", flush=True)
```

`Score` negates affinity because MHCflurry reports affinity in nM where lower means stronger binding, while `auc01` expects higher to mean more likely positive. AUC0.1 is rank-based, so the sign flip alone suffices; no log transform is needed.

Allele names need no normalization: `predict` accepts this cohort's `HLA-A01:01` format directly and normalizes internally, verified by a three-peptide probe returning sensible nM affinities. `observed`. Only a literal set-comparison against `supported_alleles` needs `mhcflurry.common.normalize_allele_name`, because that property reports the starred `HLA-A*01:01` form.

`Class1AffinityPredictor` is the right class, not `Class1PresentationPredictor`: `predict(peptides, alleles)` takes `alleles` as a list parallel to `peptides`, and binding affinity is what this cohort's `Target` label was thresholded from (`Affinity > 0.426`). The presentation predictor blends processing and cannot score per-row pairs — its `alleles` argument is a genotype list or a sample-to-alleles dict.

- [ ] **Step 2: Run a timing probe on one allele before the full pass**

```bash
uv run --no-project --python 3.11 \
  --with "mhcflurry>=2.0,<3.0" --with "tensorflow>=2.16" --with "pandas>=2.2" \
  python -c "
import pandas as pd, time
from mhcflurry import Class1AffinityPredictor
rows = pd.read_csv('data/pmhc/cohort_rows.csv')
one = rows[rows['Allele'] == 'HLA-A01:01']
p = Class1AffinityPredictor.load()
start = time.time()
p.predict(peptides=one['Peptide'].tolist(), alleles=one['Allele'].tolist())
print(f'{len(one)} rows in {time.time() - start:.1f}s')
"
```

Record the number. Extrapolate to 112,128 rows before launching the full pass; if it projects beyond two hours, stop and report rather than proceeding.

- [ ] **Step 3: Run the full prediction pass**

```bash
uv run --no-project --python 3.11 \
  --with "mhcflurry>=2.0,<3.0" --with "tensorflow>=2.16" --with "pandas>=2.2" \
  python scripts/predict_mhcflurry.py predict
```

Expected: `wrote N predictions to data/pmhc/mhcflurry_predictions.csv` where N is the row count for supported alleles.

- [ ] **Step 4: Write the failing test for the result artifact**

Add to `tests/test_pmhc_control.py`:

```python
CONTROL_RESULTS = ROOT / "data" / "pmhc" / "novelty_control_results.json"


def test_control_artifact_has_the_declared_top_level_contract() -> None:
    artifact = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    assert set(artifact) == {
        "config",
        "coverage",
        "reference",
        "control",
        "outcome",
        "support_nulls",
    }


def test_control_artifact_outcome_is_one_of_the_frozen_literals() -> None:
    artifact = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    assert artifact["outcome"] in {"novelty_effect", "intrinsic_difficulty", "mixed"}


def test_control_artifact_carries_no_cross_predictor_accuracy_comparison() -> None:
    artifact = json.loads(CONTROL_RESULTS.read_text(encoding="utf-8"))
    flat = json.dumps(artifact).lower()
    for forbidden in ("auc01_difference", "accuracy_delta", "beats", "outperforms"):
        assert forbidden not in flat
```

- [ ] **Step 5: Run to verify it fails**

Run: `uv run pytest tests/test_pmhc_control.py -k control_artifact -v`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 6: Write the orchestrating script**

Create `scripts/run_pmhc_novelty_control.py`:

```python
"""Fit the frozen novelty-control comparison and write its artifact."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.pmhc_control import (
    PINNED_REFERENCE_SLOPE,
    classify_control_outcome,
    fit_distance_slope,
    load_reference_table,
    score_per_allele_auc01,
)

ROOT = Path(__file__).resolve().parents[1]
ALLELE_ONLY_RESULTS = ROOT / "data" / "pmhc" / "loao_allele_only_results.json"
COVERAGE_PATH = ROOT / "data" / "pmhc" / "mhcflurry_allele_coverage.json"
PREDICTIONS_PATH = ROOT / "data" / "pmhc" / "mhcflurry_predictions.csv"
RESULTS_PATH = ROOT / "data" / "pmhc" / "novelty_control_results.json"
REFERENCE_ARM = "pseudo_sequence_mlp"
REFERENCE_SLOPE = PINNED_REFERENCE_SLOPE


def main() -> None:
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    supported = set(coverage["supported"])
    reference = load_reference_table(ALLELE_ONLY_RESULTS, REFERENCE_ARM)
    primary = reference.loc[reference["Allele"].isin(supported)].reset_index(drop=True)

    reference_fit = fit_distance_slope(primary["Distance"], primary["Auc01"])
    if abs(reference_fit.slope - REFERENCE_SLOPE) > 0.15:
        raise AssertionError(
            f"reference slope on supported alleles is {reference_fit.slope:.4f}, "
            f"far from the pinned {REFERENCE_SLOPE}; investigate before interpreting"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    control_auc = score_per_allele_auc01(predictions)
    control_table = primary.assign(
        ControlAuc01=[control_auc[allele] for allele in primary["Allele"]]
    )
    control_fit = fit_distance_slope(
        control_table["Distance"], control_table["ControlAuc01"]
    )

    artifact = {
        "config": {
            "reference_arm": REFERENCE_ARM,
            "pinned_reference_slope": REFERENCE_SLOPE,
            "functional_form": "ols_linear",
            "predeclaration": "docs/pmhc/novelty_control_predeclaration.md",
            "n_alleles_primary": int(len(primary)),
        },
        "coverage": coverage,
        "reference": asdict(reference_fit),
        "control": asdict(control_fit),
        "outcome": classify_control_outcome(REFERENCE_SLOPE, control_fit),
        "support_nulls": {
            "distance_vs_log_n_rows": float(
                np.corrcoef(
                    control_table["Distance"].to_numpy(dtype=float),
                    np.log(control_table["NRows"].to_numpy(dtype=float)),
                )[0, 1]
            ),
            "distance_vs_log_n_positive": float(
                np.corrcoef(
                    control_table["Distance"].to_numpy(dtype=float),
                    np.log(control_table["NPositive"].to_numpy(dtype=float)),
                )[0, 1]
            ),
            "control_vs_log_n_rows": float(
                np.corrcoef(
                    np.log(control_table["NRows"].to_numpy(dtype=float)),
                    control_table["ControlAuc01"].to_numpy(dtype=float),
                )[0, 1]
            ),
            "control_vs_log_n_positive": float(
                np.corrcoef(
                    np.log(control_table["NPositive"].to_numpy(dtype=float)),
                    control_table["ControlAuc01"].to_numpy(dtype=float),
                )[0, 1]
            ),
        },
    }
    RESULTS_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"outcome {artifact['outcome']}, wrote {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run it**

Run: `uv run python scripts/run_pmhc_novelty_control.py`
Expected: a line naming one of the three outcomes.

- [ ] **Step 8: Run the tests and local CI, then commit**

Run: `uv run pytest -q` then `uv run python -m compileall -q src scripts`

```bash
git add scripts/predict_mhcflurry.py scripts/run_pmhc_novelty_control.py \
  tests/test_pmhc_control.py data/pmhc/mhcflurry_predictions.csv \
  data/pmhc/novelty_control_results.json
git commit -m "run mhcflurry novelty control, write comparison artifact"
```

- [ ] **Step 9: Verify the integrity ordering**

```bash
git log --oneline --reverse -- docs/pmhc/novelty_control_predeclaration.md data/pmhc/novelty_control_results.json
```

Expected: the pre-declaration commit appears strictly before the control-run commit. **If it does not, the result is not trustworthy — stop and report.**

---

### Task 7: Writeup, and correct the shipped docs if the outcome requires it

**Files:**
- Create: `docs/pmhc/novelty_control.md`
- Modify: `docs/pmhc/loao_allele_only.md`, `docs/pmhc/loao_allele_peptide.md`, `docs/pmhc/loao_cluster.md` — **only** if the outcome is `intrinsic_difficulty` or `mixed`.

**Interfaces:**
- Consumes: `data/pmhc/novelty_control_results.json`.
- Produces: no code interface.

- [ ] **Step 1: Read every number out of the artifact**

Run:

```bash
uv run python -c "
import json
a = json.load(open('data/pmhc/novelty_control_results.json'))
print(json.dumps(a, indent=2, sort_keys=True))
"
```

Every figure in the writeup must be copied from this output. Do not restate any number from memory or from this plan.

- [ ] **Step 2: Write `docs/pmhc/novelty_control.md`**

Follow the structure of `docs/pmhc/loao_allele_only.md`: a one-paragraph statement of the question, a results table, a Limits section, and evidence labels. It must contain, at minimum:

- The question, and why the shipped study cannot answer it — every arm there was trained under holdout, so novelty and intrinsic difficulty predict the same curve.
- Both fitted slopes with confidence intervals, and the outcome literal.
- A Limits section stating: MHCflurry's training data overlaps these test rows so only slopes are comparable and no absolute comparison is made; the distance axis carries only 8 unique values across 47 alleles; any allele in `coverage["unsupported"]` was excluded from the primary fit, named explicitly.
- A statement of what "intrinsic difficulty" can and cannot mean here. Do **not** assert that far alleles are rare alleles: in this cohort distance is only weakly related to data volume (distance versus log `n_rows` Spearman -0.041, versus log `n_positive` -0.103, versus `training_rows` +0.041; `observed` 2026-09-16). Report the `distance_vs_*` figures from `support_nulls` and state that the rarity pathway is measurably near-closed, so an `intrinsic_difficulty` outcome would indicate difficulty arising from something other than training volume — motif degeneracy, binding promiscuity, or assay noise — none of which this study measures.
- Evidence labels per `CLAUDE.md`. The fitted slopes are `observed` with the artifact path as pointer. The interpretation is `claimed`.

- [ ] **Step 3: If the outcome is `intrinsic_difficulty` or `mixed`, correct the three shipped docs**

Add a short paragraph to the Limits section of each of `loao_allele_only.md`,
`loao_allele_peptide.md`, and `loao_cluster.md` stating that the degradation reported there
is confounded with intrinsic per-allele difficulty, with the measured control slope and a
link to `novelty_control.md`. Do not alter any existing number in those documents; the
numbers were verified against the artifacts and remain correct for their cohort. Only the
interpretation changes.

If the outcome is `novelty_effect`, add one sentence to each Limits section recording that
the novelty reading was tested against a no-novelty control and survived, with the link.

- [ ] **Step 4: Update `docs/next_candidates.md`**

Replace the "Briefed 2026-09-16 and reframed under review" passage in the
"Predictor-agnostic performance prediction" bullet with the outcome, the measured control
slope, and a link to `docs/pmhc/novelty_control.md`. Mark the candidate resolved.

- [ ] **Step 5: Run local CI and commit**

Run: `uv run pytest -q` then `uv run python -m compileall -q src scripts`

```bash
git add docs/pmhc/novelty_control.md docs/pmhc/loao_allele_only.md \
  docs/pmhc/loao_allele_peptide.md docs/pmhc/loao_cluster.md docs/next_candidates.md
git commit -m "document novelty control outcome, correct loao interpretation"
```

---

## Self-Review

**Spec coverage.** Brief's north star → Tasks 6–7. Discriminating prediction table → Task 1 Step 1 and Task 4's `classify_control_outcome`. "Earns agnostic" / `nearest_pwm` exclusion → Task 1 pre-declaration and the `REFERENCE_ARM` constant. Frozen-47 cohort → Task 2 Step 4's assertion. Covariate split → Task 1 and `support_nulls`. Freeze mechanism → Task 1 committed alone, verified in Task 6 Step 9. Open question 1 (TensorFlow) → isolated `uv run --no-project` throughout, `pyproject.toml` never touched. Open question 2 (slopes only) → Task 4 Step 1's offset-invariance test, proven able to fail in Step 5, plus Task 6's artifact test. Open question 3 (allele coverage) → Task 2, gated before any regression. Open question 4 (tolerance) → Task 1's interval-based rule with the pinned −1.0315. Non-goals are all enforced by the Global Constraints section.

**Placeholder scan.** No TBDs. Every code step carries runnable code. Task 7 Steps 2–3 prescribe document content rather than literal prose, which is correct — the numbers must come from the artifact, not from this plan.

**Type consistency.** `score_per_allele_auc01` returns `dict[str, float]`, consumed as such in Task 6. `fit_distance_slope` returns `DistanceSlope`, consumed by `classify_control_outcome` and serialized with `asdict`. `load_reference_table` produces the five columns Task 6 reads. `ControlOutcome`'s three literals match the pre-declaration table and both artifact tests.

**Resolved before execution.** An earlier revision assumed `Class1PresentationPredictor.predict` accepts a per-row `alleles` list. It does not — its `alleles` argument is a genotype list of up to six or a sample-name-to-alleles dict. Verified against the class source, `observed`. The plan now uses `Class1AffinityPredictor.predict(peptides, alleles)`, whose `alleles` "must be the same length as `peptides` and give the allele corresponding to each peptide" and which returns a numpy array of nM affinities. This is also the correct scientific match: this cohort's `Target` is a threshold on binding affinity. The `len(predicted) != len(scored)` assertion is retained regardless.
