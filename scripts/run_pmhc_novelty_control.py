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
REFERENCE_SLOPE_DRIFT_TOLERANCE = 0.15
CONTROL_COMPETENCE_FLOOR = 0.6


def _assert_control_is_skillful(control_auc: dict[str, float]) -> None:
    """Guard against the favourable outcome's own failure mode.

    `novelty_effect` fires whenever the control's slope is flat and separable from the
    reference; a control that is flat because it is broken (wrong alleles, shuffled peptides,
    an inverted sign) yields the identical verdict. This floor only asserts the control beats
    chance on this cohort — it never compares the control's absolute AUC0.1 to another arm's,
    so it does not breach the pre-declaration's slopes-only comparison rule.
    """
    incompetent = {
        allele: auc01
        for allele, auc01 in control_auc.items()
        if auc01 <= CONTROL_COMPETENCE_FLOOR
    }
    if incompetent:
        raise AssertionError(
            f"control AUC0.1 at or below the competence floor {CONTROL_COMPETENCE_FLOOR} for "
            f"{sorted(incompetent)}; a control this flat could be broken rather than merely "
            "novelty-free, and its flatness would not distinguish the two"
        )


def main() -> None:
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    supported = set(coverage["supported"])
    reference = load_reference_table(ALLELE_ONLY_RESULTS, REFERENCE_ARM)
    primary = reference.loc[reference["Allele"].isin(supported)].reset_index(drop=True)

    reference_fit = fit_distance_slope(primary["Distance"], primary["Auc01"])
    if abs(reference_fit.slope - REFERENCE_SLOPE) > REFERENCE_SLOPE_DRIFT_TOLERANCE:
        raise AssertionError(
            f"reference slope on supported alleles is {reference_fit.slope:.4f}, "
            f"far from the pinned {REFERENCE_SLOPE}; investigate before interpreting"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    control_auc = score_per_allele_auc01(predictions)
    _assert_control_is_skillful(control_auc)
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
        "control": {
            **asdict(control_fit),
            "competence_floor": CONTROL_COMPETENCE_FLOOR,
            "per_allele_auc01": {
                allele: control_auc[allele] for allele in primary["Allele"]
            },
        },
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
