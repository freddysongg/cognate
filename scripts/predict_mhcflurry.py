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
    from mhcflurry.common import normalize_allele_name

    cohort_alleles = json.loads(COHORT_ALLELES_PATH.read_text(encoding="utf-8"))
    predictor = Class1AffinityPredictor.load()
    known = set(predictor.supported_alleles)
    supported = sorted(
        allele for allele in cohort_alleles if normalize_allele_name(allele) in known
    )
    unsupported = sorted(
        allele for allele in cohort_alleles if normalize_allele_name(allele) not in known
    )
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
