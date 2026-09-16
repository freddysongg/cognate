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
