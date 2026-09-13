"""Run the fixed allele-only leave-one-allele-out (LOAO) transfer evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from cognate.pmhc import load_pmhc_dataset, load_pseudo_sequences, load_source_contract, verify_source
from cognate.pmhc_transfer import (
    build_allele_only_diagnostics,
    build_allele_only_schedule,
    evaluate_transfer_schedule,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT_PATH = ROOT / "data" / "pmhc" / "source_contract.json"
ARCHIVE_PATH = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train.tar.gz"
SOURCE_DIR = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train"
RESULTS_PATH = ROOT / "data" / "pmhc" / "loao_allele_only_results.json"

PRIMARY_COMPARISONS = (
    "pseudo_sequence_mlp_minus_peptide_only_mlp",
    "pseudo_sequence_mlp_minus_shuffled_mapping_mlp",
)
EXPECTED_ARTIFACT_KEYS = frozenset(
    {"config", "scores", "comparisons", "per_allele", "diagnostics", "criteria"}
)


def derive_primary_criterion(
    comparisons: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Declare the primary criterion true only when both required lower bounds exceed zero."""
    lower_bounds = {
        comparison: float(comparisons[comparison]["difference"]["lo"])
        for comparison in PRIMARY_COMPARISONS
    }
    return {
        "primary_criterion_met": all(bound > 0 for bound in lower_bounds.values()),
        "lower_bounds": lower_bounds,
    }


def run_allele_only(source_dir: Path, output_path: Path, *, device: str) -> dict[str, object]:
    """Score the fixed 47-allele allele-only LOAO schedule and write its aggregate artifact."""
    contract = load_source_contract(SOURCE_CONTRACT_PATH)
    verify_source(source_dir, ARCHIVE_PATH, contract)
    dataset = load_pmhc_dataset(source_dir)
    target_alleles = dataset.eligible_alleles
    pseudo_sequences = load_pseudo_sequences(source_dir / "MHC_pseudo.dat", target_alleles)
    schedule = build_allele_only_schedule(dataset.rows, target_alleles)
    result = evaluate_transfer_schedule(schedule, pseudo_sequences, target_alleles, device=device)
    artifact = {
        **result,
        "diagnostics": build_allele_only_diagnostics(schedule, pseudo_sequences),
        "criteria": derive_primary_criterion(result["comparisons"]),
    }
    if set(artifact) != EXPECTED_ARTIFACT_KEYS:
        raise AssertionError("allele-only result artifact top-level contract drifted")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    torch.set_num_threads(1)
    run_allele_only(SOURCE_DIR, RESULTS_PATH, device="cpu")
    print(f"Wrote aggregate allele-only results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
