"""Run the joint allele-and-peptide novelty LOAO track (#33).

Every target allele is held out and every training row whose peptide occurs
in that target's own test rows is also removed. This is a compound
intervention: it is never a pure allele-transfer result and never an
isolated peptide-novelty effect, because the peptide exclusion also changes
training-set composition by a target-varying amount. The deletion
diagnostics recorded per target make that composition change auditable.

This track reuses the shared LOAO foundation's five fixed arms, target-only
validation split, headline metric, bootstrap configuration, score guards, and
paired comparisons unchanged from the allele-only track (#32); only the
joint-exclusion schedule and its diagnostics are track-specific.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from cognate.pmhc import (
    load_pmhc_dataset,
    load_pseudo_sequences,
    load_source_contract,
    verify_source,
)
from cognate.pmhc_transfer import build_joint_novelty_schedule, evaluate_transfer_schedule

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT_PATH = ROOT / "data" / "pmhc" / "source_contract.json"
ARCHIVE_PATH = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train.tar.gz"
SOURCE_DIR = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train"
RESULTS_PATH = ROOT / "data" / "pmhc" / "loao_allele_peptide_results.json"

EXPECTED_TARGET_ALLELE_COUNT = 47
ESTIMAND = "joint_allele_and_peptide_novelty"
CLAIM_BOUNDARY = (
    "Joint novelty of allele and peptide under the frozen joint-exclusion rule. "
    "This is not a pure allele-transfer result and not an isolated "
    "peptide-novelty effect: removing every peptide that occurs in a target's "
    "own test rows also changes training-set composition by a target-varying "
    "amount. See diagnostics.targets for the per-target deletion accounting "
    "before drawing any conclusion from the scores below."
)
_RESULT_CORE_FIELDS = ("config", "scores", "comparisons", "per_allele")


def _require_full_cohort(eligible_alleles: tuple[str, ...]) -> None:
    """Fail the whole track rather than silently narrowing the 47-allele cohort."""
    if len(eligible_alleles) != EXPECTED_TARGET_ALLELE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TARGET_ALLELE_COUNT} target alleles, "
            f"got {len(eligible_alleles)}"
        )


def run_joint_novelty(source_dir: Path, output_path: Path, *, device: str) -> dict[str, object]:
    """Run the joint allele-and-peptide novelty track and write its aggregate artifact."""
    dataset = load_pmhc_dataset(source_dir)
    _require_full_cohort(dataset.eligible_alleles)
    pseudo_sequences = load_pseudo_sequences(
        source_dir / "MHC_pseudo.dat", dataset.eligible_alleles
    )
    schedule = build_joint_novelty_schedule(
        dataset.rows, dataset.eligible_alleles, pseudo_sequences
    )
    result_core = evaluate_transfer_schedule(
        schedule.partitions, pseudo_sequences, dataset.eligible_alleles, device=device
    )
    if tuple(result_core) != _RESULT_CORE_FIELDS:
        raise AssertionError("transfer result core fields differ from the shared contract")
    artifact = {
        "estimand": ESTIMAND,
        "claim_boundary": CLAIM_BOUNDARY,
        "diagnostics": {
            "targets": [asdict(target) for target in schedule.diagnostics],
        },
        **result_core,
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    """Verify the source, then run and write the joint-novelty artifact.

    Single-threaded intra-op execution measured faster than the default thread
    count for these small per-fit MLP tensors on this machine; process-level
    parallelism across targets measured no better or worse, so this stays a
    plain sequential loop.
    """
    torch.set_num_threads(1)
    contract = load_source_contract(SOURCE_CONTRACT_PATH)
    verify_source(SOURCE_DIR, ARCHIVE_PATH, contract)
    artifact = run_joint_novelty(SOURCE_DIR, RESULTS_PATH, device="cpu")
    n_targets = len(artifact["diagnostics"]["targets"])
    print(
        f"Wrote joint allele-and-peptide novelty results for {n_targets} targets "
        f"to {RESULTS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
