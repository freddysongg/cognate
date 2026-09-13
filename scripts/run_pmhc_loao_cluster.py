"""Run the frozen pseudo-sequence-cluster-held-out pMHC transfer evaluation.

This is cluster-held-out transfer under a frozen pseudo-sequence grouping. It is
NOT a causal claim that sequence distance alone causes any performance difference:
holding out a whole cluster also changes training-set composition (retained allele
count, class support), so any observed difference is confounded with that change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from cognate.pmhc import (
    load_pmhc_dataset,
    load_pseudo_sequences,
    load_source_contract,
    verify_source,
)
from cognate.pmhc_transfer import (
    TransferPartition,
    build_cluster_schedule,
    evaluate_transfer_schedule,
    validate_transfer_result,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT_PATH = ROOT / "data" / "pmhc" / "source_contract.json"
ARCHIVE_PATH = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train.tar.gz"
SOURCE_DIR = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train"
RESULTS_PATH = ROOT / "data" / "pmhc" / "loao_cluster_results.json"

EXPECTED_CLUSTER_SIZES = [1, 2, 2, 3, 3, 3, 5, 6, 6, 8, 8]
CUT_RULE = (
    "smallest integer Hamming threshold (raw mismatch count over the 34 "
    "pseudo-sequence positions) that leaves at most one singleton cluster under "
    "complete-linkage agglomeration with sorted-allele tie-breaking"
)
INTERPRETATION = (
    "This is cluster-held-out transfer under a frozen pseudo-sequence grouping, not "
    "a causal claim that sequence distance alone causes any performance difference. "
    "Holding out a whole cluster changes training-set composition (retained allele "
    "count and class support) at the same time as distance, so the two are "
    "confounded. The nearest-PWM comparison is a sequence-neighbour baseline, not a "
    "non-transfer control."
)
ARTIFACT_KEYS = {
    "contract",
    "config",
    "scores",
    "comparisons",
    "per_allele",
    "diagnostics",
    "criteria",
}


def _hamming(sequence_a: str, sequence_b: str) -> int:
    return sum(
        residue_a != residue_b
        for residue_a, residue_b in zip(sequence_a, sequence_b, strict=True)
    )


def _cluster_composition(partition: TransferPartition) -> dict[str, object]:
    train_targets = partition.train["Target"]
    return {
        "held_out_alleles": list(partition.held_out_alleles),
        "n_retained_alleles": len(set(partition.train["Allele"].astype(str))),
        "n_train_rows": int(len(partition.train)),
        "n_train_positive": int(train_targets.sum()),
        "n_train_negative": int((~train_targets).sum()),
        "n_test_rows": int(len(partition.test)),
    }


def _nearest_retained_distance(
    schedule: Sequence[TransferPartition], pseudo_sequences: Mapping[str, str]
) -> dict[str, int]:
    """Per held-out allele, the Hamming distance to its nearest retained allele."""
    distances: dict[str, int] = {}
    for partition in schedule:
        retained_alleles = sorted(set(partition.train["Allele"].astype(str)))
        for target_allele in partition.held_out_alleles:
            distances[target_allele] = min(
                _hamming(pseudo_sequences[target_allele], pseudo_sequences[retained])
                for retained in retained_alleles
            )
    return distances


def build_diagnostics(
    schedule: Sequence[TransferPartition], pseudo_sequences: Mapping[str, str]
) -> dict[str, object]:
    """Record exact cluster membership, the cut rule, and per-group composition."""
    clusters = [list(partition.held_out_alleles) for partition in schedule]
    sizes = sorted(len(cluster) for cluster in clusters)
    if sizes != EXPECTED_CLUSTER_SIZES:
        raise ValueError(
            f"cluster sizes {sizes} differ from the frozen {EXPECTED_CLUSTER_SIZES}"
        )
    cut_hamming_distance = max(
        _hamming(pseudo_sequences[allele_a], pseudo_sequences[allele_b])
        for cluster in clusters
        for allele_a in cluster
        for allele_b in cluster
        if allele_a != allele_b
    )
    return {
        "cut_rule": CUT_RULE,
        "cut_hamming_distance": cut_hamming_distance,
        "clusters": clusters,
        "cluster_composition": [
            _cluster_composition(partition) for partition in schedule
        ],
        "nearest_retained_distance": dict(
            sorted(_nearest_retained_distance(schedule, pseudo_sequences).items())
        ),
    }


def derive_cluster_criteria(comparisons: Mapping[str, object]) -> dict[str, object]:
    """Report the correct-mapping comparison bounds under the cluster estimand.

    Uses the same paired-lower-bound structure as the allele-only track's
    correct-mapping criterion, but this is a distinct estimand (cluster-held-out,
    not single-allele-held-out) and this function makes no causal-distance claim.
    """
    lower_bounds = {
        reference: float(
            comparisons[f"pseudo_sequence_mlp_minus_{reference}"]["difference"]["lo"]
        )
        for reference in ("peptide_only_mlp", "shuffled_mapping_mlp", "nearest_pwm")
    }
    return {
        "cluster_transfer_supported": (
            lower_bounds["peptide_only_mlp"] > 0
            and lower_bounds["shuffled_mapping_mlp"] > 0
        ),
        "lower_bounds": {
            "pseudo_sequence_mlp_minus_peptide_only_mlp": lower_bounds[
                "peptide_only_mlp"
            ],
            "pseudo_sequence_mlp_minus_shuffled_mapping_mlp": lower_bounds[
                "shuffled_mapping_mlp"
            ],
            "pseudo_sequence_mlp_minus_nearest_pwm_secondary": lower_bounds[
                "nearest_pwm"
            ],
        },
        "interpretation": INTERPRETATION,
    }


def run_cluster_holdout(source_dir: Path, output_path: Path, *, device: str) -> dict[str, object]:
    """Run the frozen cluster-held-out schedule and write the aggregate artifact."""
    contract_path = source_dir.parent.parent / "source_contract.json"
    archive_path = source_dir.parent / "NetMHCpan_train.tar.gz"
    contract = load_source_contract(contract_path)
    verify_source(source_dir, archive_path, contract)

    dataset = load_pmhc_dataset(source_dir)
    pseudo_sequences = load_pseudo_sequences(
        source_dir / "MHC_pseudo.dat", dataset.eligible_alleles
    )
    schedule = build_cluster_schedule(dataset.rows, pseudo_sequences)
    expected_alleles = tuple(sorted(dataset.eligible_alleles))

    core = evaluate_transfer_schedule(
        schedule, pseudo_sequences, expected_alleles, device=device
    )
    artifact = {
        "contract": contract,
        **core,
        "diagnostics": build_diagnostics(schedule, pseudo_sequences),
        "criteria": derive_cluster_criteria(core["comparisons"]),
    }
    if set(artifact) != ARTIFACT_KEYS:
        raise AssertionError("cluster result artifact top-level contract drifted")
    validate_transfer_result(
        {key: artifact[key] for key in ("config", "scores", "comparisons", "per_allele")},
        expected_alleles,
    )

    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    torch.set_num_threads(1)
    artifact = run_cluster_holdout(SOURCE_DIR, RESULTS_PATH, device="cpu")
    print(
        f"Wrote {len(artifact['diagnostics']['clusters'])} held-out cluster groups "
        f"covering {len(artifact['per_allele'])} alleles to {RESULTS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
