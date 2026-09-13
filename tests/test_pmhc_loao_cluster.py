from __future__ import annotations

import importlib.util
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognate import metrics, pmhc_transfer
from cognate.pmhc_transfer import (
    FROZEN_PSEUDO_SEQUENCES,
    build_cluster_schedule,
    build_pseudo_sequence_clusters,
    preflight_partitions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train"


def _load_runner_module() -> types.ModuleType:
    script_path = REPO_ROOT / "scripts" / "run_pmhc_loao_cluster.py"
    spec = importlib.util.spec_from_file_location("run_pmhc_loao_cluster", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cluster_rule_is_label_independent_and_has_one_singleton() -> None:
    clusters = build_pseudo_sequence_clusters(FROZEN_PSEUDO_SEQUENCES)

    assert sum(len(cluster) == 1 for cluster in clusters) == 1
    assert sorted(map(len, clusters)) == [1, 2, 2, 3, 3, 3, 5, 6, 6, 8, 8]


def test_cluster_rule_rejects_a_mapping_that_is_not_exactly_the_frozen_cohort() -> None:
    corrupted = dict(FROZEN_PSEUDO_SEQUENCES)
    corrupted["HLA-A01:01"] = "A" * len(corrupted["HLA-A01:01"])

    with pytest.raises(ValueError, match="frozen 47-allele cohort"):
        build_pseudo_sequence_clusters(corrupted)


def test_cluster_rule_rejects_an_incomplete_cohort() -> None:
    incomplete = dict(FROZEN_PSEUDO_SEQUENCES)
    del incomplete["HLA-A01:01"]

    with pytest.raises(ValueError, match="frozen 47-allele cohort"):
        build_pseudo_sequence_clusters(incomplete)


def _frozen_cohort_rows() -> pd.DataFrame:
    """Two rows per frozen-cohort allele, one of each target class, unique peptides."""
    alleles = sorted(FROZEN_PSEUDO_SEQUENCES)
    return pd.DataFrame(
        {
            "Allele": [allele for allele in alleles for _ in (True, False)],
            "Peptide": [
                f"PEP{index:04d}" for index in range(2 * len(alleles))
            ],
            "Target": [is_positive for _ in alleles for is_positive in (True, False)],
            "Affinity": [
                affinity for _ in alleles for affinity in (0.8, 0.2)
            ],
        }
    )


def test_cluster_schedule_groups_match_the_frozen_clusters_and_preflight_cleanly() -> None:
    rows = _frozen_cohort_rows()
    clusters = build_pseudo_sequence_clusters(FROZEN_PSEUDO_SEQUENCES)

    schedule = build_cluster_schedule(rows, FROZEN_PSEUDO_SEQUENCES)

    assert tuple(partition.held_out_alleles for partition in schedule) == clusters
    preflight_partitions(
        schedule, FROZEN_PSEUDO_SEQUENCES, tuple(sorted(FROZEN_PSEUDO_SEQUENCES))
    )


def test_preflight_rejects_group_membership_leakage() -> None:
    rows = _frozen_cohort_rows()
    schedule = list(build_cluster_schedule(rows, FROZEN_PSEUDO_SEQUENCES))
    leaked_partition = replace(
        schedule[0],
        train=pd.concat(
            [schedule[0].train, schedule[0].test.iloc[[0]]], ignore_index=True
        ),
    )
    schedule[0] = leaked_partition

    with pytest.raises(ValueError, match="held-out allele leaked"):
        preflight_partitions(
            schedule, FROZEN_PSEUDO_SEQUENCES, tuple(sorted(FROZEN_PSEUDO_SEQUENCES))
        )


def test_cluster_runner_records_membership_and_per_allele_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fixed_scores(
        partition: pmhc_transfer.TransferPartition,
        pseudo_sequences: object,
        *,
        device: str,
    ) -> dict[str, np.ndarray]:
        assert device == "cpu"
        targets = partition.test["Target"].to_numpy(dtype=float)
        scores = {
            arm: targets if arm == "pseudo_sequence_mlp" else 1 - targets
            for arm in pmhc_transfer.TRANSFER_SCORE_NAMES
        }
        scores["random"] = np.random.default_rng(0).random(len(partition.test))
        return scores

    def evaluate_small_bootstrap(
        targets: np.ndarray, scores: np.ndarray, **kwargs: object
    ) -> metrics.ScoreReport:
        return metrics.evaluate(targets, scores, **{**kwargs, "n_boot": 2})

    def compare_small_bootstrap(
        targets: np.ndarray,
        alleles: np.ndarray,
        arm: tuple[str, np.ndarray, np.ndarray | None],
        reference: tuple[str, np.ndarray, np.ndarray | None],
        **kwargs: object,
    ) -> metrics.Comparison:
        return metrics.compare_macro_auc01(
            targets, alleles, arm, reference, **{**kwargs, "n_boot": 2}
        )

    monkeypatch.setattr(pmhc_transfer, "score_transfer_partition", fixed_scores)
    monkeypatch.setattr(pmhc_transfer, "evaluate", evaluate_small_bootstrap)
    monkeypatch.setattr(pmhc_transfer, "compare_macro_auc01", compare_small_bootstrap)

    runner = _load_runner_module()
    artifact = runner.run_cluster_holdout(
        SOURCE_DIR, tmp_path / "result.json", device="cpu"
    )

    assert len(artifact["diagnostics"]["clusters"]) == 11
    assert len(artifact["per_allele"]) == 47
    assert (tmp_path / "result.json").exists()
