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
    ALLELE_ONLY_LABEL,
    CLUSTER_HOLDOUT_LABEL,
    build_allele_partition,
    build_cluster_schedule,
    build_pseudo_sequence_clusters,
    preflight_partitions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train"

# Real 47-allele pseudo-sequence cohort, fixed here only as an offline test fixture so
# clustering logic can be tested without the hash-verified source archive present.
# Production code never hardcodes this — it trusts `verify_source`'s hash chain for
# content fidelity and only checks allele-set membership (see `pmhc_transfer.py`).
FROZEN_PSEUDO_SEQUENCES: dict[str, str] = {
    "HLA-A01:01": "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY",
    "HLA-A02:01": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY",
    "HLA-A02:02": "YFAMYGEKVAHTHVDTLYLRYHYYTWAVWAYTWY",
    "HLA-A02:03": "YFAMYGEKVAHTHVDTLYVRYHYYTWAEWAYTWY",
    "HLA-A02:06": "YYAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY",
    "HLA-A02:11": "YFAMYGEKVAHIDVDTLYVRYHYYTWAVLAYTWY",
    "HLA-A02:12": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVQAYTWY",
    "HLA-A02:16": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYEWY",
    "HLA-A02:19": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVQAYTGY",
    "HLA-A03:01": "YFAMYQENVAQTDVDTLYIIYRDYTWAELAYTWY",
    "HLA-A11:01": "YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY",
    "HLA-A23:01": "YSAMYEEKVAHTDENIAYLMFHYYTWAVLAYTGY",
    "HLA-A24:02": "YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTGY",
    "HLA-A24:03": "YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTWY",
    "HLA-A26:01": "YYAMYRNNVAHTDANTLYIRYQDYTWAEWAYRWY",
    "HLA-A26:02": "YYAMYRNNVAHTDANTLYIRYQNYTWAEWAYRWY",
    "HLA-A29:02": "YTAMYLQNVAQTDANTLYIMYRDYTWAVLAYTWY",
    "HLA-A30:01": "YSAMYQENVAQTDVDTLYIIYEHYTWAWLAYTWY",
    "HLA-A30:02": "YSAMYQENVAHTDENTLYIIYEHYTWARLAYTWY",
    "HLA-A31:01": "YTAMYQENVAHIDVDTLYIMYQDYTWAVLAYTWY",
    "HLA-A32:01": "YFAMYQENVAHTDESIAYIMYQDYTWAVLAYTWY",
    "HLA-A33:01": "YTAMYRNNVAHIDVDTLYIMYQDYTWAVLAYTWH",
    "HLA-A68:01": "YYAMYRNNVAQTDVDTLYIMYRDYTWAVWAYTWY",
    "HLA-A68:02": "YYAMYRNNVAQTDVDTLYIRYHYYTWAVWAYTWY",
    "HLA-A69:01": "YYAMYRNNVAQTDVDTLYVRYHYYTWAVLAYTWY",
    "HLA-A80:01": "YFAMYEENVAHTNANTLYIIYRDYTWARLAYEGY",
    "HLA-B07:02": "YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY",
    "HLA-B08:01": "YDSEYRNIFTNTDESNLYLSYNYYTWAVDAYTWY",
    "HLA-B15:01": "YYAMYREISTNTYESNLYLRYDSYTWAEWAYLWY",
    "HLA-B15:03": "YYSEYREISTNTYESNLYLRYDSYTWAELAYLWY",
    "HLA-B15:17": "YYAMYRENMASTYENIAYLRYHDYTWAELAYLWY",
    "HLA-B18:01": "YHSTYRNISTNTYESNLYLRYDSYTWAVLAYTWH",
    "HLA-B27:05": "YHTEYREICAKTDEDTLYLNYHDYTWAVLAYEWY",
    "HLA-B35:01": "YYATYRNIFTNTYESNLYIRYDSYTWAVLAYLWY",
    "HLA-B38:01": "YYSEYRNICTNTYENIAYLRYNFYTWAVLTYTWY",
    "HLA-B39:01": "YYSEYRNICTNTDESNLYLRYNFYTWAVLTYTWY",
    "HLA-B40:01": "YHTKYREISTNTYESNLYLRYNYYSLAVLAYEWY",
    "HLA-B40:02": "YHTKYREISTNTYESNLYLSYNYYTWAVLAYEWY",
    "HLA-B44:02": "YYTKYREISTNTYENTAYIRYDDYTWAVDAYLSY",
    "HLA-B44:03": "YYTKYREISTNTYENTAYIRYDDYTWAVLAYLSY",
    "HLA-B45:01": "YHTKYREISTNTYESNLYWRYNLYTWAVDAYLSY",
    "HLA-B51:01": "YYATYRNIFTNTYENIAYWTYNYYTWAELAYLWH",
    "HLA-B53:01": "YYATYRNIFTNTYENIAYIRYDSYTWAVLAYLWY",
    "HLA-B54:01": "YYAGYRNIYAQTDESNLYWTYNLYTWAVLAYTWY",
    "HLA-B57:01": "YYAMYGENMASTYENIAYIVYDSYTWAVLAYLWY",
    "HLA-B58:01": "YYATYGENMASTYENIAYIRYDSYTWAVLAYLWY",
    "HLA-C15:02": "YYAGYRENYRQTDVNKLYIRYDLYTWAELAYTWY",
}


def _load_runner_module() -> types.ModuleType:
    script_path = REPO_ROOT / "scripts" / "run_pmhc_loao_cluster.py"
    spec = importlib.util.spec_from_file_location("run_pmhc_loao_cluster", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cluster_rule_is_label_independent_and_has_one_singleton() -> None:
    expected_alleles = sorted(FROZEN_PSEUDO_SEQUENCES)
    clusters = build_pseudo_sequence_clusters(FROZEN_PSEUDO_SEQUENCES, expected_alleles)

    assert sum(len(cluster) == 1 for cluster in clusters) == 1
    assert sorted(map(len, clusters)) == [1, 2, 2, 3, 3, 3, 5, 6, 6, 8, 8]


def test_cluster_rule_rejects_a_mismatched_allele_set() -> None:
    expected_alleles = sorted(FROZEN_PSEUDO_SEQUENCES)
    corrupted = dict(FROZEN_PSEUDO_SEQUENCES)
    del corrupted["HLA-A01:01"]
    corrupted["HLA-Z99:99"] = corrupted["HLA-A02:01"]

    with pytest.raises(ValueError, match="does not cover exactly the expected"):
        build_pseudo_sequence_clusters(corrupted, expected_alleles)


def test_cluster_rule_rejects_an_incomplete_cohort() -> None:
    expected_alleles = sorted(FROZEN_PSEUDO_SEQUENCES)
    incomplete = dict(FROZEN_PSEUDO_SEQUENCES)
    del incomplete["HLA-A01:01"]

    with pytest.raises(ValueError, match="does not cover exactly the expected"):
        build_pseudo_sequence_clusters(incomplete, expected_alleles)


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
    expected_alleles = tuple(sorted(FROZEN_PSEUDO_SEQUENCES))
    clusters = build_pseudo_sequence_clusters(FROZEN_PSEUDO_SEQUENCES, expected_alleles)

    schedule = build_cluster_schedule(rows, FROZEN_PSEUDO_SEQUENCES, expected_alleles)

    assert tuple(partition.held_out_alleles for partition in schedule) == clusters
    preflight_partitions(schedule, FROZEN_PSEUDO_SEQUENCES, expected_alleles)


def test_cluster_partitions_carry_their_own_estimand_label() -> None:
    rows = _frozen_cohort_rows()
    expected_alleles = tuple(sorted(FROZEN_PSEUDO_SEQUENCES))

    schedule = build_cluster_schedule(rows, FROZEN_PSEUDO_SEQUENCES, expected_alleles)
    allele_only = build_allele_partition(rows, expected_alleles[0])

    assert {partition.label for partition in schedule} == {CLUSTER_HOLDOUT_LABEL}
    assert allele_only.label == ALLELE_ONLY_LABEL
    assert len({*(p.label for p in schedule), allele_only.label}) == 2
    preflight_partitions(schedule, FROZEN_PSEUDO_SEQUENCES, expected_alleles)


def test_clustering_reaches_the_maximum_hamming_cut() -> None:
    maximally_distant = {"HLA-A01:01": "Q" * 34, "HLA-A02:01": "W" * 34}

    clusters = build_pseudo_sequence_clusters(
        maximally_distant, tuple(sorted(maximally_distant))
    )

    assert clusters == (("HLA-A01:01", "HLA-A02:01"),)


def test_preflight_rejects_group_membership_leakage() -> None:
    rows = _frozen_cohort_rows()
    expected_alleles = tuple(sorted(FROZEN_PSEUDO_SEQUENCES))
    schedule = list(build_cluster_schedule(rows, FROZEN_PSEUDO_SEQUENCES, expected_alleles))
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
