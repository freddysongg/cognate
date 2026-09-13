from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cognate import metrics, pmhc, pmhc_transfer

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_pmhc_loao_allele", Path(__file__).parents[1] / "scripts" / "run_pmhc_loao_allele.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
run_pmhc_loao_allele = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = run_pmhc_loao_allele
RUNNER_SPEC.loader.exec_module(run_pmhc_loao_allele)


def _cohort() -> tuple[pd.DataFrame, tuple[str, ...], dict[str, str]]:
    alleles = tuple(f"HLA-A{index:02d}:01" for index in range(47))
    rows = pd.DataFrame(
        [
            {
                "Allele": allele,
                "Peptide": "AAAAAAAAA" if index == 0 else f"{allele}-{index}",
                "Target": index % 2 == 0,
                "Affinity": 0.8 if index % 2 == 0 else 0.2,
            }
            for allele in alleles
            for index in range(4)
        ]
    )
    sequences = {
        allele: "".join("C" if index & (1 << bit) else "A" for bit in range(34))
        for index, allele in enumerate(alleles)
    }
    return rows, alleles, sequences


def test_allele_only_schedule_has_47_zero_target_leakage_partitions() -> None:
    rows, alleles, sequences = _cohort()
    original = rows.copy(deep=True)

    schedule = pmhc_transfer.build_allele_only_schedule(rows, alleles)

    assert len(schedule) == 47
    assert tuple(partition.held_out_alleles[0] for partition in schedule) == alleles
    for partition in schedule:
        assert set(partition.test["Allele"]) == set(partition.held_out_alleles)
        assert not set(partition.train["Allele"]) & set(partition.held_out_alleles)
        assert len(partition.train) + len(partition.test) == len(rows)
        assert "AAAAAAAAA" in set(partition.train["Peptide"])
    pmhc_transfer.preflight_partitions(schedule, sequences, alleles)
    pd.testing.assert_frame_equal(rows, original)


@pytest.mark.parametrize("alleles", [(), ("HLA-A00:01", "HLA-A00:01")])
def test_schedule_rejects_empty_or_duplicate_targets(alleles: tuple[str, ...]) -> None:
    rows, _, _ = _cohort()
    with pytest.raises(ValueError):
        pmhc_transfer.build_allele_only_schedule(rows, alleles)


def test_preflight_rejects_deliberate_target_allele_leakage() -> None:
    rows, alleles, sequences = _cohort()
    schedule = pmhc_transfer.build_allele_only_schedule(rows, alleles)
    leaked = replace(
        schedule[0], train=pd.concat([schedule[0].train, schedule[0].test.iloc[:1]])
    )
    with pytest.raises(ValueError, match="held-out allele leaked"):
        pmhc_transfer.preflight_partitions((leaked, *schedule[1:]), sequences, alleles)


def test_diagnostics_count_unique_peptides_and_rows_without_exclusion() -> None:
    rows, alleles, sequences = _cohort()
    rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    schedule = pmhc_transfer.build_allele_only_schedule(rows, alleles)

    diagnostics = pmhc_transfer.build_allele_only_diagnostics(schedule, sequences)

    assert set(diagnostics) == set(alleles)
    target = diagnostics[alleles[0]]
    assert target["peptide_overlap_count"] == 1
    assert target["peptide_overlap_fraction"] == pytest.approx(1 / 4)
    assert target["test_unique_peptides"] == 4
    assert target["test_rows_with_training_peptide"] == 2
    assert target["test_row_overlap_fraction"] == pytest.approx(2 / 5)
    assert target["nearest_retained_pseudo_distance"] == pytest.approx(1 / 34)
    assert target["selected_pwm_source"] == alleles[1]
    assert target["training_rows"] == 184


def test_nearest_retained_distance_does_not_require_pwm_eligibility() -> None:
    rows = pd.DataFrame(
        {
            "Allele": ["target", "target", "near", "near", "far", "far"],
            "Peptide": ["A", "B", "C", "D", "E", "F"],
            "Target": [True, False, True, True, True, False],
        }
    )
    sequences = {"target": "A" * 34, "near": "C" + "A" * 33, "far": "C" * 34}
    schedule = pmhc_transfer.build_allele_only_schedule(rows, ("target",))

    diagnostics = pmhc_transfer.build_allele_only_diagnostics(schedule, sequences)

    assert diagnostics["target"]["nearest_retained_pseudo_distance"] == pytest.approx(1 / 34)
    assert diagnostics["target"]["selected_pwm_source"] == "far"
    assert diagnostics["target"]["selected_pwm_source_distance"] == 1.0


def test_runner_writes_all_required_allele_only_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, alleles, sequences = _cohort()

    def fixed_scores(
        partition: pmhc_transfer.TransferPartition,
        pseudo_sequences: dict[str, str],
        *,
        device: str,
    ) -> dict[str, object]:
        assert device == "cpu"
        targets = partition.test["Target"].to_numpy(dtype=float)
        return {
            arm: targets if arm == "pseudo_sequence_mlp" else 1 - targets
            for arm in pmhc_transfer.TRANSFER_SCORE_NAMES
        }

    def small_bootstrap_evaluate(*args: object, **kwargs: object) -> object:
        return metrics.evaluate(*args, **{**kwargs, "n_boot": 2})

    def small_bootstrap_compare(*args: object, **kwargs: object) -> object:
        return metrics.compare_macro_auc01(*args, **{**kwargs, "n_boot": 2})

    monkeypatch.setattr(run_pmhc_loao_allele, "load_source_contract", lambda path: {})
    monkeypatch.setattr(run_pmhc_loao_allele, "verify_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_pmhc_loao_allele,
        "load_pmhc_dataset",
        lambda source_dir: pmhc.PmhcDataset(
            all_nine_mer_hla_rows=rows, rows=rows, eligible_alleles=alleles
        ),
    )
    monkeypatch.setattr(
        run_pmhc_loao_allele,
        "load_pseudo_sequences",
        lambda path, target_alleles: sequences,
    )
    monkeypatch.setattr(pmhc_transfer, "score_transfer_partition", fixed_scores)
    monkeypatch.setattr(pmhc_transfer, "evaluate", small_bootstrap_evaluate)
    monkeypatch.setattr(pmhc_transfer, "compare_macro_auc01", small_bootstrap_compare)

    output_path = tmp_path / "result.json"
    artifact = run_pmhc_loao_allele.run_allele_only(tmp_path, output_path, device="cpu")

    assert set(artifact) >= {"config", "scores", "comparisons", "per_allele", "diagnostics"}
    assert set(artifact["diagnostics"]) == set(alleles)
    assert artifact["criteria"]["primary_criterion_met"] is True
    assert output_path.exists()
    restored = json.loads(output_path.read_text(encoding="utf-8"))
    assert restored == artifact
