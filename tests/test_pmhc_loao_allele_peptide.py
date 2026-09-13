"""Joint allele-and-peptide novelty schedule and runner tests.

This track holds out one target allele and additionally removes from training
every row whose peptide occurs in that target's own test rows. It is a
compound intervention: the deletion diagnostics below describe altered
training-set composition, not an isolated peptide-novelty effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_pmhc_loao_allele_peptide as runner
from cognate import pmhc_transfer
from cognate.pmhc import PmhcDataset
from cognate.pmhc_transfer import build_joint_novelty_schedule


def _pseudo_sequences() -> dict[str, str]:
    return {
        "HLA-A01:01": "A" * 34,
        "HLA-A02:01": "C" + "A" * 33,
        "HLA-B07:02": "C" * 2 + "A" * 32,
        "HLA-C01:02": "C" * 3 + "A" * 31,
    }


def _rows() -> pd.DataFrame:
    records = [
        # "AAAAAAAAA" is shared between HLA-A01:01's positive test row and
        # HLA-A02:01's negative training row, so it is deleted from the other
        # allele's training partition whenever either becomes the target.
        ("HLA-A01:01", "AAAAAAAAA", True),
        ("HLA-A01:01", "DDDDDDDDD", True),
        ("HLA-A01:01", "EEEEEEEEE", True),
        ("HLA-A01:01", "FFFFFFFFF", False),
        ("HLA-A01:01", "GGGGGGGGG", False),
        ("HLA-A01:01", "HHHHHHHHH", False),
        ("HLA-A02:01", "AAAAAAAAA", False),
        ("HLA-A02:01", "IIIIIIIII", True),
        ("HLA-A02:01", "KKKKKKKKK", True),
        ("HLA-A02:01", "LLLLLLLLL", True),
        ("HLA-A02:01", "MMMMMMMMM", False),
        ("HLA-A02:01", "NNNNNNNNN", False),
        ("HLA-B07:02", "PPPPPPPPP", True),
        ("HLA-B07:02", "QQQQQQQQQ", True),
        ("HLA-B07:02", "RRRRRRRRR", True),
        ("HLA-B07:02", "SSSSSSSSS", False),
        ("HLA-B07:02", "TTTTTTTTT", False),
        ("HLA-B07:02", "VVVVVVVVV", False),
        ("HLA-C01:02", "WWWWWWWWW", True),
        ("HLA-C01:02", "YYYYYYYYY", True),
        ("HLA-C01:02", "ACDEFGHIK", True),
        ("HLA-C01:02", "LMNPQRSTV", False),
        ("HLA-C01:02", "WYACDEFGH", False),
        ("HLA-C01:02", "IKLMNPQRS", False),
    ]
    return pd.DataFrame(records, columns=["Allele", "Peptide", "Target"])


def test_joint_schedule_has_zero_target_and_peptide_overlap() -> None:
    rows = _rows()
    pseudo_sequences = _pseudo_sequences()

    schedule = build_joint_novelty_schedule(
        rows, sorted(pseudo_sequences), pseudo_sequences
    )

    for partition in schedule.partitions:
        assert set(partition.test["Allele"]).isdisjoint(partition.train["Allele"])
        assert set(partition.test["Peptide"]).isdisjoint(partition.train["Peptide"])
        assert partition.label == pmhc_transfer.JOINT_NOVELTY_LABEL


def test_joint_schedule_records_diagnostics_for_every_target() -> None:
    rows = _rows()
    pseudo_sequences = _pseudo_sequences()
    alleles = sorted(pseudo_sequences)

    schedule = build_joint_novelty_schedule(rows, alleles, pseudo_sequences)

    assert [diagnostic.target_allele for diagnostic in schedule.diagnostics] == alleles
    by_target = {diagnostic.target_allele: diagnostic for diagnostic in schedule.diagnostics}

    a01 = by_target["HLA-A01:01"]
    assert a01.deleted_training_rows == 1
    assert a01.retained_positive_rows == 9
    assert a01.retained_negative_rows == 8
    assert a01.retained_allele_count == 3
    assert a01.nearest_pwm_source == "HLA-A02:01"
    assert a01.nearest_pseudo_sequence_distance == pytest.approx(1 / 34)

    a02 = by_target["HLA-A02:01"]
    assert a02.deleted_training_rows == 1
    assert a02.retained_positive_rows == 8
    assert a02.retained_negative_rows == 9
    assert a02.retained_allele_count == 3
    assert a02.nearest_pwm_source == "HLA-A01:01"
    assert a02.nearest_pseudo_sequence_distance == pytest.approx(1 / 34)

    b07 = by_target["HLA-B07:02"]
    assert b07.deleted_training_rows == 0
    assert b07.retained_positive_rows == 9
    assert b07.retained_negative_rows == 9
    assert b07.retained_allele_count == 3
    assert b07.nearest_pwm_source == "HLA-A02:01"
    assert b07.nearest_pseudo_sequence_distance == pytest.approx(1 / 34)

    c01 = by_target["HLA-C01:02"]
    assert c01.deleted_training_rows == 0
    assert c01.retained_positive_rows == 9
    assert c01.retained_negative_rows == 9
    assert c01.retained_allele_count == 3
    assert c01.nearest_pwm_source == "HLA-B07:02"
    assert c01.nearest_pseudo_sequence_distance == pytest.approx(1 / 34)


def test_joint_schedule_rejects_peptide_leakage(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows()
    pseudo_sequences = _pseudo_sequences()
    alleles = sorted(pseudo_sequences)
    leaked_target = alleles[0]
    original_builder = pmhc_transfer.build_joint_novelty_partition

    def leaking_builder(rows: pd.DataFrame, target_allele: str) -> pmhc_transfer.TransferPartition:
        partition = original_builder(rows, target_allele)
        if target_allele != leaked_target:
            return partition
        leaked_train = partition.train.copy()
        leaked_train.loc[leaked_train.index[0], "Peptide"] = partition.test["Peptide"].iloc[0]
        return pmhc_transfer.TransferPartition(
            label=partition.label,
            held_out_alleles=partition.held_out_alleles,
            train=leaked_train,
            test=partition.test,
        )

    monkeypatch.setattr(pmhc_transfer, "build_joint_novelty_partition", leaking_builder)

    with pytest.raises(ValueError, match="test peptide leaked"):
        build_joint_novelty_schedule(rows, alleles, pseudo_sequences)


def test_joint_schedule_fails_entirely_when_any_target_loses_a_training_class() -> None:
    rows = pd.DataFrame(
        {
            "Allele": [
                "HLA-A01:01", "HLA-A01:01",
                "HLA-A02:01", "HLA-A02:01",
                "HLA-B07:02", "HLA-B07:02",
            ],
            "Peptide": [
                "AAAAAAAAA", "CCCCCCCCC",
                "AAAAAAAAA", "YYYYYYYYY",
                "AAAAAAAAA", "ZZZZZZZZZ",
            ],
            "Target": [True, False, True, False, True, False],
        }
    )
    pseudo_sequences = {
        "HLA-A01:01": "A" * 34,
        "HLA-A02:01": "C" + "A" * 33,
        "HLA-B07:02": "C" * 2 + "A" * 32,
    }

    with pytest.raises(ValueError, match="both target classes"):
        build_joint_novelty_schedule(rows, list(pseudo_sequences), pseudo_sequences)


def _synthetic_pseudo_sequence(index: int) -> str:
    """Return a deterministic 34-residue sequence unique for each of 47 indices."""
    residues = ["A"] * 34
    primary_position = index % 34
    residues[primary_position] = "C"
    if index >= 34:
        residues[(primary_position + 17) % 34] = "C"
    return "".join(residues)


def _synthetic_dataset(n_alleles: int = 47) -> PmhcDataset:
    alleles = tuple(f"HLA-A{index:02d}:01" for index in range(n_alleles))
    records = []
    for allele in alleles:
        records.append((allele, f"{allele}POS", True))
        records.append((allele, f"{allele}NEG", False))
    rows = pd.DataFrame(records, columns=["Allele", "Peptide", "Target"])
    return PmhcDataset(all_nine_mer_hla_rows=rows, rows=rows, eligible_alleles=alleles)


def test_joint_runner_records_retention_for_each_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _synthetic_dataset()
    pseudo_sequences = {
        allele: _synthetic_pseudo_sequence(index)
        for index, allele in enumerate(dataset.eligible_alleles)
    }

    def fake_evaluate_transfer_schedule(
        partitions: object,
        pseudo_sequences_arg: object,
        expected_target_alleles: object,
        *,
        device: str,
    ) -> dict[str, object]:
        assert device == "cpu"
        assert tuple(sorted(expected_target_alleles)) == dataset.eligible_alleles
        return {"config": {}, "scores": {}, "comparisons": {}, "per_allele": {}}

    monkeypatch.setattr(runner, "load_pmhc_dataset", lambda source_dir: dataset)
    monkeypatch.setattr(
        runner, "load_pseudo_sequences", lambda path, alleles: pseudo_sequences
    )
    monkeypatch.setattr(
        runner, "evaluate_transfer_schedule", fake_evaluate_transfer_schedule
    )

    output_path = tmp_path / "result.json"
    artifact = runner.run_joint_novelty(tmp_path / "source", output_path, device="cpu")

    assert len(artifact["diagnostics"]["targets"]) == 47
    assert artifact["estimand"] == "joint_allele_and_peptide_novelty"
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == artifact


def test_joint_runner_rejects_a_cohort_that_is_not_47_alleles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _synthetic_dataset(n_alleles=5)
    monkeypatch.setattr(runner, "load_pmhc_dataset", lambda source_dir: dataset)

    with pytest.raises(ValueError, match="47 target alleles"):
        runner.run_joint_novelty(tmp_path / "source", tmp_path / "result.json", device="cpu")
