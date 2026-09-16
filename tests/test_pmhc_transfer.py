from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cognate import metrics, pmhc, pmhc_transfer
from cognate.pmhc_transfer import (
    TransferPartition,
    build_allele_partition,
    build_joint_novelty_partition,
    build_transfer_partition,
    build_shuffled_pseudo_mapping,
    preflight_partitions,
    select_nearest_pwm_source,
    split_transfer_fit_validation,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Allele": [
                "HLA-A02:01",
                "HLA-A02:01",
                "HLA-A01:01",
                "HLA-A01:01",
                "HLA-A01:01",
                "HLA-A01:01",
            ],
            "Peptide": [
                "AAAAAAAAA",
                "CCCCCCCCC",
                "AAAAAAAAA",
                "CCCCCCCCC",
                "DDDDDDDDD",
                "EEEEEEEEE",
            ],
            "Target": [True, False, True, False, True, False],
        }
    )


def test_joint_partition_removes_target_rows_and_test_peptides() -> None:
    rows = _rows()

    partition = build_joint_novelty_partition(rows, "HLA-A02:01")

    assert "HLA-A02:01" not in set(partition.train["Allele"])
    assert set(partition.test["Peptide"]).isdisjoint(partition.train["Peptide"])


def test_allele_partition_holds_out_only_the_target_allele() -> None:
    partition = build_allele_partition(_rows(), "HLA-A02:01")

    assert partition.held_out_alleles == ("HLA-A02:01",)
    assert set(partition.test["Allele"]) == {"HLA-A02:01"}
    assert set(partition.test["Peptide"]) & set(partition.train["Peptide"])


def test_partition_does_not_mutate_source_rows() -> None:
    rows = _rows()
    expected_rows = rows.copy(deep=True)

    partition = build_joint_novelty_partition(rows, "HLA-A02:01")
    partition.train.loc[:, "Peptide"] = "FFFFFFFFF"

    pd.testing.assert_frame_equal(rows, expected_rows)


def test_partition_rejects_a_test_set_without_both_classes() -> None:
    rows = _rows().loc[lambda frame: frame["Peptide"] != "CCCCCCCCC"]

    with pytest.raises(ValueError, match="both target classes"):
        build_allele_partition(rows, "HLA-A02:01")


def test_partition_rejects_an_empty_train_partition() -> None:
    rows = _rows().loc[lambda frame: frame["Allele"] == "HLA-A02:01"]

    with pytest.raises(ValueError, match="training partition is empty"):
        build_allele_partition(rows, "HLA-A02:01")


def test_partition_rejects_an_empty_test_partition() -> None:
    with pytest.raises(ValueError, match="test partition is empty"):
        build_transfer_partition(
            _rows(), ("HLA-A03:01",), exclude_test_peptides=False
        )


def _pseudo_sequences() -> dict[str, str]:
    return {
        "HLA-A01:01": "C" + "A" * 33,
        "HLA-A02:01": "A" + "C" + "A" * 32,
        "HLA-B07:02": "C" * 2 + "A" * 32,
    }


def _transfer_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Allele": [
                "HLA-A01:01",
                "HLA-A01:01",
                "HLA-A02:01",
                "HLA-A02:01",
                "HLA-B07:02",
                "HLA-B07:02",
            ],
            "Peptide": [
                "AAAAAAAAA",
                "CCCCCCCCC",
                "DDDDDDDDD",
                "EEEEEEEEE",
                "FFFFFFFFF",
                "GGGGGGGGG",
            ],
            "Target": [True, False, True, False, True, False],
        }
    )


def _validation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Allele": ["HLA-A01:01"] * 10 + ["HLA-A02:01"] * 10,
            "Peptide": [f"PEPTIDE{index:02d}" for index in range(20)],
            "Target": [False] * 9 + [True, False] + [True] * 9,
        }
    )


def _sufficient_transfer_rows() -> pd.DataFrame:
    rows: list[dict[str, str | bool]] = []
    for allele_index, allele in enumerate(_pseudo_sequences()):
        for row_index in range(10):
            rows.append(
                {
                    "Allele": allele,
                    "Peptide": "AAAAAAAA" + "ACDEFGHIKL"[row_index],
                    "Target": row_index % 2 == 0,
                    "Affinity": 0.8 if row_index % 2 == 0 else 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_transfer_validation_split_is_target_stratified() -> None:
    train = _validation_rows()

    fit_indices, validation_indices = split_transfer_fit_validation(train)

    assert len(fit_indices) == 18
    assert len(validation_indices) == 2
    assert set(train.iloc[fit_indices]["Target"]) == {False, True}
    assert set(train.iloc[validation_indices]["Target"]) == {False, True}


def test_transfer_validation_split_rejects_a_nonzero_seed() -> None:
    with pytest.raises(ValueError, match="seed 0"):
        split_transfer_fit_validation(_validation_rows(), seed=1)


def test_shuffled_pseudo_mapping_is_a_cyclic_derangement() -> None:
    pseudo_sequences = _pseudo_sequences()

    shuffled_mapping = build_shuffled_pseudo_mapping(pseudo_sequences)

    assert shuffled_mapping == {
        "HLA-A01:01": "C" * 2 + "A" * 32,
        "HLA-A02:01": "C" + "A" * 33,
        "HLA-B07:02": "A" + "C" + "A" * 32,
    }
    assert all(
        shuffled_mapping[allele] != pseudo_sequence
        for allele, pseudo_sequence in pseudo_sequences.items()
    )


def test_shuffled_pseudo_mapping_rejects_a_non_derangement() -> None:
    pseudo_sequences = {
        "HLA-A01:01": "A" * 34,
        "HLA-A02:01": "A" * 34,
    }

    with pytest.raises(ValueError, match="derangement"):
        build_shuffled_pseudo_mapping(pseudo_sequences)


def test_nearest_pwm_source_uses_distance_then_allele_name() -> None:
    train = _transfer_rows().loc[
        lambda frame: frame["Allele"] != "HLA-B07:02"
    ].reset_index(drop=True)

    source_allele = select_nearest_pwm_source(
        train,
        "HLA-B07:02",
        _pseudo_sequences(),
    )

    assert source_allele == "HLA-A01:01"


def test_preflight_rejects_a_partition_without_a_pwm_source() -> None:
    rows = _transfer_rows()
    rows.loc[rows["Allele"] == "HLA-A01:01", "Target"] = True
    rows.loc[rows["Allele"] == "HLA-A02:01", "Target"] = False
    partitions = [build_allele_partition(rows, "HLA-B07:02")]

    with pytest.raises(ValueError, match="eligible PWM source"):
        preflight_partitions(partitions, _pseudo_sequences(), ("HLA-B07:02",))


def test_preflight_rejects_mutated_joint_partition_peptide_overlap() -> None:
    rows = _sufficient_transfer_rows()
    rows.loc[rows["Allele"] == "HLA-B07:02", "Peptide"] = "FFFFFFFFF"
    partition = build_joint_novelty_partition(rows, "HLA-B07:02")
    partition.train.loc[0, "Peptide"] = partition.test.loc[0, "Peptide"]

    with pytest.raises(ValueError, match="test peptide leaked"):
        preflight_partitions([partition], _pseudo_sequences(), ("HLA-B07:02",))


@pytest.mark.parametrize(
    ("label", "error"),
    [
        (pmhc_transfer.JOINT_NOVELTY_LABEL, "test peptide leaked"),
        ("unknown", "unsupported transfer partition label"),
    ],
)
def test_preflight_rejects_invalid_direct_partition(label: str, error: str) -> None:
    partition = build_allele_partition(_sufficient_transfer_rows(), "HLA-B07:02")
    direct_partition = TransferPartition(
        label=label,
        held_out_alleles=partition.held_out_alleles,
        train=partition.train,
        test=partition.test,
    )

    with pytest.raises(ValueError, match=error):
        preflight_partitions([direct_partition], _pseudo_sequences(), ("HLA-B07:02",))


def test_preflight_allows_allele_only_peptide_overlap() -> None:
    partition = build_allele_partition(_sufficient_transfer_rows(), "HLA-B07:02")
    assert set(partition.train["Peptide"]) & set(partition.test["Peptide"])

    preflight_partitions([partition], _pseudo_sequences(), ("HLA-B07:02",))


def test_nearest_pwm_preserves_interleaved_target_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _transfer_rows().loc[lambda frame: frame["Allele"] != "HLA-B07:02"]
    test = pd.DataFrame(
        {
            "Allele": ["HLA-C07:02", "HLA-B07:02", "HLA-C07:02", "HLA-B07:02"],
            "Peptide": ["FFFFFFFFF", "GGGGGGGGG", "HHHHHHHHH", "IIIIIIIII"],
            "Target": [True, False, False, True],
        },
        index=[41, 13, 8, 99],
    )
    partition = TransferPartition(
        label=pmhc_transfer.ALLELE_ONLY_LABEL,
        held_out_alleles=("HLA-B07:02", "HLA-C07:02"),
        train=train,
        test=test,
    )
    expected_test = test.copy(deep=True)
    pseudo_sequences = _pseudo_sequences()
    pseudo_sequences["HLA-B07:02"] = pseudo_sequences["HLA-A01:01"]
    pseudo_sequences["HLA-C07:02"] = pseudo_sequences["HLA-A02:01"]
    scored_frames: list[pd.DataFrame] = []
    source_scores = {"HLA-A01:01": 10.0, "HLA-A02:01": 20.0}

    def score_source_pwm(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        pd.testing.assert_frame_equal(train_rows, train)
        scored_frames.append(test_rows.copy(deep=True))
        return test_rows["Allele"].map(source_scores).to_numpy() + np.arange(len(test_rows))

    monkeypatch.setattr(pmhc_transfer, "score_pwm_fold", score_source_pwm)

    scores = pmhc_transfer._score_nearest_pwm(partition, pseudo_sequences)

    np.testing.assert_array_equal(scores, [20.0, 10.0, 21.0, 11.0])
    assert len(scored_frames) == 2
    for scored_frame, row_positions, source_allele in zip(
        scored_frames,
        ([1, 3], [0, 2]),
        ("HLA-A01:01", "HLA-A02:01"),
        strict=True,
    ):
        expected_source_test = expected_test.iloc[row_positions].assign(Allele=source_allele)
        pd.testing.assert_frame_equal(scored_frame, expected_source_test)
    pd.testing.assert_frame_equal(partition.test, expected_test)


def test_preflight_rejects_target_allele_leakage() -> None:
    rows = _transfer_rows()
    partition = build_allele_partition(rows, "HLA-B07:02")
    leaked_train = pd.concat((partition.train, partition.test.iloc[[0]]), ignore_index=True)
    leaked_partition = TransferPartition(
        label=partition.label,
        held_out_alleles=partition.held_out_alleles,
        train=leaked_train,
        test=partition.test,
    )

    with pytest.raises(ValueError, match="held-out allele leaked"):
        preflight_partitions(
            [leaked_partition], _pseudo_sequences(), ("HLA-B07:02",)
        )


def test_preflight_rejects_a_target_without_both_test_classes() -> None:
    rows = _transfer_rows().loc[
        lambda frame: ~(
            (frame["Allele"] == "HLA-B07:02") & (frame["Target"] == False)
        )
    ].reset_index(drop=True)
    partition = TransferPartition(
        label="allele_only",
        held_out_alleles=("HLA-B07:02",),
        train=rows.loc[rows["Allele"] != "HLA-B07:02"].reset_index(drop=True),
        test=rows.loc[rows["Allele"] == "HLA-B07:02"].reset_index(drop=True),
    )

    with pytest.raises(ValueError, match="both target classes"):
        preflight_partitions([partition], _pseudo_sequences(), ("HLA-B07:02",))


def test_preflight_rejects_a_single_class_training_partition() -> None:
    partition = build_allele_partition(_transfer_rows(), "HLA-B07:02")
    positive_only_partition = TransferPartition(
        label=partition.label,
        held_out_alleles=partition.held_out_alleles,
        train=partition.train.loc[partition.train["Target"]].reset_index(drop=True),
        test=partition.test,
    )

    with pytest.raises(ValueError, match="both target classes"):
        preflight_partitions(
            [positive_only_partition], _pseudo_sequences(), ("HLA-B07:02",)
        )


def test_preflight_rejects_missing_target_pseudo_sequence() -> None:
    partition = build_allele_partition(_transfer_rows(), "HLA-B07:02")
    pseudo_sequences = _pseudo_sequences()
    del pseudo_sequences["HLA-B07:02"]

    with pytest.raises(ValueError, match="missing pseudo-sequence"):
        preflight_partitions([partition], pseudo_sequences, ("HLA-B07:02",))


def test_preflight_rejects_an_omitted_eligible_target() -> None:
    partition = build_allele_partition(_transfer_rows(), "HLA-B07:02")

    with pytest.raises(ValueError, match="expected eligible alleles"):
        preflight_partitions(
            [partition],
            _pseudo_sequences(),
            ("HLA-A02:01", "HLA-B07:02"),
        )


def test_preflight_rejects_a_missing_retained_allele_pseudo_sequence() -> None:
    rows = _sufficient_transfer_rows()
    rows.loc[rows["Allele"] == "HLA-A02:01", "Target"] = True
    partition = build_allele_partition(rows, "HLA-B07:02")
    pseudo_sequences = _pseudo_sequences()
    del pseudo_sequences["HLA-A02:01"]

    with pytest.raises(ValueError, match="missing pseudo-sequence"):
        preflight_partitions(
            [partition],
            pseudo_sequences,
            ("HLA-B07:02",),
        )


def test_transfer_scores_share_test_row_order() -> None:
    partition = build_allele_partition(
        _sufficient_transfer_rows(), "HLA-B07:02"
    )

    scores = pmhc_transfer.score_transfer_partition(
        partition,
        _pseudo_sequences(),
        device="cpu",
    )

    assert set(scores) == {
        "random",
        "peptide_only_mlp",
        "shuffled_mapping_mlp",
        "nearest_pwm",
        "pseudo_sequence_mlp",
    }
    assert all(values.shape == (len(partition.test),) for values in scores.values())


def test_transfer_scores_reject_nonfinite_arm_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = build_allele_partition(
        _sufficient_transfer_rows(), "HLA-B07:02"
    )
    test_rows = len(partition.test)
    monkeypatch.setattr(
        pmhc_transfer,
        "random_scores",
        lambda _: np.zeros(test_rows),
    )
    monkeypatch.setattr(
        pmhc_transfer,
        "_score_peptide_only_mlp",
        lambda *args, **kwargs: np.full(test_rows, np.nan),
    )
    monkeypatch.setattr(
        pmhc_transfer,
        "score_mlp_fold",
        lambda *args, **kwargs: np.zeros(test_rows),
    )
    monkeypatch.setattr(
        pmhc_transfer,
        "_score_nearest_pwm",
        lambda *args, **kwargs: np.zeros(test_rows),
    )

    with pytest.raises(ValueError, match="peptide_only_mlp contains non-finite"):
        pmhc_transfer.score_transfer_partition(
            partition,
            _pseudo_sequences(),
            device="cpu",
        )

def test_schedule_rejects_a_partition_missing_affinity_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _sufficient_transfer_rows()
    partition = build_allele_partition(rows, "HLA-B07:02")
    without_affinity = replace(
        partition, train=partition.train.drop(columns=["Affinity"])
    )

    def unexpected_score(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        raise AssertionError("scoring started on a partition the MLP arms cannot fit")

    monkeypatch.setattr(pmhc_transfer, "score_transfer_partition", unexpected_score)

    with pytest.raises(ValueError, match="missing the Affinity column"):
        pmhc_transfer.evaluate_transfer_schedule(
            [without_affinity], _pseudo_sequences(), ("HLA-B07:02",), device="cpu"
        )


def test_schedule_preflights_every_target_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _sufficient_transfer_rows()
    expected_alleles = tuple(_pseudo_sequences())
    partitions = [build_allele_partition(rows, allele) for allele in expected_alleles]
    partitions[-1].train.loc[0, "Allele"] = expected_alleles[-1]
    scored_targets: list[str] = []

    def unexpected_score(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        scored_targets.append("scored")
        raise AssertionError("scoring started before full preflight")

    monkeypatch.setattr(pmhc_transfer, "score_transfer_partition", unexpected_score)
    with pytest.raises(ValueError, match="held-out allele leaked"):
        pmhc_transfer.evaluate_transfer_schedule(
            partitions, _pseudo_sequences(), expected_alleles, device="cpu"
        )
    assert scored_targets == []

def test_transfer_result_validator_rejects_empty_core() -> None:
    with pytest.raises(ValueError, match="result core"):
        pmhc_transfer.validate_transfer_result({}, tuple(_pseudo_sequences()))


def test_mlp_adapters_wire_features_affinities_and_ensemble_without_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _sufficient_transfer_rows()
    rows["Affinity"] = np.linspace(0.11, 0.91, len(rows))
    partition = build_allele_partition(rows, "HLA-B07:02")
    partition = replace(partition, test=partition.test.iloc[[7, 0, 5, 2, 9, 4, 1, 8, 3, 6]])
    expected_train = partition.train.copy(deep=True)
    expected_test = partition.test.copy(deep=True)
    fit_indices, validation_indices = split_transfer_fit_validation(partition.train)
    pseudo_sequences = _pseudo_sequences()
    shuffled = build_shuffled_pseudo_mapping(pseudo_sequences)
    train_features = [
        pmhc.encode_blosum50(partition.train["Peptide"]),
        pmhc.build_mlp_features(partition.train, shuffled, (), use_one_hot_allele=False),
        pmhc.build_mlp_features(partition.train, pseudo_sequences, (), use_one_hot_allele=False),
    ]
    test_features = [
        pmhc.encode_blosum50(partition.test["Peptide"]),
        pmhc.build_mlp_features(partition.test, shuffled, (), use_one_hot_allele=False),
        pmhc.build_mlp_features(partition.test, pseudo_sequences, (), use_one_hot_allele=False),
    ]
    calls: list[tuple[int, int, int]] = []

    def inspect_model(
        fit_features: np.ndarray,
        fit_affinities: np.ndarray,
        validation_features: np.ndarray,
        validation_affinities: np.ndarray,
        held_out_features: np.ndarray,
        *,
        hidden: int,
        seed: int,
        device: str,
    ) -> np.ndarray:
        arm_index = len(calls) // 10
        calls.append((arm_index, hidden, seed))
        assert device == "cpu"
        assert fit_features.shape[1] == (180 if arm_index == 0 else 860)
        np.testing.assert_array_equal(fit_features, train_features[arm_index][fit_indices])
        np.testing.assert_array_equal(
            validation_features, train_features[arm_index][validation_indices]
        )
        np.testing.assert_array_equal(held_out_features, test_features[arm_index])
        np.testing.assert_array_equal(
            fit_affinities, partition.train["Affinity"].to_numpy()[fit_indices]
        )
        np.testing.assert_array_equal(
            validation_affinities, partition.train["Affinity"].to_numpy()[validation_indices]
        )
        assert not np.array_equal(fit_affinities, partition.train["Target"].to_numpy()[fit_indices])
        return held_out_features[:, 160] + hidden + seed + 100 * arm_index

    monkeypatch.setattr(pmhc_transfer, "_score_single_mlp", inspect_model)
    monkeypatch.setattr(pmhc, "_score_single_mlp", inspect_model)
    scores = pmhc_transfer.score_transfer_partition(partition, pseudo_sequences, device="cpu")

    assert calls == [
        (arm_index, hidden, seed)
        for arm_index in range(3)
        for hidden in (55, 66)
        for seed in range(5)
    ]
    for arm_index, arm in enumerate(("peptide_only_mlp", "shuffled_mapping_mlp", "pseudo_sequence_mlp")):
        np.testing.assert_array_equal(
            scores[arm], test_features[arm_index][:, 160] + 62.5 + 100 * arm_index
        )
    for encoded_rows in (train_features, test_features):
        np.testing.assert_array_equal(encoded_rows[1][:, :180], encoded_rows[2][:, :180])
        assert not np.array_equal(encoded_rows[1][:, 180:], encoded_rows[2][:, 180:])
    pd.testing.assert_frame_equal(partition.train, expected_train)
    pd.testing.assert_frame_equal(partition.test, expected_test)


@pytest.fixture
def aggregate_core(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    rows = _sufficient_transfer_rows()
    expected_alleles = tuple(_pseudo_sequences())
    rows = pd.concat((rows, rows.loc[rows["Allele"] == expected_alleles[0]]), ignore_index=True)
    partitions = [
        build_transfer_partition(rows, expected_alleles[1:], exclude_test_peptides=False),
        build_allele_partition(rows, expected_alleles[0]),
    ]

    def fixed_scores(
        partition: TransferPartition, pseudo_sequences: object, *, device: str
    ) -> dict[str, np.ndarray]:
        assert device == "cpu"
        targets = partition.test["Target"].to_numpy(dtype=float)
        scores = {
            arm: targets if arm == "pseudo_sequence_mlp" else 1 - targets
            for arm in pmhc_transfer.TRANSFER_SCORE_NAMES
        }
        scores["random"] = np.where(
            partition.test["Allele"].to_numpy() == expected_alleles[0], targets, 1 - targets
        )
        return scores

    def evaluate_small_bootstrap(
        targets: np.ndarray, scores: np.ndarray, **kwargs: object
    ) -> metrics.ScoreReport:
        assert kwargs["mode"] == "both"
        assert kwargs["n_boot"] == 20_000
        assert kwargs["seed"] == 0
        assert kwargs["confidence"] == 0.95
        return metrics.evaluate(targets, scores, **{**kwargs, "n_boot": 2})

    def compare_small_bootstrap(
        targets: np.ndarray, alleles: np.ndarray,
        arm: tuple[str, np.ndarray, np.ndarray | None],
        reference: tuple[str, np.ndarray, np.ndarray | None],
        **kwargs: object,
    ) -> metrics.Comparison:
        assert kwargs == {"mode": "both", "n_boot": 20_000, "confidence": 0.95, "seed": 0}
        assert arm[0] == "pseudo_sequence_mlp"
        assert reference[0] in pmhc_transfer.TRANSFER_REFERENCES
        assert arm[2] is None and reference[2] is None
        np.testing.assert_array_equal(arm[1], targets)
        np.testing.assert_array_equal(reference[1], 1 - targets)
        return metrics.compare_macro_auc01(
            targets, alleles, arm, reference, **{**kwargs, "n_boot": 2}
        )

    monkeypatch.setattr(pmhc_transfer, "score_transfer_partition", fixed_scores)
    monkeypatch.setattr(pmhc_transfer, "evaluate", evaluate_small_bootstrap)
    monkeypatch.setattr(pmhc_transfer, "compare_macro_auc01", compare_small_bootstrap)
    return pmhc_transfer.evaluate_transfer_schedule(
        partitions, _pseudo_sequences(), expected_alleles, device="cpu"
    )


def test_schedule_produces_serializable_aggregate_core(aggregate_core: dict[str, object]) -> None:
    restored = json.loads(json.dumps(aggregate_core, allow_nan=False))
    pmhc_transfer.validate_transfer_result(restored, tuple(_pseudo_sequences()))
    assert set(restored) == {"config", "scores", "comparisons", "per_allele"}
    assert restored["scores"]["pseudo_sequence_mlp"]["macro_auc01"]["point"] == 1.0
    assert restored["scores"]["pseudo_sequence_mlp"]["n_alleles"] == 3
    random_per_allele = [
        allele["arms"]["random"]["auc01"] for allele in restored["per_allele"].values()
    ]
    assert restored["scores"]["random"]["macro_auc01"]["point"] == pytest.approx(
        np.mean(random_per_allele)
    )
    assert restored["scores"]["random"]["macro_auc01"]["point"] != pytest.approx(
        np.average(random_per_allele, weights=[20, 10, 10])
    )
    for comparison in restored["comparisons"].values():
        assert comparison["difference"]["lo"] > 0


@pytest.mark.parametrize(
    ("path", "replacement", "error"),
    [
        (("config", "bootstrap", "draws"), 1000, "frozen contract"),
        (("config", "bootstrap", "seed"), 1, "frozen contract"),
        (("config", "bootstrap", "mode"), "groups", "frozen contract"),
        (("config", "target_alleles"), ["HLA-B07:02"], "frozen contract"),
        (("scores", "one_hot_mlp"), {}, "scores fields"),
        (("scores", "random", "macro_auc01", "point"), float("nan"), "finite number"),
        (("scores", "random", "macro_auc01", "hi"), -0.1, "metric bounds"),
        (("scores", "random", "macro_auc01", "n_replicates"), 20001, "replicate count"),
        (("scores", "random", "n_alleles_skipped"), 1, "frozen cohort"),
        (("scores", "random", "n_rows"), 5, "frozen cohort"),
        (("per_allele", "HLA-A01:01", "n_positive"), 0, "both target classes"),
        (("per_allele", "HLA-A01:01", "arms", "random", "auc01"), 0.9, "held-out allele mean"),
        (("per_allele", "HLA-A01:01", "predictions"), [0.1], "fields"),
        (("comparisons", "pseudo_sequence_mlp_minus_nearest_pwm", "n_alleles"), 2, "frozen cohort"),
        (("comparisons", "pseudo_sequence_mlp_minus_nearest_pwm", "difference", "point"), 0.0, "paired difference"),
        (("comparisons", "pseudo_sequence_mlp_minus_nearest_pwm", "reference"), "pwm", "fixed contract"),
    ],
)
def test_transfer_result_validator_rejects_contract_drift(
    aggregate_core: dict[str, object], path: tuple[str, ...], replacement: object, error: str
) -> None:
    corrupted = json.loads(json.dumps(aggregate_core))
    container = corrupted
    for field in path[:-1]:
        container = container[field]
    container[path[-1]] = replacement
    with pytest.raises(ValueError, match=error):
        pmhc_transfer.validate_transfer_result(corrupted, tuple(_pseudo_sequences()))


@pytest.mark.parametrize("is_duplicate", [False, True])
def test_schedule_rejects_incomplete_or_duplicate_cohort_before_scoring(
    monkeypatch: pytest.MonkeyPatch, is_duplicate: bool
) -> None:
    rows = _sufficient_transfer_rows()
    expected_alleles = tuple(_pseudo_sequences())
    partitions = [build_allele_partition(rows, allele) for allele in expected_alleles]
    invalid_schedule = partitions + partitions[:1] if is_duplicate else partitions[:-1]

    def unexpected_score(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        raise AssertionError("scoring started before cohort validation")

    monkeypatch.setattr(pmhc_transfer, "score_transfer_partition", unexpected_score)
    with pytest.raises(ValueError, match="more than one|expected eligible alleles"):
        pmhc_transfer.evaluate_transfer_schedule(
            invalid_schedule, _pseudo_sequences(), expected_alleles, device="cpu"
        )


def test_schedule_rejects_malformed_scores_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = build_allele_partition(_sufficient_transfer_rows(), "HLA-B07:02")

    def malformed_scores(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        return {arm: np.zeros((10, 1)) for arm in pmhc_transfer.TRANSFER_SCORE_NAMES}

    def unexpected_evaluate(*args: object, **kwargs: object) -> metrics.ScoreReport:
        raise AssertionError("evaluation started with invalid scores")

    monkeypatch.setattr(pmhc_transfer, "score_transfer_partition", malformed_scores)
    monkeypatch.setattr(pmhc_transfer, "evaluate", unexpected_evaluate)
    with pytest.raises(ValueError, match="cover each test row once"):
        pmhc_transfer.evaluate_transfer_schedule(
            [partition], _pseudo_sequences(), ("HLA-B07:02",), device="cpu"
        )
