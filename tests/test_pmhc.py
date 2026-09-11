from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import cognate.pmhc as pmhc
from cognate.baseline_knn import KnnResult
from cognate.embed import EmbeddingCache, MODELS
from cognate.metrics import Comparison, GroupedScore, Interval, ScoreDiagnostics, ScoreReport
from cognate.pmhc import (
    PmhcDataset,
    build_composition_features,
    build_mlp_features,
    encode_blosum50,
    iter_pmhc_folds,
    load_pmhc_dataset,
    load_pseudo_sequences,
    load_source_contract,
    random_scores,
    score_mlp_fold,
    score_pwm_fold,
    score_retrieval_fold,
    split_fit_validation,
    verify_source,
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
FOLD_NAMES = tuple(f"c00{index}_ba" for index in range(5))
PSEUDO_SEQUENCE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ"

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_pmhc", Path(__file__).parents[1] / "scripts" / "run_pmhc.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
run_pmhc = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = run_pmhc
RUNNER_SPEC.loader.exec_module(run_pmhc)


def test_random_is_seeded_uniform() -> None:
    expected = np.array(
        [0.6369616873214543, 0.2697867137638703, 0.04097352393619469]
    )

    np.testing.assert_array_equal(random_scores(3), expected)


def test_composition_columns_are_fixed() -> None:
    expected = np.array(
        [[9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    features = build_composition_features(["ACDEFGHIK"])

    assert features.shape == (1, 21)
    np.testing.assert_array_equal(features, expected)


def test_composition_fold_fits_train_target_and_scores_test_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted: list[tuple[np.ndarray, np.ndarray]] = []
    scored: list[tuple[object, np.ndarray]] = []
    model = object()
    expected_scores = np.array([0.25, 0.75])

    def record_fit(features: np.ndarray, targets: np.ndarray) -> object:
        fitted.append((features.copy(), targets.copy()))
        return model

    def record_scores(
        fitted_model: object, features: np.ndarray
    ) -> np.ndarray:
        scored.append((fitted_model, features.copy()))
        return expected_scores

    monkeypatch.setattr(pmhc, "fit_logistic", record_fit)
    monkeypatch.setattr(pmhc, "logistic_scores", record_scores)
    train = pd.DataFrame(
        {
            "Peptide": ["AAAAAAAAA", "CCCCCCCCC", "ACDEFGHIK"],
            "Target": [True, False, True],
        }
    )
    test = pd.DataFrame(
        {
            "Peptide": ["DDDDDDDDD", "EEEEEEEEE"],
            "Target": [False, False],
        }
    )

    scores = pmhc.score_composition_fold(train, test)

    assert len(fitted) == 1
    np.testing.assert_array_equal(
        fitted[0][0], build_composition_features(train["Peptide"])
    )
    np.testing.assert_array_equal(fitted[0][1], np.array([1, 0, 1]))
    assert len(scored) == 1
    assert scored[0][0] is model
    np.testing.assert_array_equal(
        scored[0][1], build_composition_features(test["Peptide"])
    )
    assert scores is expected_scores


def test_retrieval_is_per_allele() -> None:
    train = pd.DataFrame(
        {
            "Allele": ["HLA-A02:01", "HLA-B07:02"],
            "Peptide": ["AAAAAAAAA", "CCCCCCCCC"],
            "Target": [True, True],
        }
    )
    test = pd.DataFrame(
        {"Allele": ["HLA-B07:02"], "Peptide": ["AAAAAAAAA"]}
    )

    result = score_retrieval_fold(test, train)

    assert result.score[0] == pytest.approx(0.0)
    assert result.nearest_sequence[0] == "CCCCCCCCC"


def test_retrieval_operator_is_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def record_call(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(pmhc, "score_by_nearest_positive", record_call)
    train = pd.DataFrame({"Allele": [], "Peptide": [], "Target": []})
    test = pd.DataFrame({"Allele": [], "Peptide": []})

    edit_result = score_retrieval_fold(test, train)
    def similarity(queries: np.ndarray, entries: np.ndarray) -> np.ndarray:
        return np.zeros((len(queries), len(entries)))

    cosine_result = score_retrieval_fold(test, train, similarity_fn=similarity)

    assert edit_result is sentinel
    assert cosine_result is sentinel
    assert calls == [
        (
            (test, train),
            {"peptide_column": "Allele", "sequence_column": "Peptide"},
        ),
        (
            (test, train),
            {
                "peptide_column": "Allele",
                "sequence_column": "Peptide",
                "similarity_fn": similarity,
            },
        ),
    ]


def test_pwm_matches_hand_calculation() -> None:
    train = pd.DataFrame(
        {
            "Allele": ["HLA-A02:01"] * 3,
            "Peptide": ["AAAAAAAAA", "CAAAAAAAA", "DAAAAAAAA"],
            "Target": [True, True, False],
        }
    )
    test = pd.DataFrame(
        {"Allele": ["HLA-A02:01"], "Peptide": ["AAAAAAAAA"]}
    )
    position_zero_log_odds = np.log((2 / 22) / (1 / 21))
    repeated_position_log_odds = np.log((3 / 22) / (2 / 21))
    expected_score = position_zero_log_odds + 8 * repeated_position_log_odds

    scores = score_pwm_fold(train, test)

    assert position_zero_log_odds == pytest.approx(np.log(42 / 22))
    assert scores[0] == pytest.approx(expected_score)


def test_blosum50_values_are_raw() -> None:
    expected_a = np.array(
        [
            5, -1, -2, -1, -3, 0, -2, -1, -1, -2,
            -1, -1, -1, -1, -2, 1, 0, 0, -3, -2,
        ]
    )
    expected_w = np.array(
        [
            -3, -5, -5, -3, 1, -3, -3, -3, -3, -2,
            -1, -4, -4, -1, -3, -4, -3, -3, 15, 2,
        ]
    )
    encoded = encode_blosum50(["AW"])
    allele_order = tuple(f"HLA-A{index:02d}:01" for index in range(47))
    rows = pd.DataFrame({"Peptide": ["AAAAAAAAA"], "Allele": [allele_order[0]]})
    pseudo_sequences = {allele: "A" * 34 for allele in allele_order}

    pseudo_features = build_mlp_features(
        rows, pseudo_sequences, allele_order, use_one_hot_allele=False
    )
    one_hot_features = build_mlp_features(
        rows, pseudo_sequences, allele_order, use_one_hot_allele=True
    )

    np.testing.assert_array_equal(encoded[0, :20], expected_a)
    np.testing.assert_array_equal(encoded[0, 20:], expected_w)
    assert pseudo_features.shape == (1, 860)
    assert one_hot_features.shape == (1, 227)
    assert one_hot_features[0, 180] == 1.0


def test_validation_membership_is_shared() -> None:
    rows = pd.DataFrame(
        {
            "Allele": np.repeat(["HLA-A02:01", "HLA-B07:02"], 20),
            "Target": np.tile(np.repeat([False, True], 10), 2),
        }
    )

    fit_indices, validation_indices = split_fit_validation(rows, seed=0)
    repeated_fit, repeated_validation = split_fit_validation(rows, seed=0)

    np.testing.assert_array_equal(fit_indices, repeated_fit)
    np.testing.assert_array_equal(validation_indices, repeated_validation)
    validation_strata = (
        rows.iloc[validation_indices].groupby(["Allele", "Target"]).size()
    )
    assert len(fit_indices) == 36
    assert len(validation_indices) == 4
    assert validation_strata.to_dict() == {
        ("HLA-A02:01", False): 1,
        ("HLA-A02:01", True): 1,
        ("HLA-B07:02", False): 1,
        ("HLA-B07:02", True): 1,
    }


def _mlp_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], tuple[str, ...]]:
    allele_order = ("HLA-A02:01",)
    train = pd.DataFrame(
        {
            "Allele": [allele_order[0]] * 4,
            "Peptide": ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD", "EEEEEEEEE"],
            "Affinity": [0.1, 0.2, 0.3, 0.9],
            "Target": [True, False, True, False],
        }
    )
    test = pd.DataFrame(
        {"Allele": [allele_order[0]], "Peptide": ["FFFFFFFFF"]}
    )
    return train, test, {allele_order[0]: "A" * 34}, allele_order


def test_scaler_uses_fit_partition_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fitted_features: list[np.ndarray] = []

    class RecordingScaler:
        def fit(self, features: np.ndarray) -> RecordingScaler:
            fitted_features.append(features.copy())
            return self

        def transform(self, features: np.ndarray) -> np.ndarray:
            return features

    monkeypatch.setattr(pmhc, "StandardScaler", RecordingScaler)
    monkeypatch.setattr(pmhc, "MLP_MAX_EPOCHS", 1)
    train, test, pseudo_sequences, allele_order = _mlp_rows()

    score_mlp_fold(
        train,
        test,
        pseudo_sequences,
        allele_order,
        np.array([0, 1, 2]),
        np.array([3]),
        use_one_hot_allele=True,
        device="cpu",
    )

    expected_fit_features = build_mlp_features(
        train.iloc[[0, 1, 2]],
        pseudo_sequences,
        allele_order,
        use_one_hot_allele=True,
    )
    assert fitted_features
    for features in fitted_features:
        np.testing.assert_array_equal(features, expected_fit_features)


def test_mlp_uses_adam_mse_with_sigmoid_and_affinity_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adam_learning_rates: list[float] = []
    logits: list[np.ndarray] = []
    mse_calls: list[tuple[np.ndarray, np.ndarray]] = []
    real_adam = torch.optim.Adam
    real_mse = torch.nn.MSELoss

    class ProbeHead(torch.nn.Module):
        def __init__(self, n_features: int, *, hidden: int, dropout: float) -> None:
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            output = self.logit.expand(len(features))
            logits.append(output.detach().cpu().numpy().copy())
            return output

    class RecordingMse(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.criterion = real_mse()

        def forward(
            self, predictions: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            mse_calls.append(
                (
                    predictions.detach().cpu().numpy().copy(),
                    targets.detach().cpu().numpy().copy(),
                )
            )
            return self.criterion(predictions, targets)

    def record_adam(
        parameters: Iterable[torch.nn.Parameter], *, lr: float
    ) -> torch.optim.Optimizer:
        adam_learning_rates.append(lr)
        return real_adam(parameters, lr=lr)

    def record_mse() -> torch.nn.Module:
        return RecordingMse()

    monkeypatch.setattr(pmhc, "MlpHead", ProbeHead)
    monkeypatch.setattr(pmhc.torch.optim, "Adam", record_adam)
    monkeypatch.setattr(pmhc.torch.nn, "MSELoss", record_mse)
    monkeypatch.setattr(pmhc, "MLP_HIDDEN_SIZES", (2,))
    monkeypatch.setattr(pmhc, "MLP_SEEDS", (0,))
    monkeypatch.setattr(pmhc, "MLP_MAX_EPOCHS", 1)
    train, test, pseudo_sequences, allele_order = _mlp_rows()

    score_mlp_fold(
        train,
        test,
        pseudo_sequences,
        allele_order,
        np.array([0, 1, 2]),
        np.array([3]),
        use_one_hot_allele=True,
        device="cpu",
    )

    assert adam_learning_rates == [1e-3]
    assert len(mse_calls) == 2
    assert len(logits) == 3
    for call_index, (predictions, _) in enumerate(mse_calls):
        expected_probabilities = 1 / (1 + np.exp(-logits[call_index]))
        np.testing.assert_allclose(predictions, expected_probabilities)
    np.testing.assert_allclose(np.sort(mse_calls[0][1]), np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(mse_calls[1][1], np.array([0.9]))


def test_mlp_restores_best_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_weights: list[float] = []

    class ProbeHead(torch.nn.Module):
        def __init__(self, n_features: int, *, hidden: int, dropout: float) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.weight.expand(len(features))

        def load_state_dict(
            self, state_dict: dict[str, torch.Tensor], strict: bool = True
        ) -> object:
            loaded_weights.append(float(state_dict["weight"]))
            return super().load_state_dict(state_dict, strict=strict)

    class StepOptimiser:
        def __init__(self, parameters: Iterable[torch.nn.Parameter]) -> None:
            self.parameters = list(parameters)

        def zero_grad(self) -> None:
            for parameter in self.parameters:
                parameter.grad = None

        def step(self) -> None:
            with torch.no_grad():
                for parameter in self.parameters:
                    parameter.add_(1.0)

    def make_optimiser(
        parameters: Iterable[torch.nn.Parameter], *, lr: float
    ) -> StepOptimiser:
        return StepOptimiser(parameters)

    monkeypatch.setattr(pmhc, "MlpHead", ProbeHead)
    monkeypatch.setattr(pmhc.torch.optim, "Adam", make_optimiser)
    monkeypatch.setattr(pmhc, "MLP_MAX_EPOCHS", 2)

    score = pmhc._score_single_mlp(
        np.array([[0.0]]),
        np.array([0.0]),
        np.array([[0.0]]),
        np.array([0.7]),
        np.array([[0.0]]),
        hidden=2,
        seed=0,
        device="cpu",
    )

    assert loaded_weights == [1.0]
    assert score[0] == pytest.approx(float(torch.sigmoid(torch.tensor(1.0))))


def test_ensemble_uses_all_ten_models(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def record_model(
        fit_features: np.ndarray,
        fit_affinities: np.ndarray,
        validation_features: np.ndarray,
        validation_affinities: np.ndarray,
        test_features: np.ndarray,
        *,
        hidden: int,
        seed: int,
        device: str,
    ) -> np.ndarray:
        calls.append((hidden, seed))
        return np.full(len(test_features), hidden + seed, dtype=float)

    monkeypatch.setattr(pmhc, "_score_single_mlp", record_model)
    train, test, pseudo_sequences, allele_order = _mlp_rows()

    scores = score_mlp_fold(
        train,
        test,
        pseudo_sequences,
        allele_order,
        np.array([0, 1, 2]),
        np.array([3]),
        use_one_hot_allele=False,
        device="cpu",
    )

    assert calls == [(hidden, seed) for hidden in (55, 66) for seed in range(5)]
    np.testing.assert_array_equal(scores, np.array([62.5]))


def _peptide(number: int, length: int = 9) -> str:
    residues: list[str] = []
    current = number
    for _ in range(length):
        residues.append(AMINO_ACIDS[current % len(AMINO_ACIDS)])
        current //= len(AMINO_ACIDS)
    return "".join(residues)


def _write_valid_source(
    root: Path,
) -> tuple[Path, Path, Path, str, str, str, str]:
    source_dir = root / "source"
    source_dir.mkdir()
    peptide_number = 0
    boundary_positive_peptide = ""
    boundary_negative_peptide = ""
    near_zero_negative_peptide = ""
    mid_range_negative_peptide = ""
    shared_peptide = ""
    for fold_name in FOLD_NAMES:
        lines: list[str] = []
        for index in range(25):
            peptide = _peptide(peptide_number)
            affinity = "0.500"
            if fold_name == FOLD_NAMES[0] and index == 0:
                affinity = "0.427"
                boundary_positive_peptide = peptide
            if fold_name == FOLD_NAMES[0] and index == 1:
                shared_peptide = peptide
            lines.append(f"{peptide} {affinity} HLA-A02:01")
            peptide_number += 1
        for index in range(12):
            peptide = _peptide(peptide_number)
            affinity = "0.200"
            if fold_name == FOLD_NAMES[0] and index == 0:
                affinity = "0.426"
                boundary_negative_peptide = peptide
            elif fold_name == FOLD_NAMES[0] and index == 1:
                affinity = "0.005"
                near_zero_negative_peptide = peptide
            elif fold_name == FOLD_NAMES[0] and index == 2:
                affinity = "0.100"
                mid_range_negative_peptide = peptide
            lines.append(f"{peptide} {affinity} HLA-A02:01")
            peptide_number += 1
        for _ in range(8):
            lines.append(f"{_peptide(peptide_number)} 0.010 HLA-A02:01")
            peptide_number += 1
        lines.append(f"{_peptide(peptide_number, 10)} 0.500 HLA-A02:01")
        peptide_number += 1
        lines.append(f"{_peptide(peptide_number)} 0.010 BoLA-1:00901")
        peptide_number += 1
        if fold_name == FOLD_NAMES[0]:
            lines.append(f"{shared_peptide} 0.500 HLA-C07:01")
        for _ in range(20):
            lines.append(f"{_peptide(peptide_number)} 0.500 HLA-B07:02")
            peptide_number += 1
        for _ in range(5):
            lines.append(f"{_peptide(peptide_number)} 0.200 HLA-B07:02")
            peptide_number += 1
        for _ in range(5):
            lines.append(f"{_peptide(peptide_number)} 0.010 HLA-B07:02")
            peptide_number += 1
        for _ in range(10):
            lines.append(f"{_peptide(peptide_number)} 0.500 HLA-C07:01")
            peptide_number += 1
        for _ in range(10):
            lines.append(f"{_peptide(peptide_number)} 0.200 HLA-C07:01")
            peptide_number += 1
        for _ in range(10):
            lines.append(f"{_peptide(peptide_number)} 0.010 HLA-C07:01")
            peptide_number += 1
        (source_dir / fold_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (source_dir / "c000_el").write_text(
        "AAAAAAAAA 0.500 HLA-A02:01\n", encoding="utf-8"
    )
    pseudo_path = source_dir / "MHC_pseudo.dat"
    pseudo_path.write_text(f"HLA-A0201 {PSEUDO_SEQUENCE}\n", encoding="utf-8")
    archive_path = root / "source.tar.gz"
    archive_path.write_bytes(b"synthetic archive")
    contract_path = root / "source_contract.json"
    contract_path.write_text(
        json.dumps(_valid_contract(source_dir, archive_path)), encoding="utf-8"
    )
    return (
        source_dir,
        archive_path,
        contract_path,
        boundary_positive_peptide,
        boundary_negative_peptide,
        near_zero_negative_peptide,
        mid_range_negative_peptide,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_contract(source_dir: Path, archive_path: Path) -> dict[str, object]:
    return {
        "source_url": "https://example.test/source.tar.gz",
        "retrieval_date": "2026-09-10",
        "archive_sha256": _sha256(archive_path),
        "file_sha256": {
            name: _sha256(source_dir / name)
            for name in (*FOLD_NAMES, "MHC_pseudo.dat")
        },
        "source_counts": {
            "ba_rows_all_species": 536,
            "hla_abc_rows": 531,
            "hla_abc_positive": 281,
            "hla_abc_artificial_negative": 115,
            "nine_mer_hla_rows": 526,
            "nine_mer_hla_positive": 276,
            "nine_mer_hla_negative": 250,
        },
        "headline_counts": {
            "alleles": 1,
            "rows": 225,
            "positive": 125,
            "negative": 100,
            "measured_nonbinder": 60,
            "artificial_negative": 40,
        },
    }


def _replace_first_line(path: Path, transform: Callable[[list[str]], list[str]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = " ".join(transform(lines[0].split()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_valid_source_is_parsed_and_partitioned(tmp_path: Path) -> None:
    (
        source_dir,
        archive_path,
        contract_path,
        boundary_positive_peptide,
        boundary_negative_peptide,
        near_zero_negative_peptide,
        mid_range_negative_peptide,
    ) = _write_valid_source(tmp_path)
    contract = load_source_contract(contract_path)
    verify_source(source_dir, archive_path, contract)

    dataset = load_pmhc_dataset(source_dir)

    assert dataset.eligible_alleles == ("HLA-A02:01",)
    assert len(dataset.all_nine_mer_hla_rows) == 526
    assert len(dataset.rows) == 225
    assert dataset.rows["Target"].value_counts().to_dict() == {True: 125, False: 100}
    assert dataset.rows["RowType"].value_counts().to_dict() == {
        "positive": 125,
        "measured_nonbinder": 60,
        "artificial_negative": 40,
    }
    boundary_positive_row = dataset.rows.loc[
        dataset.rows["Peptide"] == boundary_positive_peptide
    ].iloc[0]
    assert bool(boundary_positive_row["Target"]) is True
    assert boundary_positive_row["RowType"] == "positive"
    boundary_negative_row = dataset.rows.loc[
        dataset.rows["Peptide"] == boundary_negative_peptide
    ].iloc[0]
    assert bool(boundary_negative_row["Target"]) is False
    assert boundary_negative_row["RowType"] == "measured_nonbinder"
    near_zero_negative_row = dataset.rows.loc[
        dataset.rows["Peptide"] == near_zero_negative_peptide
    ].iloc[0]
    assert bool(near_zero_negative_row["Target"]) is False
    assert near_zero_negative_row["RowType"] == "measured_nonbinder"
    mid_range_negative_row = dataset.rows.loc[
        dataset.rows["Peptide"] == mid_range_negative_peptide
    ].iloc[0]
    assert bool(mid_range_negative_row["Target"]) is False
    assert mid_range_negative_row["RowType"] == "measured_nonbinder"
    folds = list(iter_pmhc_folds(dataset))
    assert [name for name, _, _ in folds] == list(FOLD_NAMES)
    assert all(len(train_rows) == 180 for _, train_rows, _ in folds)
    assert all(len(test_rows) == 45 for _, _, test_rows in folds)
    for name, train_rows, test_rows in folds:
        assert set(test_rows["Fold"]) == {name}
        assert set(train_rows["Fold"]) == set(FOLD_NAMES) - {name}
    pseudo_sequences = load_pseudo_sequences(
        source_dir / "MHC_pseudo.dat", dataset.eligible_alleles
    )
    assert pseudo_sequences == {"HLA-A02:01": PSEUDO_SEQUENCE}


def _run_corruption(case: str, root: Path) -> None:
    source_dir, archive_path, contract_path, _, _, _, _ = _write_valid_source(root)
    contract = load_source_contract(contract_path)
    if case == "archive-hash":
        archive_path.write_bytes(b"corrupted archive")
        verify_source(source_dir, archive_path, contract)
    elif case == "extracted-file-hash":
        with (source_dir / "c000_ba").open("a", encoding="utf-8") as stream:
            stream.write(f"{_peptide(1000)} 0.500 HLA-A02:01\n")
        verify_source(source_dir, archive_path, contract)
    elif case == "extracted-file-hash-pseudo":
        with (source_dir / "MHC_pseudo.dat").open("a", encoding="utf-8") as stream:
            stream.write(f"HLA-B0702 {PSEUDO_SEQUENCE}\n")
        verify_source(source_dir, archive_path, contract)
    elif case == "fold-set":
        (source_dir / "c005_ba").write_text("", encoding="utf-8")
        load_pmhc_dataset(source_dir)
    elif case == "malformed-row-too-few":
        with (source_dir / "c000_ba").open("a", encoding="utf-8") as stream:
            stream.write("PEPTIDE 0.5\n")
        load_pmhc_dataset(source_dir)
    elif case == "malformed-row-too-many":
        with (source_dir / "c000_ba").open("a", encoding="utf-8") as stream:
            stream.write(f"{_peptide(1000)} 0.500 HLA-A02:01 EXTRA\n")
        load_pmhc_dataset(source_dir)
    elif case == "invalid-affinity":
        _replace_first_line(
            source_dir / "c000_ba", lambda fields: [fields[0], "1.1", fields[2]]
        )
        load_pmhc_dataset(source_dir)
    elif case == "invalid-residue":
        _replace_first_line(
            source_dir / "c000_ba",
            lambda fields: [f"Z{fields[0][1:]}", fields[1], fields[2]],
        )
        load_pmhc_dataset(source_dir)
    elif case == "fold-overlap":
        first_peptide = (source_dir / "c000_ba").read_text(encoding="utf-8").split()[0]
        _replace_first_line(
            source_dir / "c001_ba",
            lambda fields: [first_peptide, fields[1], fields[2]],
        )
        load_pmhc_dataset(source_dir)
    elif case == "minimum-class-support":
        load_pmhc_dataset(source_dir, minimum_class_rows=101)
    elif case == "missing-class-test-fold":
        path = source_dir / FOLD_NAMES[3]
        path.write_text(
            path.read_text(encoding="utf-8").replace(" 0.500 ", " 0.200 "),
            encoding="utf-8",
        )
        load_pmhc_dataset(source_dir, minimum_class_rows=2)
    elif case == "missing-class-training-complement":
        for fold_name in FOLD_NAMES[1:]:
            path = source_dir / fold_name
            path.write_text(
                path.read_text(encoding="utf-8").replace(" 0.500 ", " 0.200 "),
                encoding="utf-8",
            )
        load_pmhc_dataset(source_dir, minimum_class_rows=2)
    elif case == "missing-pseudo-sequence":
        load_pseudo_sequences(
            source_dir / "MHC_pseudo.dat", ["HLA-A02:01", "HLA-B07:02"]
        )
    elif case == "short-pseudo-sequence":
        (source_dir / "MHC_pseudo.dat").write_text(
            f"HLA-A0201 {PSEUDO_SEQUENCE[:-1]}\n", encoding="utf-8"
        )
        load_pseudo_sequences(source_dir / "MHC_pseudo.dat", ["HLA-A02:01"])
    elif case == "long-pseudo-sequence":
        (source_dir / "MHC_pseudo.dat").write_text(
            f"HLA-A0201 {PSEUDO_SEQUENCE}A\n", encoding="utf-8"
        )
        load_pseudo_sequences(source_dir / "MHC_pseudo.dat", ["HLA-A02:01"])
    elif case == "malformed-pseudo-row":
        with (source_dir / "MHC_pseudo.dat").open("a", encoding="utf-8") as stream:
            stream.write(f"HLA-B0702 {PSEUDO_SEQUENCE} EXTRA\n")
        load_pseudo_sequences(source_dir / "MHC_pseudo.dat", ["HLA-A02:01"])
    else:
        raise AssertionError(f"Unknown corruption case: {case}")


CORRUPTION_CASES: tuple[tuple[str, str], ...] = (
    ("archive-hash", "archive SHA-256 mismatch"),
    ("extracted-file-hash", "extracted file SHA-256 mismatch"),
    ("extracted-file-hash-pseudo", "extracted file SHA-256 mismatch"),
    ("fold-set", "fold set mismatch"),
    ("malformed-row-too-few", "malformed row"),
    ("malformed-row-too-many", "malformed row"),
    ("invalid-affinity", "affinity outside"),
    ("invalid-residue", "invalid peptide residue"),
    ("fold-overlap", "peptide overlap across folds"),
    ("minimum-class-support", "no allele meets minimum class support"),
    ("missing-class-test-fold", "missing class in test fold"),
    (
        "missing-class-training-complement",
        "missing class in training complement",
    ),
    ("missing-pseudo-sequence", "missing pseudo-sequence"),
    ("short-pseudo-sequence", "expected 34 residues"),
    ("long-pseudo-sequence", "expected 34 residues"),
    ("malformed-pseudo-row", "malformed pseudo-sequence row"),
)


@pytest.mark.parametrize(
    ("case", "message"),
    CORRUPTION_CASES,
    ids=[case for case, _ in CORRUPTION_CASES],
)
def test_contract_corruption_is_rejected(
    case: str, message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        _run_corruption(case, tmp_path)


CONTRACT_COUNT_FIELDS: tuple[tuple[str, str], ...] = (
    ("source_counts", "ba_rows_all_species"),
    ("source_counts", "hla_abc_rows"),
    ("source_counts", "hla_abc_positive"),
    ("source_counts", "hla_abc_artificial_negative"),
    ("source_counts", "nine_mer_hla_rows"),
    ("source_counts", "nine_mer_hla_positive"),
    ("source_counts", "nine_mer_hla_negative"),
    ("headline_counts", "alleles"),
    ("headline_counts", "rows"),
    ("headline_counts", "positive"),
    ("headline_counts", "negative"),
    ("headline_counts", "measured_nonbinder"),
    ("headline_counts", "artificial_negative"),
)


CONTRACT_COUNT_FIELD_CASES: tuple[tuple[str, str, str], ...] = (
    *((parent, field, "increment") for parent, field in CONTRACT_COUNT_FIELDS),
    ("source_counts", "ba_rows_all_species", "delete"),
    ("source_counts", "ba_rows_all_species", "extra"),
)


@pytest.mark.parametrize(
    ("parent", "field", "mode"),
    CONTRACT_COUNT_FIELD_CASES,
    ids=[f"{parent}.{field}-{mode}" for parent, field, mode in CONTRACT_COUNT_FIELD_CASES],
)
def test_contract_count_field_mismatch_is_rejected(
    parent: str, field: str, mode: str, tmp_path: Path
) -> None:
    source_dir, archive_path, contract_path, _, _, _, _ = _write_valid_source(tmp_path)
    contract = load_source_contract(contract_path)
    counts = contract[parent]
    assert isinstance(counts, dict)
    if mode == "increment":
        counts[field] += 1
        expected_message = f"{parent} mismatch"
    elif mode == "delete":
        del counts[field]
        expected_message = f"{parent} keys mismatch"
    else:
        counts["unexpected_count_field"] = 0
        expected_message = f"{parent} keys mismatch"
    with pytest.raises(ValueError, match=expected_message):
        verify_source(source_dir, archive_path, contract)


def _tiny_runner_dataset() -> PmhcDataset:
    records: list[dict[str, object]] = []
    for fold_index, fold_name in enumerate(FOLD_NAMES):
        for is_positive in (False, True):
            row_id = 2 * fold_index + int(is_positive)
            records.append(
                {
                    "RowId": row_id,
                    "Peptide": _peptide(10_000 + row_id),
                    "Affinity": 0.8 if is_positive else 0.2,
                    "Allele": "HLA-A02:01",
                    "Fold": fold_name,
                    "Target": is_positive,
                    "RowType": "positive" if is_positive else (
                        "measured_nonbinder"
                        if fold_index % 2 == 0
                        else "artificial_negative"
                    ),
                }
            )
    rows = pd.DataFrame.from_records(records)
    return PmhcDataset(
        all_nine_mer_hla_rows=rows.copy(),
        rows=rows,
        eligible_alleles=("HLA-A02:01",),
    )


def _runner_scores(n_rows: int) -> dict[str, np.ndarray]:
    return {
        arm: np.linspace(0.05, 0.95, n_rows) + arm_index * 1e-4
        for arm_index, arm in enumerate(run_pmhc.ARM_NAMES)
    }


def _fake_report(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    groups: object = None,
    *,
    subset: np.ndarray | pd.Series | None = None,
    label: str = "all",
    mode: str = "both",
    n_boot: int = 20_000,
    confidence: float = 0.95,
    seed: int = 0,
    on_degenerate: str = "raise",
) -> ScoreReport:
    del mode, confidence, seed, on_degenerate
    targets = np.asarray(y_true, dtype=bool)
    scores = np.asarray(y_score, dtype=float)
    alleles = np.asarray(groups, dtype=object)
    if subset is not None:
        mask = np.asarray(subset, dtype=bool)
        targets = targets[mask]
        scores = scores[mask]
        alleles = alleles[mask]
    point = 0.55 + float(np.mean(scores)) * 1e-3
    interval = Interval(point, point - 0.01, point + 0.01, n_boot)
    per_allele = {str(allele): point for allele in np.unique(alleles)}
    return ScoreReport(
        label=label,
        n_rows=len(targets),
        n_positive=int(targets.sum()),
        n_groups_scored=len(per_allele),
        n_groups_skipped=0,
        skipped_groups=(),
        auroc=interval,
        auprc=interval,
        macro_auc01=interval,
        per_group_auc01=per_allele,
        diagnostics=ScoreDiagnostics(
            n_rows=len(targets),
            n_unique_scores=len(np.unique(scores)),
            n_nonfinite=0,
            modal_fraction=1 / len(scores),
            constant_groups=(),
        ),
    )


def _fake_macro_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: object,
    scorer: object,
) -> GroupedScore:
    del y_true, y_score, scorer
    per_allele = {str(allele): 0.6 for allele in np.unique(np.asarray(groups))}
    return GroupedScore(0.6, per_allele, 0, ())


def _fake_comparison(
    y_true: np.ndarray | pd.Series,
    groups: object,
    a: tuple[str, np.ndarray, np.ndarray | None],
    b: tuple[str, np.ndarray, np.ndarray | None],
    *,
    mode: str = "both",
    n_boot: int = 20_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    del y_true, groups, mode, confidence, seed
    difference = Interval(0.01, 0.0, 0.015, n_boot)
    return Comparison(a[0], b[0], 0.56, 0.55, difference, 0.04, 1)


def test_each_row_scored_once() -> None:
    scores = _runner_scores(4)
    assignment_counts = {
        arm: np.ones(4, dtype=np.int8) for arm in run_pmhc.ARM_NAMES
    }
    assignment_counts["pwm"][2] = 0

    with pytest.raises(ValueError, match="exactly once"):
        run_pmhc.validate_score_arrays(scores, assignment_counts, 4)


def test_invalid_scores_stop_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _tiny_runner_dataset()
    scores = _runner_scores(len(dataset.rows))
    scores["edit_retrieval"][3] = np.nan
    monkeypatch.setattr(
        run_pmhc,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("evaluation accepted invalid scores"),
    )

    with pytest.raises(ValueError, match="non-finite"):
        run_pmhc.evaluate_predictions(
            dataset.rows,
            scores,
            np.zeros(len(dataset.rows), dtype=float),
        )


@pytest.mark.parametrize("corruption", ["model", "layer", "peptides"])
def test_cache_rejects_wrong_model_layer_or_peptide_set(corruption: str) -> None:
    expected_peptides = ("AAAAAAAAA", "CCCCCCCCC")
    model_name = MODELS["35M"] if corruption != "model" else "wrong/model"
    n_layers = 11 if corruption != "layer" else 10
    peptides = expected_peptides if corruption != "peptides" else ("AAAAAAAAA",)
    cache = EmbeddingCache(
        model_name=model_name,
        sequences=np.array(peptides, dtype=object),
        layers=np.zeros((n_layers, len(peptides), 2), dtype=np.float32),
        index={peptide: index for index, peptide in enumerate(peptides)},
    )

    with pytest.raises(ValueError, match=corruption):
        run_pmhc.validate_embedding_cache(cache, expected_peptides)


def test_headline_uses_alleles_and_20000_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _tiny_runner_dataset()
    calls: list[dict[str, object]] = []

    def record_evaluate(*args: object, **kwargs: object) -> ScoreReport:
        calls.append(kwargs)
        return _fake_report(*args, **kwargs)

    monkeypatch.setattr(run_pmhc, "evaluate", record_evaluate)
    monkeypatch.setattr(run_pmhc, "macro_by_group", _fake_macro_by_group)
    monkeypatch.setattr(run_pmhc, "compare_macro_auc01", _fake_comparison)

    run_pmhc.evaluate_predictions(
        dataset.rows,
        _runner_scores(len(dataset.rows)),
        np.zeros(len(dataset.rows), dtype=float),
    )

    headline_calls = calls[: len(run_pmhc.ARM_NAMES)]
    assert len(headline_calls) == len(run_pmhc.ARM_NAMES)
    for call in headline_calls:
        np.testing.assert_array_equal(call["groups"], dataset.rows["Allele"])
        assert call["mode"] == "both"
        assert call["n_boot"] == 20_000
        assert call["seed"] == 0


def test_predictive_requires_lower_bound_above_half() -> None:
    assert run_pmhc.is_predictive({"point": 0.7, "lo": 0.500001, "hi": 0.8})
    assert not run_pmhc.is_predictive({"point": 0.7, "lo": 0.5, "hi": 0.8})
    assert not run_pmhc.is_predictive({"point": 0.7, "lo": 0.49, "hi": 0.8})


@pytest.mark.parametrize(
    ("pwm_lower_bound", "mlp_minus_pwm_upper_bound", "expected"),
    [(0.51, 0.019, True), (0.50, 0.019, False), (0.51, 0.020, False)],
)
def test_pwm_sufficiency_requires_both_conditions(
    pwm_lower_bound: float,
    mlp_minus_pwm_upper_bound: float,
    expected: bool,
) -> None:
    scores = {
        arm: {"macro_auc01": {"point": 0.6, "lo": 0.51, "hi": 0.7}}
        for arm in run_pmhc.ARM_NAMES
    }
    scores["pwm"]["macro_auc01"]["lo"] = pwm_lower_bound
    comparisons = {
        "mlp_pseudo_sequence_minus_pwm": {
            "difference": {
                "point": 0.01,
                "lo": -0.01,
                "hi": mlp_minus_pwm_upper_bound,
            }
        }
    }

    criteria = run_pmhc.derive_criteria(scores, comparisons)

    assert criteria["pwm_sufficient_relative_to_mlp"] is expected


def test_comparisons_are_paired_against_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _tiny_runner_dataset()
    calls: list[tuple[tuple[str, np.ndarray, np.ndarray | None], tuple[str, np.ndarray, np.ndarray | None]]] = []

    def record_comparison(
        y_true: np.ndarray | pd.Series,
        groups: object,
        a: tuple[str, np.ndarray, np.ndarray | None],
        b: tuple[str, np.ndarray, np.ndarray | None],
        **kwargs: object,
    ) -> Comparison:
        calls.append((a, b))
        return _fake_comparison(y_true, groups, a, b, **kwargs)

    monkeypatch.setattr(run_pmhc, "evaluate", _fake_report)
    monkeypatch.setattr(run_pmhc, "macro_by_group", _fake_macro_by_group)
    monkeypatch.setattr(run_pmhc, "compare_macro_auc01", record_comparison)

    result = run_pmhc.evaluate_predictions(
        dataset.rows,
        _runner_scores(len(dataset.rows)),
        np.zeros(len(dataset.rows), dtype=float),
    )

    random_comparisons = calls[:-1]
    assert len(random_comparisons) == len(run_pmhc.ARM_NAMES) - 1
    assert all(first[0] != "random" and second[0] == "random" for first, second in random_comparisons)
    assert set(result["comparisons"]) == {
        *(f"{arm}_minus_random" for arm in run_pmhc.ARM_NAMES if arm != "random"),
        "mlp_pseudo_sequence_minus_pwm",
    }


def test_negative_sources_are_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _tiny_runner_dataset()
    monkeypatch.setattr(run_pmhc, "evaluate", _fake_report)
    monkeypatch.setattr(run_pmhc, "macro_by_group", _fake_macro_by_group)
    monkeypatch.setattr(run_pmhc, "compare_macro_auc01", _fake_comparison)

    result = run_pmhc.evaluate_predictions(
        dataset.rows,
        _runner_scores(len(dataset.rows)),
        np.arange(len(dataset.rows), dtype=float),
    )

    measured = result["negative_sources"]["measured_nonbinder"]
    artificial = result["negative_sources"]["artificial_negative"]
    assert measured["n_negative"] == 3
    assert artificial["n_negative"] == 2
    assert measured["n_rows"] == 8
    assert artificial["n_rows"] == 7
    assert set(measured["scores"]) == set(run_pmhc.ARM_NAMES)
    assert set(artificial["scores"]) == set(run_pmhc.ARM_NAMES)


def test_tcr_uplift_uses_observed_random(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval.json"
    random_path = tmp_path / "random.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "scores": {
                    "edit/seen": {"macro_auc01": {"point": 0.7}},
                    "35M layer10 (headline)/seen": {
                        "macro_auc01": {"point": 0.65}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    random_path.write_text(
        json.dumps(
            {"scores": {"random/seen": {"macro_auc01": {"point": 0.6}}}}
        ),
        encoding="utf-8",
    )

    result = run_pmhc.load_tcr_retrieval_contrast(retrieval_path, random_path)

    assert result["edit_retrieval"]["uplift_over_observed_random"] == pytest.approx(0.1)
    assert result["esm_retrieval"]["uplift_over_observed_random"] == pytest.approx(0.05)


def test_tiny_five_fold_integration_emits_all_fixed_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _tiny_runner_dataset()
    peptides = tuple(dataset.rows["Peptide"].astype(str))
    embedding_calls: list[tuple[str, ...]] = []
    mlp_calls: list[bool] = []

    def fake_embed_sequences(
        sequences: Iterable[str],
        model_key: str,
        *,
        device: str,
        progress_every: int,
    ) -> tuple[EmbeddingCache, float]:
        embedded = tuple(sorted(set(sequences)))
        embedding_calls.append(embedded)
        assert model_key == "35M"
        assert device == "cpu"
        assert progress_every == 20
        cache = EmbeddingCache(
            model_name=MODELS["35M"],
            sequences=np.array(embedded, dtype=object),
            layers=np.ones((11, len(embedded), 2), dtype=np.float32),
            index={peptide: index for index, peptide in enumerate(embedded)},
        )
        return cache, 0.01

    def fold_values(test: pd.DataFrame, offset: float) -> np.ndarray:
        return test["RowId"].to_numpy(dtype=float) + offset

    def fake_retrieval(
        test: pd.DataFrame,
        train: pd.DataFrame,
        *,
        similarity_fn: object = None,
    ) -> KnnResult:
        del train
        offset = 3.0 if similarity_fn is None else 4.0
        return KnnResult(
            score=fold_values(test, offset),
            has_database=np.ones(len(test), dtype=bool),
            database_size=np.ones(len(test), dtype=int),
            nearest_sequence=np.array(["AAAAAAAAA"] * len(test), dtype=object),
        )

    def fake_mlp(
        train: pd.DataFrame,
        test: pd.DataFrame,
        pseudo_sequences: dict[str, str],
        allele_order: tuple[str, ...],
        fit_indices: np.ndarray,
        validation_indices: np.ndarray,
        *,
        use_one_hot_allele: bool,
        device: str,
    ) -> np.ndarray:
        del train, pseudo_sequences, allele_order, fit_indices, validation_indices, device
        mlp_calls.append(use_one_hot_allele)
        return fold_values(test, 6.0 if not use_one_hot_allele else 7.0)

    monkeypatch.setattr(run_pmhc, "embed_sequences", fake_embed_sequences)
    monkeypatch.setattr(
        run_pmhc,
        "random_scores",
        lambda n_rows: np.arange(n_rows, dtype=float) + 1.0,
    )
    monkeypatch.setattr(run_pmhc, "score_composition_fold", lambda train, test: fold_values(test, 2.0))
    monkeypatch.setattr(run_pmhc, "score_retrieval_fold", fake_retrieval)
    monkeypatch.setattr(run_pmhc, "score_pwm_fold", lambda train, test: fold_values(test, 5.0))
    monkeypatch.setattr(run_pmhc, "score_mlp_fold", fake_mlp)
    monkeypatch.setattr(
        run_pmhc,
        "split_fit_validation",
        lambda train: (np.arange(len(train) - 1), np.array([len(train) - 1])),
    )
    monkeypatch.setattr(run_pmhc, "evaluate", _fake_report)
    monkeypatch.setattr(run_pmhc, "macro_by_group", _fake_macro_by_group)
    monkeypatch.setattr(run_pmhc, "compare_macro_auc01", _fake_comparison)

    cache = run_pmhc.load_or_build_embedding_cache(
        peptides,
        tmp_path / "peptides.npz",
        device="cpu",
    )
    scored = run_pmhc.score_dataset(
        dataset,
        {"HLA-A02:01": PSEUDO_SEQUENCE},
        cache,
        device="cpu",
    )
    result = run_pmhc.evaluate_predictions(
        dataset.rows,
        scored.scores,
        scored.nearest_positive_edit_distance,
    )

    assert embedding_calls == [tuple(sorted(peptides))]
    assert len(mlp_calls) == 10
    assert mlp_calls.count(False) == 5
    assert mlp_calls.count(True) == 5
    assert set(scored.scores) == set(run_pmhc.ARM_NAMES)
    for arm_index, arm in enumerate(run_pmhc.ARM_NAMES):
        offset = float(arm_index + 1)
        np.testing.assert_array_equal(
            scored.scores[arm],
            dataset.rows["RowId"].to_numpy(dtype=float) + offset,
        )
    assert set(result) == {
        "scores",
        "per_allele",
        "comparisons",
        "negative_sources",
        "retrieval_diagnostics",
        "criteria",
    }
