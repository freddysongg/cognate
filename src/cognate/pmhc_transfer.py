"""Immutable leave-one-allele-out partitions for pMHC transfer experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from cognate.metrics import compare_macro_auc01, evaluate
from cognate.pmhc import (
    MLP_HIDDEN_SIZES,
    MLP_SEEDS,
    PSEUDO_SEQUENCE_LENGTH,
    _score_single_mlp,
    encode_blosum50,
    random_scores,
    score_mlp_fold,
    score_pwm_fold,
)

ALLELE_ONLY_LABEL = "allele_only"
TRANSFER_REFERENCES = ("peptide_only_mlp", "shuffled_mapping_mlp", "nearest_pwm")
TRANSFER_BOOTSTRAP_DRAWS = 20_000
TRANSFER_BOOTSTRAP_SEED = 0
JOINT_NOVELTY_LABEL = "joint_novelty"
_REQUIRED_COLUMNS = frozenset({"Allele", "Peptide", "Target"})
TRANSFER_SCORE_NAMES = (
    "random",
    "peptide_only_mlp",
    "shuffled_mapping_mlp",
    "nearest_pwm",
    "pseudo_sequence_mlp",
)


@dataclass(frozen=True)
class TransferPartition:
    """One transfer evaluation partition with all held-out rows in its test frame."""

    label: str
    held_out_alleles: tuple[str, ...]
    train: pd.DataFrame
    test: pd.DataFrame


def build_transfer_partition(
    rows: pd.DataFrame,
    held_out_alleles: Sequence[str],
    *,
    exclude_test_peptides: bool,
) -> TransferPartition:
    """Build a validated, non-mutating transfer partition from pMHC rows."""
    _require_partition_columns(rows)
    held_out_allele_tuple = tuple(held_out_alleles)
    _require_held_out_alleles(held_out_allele_tuple)

    is_held_out_allele = rows["Allele"].isin(held_out_allele_tuple)
    test = rows.loc[is_held_out_allele].copy().reset_index(drop=True)
    train = rows.loc[~is_held_out_allele].copy()
    if exclude_test_peptides:
        train = train.loc[~train["Peptide"].isin(test["Peptide"])]
    train = train.reset_index(drop=True)

    _require_valid_partition(
        train,
        test,
        held_out_allele_tuple,
        exclude_test_peptides=exclude_test_peptides,
    )
    label = JOINT_NOVELTY_LABEL if exclude_test_peptides else ALLELE_ONLY_LABEL
    return TransferPartition(
        label=label,
        held_out_alleles=held_out_allele_tuple,
        train=train,
        test=test,
    )


def build_allele_partition(rows: pd.DataFrame, target_allele: str) -> TransferPartition:
    """Hold out one allele while retaining non-target rows for shared peptides."""
    return build_transfer_partition(
        rows,
        (target_allele,),
        exclude_test_peptides=False,
    )


def build_joint_novelty_partition(
    rows: pd.DataFrame, target_allele: str
) -> TransferPartition:
    """Hold out one allele and remove every one of its test peptides from training."""
    return build_transfer_partition(
        rows,
        (target_allele,),
        exclude_test_peptides=True,
    )


def split_transfer_fit_validation(
    train: pd.DataFrame, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed 90/10 transfer split stratified only by target class."""
    _require_partition_columns(train)
    if seed != 0:
        raise ValueError("transfer validation split requires seed 0")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=0,
    )
    target_strata = train["Target"].to_numpy()
    return next(splitter.split(np.zeros(len(train)), target_strata))


def build_shuffled_pseudo_mapping(
    pseudo_sequences: Mapping[str, str],
) -> dict[str, str]:
    """Shift sorted unique pseudo-sequences by one to make the control mapping."""
    _require_pseudo_sequence_mapping(pseudo_sequences)
    unique_sequences = tuple(sorted(set(pseudo_sequences.values())))
    if len(unique_sequences) < 2:
        raise ValueError("cannot construct a shuffled pseudo-sequence derangement")
    shifted_sequences = unique_sequences[1:] + unique_sequences[:1]
    shifted_by_sequence = dict(zip(unique_sequences, shifted_sequences, strict=True))
    shuffled_mapping = {
        allele: shifted_by_sequence[pseudo_sequence]
        for allele, pseudo_sequence in pseudo_sequences.items()
    }
    if any(
        shuffled_mapping[allele] == pseudo_sequence
        for allele, pseudo_sequence in pseudo_sequences.items()
    ):
        raise ValueError("shuffled pseudo-sequence mapping is not a derangement")
    return shuffled_mapping


def select_nearest_pwm_source(
    train: pd.DataFrame,
    target_allele: str,
    pseudo_sequences: Mapping[str, str],
) -> str:
    """Return the nearest retained allele with both PWM training classes."""
    _require_partition_columns(train)
    _require_pseudo_sequence(target_allele, pseudo_sequences)
    if (train["Allele"] == target_allele).any():
        raise ValueError("target allele leaked into PWM training partition")

    nearest_sources: list[tuple[float, str]] = []
    target_sequence = pseudo_sequences[target_allele]
    for source_allele, source_rows in train.groupby("Allele", sort=True):
        source_allele_name = str(source_allele)
        if set(source_rows["Target"]) != {True, False}:
            continue
        _require_pseudo_sequence(source_allele_name, pseudo_sequences)
        source_sequence = pseudo_sequences[source_allele_name]
        distance = _normalized_hamming_distance(target_sequence, source_sequence)
        nearest_sources.append((distance, source_allele_name))
    if not nearest_sources:
        raise ValueError(f"no eligible PWM source for target allele {target_allele}")
    return min(nearest_sources)[1]


def _normalized_hamming_distance(sequence_a: str, sequence_b: str) -> float:
    """Fraction of differing residues across the fixed 34 pseudo-sequence positions."""
    return sum(
        residue_a != residue_b
        for residue_a, residue_b in zip(sequence_a, sequence_b, strict=True)
    ) / PSEUDO_SEQUENCE_LENGTH


def preflight_partitions(
    partitions: Sequence[TransferPartition],
    pseudo_sequences: Mapping[str, str],
    expected_target_alleles: Sequence[str],
) -> None:
    """Validate every target partition before a transfer track scores any arm."""
    if not partitions:
        raise ValueError("transfer track has no target partitions")
    expected_targets = tuple(expected_target_alleles)
    _require_held_out_alleles(expected_targets)
    partition_targets = {
        allele
        for partition in partitions
        for allele in partition.held_out_alleles
    }
    if partition_targets != set(expected_targets):
        raise ValueError("held-out targets differ from expected eligible alleles")
    _require_pseudo_sequence_mapping(pseudo_sequences)
    build_shuffled_pseudo_mapping(pseudo_sequences)

    checked_targets: set[str] = set()
    for partition in partitions:
        if partition.label not in (ALLELE_ONLY_LABEL, JOINT_NOVELTY_LABEL):
            raise ValueError(f"unsupported transfer partition label: {partition.label}")
        _require_partition_columns(partition.train)
        _require_partition_columns(partition.test)
        _require_held_out_alleles(partition.held_out_alleles)
        held_out_alleles = set(partition.held_out_alleles)
        if held_out_alleles & checked_targets:
            raise ValueError("target allele appears in more than one transfer partition")
        if set(partition.test["Allele"]) != held_out_alleles:
            raise ValueError("test partition does not match held-out alleles")
        if partition.train["Allele"].isin(held_out_alleles).any():
            raise ValueError("held-out allele leaked into training partition")
        if partition.label == JOINT_NOVELTY_LABEL and (
            set(partition.train["Peptide"]) & set(partition.test["Peptide"])
        ):
            raise ValueError("test peptide leaked into training partition")
        _require_partition_pseudo_sequences(partition, pseudo_sequences)
        if set(partition.train["Target"]) != {True, False}:
            raise ValueError("training partition must contain both target classes")

        for target_allele in sorted(held_out_alleles):
            target_rows = partition.test.loc[
                partition.test["Allele"] == target_allele
            ]
            if set(target_rows["Target"]) != {True, False}:
                raise ValueError("test partition must contain both target classes")
            _require_pseudo_sequence(target_allele, pseudo_sequences)
            select_nearest_pwm_source(
                partition.train,
                target_allele,
                pseudo_sequences,
            )
        split_transfer_fit_validation(partition.train)
        checked_targets.update(held_out_alleles)


@dataclass(frozen=True)
class JointNoveltyTargetDiagnostics:
    """Deletion and transfer diagnostics for one joint allele-and-peptide target."""

    target_allele: str
    deleted_training_rows: int
    retained_positive_rows: int
    retained_negative_rows: int
    retained_allele_count: int
    nearest_pwm_source: str
    nearest_pseudo_sequence_distance: float


def build_joint_novelty_schedule(
    rows: pd.DataFrame,
    alleles: Sequence[str],
    pseudo_sequences: Mapping[str, str],
) -> tuple[TransferPartition, ...]:
    """Build the joint allele-and-peptide novelty partition schedule for every target.

    Every target additionally loses training rows whose peptide occurs in its
    own held-out rows. This is a compound intervention, not an isolated
    peptide-novelty effect — see `build_joint_novelty_diagnostics` for the
    per-target deletion accounting. The shared all-or-fail preflight runs
    before this returns, so one invalid target aborts the whole schedule.
    """
    target_alleles = tuple(sorted(alleles))
    _require_held_out_alleles(target_alleles)
    partitions = tuple(
        build_joint_novelty_partition(rows, target_allele)
        for target_allele in target_alleles
    )
    preflight_partitions(partitions, pseudo_sequences, target_alleles)
    return partitions


def build_joint_novelty_diagnostics(
    partitions: Sequence[TransferPartition],
    rows: pd.DataFrame,
    pseudo_sequences: Mapping[str, str],
) -> tuple[JointNoveltyTargetDiagnostics, ...]:
    """Report deletion and transfer diagnostics for every joint-novelty target."""
    return tuple(
        _joint_novelty_target_diagnostics(rows, partition, pseudo_sequences)
        for partition in partitions
    )


def _joint_novelty_target_diagnostics(
    rows: pd.DataFrame,
    partition: TransferPartition,
    pseudo_sequences: Mapping[str, str],
) -> JointNoveltyTargetDiagnostics:
    (target_allele,) = partition.held_out_alleles
    non_target_rows = rows.loc[rows["Allele"] != target_allele]
    nearest_source = select_nearest_pwm_source(
        partition.train, target_allele, pseudo_sequences
    )
    return JointNoveltyTargetDiagnostics(
        target_allele=target_allele,
        deleted_training_rows=len(non_target_rows) - len(partition.train),
        retained_positive_rows=int(partition.train["Target"].sum()),
        retained_negative_rows=int((~partition.train["Target"]).sum()),
        retained_allele_count=int(partition.train["Allele"].nunique()),
        nearest_pwm_source=nearest_source,
        nearest_pseudo_sequence_distance=_normalized_hamming_distance(
            pseudo_sequences[target_allele], pseudo_sequences[nearest_source]
        ),
    )


def score_transfer_partition(
    partition: TransferPartition,
    pseudo_sequences: Mapping[str, str],
    *,
    device: str,
) -> dict[str, np.ndarray]:
    """Score one validated transfer partition with the fixed compatible arms."""
    preflight_partitions(
        (partition,),
        pseudo_sequences,
        partition.held_out_alleles,
    )
    fit_indices, validation_indices = split_transfer_fit_validation(partition.train)
    shuffled_mapping = build_shuffled_pseudo_mapping(pseudo_sequences)
    scores = {
        "random": random_scores(len(partition.test)),
        "peptide_only_mlp": _score_peptide_only_mlp(
            partition,
            fit_indices,
            validation_indices,
            device=device,
        ),
        "shuffled_mapping_mlp": score_mlp_fold(
            partition.train,
            partition.test,
            shuffled_mapping,
            (),
            fit_indices,
            validation_indices,
            use_one_hot_allele=False,
            device=device,
        ),
        "nearest_pwm": _score_nearest_pwm(
            partition,
            pseudo_sequences,
        ),
        "pseudo_sequence_mlp": score_mlp_fold(
            partition.train,
            partition.test,
            pseudo_sequences,
            (),
            fit_indices,
            validation_indices,
            use_one_hot_allele=False,
            device=device,
        ),
    }
    _require_valid_score_arrays(scores, len(partition.test))
    return scores


def evaluate_transfer_schedule(
    partitions: Sequence[TransferPartition],
    pseudo_sequences: Mapping[str, str],
    expected_target_alleles: Sequence[str],
    *,
    device: str,
) -> dict[str, object]:
    """Preflight the complete frozen schedule and return its aggregate result core.

    Callers supply the frozen source cohort, never a subset inferred from successful
    fits. Track diagnostics and source provenance belong outside this core.
    """
    preflight_partitions(partitions, pseudo_sequences, expected_target_alleles)
    if len({partition.label for partition in partitions}) != 1:
        raise ValueError("transfer schedule must not combine different estimands")
    partition_scores = []
    for partition in partitions:
        scores = score_transfer_partition(partition, pseudo_sequences, device=device)
        _require_valid_score_arrays(scores, len(partition.test))
        partition_scores.append(scores)
    rows = pd.concat([partition.test for partition in partitions], ignore_index=True)
    targets = rows["Target"].to_numpy(dtype=bool)
    alleles = rows["Allele"].astype(str).to_numpy()
    combined_scores = {
        arm: np.concatenate([scores[arm] for scores in partition_scores])
        for arm in TRANSFER_SCORE_NAMES
    }
    reports = {
        arm: evaluate(
            targets, combined_scores[arm], groups=alleles, label=arm,
            mode="both", n_boot=TRANSFER_BOOTSTRAP_DRAWS,
            confidence=0.95, seed=TRANSFER_BOOTSTRAP_SEED, on_degenerate="warn",
        )
        for arm in TRANSFER_SCORE_NAMES
    }
    comparisons = {}
    for reference in TRANSFER_REFERENCES:
        comparison = compare_macro_auc01(
            targets, alleles,
            ("pseudo_sequence_mlp", combined_scores["pseudo_sequence_mlp"], None),
            (reference, combined_scores[reference], None),
            mode="both", n_boot=TRANSFER_BOOTSTRAP_DRAWS,
            confidence=0.95, seed=TRANSFER_BOOTSTRAP_SEED,
        )
        comparisons[f"pseudo_sequence_mlp_minus_{reference}"] = {
            "arm": comparison.label_a,
            "reference": comparison.label_b,
            "arm_point": comparison.point_a,
            "reference_point": comparison.point_b,
            "difference": asdict(comparison.difference),
            "p_two_sided": comparison.p_two_sided,
            "n_alleles": comparison.n_groups,
        }
    result = {
        "config": {
            "arms": list(TRANSFER_SCORE_NAMES),
            "target_alleles": sorted(expected_target_alleles),
            "classification_threshold": 0.426,
            "headline": "macro_standardized_auc01",
            "bootstrap": {
                "mode": "both", "draws": TRANSFER_BOOTSTRAP_DRAWS,
                "seed": TRANSFER_BOOTSTRAP_SEED, "group": "Allele", "confidence": 0.95,
            },
        },
        "scores": {
            arm: {
                "macro_auc01": asdict(report.macro_auc01),
                "n_rows": report.n_rows,
                "n_positive": report.n_positive,
                "n_alleles": report.n_groups_scored,
                "n_alleles_skipped": report.n_groups_skipped,
            }
            for arm, report in reports.items()
        },
        "comparisons": comparisons,
        "per_allele": {
            str(allele): {
                "n_rows": len(allele_rows),
                "n_positive": int(allele_rows["Target"].sum()),
                "arms": {
                    arm: {"auc01": report.per_group_auc01[str(allele)]}
                    for arm, report in reports.items()
                },
            }
            for allele, allele_rows in rows.groupby("Allele", sort=True)
        },
    }
    validate_transfer_result(result, expected_target_alleles)
    return result


def _require_result_fields(
    payload: object, fields: Sequence[str], label: str
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != set(fields):
        raise ValueError(f"invalid {label} fields")
    return payload


def _require_result_number(number: object, label: str) -> float:
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not np.isfinite(number)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(number)


def _require_result_count(count: object, label: str) -> int:
    if type(count) is not int or count < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return count


def _require_result_interval(payload: object, *, is_difference: bool) -> float:
    interval = _require_result_fields(
        payload, ("point", "lo", "hi", "n_replicates"), "interval"
    )
    point, lower, upper = (
        _require_result_number(interval[field], field) for field in ("point", "lo", "hi")
    )
    minimum = -1.0 if is_difference else 0.0
    if not minimum <= point <= 1.0 or not minimum <= lower <= upper <= 1.0:
        raise ValueError("interval is outside its metric bounds")
    replicates = _require_result_count(interval["n_replicates"], "n_replicates")
    if not 0 < replicates <= TRANSFER_BOOTSTRAP_DRAWS:
        raise ValueError("invalid successful bootstrap replicate count")
    return point


def validate_transfer_result(
    result: Mapping[str, object], expected_target_alleles: Sequence[str]
) -> None:
    """Reject incomplete, nonfinite, or nonaggregate transfer result cores."""
    expected_targets = tuple(expected_target_alleles)
    _require_held_out_alleles(expected_targets)
    core = _require_result_fields(
        result, ("config", "scores", "comparisons", "per_allele"), "result core"
    )
    config = _require_result_fields(
        core["config"],
        ("arms", "target_alleles", "classification_threshold", "headline", "bootstrap"),
        "config",
    )
    if (
        config["arms"] != list(TRANSFER_SCORE_NAMES)
        or config["target_alleles"] != sorted(expected_targets)
        or config["classification_threshold"] != 0.426
        or config["headline"] != "macro_standardized_auc01"
        or config["bootstrap"] != {
            "mode": "both", "draws": TRANSFER_BOOTSTRAP_DRAWS,
            "seed": TRANSFER_BOOTSTRAP_SEED, "group": "Allele", "confidence": 0.95,
        }
    ):
        raise ValueError("transfer result config differs from the frozen contract")
    per_allele = _require_result_fields(core["per_allele"], expected_targets, "per_allele")
    arm_points: dict[str, list[float]] = {arm: [] for arm in TRANSFER_SCORE_NAMES}
    total_rows = 0
    total_positive = 0
    for allele, payload in per_allele.items():
        allele_payload = _require_result_fields(
            payload, ("n_rows", "n_positive", "arms"), str(allele)
        )
        row_count = _require_result_count(allele_payload["n_rows"], "n_rows")
        positive_count = _require_result_count(allele_payload["n_positive"], "n_positive")
        if not 0 < positive_count < row_count:
            raise ValueError("per-allele result must retain both target classes")
        total_rows += row_count
        total_positive += positive_count
        arms = _require_result_fields(allele_payload["arms"], TRANSFER_SCORE_NAMES, "arms")
        for arm, score in arms.items():
            allele_score = _require_result_fields(score, ("auc01",), "per-allele score")
            point = _require_result_number(allele_score["auc01"], "auc01")
            if not 0 <= point <= 1:
                raise ValueError("per-allele AUC0.1 is outside metric bounds")
            arm_points[arm].append(point)
    scores = _require_result_fields(core["scores"], TRANSFER_SCORE_NAMES, "scores")
    macro_points: dict[str, float] = {}
    for arm, payload in scores.items():
        score = _require_result_fields(
            payload,
            ("macro_auc01", "n_rows", "n_positive", "n_alleles", "n_alleles_skipped"),
            "score",
        )
        for field, expected_count in (
            ("n_rows", total_rows), ("n_positive", total_positive),
            ("n_alleles", len(expected_targets)), ("n_alleles_skipped", 0),
        ):
            if _require_result_count(score[field], field) != expected_count:
                raise ValueError(f"{arm} {field} differs from the frozen cohort")
        macro_points[arm] = _require_result_interval(score["macro_auc01"], is_difference=False)
        if not np.isclose(macro_points[arm], np.mean(arm_points[arm]), rtol=0, atol=1e-12):
            raise ValueError("macro AUC0.1 differs from the held-out allele mean")
    comparisons = _require_result_fields(
        core["comparisons"],
        tuple(f"pseudo_sequence_mlp_minus_{reference}" for reference in TRANSFER_REFERENCES),
        "comparisons",
    )
    for reference in TRANSFER_REFERENCES:
        comparison = _require_result_fields(
            comparisons[f"pseudo_sequence_mlp_minus_{reference}"],
            ("arm", "reference", "arm_point", "reference_point", "difference",
             "p_two_sided", "n_alleles"),
            "comparison",
        )
        if comparison["arm"] != "pseudo_sequence_mlp" or comparison["reference"] != reference:
            raise ValueError("comparison arms differ from the fixed contract")
        if _require_result_count(comparison["n_alleles"], "n_alleles") != len(expected_targets):
            raise ValueError("comparison does not cover the frozen cohort")
        for field, arm in (("arm_point", "pseudo_sequence_mlp"), ("reference_point", reference)):
            point = _require_result_number(comparison[field], field)
            if not np.isclose(point, macro_points[arm], rtol=0, atol=1e-12):
                raise ValueError("comparison point differs from the arm score")
        difference = _require_result_interval(comparison["difference"], is_difference=True)
        if not np.isclose(
            difference, macro_points["pseudo_sequence_mlp"] - macro_points[reference],
            rtol=0, atol=1e-12,
        ):
            raise ValueError("paired difference does not match the arm points")
        if not 0 <= _require_result_number(comparison["p_two_sided"], "p_two_sided") <= 1:
            raise ValueError("p_two_sided must be between zero and one")


def _score_peptide_only_mlp(
    partition: TransferPartition,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Average the fixed MLP ensemble using BLOSUM50 peptide features only."""
    train_features = encode_blosum50(partition.train["Peptide"].astype(str))
    test_features = encode_blosum50(partition.test["Peptide"].astype(str))
    affinities = partition.train["Affinity"].to_numpy(dtype=float)
    predictions = [
        _score_single_mlp(
            train_features[fit_indices],
            affinities[fit_indices],
            train_features[validation_indices],
            affinities[validation_indices],
            test_features,
            hidden=hidden,
            seed=seed,
            device=device,
        )
        for hidden in MLP_HIDDEN_SIZES
        for seed in MLP_SEEDS
    ]
    return np.mean(predictions, axis=0)


def _score_nearest_pwm(
    partition: TransferPartition,
    pseudo_sequences: Mapping[str, str],
) -> np.ndarray:
    """Score each target allele with its nearest eligible retained PWM."""
    scores = np.empty(len(partition.test), dtype=float)
    test_alleles = partition.test["Allele"].astype(str).to_numpy()
    for target_allele in sorted(set(test_alleles)):
        row_indices = np.flatnonzero(test_alleles == target_allele)
        source_allele = select_nearest_pwm_source(
            partition.train,
            target_allele,
            pseudo_sequences,
        )
        source_test = partition.test.iloc[row_indices].copy()
        source_test.loc[:, "Allele"] = source_allele
        scores[row_indices] = score_pwm_fold(partition.train, source_test)
    return scores


def _require_valid_score_arrays(
    scores: Mapping[str, np.ndarray], test_rows: int
) -> None:
    if tuple(scores) != TRANSFER_SCORE_NAMES:
        raise ValueError("transfer scores do not match the fixed arm contract")
    for score_name, score_values in scores.items():
        if score_values.shape != (test_rows,):
            raise ValueError(f"transfer score {score_name} does not cover each test row once")
        if not np.isfinite(score_values).all():
            raise ValueError(f"transfer score {score_name} contains non-finite values")


def _require_partition_columns(rows: pd.DataFrame) -> None:
    missing_columns = sorted(_REQUIRED_COLUMNS - set(rows.columns))
    if missing_columns:
        raise ValueError(f"pMHC rows missing required columns: {missing_columns}")


def _require_pseudo_sequence_mapping(pseudo_sequences: Mapping[str, str]) -> None:
    if not pseudo_sequences:
        raise ValueError("pseudo-sequence mapping is empty")
    for allele, pseudo_sequence in pseudo_sequences.items():
        if not isinstance(allele, str):
            raise ValueError("pseudo-sequence mapping contains a non-string allele")
        if not isinstance(pseudo_sequence, str):
            raise ValueError(f"pseudo-sequence for allele {allele} must be a string")
        if len(pseudo_sequence) != PSEUDO_SEQUENCE_LENGTH:
            raise ValueError(
                f"pseudo-sequence for allele {allele} has {len(pseudo_sequence)} residues, "
                f"expected {PSEUDO_SEQUENCE_LENGTH} residues"
            )


def _require_pseudo_sequence(
    allele: str, pseudo_sequences: Mapping[str, str]
) -> None:
    pseudo_sequence = pseudo_sequences.get(allele)
    if pseudo_sequence is None:
        raise ValueError(f"missing pseudo-sequence for allele {allele}")
    if not isinstance(pseudo_sequence, str):
        raise ValueError(f"pseudo-sequence for allele {allele} must be a string")
    if len(pseudo_sequence) != PSEUDO_SEQUENCE_LENGTH:
        raise ValueError(
            f"pseudo-sequence for allele {allele} has {len(pseudo_sequence)} residues, "
            f"expected {PSEUDO_SEQUENCE_LENGTH} residues"
        )


def _require_partition_pseudo_sequences(
    partition: TransferPartition,
    pseudo_sequences: Mapping[str, str],
) -> None:
    partition_alleles = set(partition.train["Allele"]) | set(partition.test["Allele"])
    for allele in sorted(partition_alleles):
        _require_pseudo_sequence(str(allele), pseudo_sequences)


def _require_held_out_alleles(held_out_alleles: tuple[str, ...]) -> None:
    if not held_out_alleles:
        raise ValueError("held-out allele set is empty")
    if len(set(held_out_alleles)) != len(held_out_alleles):
        raise ValueError("held-out alleles must be unique")


def _require_valid_partition(
    train: pd.DataFrame,
    test: pd.DataFrame,
    held_out_alleles: tuple[str, ...],
    *,
    exclude_test_peptides: bool,
) -> None:
    if train.empty:
        raise ValueError("training partition is empty")
    if test.empty:
        raise ValueError("test partition is empty")
    if set(test["Target"]) != {True, False}:
        raise ValueError("test partition must contain both target classes")
    if train["Allele"].isin(held_out_alleles).any():
        raise ValueError("held-out allele leaked into training partition")
    if exclude_test_peptides and set(train["Peptide"]) & set(test["Peptide"]):
        raise ValueError("test peptide leaked into training partition")
