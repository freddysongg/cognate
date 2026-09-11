"""Run the fixed pMHC binding replication experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.baseline_knn import cosine_similarity
from cognate.embed import (
    MODELS,
    EmbeddingCache,
    embed_sequences,
    load_cache,
    pick_device,
    save_cache,
)
from cognate.metrics import (
    Comparison,
    Interval,
    ScoreReport,
    auroc,
    compare_macro_auc01,
    evaluate,
    macro_by_group,
)
from cognate.pmhc import (
    FOLD_NAMES,
    MLP_HIDDEN_SIZES,
    MLP_LEARNING_RATE,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    MLP_SEEDS,
    PmhcDataset,
    iter_pmhc_folds,
    load_pmhc_dataset,
    load_pseudo_sequences,
    load_source_contract,
    random_scores,
    score_composition_fold,
    score_mlp_fold,
    score_pwm_fold,
    score_retrieval_fold,
    split_fit_validation,
    verify_source,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT_PATH = ROOT / "data" / "pmhc" / "source_contract.json"
ARCHIVE_PATH = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train.tar.gz"
SOURCE_DIR = ROOT / "data" / "pmhc" / "raw" / "NetMHCpan_train"
EMBEDDING_CACHE_PATH = ROOT / "shared" / "pmhc" / "esm2_35M_peptides.npz"
RESULTS_PATH = ROOT / "data" / "pmhc" / "results.json"

ESM_MODEL_KEY = "35M"
ESM_LAYER = 10
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 0
NON_INFERIORITY_MARGIN = 0.02
EXPECTED_HEADLINE_COUNTS = {
    "alleles": 47,
    "rows": 112_128,
    "positive": 28_538,
    "negative": 83_590,
    "measured_nonbinder": 82_448,
    "artificial_negative": 1_142,
}
ARM_NAMES = (
    "random",
    "length_composition",
    "edit_retrieval",
    "esm_retrieval",
    "pwm",
    "mlp_pseudo_sequence",
    "mlp_one_hot",
)
NEGATIVE_SOURCES = ("measured_nonbinder", "artificial_negative")


@dataclass(frozen=True)
class ScoredDataset:
    scores: dict[str, np.ndarray]
    nearest_positive_edit_distance: np.ndarray


def validate_score_arrays(
    scores: Mapping[str, np.ndarray],
    assignment_counts: Mapping[str, np.ndarray],
    n_rows: int,
) -> None:
    """Reject incomplete, duplicate, malformed, or non-finite score arrays."""
    expected_arms = set(ARM_NAMES)
    if set(scores) != expected_arms:
        raise ValueError(
            f"score arm mismatch: expected {sorted(expected_arms)}, got {sorted(scores)}"
        )
    if set(assignment_counts) != expected_arms:
        raise ValueError("score assignment arm mismatch")
    for arm in ARM_NAMES:
        arm_scores = np.asarray(scores[arm], dtype=float)
        counts = np.asarray(assignment_counts[arm], dtype=int)
        if arm_scores.shape != (n_rows,) or counts.shape != (n_rows,):
            raise ValueError(f"{arm} score shape mismatch for {n_rows} rows")
        incorrectly_assigned = np.flatnonzero(counts != 1)
        if len(incorrectly_assigned):
            raise ValueError(
                f"{arm} did not score every row exactly once; "
                f"first bad rows {incorrectly_assigned[:5].tolist()}"
            )
        nonfinite = np.flatnonzero(~np.isfinite(arm_scores))
        if len(nonfinite):
            raise ValueError(
                f"{arm} contains non-finite scores at rows {nonfinite[:5].tolist()}"
            )


def validate_embedding_cache(
    cache: EmbeddingCache, expected_peptides: Sequence[str]
) -> None:
    """Require the fixed checkpoint, headline layer, and exact peptide set."""
    if cache.model_name != MODELS[ESM_MODEL_KEY]:
        raise ValueError(
            f"embedding cache model mismatch: expected {MODELS[ESM_MODEL_KEY]}, "
            f"got {cache.model_name}"
        )
    if cache.n_layers <= ESM_LAYER:
        raise ValueError(
            f"embedding cache layer mismatch: layer {ESM_LAYER} is unavailable"
        )
    expected = tuple(sorted(set(map(str, expected_peptides))))
    cached = tuple(sorted(map(str, cache.sequences)))
    if cached != expected:
        raise ValueError(
            "embedding cache peptides mismatch: "
            f"expected {len(expected)}, got {len(cached)}"
        )
    if not np.isfinite(cache.layer(ESM_LAYER)).all():
        raise ValueError(f"embedding cache layer {ESM_LAYER} contains non-finite values")


def load_or_build_embedding_cache(
    peptides: Sequence[str], cache_path: Path, *, device: str
) -> EmbeddingCache:
    """Load a matching fixed ESM cache or build it once when absent."""
    expected_peptides = tuple(sorted(set(map(str, peptides))))
    if cache_path.exists():
        cache = load_cache(cache_path)
        validate_embedding_cache(cache, expected_peptides)
        print(f"Reused ESM-2 cache with {cache.n_sequences:,} peptides", flush=True)
        return cache

    cache, elapsed_seconds = embed_sequences(
        expected_peptides,
        ESM_MODEL_KEY,
        device=device,
        progress_every=20,
    )
    validate_embedding_cache(cache, expected_peptides)
    size_bytes = save_cache(cache, cache_path)
    print(
        f"Built ESM-2 cache with {cache.n_sequences:,} peptides in "
        f"{elapsed_seconds:.1f}s ({size_bytes / 1e9:.2f} GB)",
        flush=True,
    )
    return cache


def _assign_fold_scores(
    scores: dict[str, np.ndarray],
    assignment_counts: dict[str, np.ndarray],
    arm: str,
    positions: np.ndarray,
    fold_scores: np.ndarray,
) -> None:
    values = np.asarray(fold_scores, dtype=float)
    if values.shape != (len(positions),):
        raise ValueError(
            f"{arm} returned shape {values.shape} for {len(positions)} test rows"
        )
    scores[arm][positions] = values
    assignment_counts[arm][positions] += 1


def score_dataset(
    dataset: PmhcDataset,
    pseudo_sequences: Mapping[str, str],
    embedding_cache: EmbeddingCache,
    *,
    device: str,
) -> ScoredDataset:
    """Score every supplied outer fold and restore original dataset row order."""
    n_rows = len(dataset.rows)
    peptides = dataset.rows["Peptide"].astype(str)
    validate_embedding_cache(embedding_cache, tuple(peptides))
    scores = {arm: np.full(n_rows, np.nan, dtype=float) for arm in ARM_NAMES}
    assignment_counts = {
        arm: np.zeros(n_rows, dtype=np.int8) for arm in ARM_NAMES
    }
    nearest_positive_edit_distance = np.full(n_rows, np.nan, dtype=float)
    fixed_random_scores = random_scores(n_rows)
    esm_similarity = cosine_similarity(embedding_cache, ESM_LAYER)

    for fold_name, train_rows, test_rows in iter_pmhc_folds(dataset):
        positions = np.flatnonzero(dataset.rows["Fold"].to_numpy() == fold_name)
        if len(positions) != len(test_rows):
            raise ValueError(f"fold {fold_name} row-order mismatch")
        print(
            f"Scoring {fold_name}: {len(train_rows):,} train, {len(test_rows):,} test",
            flush=True,
        )
        edit_result = score_retrieval_fold(test_rows, train_rows)
        esm_result = score_retrieval_fold(
            test_rows,
            train_rows,
            similarity_fn=esm_similarity,
        )
        fit_indices, validation_indices = split_fit_validation(train_rows)
        fold_results = {
            "random": fixed_random_scores[positions],
            "length_composition": score_composition_fold(train_rows, test_rows),
            "edit_retrieval": edit_result.score,
            "esm_retrieval": esm_result.score,
            "pwm": score_pwm_fold(train_rows, test_rows),
            "mlp_pseudo_sequence": score_mlp_fold(
                train_rows,
                test_rows,
                pseudo_sequences,
                dataset.eligible_alleles,
                fit_indices,
                validation_indices,
                use_one_hot_allele=False,
                device=device,
            ),
            "mlp_one_hot": score_mlp_fold(
                train_rows,
                test_rows,
                pseudo_sequences,
                dataset.eligible_alleles,
                fit_indices,
                validation_indices,
                use_one_hot_allele=True,
                device=device,
            ),
        }
        for arm, fold_scores in fold_results.items():
            _assign_fold_scores(scores, assignment_counts, arm, positions, fold_scores)
        peptide_lengths = test_rows["Peptide"].astype(str).str.len().to_numpy()
        nearest_positive_edit_distance[positions] = np.rint(
            (1.0 - edit_result.score) * peptide_lengths
        )

    validate_score_arrays(scores, assignment_counts, n_rows)
    if not np.isfinite(nearest_positive_edit_distance).all():
        raise ValueError("nearest-positive edit distances contain non-finite values")
    return ScoredDataset(scores, nearest_positive_edit_distance)


def _interval_payload(interval: Interval) -> dict[str, float | int]:
    return {
        "point": float(interval.point),
        "lo": float(interval.lo),
        "hi": float(interval.hi),
        "n_replicates": int(interval.n_replicates),
    }


def _score_payload(
    report: ScoreReport, macro_auroc_point: float
) -> dict[str, object]:
    return {
        "macro_auc01": _interval_payload(report.macro_auc01),
        "pooled_auroc": _interval_payload(report.auroc),
        "macro_auroc_point": float(macro_auroc_point),
        "pooled_auprc": _interval_payload(report.auprc),
        "n_rows": int(report.n_rows),
        "n_positive": int(report.n_positive),
        "n_alleles": int(report.n_groups_scored),
        "n_alleles_skipped": int(report.n_groups_skipped),
        "is_degenerate": bool(report.diagnostics.is_degenerate),
    }


def _comparison_payload(comparison: Comparison) -> dict[str, object]:
    return {
        "arm": comparison.label_a,
        "reference": comparison.label_b,
        "arm_point": float(comparison.point_a),
        "reference_point": float(comparison.point_b),
        "difference": _interval_payload(comparison.difference),
        "p_two_sided": float(comparison.p_two_sided),
        "n_alleles": int(comparison.n_groups),
    }


def _evaluate_arm(
    rows: pd.DataFrame,
    arm: str,
    arm_scores: np.ndarray,
    *,
    subset: np.ndarray | None = None,
    label: str,
) -> tuple[ScoreReport, dict[str, float]]:
    targets = rows["Target"].to_numpy(dtype=bool)
    alleles = rows["Allele"].astype(str).to_numpy()
    report = evaluate(
        targets,
        arm_scores,
        groups=alleles,
        subset=subset,
        label=label,
        mode="both",
        n_boot=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
        on_degenerate="warn",
    )
    active = np.ones(len(rows), dtype=bool) if subset is None else subset
    macro_auroc = macro_by_group(
        targets[active], arm_scores[active], alleles[active], auroc
    )
    return report, macro_auroc.per_group


def evaluate_predictions(
    rows: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    nearest_positive_edit_distance: np.ndarray,
) -> dict[str, object]:
    """Evaluate fixed headline, paired, and negative-source contracts."""
    assignment_counts = {
        arm: np.ones(len(rows), dtype=np.int8) for arm in ARM_NAMES
    }
    validate_score_arrays(scores, assignment_counts, len(rows))
    distances = np.asarray(nearest_positive_edit_distance, dtype=float)
    if distances.shape != (len(rows),) or not np.isfinite(distances).all():
        raise ValueError("nearest-positive edit distances contain non-finite values")

    reports: dict[str, ScoreReport] = {}
    macro_aurocs: dict[str, dict[str, float]] = {}
    score_payloads: dict[str, object] = {}
    for arm in ARM_NAMES:
        report, per_allele_auroc = _evaluate_arm(
            rows, arm, np.asarray(scores[arm]), label=arm
        )
        reports[arm] = report
        macro_aurocs[arm] = per_allele_auroc
        macro_auroc_point = float(np.mean(list(per_allele_auroc.values())))
        score_payloads[arm] = _score_payload(report, macro_auroc_point)

    per_allele: dict[str, object] = {}
    for allele, allele_rows in rows.groupby("Allele", sort=True):
        per_allele[str(allele)] = {
            "n_rows": int(len(allele_rows)),
            "n_positive": int(allele_rows["Target"].sum()),
            "arms": {
                arm: {
                    "auc01": float(reports[arm].per_group_auc01[str(allele)]),
                    "auroc": float(macro_aurocs[arm][str(allele)]),
                }
                for arm in ARM_NAMES
            },
        }

    targets = rows["Target"].to_numpy(dtype=bool)
    alleles = rows["Allele"].astype(str).to_numpy()
    comparisons: dict[str, object] = {}
    for arm in ARM_NAMES:
        if arm == "random":
            continue
        comparison = compare_macro_auc01(
            targets,
            alleles,
            (arm, np.asarray(scores[arm]), None),
            ("random", np.asarray(scores["random"]), None),
            mode="both",
            n_boot=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        comparisons[f"{arm}_minus_random"] = _comparison_payload(comparison)
    mlp_minus_pwm = compare_macro_auc01(
        targets,
        alleles,
        ("mlp_pseudo_sequence", np.asarray(scores["mlp_pseudo_sequence"]), None),
        ("pwm", np.asarray(scores["pwm"]), None),
        mode="both",
        n_boot=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )
    comparisons["mlp_pseudo_sequence_minus_pwm"] = _comparison_payload(
        mlp_minus_pwm
    )

    negative_sources: dict[str, object] = {}
    retrieval_diagnostics: dict[str, object] = {}
    row_types = rows["RowType"].astype(str).to_numpy()
    for source in NEGATIVE_SOURCES:
        negative_mask = row_types == source
        subset = targets | negative_mask
        source_scores: dict[str, object] = {}
        for arm in ARM_NAMES:
            report, per_allele_auroc = _evaluate_arm(
                rows,
                arm,
                np.asarray(scores[arm]),
                subset=subset,
                label=f"{arm}/{source}",
            )
            source_scores[arm] = _score_payload(
                report, float(np.mean(list(per_allele_auroc.values())))
            )
        source_distances = distances[negative_mask]
        negative_sources[source] = {
            "n_rows": int(subset.sum()),
            "n_positive": int(targets.sum()),
            "n_negative": int(negative_mask.sum()),
            "scores": source_scores,
        }
        retrieval_diagnostics[source] = {
            "n_negative": int(len(source_distances)),
            "mean_nearest_positive_edit_distance": float(source_distances.mean()),
            "median_nearest_positive_edit_distance": float(np.median(source_distances)),
            "exact_duplicate_count": int((source_distances == 0).sum()),
        }

    criteria = derive_criteria(score_payloads, comparisons)
    return {
        "scores": score_payloads,
        "per_allele": per_allele,
        "comparisons": comparisons,
        "negative_sources": negative_sources,
        "retrieval_diagnostics": retrieval_diagnostics,
        "criteria": criteria,
    }


def is_predictive(interval: Mapping[str, float]) -> bool:
    """Apply the predeclared strict lower-bound criterion."""
    return float(interval["lo"]) > 0.5


def derive_criteria(
    scores: Mapping[str, Mapping[str, object]],
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Derive predictive and PWM-sufficiency verdicts from saved intervals."""
    predictive: dict[str, bool] = {}
    for arm in ARM_NAMES:
        headline = scores[arm]["macro_auc01"]
        if not isinstance(headline, Mapping):
            raise ValueError(f"missing macro-AUC0.1 interval for {arm}")
        predictive[arm] = is_predictive(headline)
    pwm_comparison = comparisons["mlp_pseudo_sequence_minus_pwm"]["difference"]
    if not isinstance(pwm_comparison, Mapping):
        raise ValueError("missing MLP-minus-PWM difference interval")
    comparison_upper_bound = float(pwm_comparison["hi"])
    return {
        "predictive": predictive,
        "pwm_sufficient_relative_to_mlp": (
            predictive["pwm"]
            and comparison_upper_bound < NON_INFERIORITY_MARGIN
        ),
        "non_inferiority_margin": NON_INFERIORITY_MARGIN,
        "mlp_pseudo_sequence_minus_pwm_upper_bound": comparison_upper_bound,
    }


def load_tcr_retrieval_contrast(
    retrieval_path: Path, random_path: Path
) -> dict[str, dict[str, float]]:
    """Read closed-TCR points and calculate uplift over its observed random arm."""
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    observed_random = float(
        random["scores"]["random/seen"]["macro_auc01"]["point"]
    )
    result: dict[str, dict[str, float]] = {}
    for arm, source_key in (
        ("edit_retrieval", "edit/seen"),
        ("esm_retrieval", "35M layer10 (headline)/seen"),
    ):
        point = float(retrieval["scores"][source_key]["macro_auc01"]["point"])
        result[arm] = {
            "observed_random": observed_random,
            "macro_auc01": point,
            "uplift_over_observed_random": point - observed_random,
        }
    return result


def _require_fixed_cohort(dataset: PmhcDataset) -> None:
    rows = dataset.rows
    observed = {
        "alleles": len(dataset.eligible_alleles),
        "rows": len(rows),
        "positive": int(rows["Target"].sum()),
        "negative": int((~rows["Target"]).sum()),
        "measured_nonbinder": int((rows["RowType"] == "measured_nonbinder").sum()),
        "artificial_negative": int(
            (rows["RowType"] == "artificial_negative").sum()
        ),
    }
    if observed != EXPECTED_HEADLINE_COUNTS:
        raise ValueError(
            f"fixed headline cohort mismatch: expected {EXPECTED_HEADLINE_COUNTS}, "
            f"got {observed}"
        )


def _experiment_config(device: str) -> dict[str, object]:
    return {
        "folds": list(FOLD_NAMES),
        "arms": list(ARM_NAMES),
        "classification_threshold": 0.426,
        "bootstrap": {
            "mode": "both",
            "draws": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "group": "Allele",
        },
        "random_seed": 0,
        "esm": {
            "model": MODELS[ESM_MODEL_KEY],
            "layer": ESM_LAYER,
            "pooling": "residue_mean",
            "similarity": "cosine",
        },
        "retrieval": {
            "operator": "nearest_positive",
            "group": "Allele",
            "edit_similarity": "normalized_levenshtein",
        },
        "pwm": {"positions": 9, "pseudocount": 1},
        "mlp": {
            "hidden_sizes": list(MLP_HIDDEN_SIZES),
            "seeds": list(MLP_SEEDS),
            "encodings": ["pseudo_sequence", "one_hot"],
            "fits_per_encoding": 50,
            "learning_rate": MLP_LEARNING_RATE,
            "loss": "mse_continuous_affinity",
            "max_epochs": MLP_MAX_EPOCHS,
            "patience": MLP_PATIENCE,
            "validation_fraction": 0.1,
        },
        "device": device,
    }


def main() -> None:
    contract = load_source_contract(SOURCE_CONTRACT_PATH)
    verify_source(SOURCE_DIR, ARCHIVE_PATH, contract)
    dataset = load_pmhc_dataset(SOURCE_DIR)
    _require_fixed_cohort(dataset)
    print(
        "Verified contract before scoring: 5 BA folds, 47 alleles, "
        "112,128 rows, 28,538 positives, 83,590 negatives, "
        "zero cross-fold peptide overlap",
        flush=True,
    )
    pseudo_sequences = load_pseudo_sequences(
        SOURCE_DIR / "MHC_pseudo.dat", dataset.eligible_alleles
    )
    device = pick_device()
    cache = load_or_build_embedding_cache(
        tuple(dataset.rows["Peptide"].astype(str)),
        EMBEDDING_CACHE_PATH,
        device=device,
    )
    scored = score_dataset(
        dataset,
        pseudo_sequences,
        cache,
        device=device,
    )
    evaluated = evaluate_predictions(
        dataset.rows,
        scored.scores,
        scored.nearest_positive_edit_distance,
    )
    artifact = {
        "contract": contract,
        "config": _experiment_config(device),
        **evaluated,
    }
    expected_keys = {
        "contract",
        "config",
        "scores",
        "per_allele",
        "comparisons",
        "negative_sources",
        "retrieval_diagnostics",
        "criteria",
    }
    if set(artifact) != expected_keys:
        raise AssertionError("result artifact top-level contract drifted")
    RESULTS_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote aggregate results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
