"""B3 -- rerun the Phase 1 analyses on the Phase B evaluation set and verdict the belief list.

Same models, same headline config (ESM-2 35M layer 10, `matched` negatives, ratio 5, seed 0),
same metric code. The only thing that changes is the evaluation set: 88 peptides instead of 20,
48 seen in training instead of 13, 40 unseen instead of 7, and every IMMREP23 training TCR
removed.

Each of the five claims in docs/belief_list.md is verdicted held / weakened / reversed against
its own recorded 'what would overturn it' line. Per-peptide sample size is reported alongside
every per-peptide number, and each correlation is refit twice more -- on well-supported peptides
only, and weighted by support -- because macro AUC0.1 weights a 52-TCR peptide identically to a
500-TCR one and 34 of the 88 sit under 100 positives.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
from scipy.stats import rankdata

from cognate.baseline_knn import score_by_nearest_positive
from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.features import build_features
from cognate.metrics import (
    auc01,
    auroc,
    compare_macro_auc01,
    correlation_difference_interval,
    correlation_interval,
    diagnose_scores,
    evaluate,
    leave_one_out_correlations,
    macro_auc01,
    report_table,
)
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, fit_mlp, logistic_scores, mlp_scores

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = default_cache_dir()
VDJDB_CACHE = CACHE_DIR / "esm2_35M_vdjdb.npz"
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
OUT_JSON = ROOT / "data" / "b3_results.json"
OUT_PER_PEPTIDE = ROOT / "data" / "b3_per_peptide.csv"

MODEL_KEY = "35M"
LAYER = 10
STRATEGY = "matched"
RATIO = 5.0
SEED = 0
WELL_SUPPORTED_MIN_POSITIVES = 100


def nearest_training_distance(frame: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    queries = np.array(sorted(set(frame["CDR3b"].astype(str))), dtype=object)
    database = np.array(sorted(set(train["CDR3b"].astype(str))), dtype=object)
    similarity = cdist(
        queries, database, scorer=Levenshtein.normalized_similarity, workers=-1
    )
    closest = dict(zip(queries, 1.0 - similarity.max(axis=1)))
    return np.array([closest[s] for s in frame["CDR3b"].astype(str)])


def interval_dict(interval) -> dict:
    return {
        "point": round(interval.point, 4),
        "lo": round(interval.lo, 4),
        "hi": round(interval.hi, 4),
    }


def correlation_block(
    label: str, x: np.ndarray, y: np.ndarray, support: np.ndarray
) -> dict:
    """Point estimate, CI, leave-one-out range, and the two support-sensitivity refits."""
    interval = correlation_interval(x, y, seed=SEED)
    loo = leave_one_out_correlations(x, y)
    well_supported = support >= WELL_SUPPORTED_MIN_POSITIVES
    refit = (
        correlation_interval(x[well_supported], y[well_supported], seed=SEED)
        if well_supported.sum() >= 3
        else None
    )
    weights = support / support.sum()
    weighted_x = x - np.average(x, weights=weights)
    weighted_y = y - np.average(y, weights=weights)
    weighted_r = float(
        np.average(weighted_x * weighted_y, weights=weights)
        / np.sqrt(
            np.average(weighted_x**2, weights=weights)
            * np.average(weighted_y**2, weights=weights)
        )
    )
    return {
        "label": label,
        "n_points": int(len(x)),
        "r": interval_dict(interval),
        "spans_zero": bool(interval.lo <= 0.0 <= interval.hi),
        "leave_one_out_min": round(float(loo.min()), 4),
        "leave_one_out_max": round(float(loo.max()), 4),
        "n_well_supported": int(well_supported.sum()),
        "r_well_supported": interval_dict(refit) if refit else None,
        "r_support_weighted": round(weighted_r, 4),
    }


def main() -> None:
    positives = load_train()
    evalset = pd.read_csv(EVAL_CSV)
    y_eval = evalset["Label"].to_numpy()
    peptides = evalset["Peptide"]
    train_peptides = set(positives["Peptide"])
    is_seen = peptides.isin(train_peptides).to_numpy()

    print(f"eval: {len(evalset):,} rows, {peptides.nunique()} peptides "
          f"({is_seen.sum():,} seen rows / {(~is_seen).sum():,} unseen rows)")
    print(f"      {len(set(peptides[is_seen]))} seen peptides, "
          f"{len(set(peptides[~is_seen]))} unseen peptides\n")

    split = component_split(positives, validation_fraction=0.2, seed=SEED)
    train = pd.concat(
        [split.train, make_negatives(split.train, STRATEGY, RATIO, SEED)], ignore_index=True
    )
    validation = pd.concat(
        [split.validation, make_negatives(split.validation, STRATEGY, RATIO, SEED)],
        ignore_index=True,
    )
    train_cache = load_cache(cache_path(MODEL_KEY, CACHE_DIR))
    eval_cache = load_cache(VDJDB_CACHE)

    x_train = build_features(train, train_cache, LAYER)
    x_val = build_features(validation, train_cache, LAYER)
    x_eval = build_features(evalset, eval_cache, LAYER)
    y_train = train["Target"].to_numpy()

    logistic = fit_logistic(x_train, y_train, seed=SEED)
    logistic_eval = logistic_scores(logistic, x_eval)
    mlp, scaler, history = fit_mlp(
        x_train, y_train, x_val, validation["Target"].to_numpy(),
        validation["Peptide"].to_numpy(), seed=SEED,
    )
    mlp_eval = mlp_scores(mlp, scaler, x_eval)
    print(f"MLP: {history.n_epochs} epochs, best {history.best_epoch}, "
          f"val loss {min(history.val_loss):.4f}")

    knn_result = score_by_nearest_positive(evalset, positives)
    knn_eval = knn_result.score
    rng = np.random.default_rng(SEED)
    random_eval = rng.random(len(evalset))

    models = [
        ("random", random_eval),
        ("k-NN", knn_eval),
        ("logistic", logistic_eval),
        ("MLP", mlp_eval),
    ]

    print("\n=== degeneracy ===")
    degeneracy = {}
    for name, scores in models:
        for slice_name, mask in [("seen", is_seen), ("unseen", ~is_seen)]:
            d = diagnose_scores(scores[mask], peptides[mask])
            degeneracy[f"{name}/{slice_name}"] = {
                "is_degenerate": bool(d.is_degenerate),
                "distinct_scores": int(d.n_unique_scores),
                "constant_groups": int(d.n_constant_groups),
            }
            print(f"  {name:<9} {slice_name:<7} "
                  f"{'DEGENERATE' if d.is_degenerate else 'ok':<11} {d}")

    print("\n=== scores ===")
    reports, scored = [], {}
    for name, scores in models:
        for slice_name, mask in [("all", np.ones(len(evalset), bool)),
                                 ("seen", is_seen), ("unseen", ~is_seen)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = evaluate(y_eval, scores, peptides, subset=mask,
                                  label=f"{name} / {slice_name}", seed=SEED)
            reports.append(report)
            scored[f"{name}/{slice_name}"] = report
    print(report_table(reports).to_string())

    print("\n=== paired comparisons, k-NN minus head, same peptides both arms ===")
    comparisons = {}
    for slice_name, mask in [("seen", is_seen), ("unseen", ~is_seen)]:
        for head_name, head_scores_array in [("logistic", logistic_eval), ("MLP", mlp_eval)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comparison = compare_macro_auc01(
                    y_eval, peptides,
                    (f"k-NN {slice_name}", knn_eval, mask),
                    (f"{head_name} {slice_name}", head_scores_array, mask), seed=SEED,
                )
            comparisons[f"knn_minus_{head_name}/{slice_name}"] = {
                "point": round(comparison.difference.point, 4),
                "lo": round(comparison.difference.lo, 4),
                "hi": round(comparison.difference.hi, 4),
                "p": round(comparison.p_two_sided, 4),
                "n_peptides": int(comparison.n_groups),
            }
            print("  " + str(comparison))

    # --- per-peptide table -------------------------------------------------------------------
    distance = nearest_training_distance(evalset, positives)
    support = positives.groupby("Peptide").size()
    rows = []
    for peptide in sorted(set(peptides)):
        mask = (peptides == peptide).to_numpy()
        labels = y_eval[mask]
        binder = mask & (y_eval == 1)
        row = {
            "peptide": peptide,
            "n_positive": int(labels.sum()),
            "n_rows": int(mask.sum()),
            "seen": peptide in train_peptides,
            "train_support": int(support.get(peptide, 0)),
            "median_distance_to_train": float(np.median(distance[binder])),
            "knn_database_size": int(knn_result.database_size[mask].max()),
        }
        for name, scores in models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row[f"{name}_auc01"] = float(auc01(labels, scores[mask]))
            row[f"{name}_mean_score"] = float(scores[mask].mean())
            row[f"{name}_mean_score_neg"] = float(scores[mask & (y_eval == 0)][
                : int((~labels.astype(bool)).sum())
            ].mean()) if (mask & (y_eval == 0)).any() else float("nan")
        rows.append(row)
    per_peptide = pd.DataFrame(rows)
    per_peptide.to_csv(OUT_PER_PEPTIDE, index=False)

    seen_rows = per_peptide[per_peptide["seen"]].reset_index(drop=True)
    log_support = np.log10(seen_rows["train_support"].to_numpy())
    seen_support_n = seen_rows["n_positive"].to_numpy()
    all_support_n = per_peptide["n_positive"].to_numpy()

    print(f"\n=== per-peptide correlations "
          f"(seen peptides n={len(seen_rows)}, all n={len(per_peptide)}) ===")
    correlations = {
        "knn_auc01_vs_distance_seen": correlation_block(
            "2  k-NN AUC0.1 vs median distance to training (seen)",
            seen_rows["median_distance_to_train"].to_numpy(),
            seen_rows["k-NN_auc01"].to_numpy(), seen_support_n),
        "knn_auc01_vs_distance_all": correlation_block(
            "2b k-NN AUC0.1 vs median distance to training (all 88)",
            per_peptide["median_distance_to_train"].to_numpy(),
            per_peptide["k-NN_auc01"].to_numpy(), all_support_n),
        "head_auc01_vs_support_seen": correlation_block(
            "3  logistic AUC0.1 vs log10(training support) (seen)",
            log_support, seen_rows["logistic_auc01"].to_numpy(), seen_support_n),
        "knn_auc01_vs_support_seen": correlation_block(
            "3b k-NN AUC0.1 vs log10(training support) (seen)",
            log_support, seen_rows["k-NN_auc01"].to_numpy(), seen_support_n),
        "knn_mean_score_vs_support_seen": correlation_block(
            "5  k-NN mean raw score vs log10(training support) (seen)",
            log_support, seen_rows["k-NN_mean_score"].to_numpy(), seen_support_n),
        "head_auc01_vs_distance_seen": correlation_block(
            "-  logistic AUC0.1 vs median distance to training (seen)",
            seen_rows["median_distance_to_train"].to_numpy(),
            seen_rows["logistic_auc01"].to_numpy(), seen_support_n),
    }
    for block in correlations.values():
        marker = "spans zero" if block["spans_zero"] else "excludes zero"
        refit = block["r_well_supported"]
        refit_text = (f"  n>={WELL_SUPPORTED_MIN_POSITIVES}: {refit['point']:+.3f} "
                      f"[{refit['lo']:+.3f}, {refit['hi']:+.3f}]" if refit else "  n>=100: n/a")
        print(f"  {block['label']:<58} r = {block['r']['point']:+.3f} "
              f"[{block['r']['lo']:+.3f}, {block['r']['hi']:+.3f}]  {marker}")
        print(f"    leave-one-out {block['leave_one_out_min']:+.3f} to "
              f"{block['leave_one_out_max']:+.3f}{refit_text}  "
              f"support-weighted {block['r_support_weighted']:+.3f}")

    print("\n=== claim 3: paired test on the difference of correlations ===")
    difference, p_value = correlation_difference_interval(
        log_support,
        seen_rows["logistic_auc01"].to_numpy(),
        seen_rows["k-NN_auc01"].to_numpy(),
        seed=SEED,
    )
    correlation_difference = {
        "description": "corr(log support, logistic AUC0.1) - corr(log support, k-NN AUC0.1)",
        "difference": interval_dict(difference),
        "p_two_sided": round(p_value, 4),
        "n_peptides": int(len(seen_rows)),
        "spans_zero": bool(difference.lo <= 0.0 <= difference.hi),
    }
    print(f"  head minus k-NN: {difference}  p={p_value:.4f}  "
          f"over {len(seen_rows)} seen peptides")

    # --- claim 4, analytic, re-verified ------------------------------------------------------
    print("\n=== claim 4: within-peptide rescaling invariance ===")
    groups = peptides.to_numpy()
    frame = pd.DataFrame({"g": groups, "s": knn_eval})
    z_scored = frame.groupby("g")["s"].transform(
        lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * 0.0
    ).to_numpy()
    rank_normalised = frame.groupby("g")["s"].transform(
        lambda v: rankdata(v) / len(v)
    ).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        invariance = {
            "macro_auc01_raw": float(macro_auc01(y_eval, knn_eval, groups).value),
            "macro_auc01_zscored": float(macro_auc01(y_eval, z_scored, groups).value),
            "macro_auc01_rank_normalised": float(
                macro_auc01(y_eval, rank_normalised, groups).value),
            "pooled_auroc_raw": float(auroc(y_eval, knn_eval)),
            "pooled_auroc_zscored": float(auroc(y_eval, z_scored)),
            "pooled_auroc_rank_normalised": float(auroc(y_eval, rank_normalised)),
        }
    print(f"  macro AUC0.1   raw {invariance['macro_auc01_raw']:.6f}  "
          f"z {invariance['macro_auc01_zscored']:.6f}  "
          f"rank {invariance['macro_auc01_rank_normalised']:.6f}")
    print(f"  pooled AUROC   raw {invariance['pooled_auroc_raw']:.6f}  "
          f"z {invariance['pooled_auroc_zscored']:.6f}  "
          f"rank {invariance['pooled_auroc_rank_normalised']:.6f}")

    OUT_JSON.write_text(json.dumps({
        "config": {"model": MODEL_KEY, "layer": LAYER, "strategy": STRATEGY,
                   "ratio": RATIO, "seed": SEED,
                   "well_supported_min_positives": WELL_SUPPORTED_MIN_POSITIVES},
        "composition": {
            "n_peptides": int(peptides.nunique()),
            "n_seen_peptides": int(len(set(peptides[is_seen]))),
            "n_unseen_peptides": int(len(set(peptides[~is_seen]))),
            "n_rows": int(len(evalset)),
        },
        "scores": {k: {"macro_auc01": interval_dict(r.macro_auc01),
                       "auroc": interval_dict(r.auroc),
                       "auprc": interval_dict(r.auprc),
                       "n_rows": int(r.n_rows),
                       "n_positive": int(r.n_positive),
                       "n_peptides": int(r.n_groups_scored),
                       "is_degenerate": bool(r.diagnostics.is_degenerate)}
                   for k, r in scored.items()},
        "degeneracy": degeneracy,
        "comparisons": comparisons,
        "correlations": correlations,
        "correlation_difference": correlation_difference,
        "invariance": invariance,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
