"""Regenerate every headline number in one deterministic pass.

Writes data/headline.json. Two runs of this script must produce byte-identical output;
that is the reproducibility acceptance test for the whole project.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.baseline_knn import exact_match_mask, score_by_nearest_positive
from cognate.data import load_solutions, load_train
from cognate.embed import cache_path, load_cache
from cognate.features import build_features
from cognate.metrics import auc01, compare_macro_auc01, diagnose_scores, evaluate
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, logistic_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "headline.json"
MODEL_KEY = "35M"
LAYER = 10
STRATEGY = "matched"
RATIO = 5.0
SEED = 0


def _interval(report) -> dict[str, float]:
    return {
        "point": round(report.point, 4),
        "lo": round(report.lo, 4),
        "hi": round(report.hi, 4),
    }


def head_scores(test: pd.DataFrame, positives: pd.DataFrame, strategy: str) -> np.ndarray:
    split = component_split(positives, validation_fraction=0.2, seed=SEED)
    train = pd.concat(
        [split.train, make_negatives(split.train, strategy, RATIO, SEED)],
        ignore_index=True,
    )
    cache = load_cache(cache_path(MODEL_KEY))
    model = fit_logistic(
        build_features(train, cache, LAYER), train["Target"].to_numpy(), seed=SEED
    )
    return logistic_scores(model, build_features(test, cache, LAYER))


def main() -> None:
    positives, test = load_train(), load_solutions()
    y_true = test["Label"].to_numpy()
    peptides = test["Peptide"]
    is_seen = test["Peptide"].isin(set(positives["Peptide"])).to_numpy()
    is_exact = exact_match_mask(test, positives)

    knn = score_by_nearest_positive(test, positives).score
    head = head_scores(test, positives, STRATEGY)
    rng_scores = np.random.default_rng(SEED).random(len(test))

    slices = {
        "seen": is_seen,
        "unseen": ~is_seen,
        "seen_exact_excluded": is_seen & ~is_exact,
    }
    result: dict[str, object] = {"config": {
        "model": MODEL_KEY, "layer": LAYER, "strategy": STRATEGY,
        "ratio": RATIO, "seed": SEED,
    }}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model_name, scores in [("random", rng_scores), ("knn", knn), ("head", head)]:
            for slice_name, mask in slices.items():
                report = evaluate(
                    y_true, scores, peptides, subset=mask,
                    label=f"{model_name}/{slice_name}", seed=SEED,
                )
                result[f"{model_name}/{slice_name}"] = {
                    "n": report.n_rows,
                    "macro_auc01": _interval(report.macro_auc01),
                    "auroc": _interval(report.auroc),
                    "auprc": _interval(report.auprc),
                    "degenerate": report.diagnostics.is_degenerate,
                }

        for slice_name in ["seen", "unseen"]:
            comparison = compare_macro_auc01(
                y_true, peptides,
                ("knn", knn, slices[slice_name]),
                ("head", head, slices[slice_name]),
                seed=SEED,
            )
            result[f"knn_minus_head/{slice_name}"] = {
                **_interval(comparison.difference),
                "p": round(comparison.p_two_sided, 4),
                "n_peptides": comparison.n_groups,
            }

        result["knn_exact_row_ablation"] = {
            **_interval(compare_macro_auc01(
                y_true, peptides,
                ("seen", knn, slices["seen"]),
                ("seen_minus_exact", knn, slices["seen_exact_excluded"]),
                seed=SEED,
            ).difference)
        }

    points = pd.DataFrame(
        json.loads((ROOT / "data" / "t6a_points.json").read_text(encoding="utf-8"))
    )
    seen_points = points[points["seen"]]
    usable = points[points["knn_usable"]]
    result["correlations"] = {
        "knn_vs_distance": round(float(np.corrcoef(
            usable["distance_positives"], usable["knn_auc01"])[0, 1]), 4),
        "knn_vs_distance_drop_max": round(float(np.corrcoef(
            *[usable[usable.peptide != "GILGFVFTL"][c]
              for c in ["distance_positives", "knn_auc01"]])[0, 1]), 4),
        "head_vs_support": round(float(np.corrcoef(
            np.log10(seen_points["support"]), seen_points["head_matched_auc01"])[0, 1]), 4),
        "knn_vs_support": round(float(np.corrcoef(
            np.log10(usable["support"]), usable["knn_auc01"])[0, 1]), 4),
    }
    result["degeneracy"] = {
        "knn_unseen_distinct_scores": diagnose_scores(knn[~is_seen]).n_unique_scores,
        "head_unseen_distinct_scores": diagnose_scores(head[~is_seen]).n_unique_scores,
        "auc01_of_constant_column": round(auc01(np.array([1, 0, 0, 1, 0, 0]), np.zeros(6)), 6),
    }

    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT}")
    for key in ["knn/seen", "head/seen", "knn/unseen", "head/unseen"]:
        entry = result[key]
        print(f"  {key:<22} macro AUC0.1 {entry['macro_auc01']['point']:.3f} "
              f"[{entry['macro_auc01']['lo']:.3f}, {entry['macro_auc01']['hi']:.3f}]")
    print(f"  knn - head (seen)      {result['knn_minus_head/seen']['point']:+.3f} "
          f"p={result['knn_minus_head/seen']['p']:.3f}")


main()
