"""T5b -- train the heads on the validation-selected layer and report against the baseline.

Every number is reported seen/unseen separately with a bootstrap CI. Head-vs-baseline uses
the paired bootstrap, which is valid here because both arms score the same peptides.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.baseline_knn import score_by_nearest_positive
from cognate.data import load_solutions, load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.features import build_features
from cognate.metrics import compare_macro_auc01, diagnose_scores, evaluate, report_table
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, fit_mlp, logistic_scores, mlp_scores

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = default_cache_dir()
SWEEP = ROOT / "data" / "layer_sweep.json"
STRATEGY = "matched"
RATIO = 5.0
SEED = 0


def main() -> None:
    positives = load_train()
    test = load_solutions()
    split = component_split(positives, validation_fraction=0.2, seed=SEED)
    train = pd.concat(
        [split.train, make_negatives(split.train, STRATEGY, RATIO, SEED)],
        ignore_index=True,
    )
    validation = pd.concat(
        [split.validation, make_negatives(split.validation, STRATEGY, RATIO, SEED)],
        ignore_index=True,
    )

    y_train = train["Target"].to_numpy()
    y_val = validation["Target"].to_numpy()
    y_test = test["Label"].to_numpy()
    peptides = test["Peptide"]
    is_seen = test["Peptide"].isin(set(positives["Peptide"])).to_numpy()

    sweep = pd.DataFrame(json.loads(SWEEP.read_text(encoding="utf-8")))
    best = sweep.loc[sweep.groupby("model")["val_macro_auc01"].idxmax()]
    model_key = str(best.loc[best["val_macro_auc01"].idxmax(), "model"])
    layer = int(best.loc[best["val_macro_auc01"].idxmax(), "layer"])
    print(f"validation selected: {model_key} layer {layer}")
    print(best[["model", "layer", "val_macro_auc01", "test_macro_auc01"]].to_string(index=False))

    cache = load_cache(cache_path(model_key, CACHE_DIR))
    x_train = build_features(train, cache, layer)
    x_val = build_features(validation, cache, layer)
    x_test = build_features(test, cache, layer)

    logistic = fit_logistic(x_train, y_train, seed=SEED)
    logistic_test = logistic_scores(logistic, x_test)

    mlp, scaler, history = fit_mlp(
        x_train, y_train, x_val, y_val, validation["Peptide"].to_numpy(), seed=SEED
    )
    mlp_test = mlp_scores(mlp, scaler, x_test)
    print(
        f"\nMLP: {history.n_epochs} epochs, best epoch {history.best_epoch}, "
        f"val loss {min(history.val_loss):.4f}, "
        f"val macro AUC0.1 at best {history.val_macro_auc01[history.best_epoch]:.3f}"
    )

    knn_test = score_by_nearest_positive(test, positives).score

    print("\n=== degeneracy check ===")
    for name, scores in [("k-NN", knn_test), ("logistic", logistic_test), ("MLP", mlp_test)]:
        for slice_name, mask in [("seen", is_seen), ("unseen", ~is_seen)]:
            d = diagnose_scores(scores[mask], peptides[mask])
            flag = "DEGENERATE" if d.is_degenerate else "ok"
            print(f"  {name:<9} {slice_name:<7} {flag:<11} {d}")

    print("\n=== scores ===")
    reports = []
    for name, scores in [("k-NN", knn_test), ("logistic", logistic_test), ("MLP", mlp_test)]:
        for slice_name, mask in [("all", np.ones(len(test), bool)), ("seen", is_seen), ("unseen", ~is_seen)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reports.append(
                    evaluate(y_test, scores, peptides, subset=mask,
                             label=f"{name} / {slice_name}", seed=SEED)
                )
    print(report_table(reports).to_string())

    print("\n=== paired comparisons (same peptides in both arms) ===")
    for slice_name, mask in [("seen", is_seen), ("unseen", ~is_seen)]:
        for a_name, a_scores in [("logistic", logistic_test), ("MLP", mlp_test)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print("  " + str(compare_macro_auc01(
                    y_test, peptides,
                    (f"{a_name} {slice_name}", a_scores, mask),
                    (f"k-NN {slice_name}", knn_test, mask), seed=SEED)))

    print("\n=== follow-up 2: does the head inflate scores with training support? ===")
    support = positives.groupby("Peptide").size()
    rows = []
    for peptide in sorted(set(test["Peptide"][is_seen])):
        mask = (test["Peptide"] == peptide).to_numpy()
        rows.append({
            "peptide": peptide,
            "log_support": np.log10(support[peptide]),
            "knn_mean": knn_test[mask].mean(),
            "logistic_mean": logistic_test[mask].mean(),
            "mlp_mean": mlp_test[mask].mean(),
        })
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  correlation of per-peptide MEAN RAW SCORE with log10(training support):")
    for column in ["knn_mean", "logistic_mean", "mlp_mean"]:
        r = np.corrcoef(frame["log_support"], frame[column])[0, 1]
        print(f"    {column:<15} r = {r:+.3f}")


main()
