"""T5b + T6b -- logistic head over every layer of both ESM-2 models.

The layer is chosen on the peptide-and-TCR-disjoint validation arm, never on test.
Test scores for every layer are recorded too, but only so the T6a/T6b plots can show
whether validation selection actually picked the test optimum.
"""

import json
import time
import warnings
from pathlib import Path

import pandas as pd

from cognate.data import load_solutions, load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.features import build_features
from cognate.metrics import macro_auc01
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, logistic_scores

CACHE_DIR = default_cache_dir()
RESULTS = Path(__file__).resolve().parents[1] / "data" / "layer_sweep.json"
NEGATIVE_RATIO = 5.0
NEGATIVE_STRATEGY = "matched"
SEED = 0


def build_arms():
    train_positives = load_train()
    split = component_split(train_positives, validation_fraction=0.2, seed=SEED)
    print(split)

    train = pd.concat(
        [split.train, make_negatives(split.train, NEGATIVE_STRATEGY, NEGATIVE_RATIO, SEED)],
        ignore_index=True,
    )
    validation = pd.concat(
        [
            split.validation,
            make_negatives(split.validation, NEGATIVE_STRATEGY, NEGATIVE_RATIO, SEED),
        ],
        ignore_index=True,
    )
    test = load_solutions()
    print(
        f"train {len(train):,} rows | val {len(validation):,} rows | test {len(test):,} rows"
    )
    return train, validation, test, set(train_positives["Peptide"])


def main() -> None:
    train, validation, test, train_peptides = build_arms()
    y_train = train["Target"].to_numpy()
    y_val = validation["Target"].to_numpy()
    y_test = test["Label"].to_numpy()
    is_seen = test["Peptide"].isin(train_peptides).to_numpy()

    records = []
    for model_key in ["8M", "35M"]:
        cache = load_cache(cache_path(model_key, CACHE_DIR))
        for layer in range(cache.n_layers):
            started = time.perf_counter()
            model = fit_logistic(
                build_features(train, cache, layer), y_train, seed=SEED
            )
            val_scores = logistic_scores(model, build_features(validation, cache, layer))
            test_scores = logistic_scores(model, build_features(test, cache, layer))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                record = {
                    "model": model_key,
                    "layer": layer,
                    "val_macro_auc01": macro_auc01(
                        y_val, val_scores, validation["Peptide"]
                    ).value,
                    "test_macro_auc01": macro_auc01(
                        y_test, test_scores, test["Peptide"]
                    ).value,
                    "test_seen": macro_auc01(
                        y_test[is_seen], test_scores[is_seen], test["Peptide"][is_seen]
                    ).value,
                    "test_unseen": macro_auc01(
                        y_test[~is_seen],
                        test_scores[~is_seen],
                        test["Peptide"][~is_seen],
                    ).value,
                    "seconds": round(time.perf_counter() - started, 1),
                    "degenerate_warnings": len(caught),
                }
            records.append(record)
            print(
                f"{model_key:>3} L{layer:<2} val={record['val_macro_auc01']:.3f} "
                f"test={record['test_macro_auc01']:.3f} "
                f"seen={record['test_seen']:.3f} unseen={record['test_unseen']:.3f} "
                f"({record['seconds']}s)",
                flush=True,
            )

    frame = pd.DataFrame(records)
    RESULTS.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print("\n=== layer selected on validation ===")
    for model_key in ["8M", "35M"]:
        subset = frame[frame["model"] == model_key]
        best_val = subset.loc[subset["val_macro_auc01"].idxmax()]
        best_test = subset.loc[subset["test_macro_auc01"].idxmax()]
        print(
            f"{model_key}: val picks L{int(best_val['layer'])} "
            f"(val {best_val['val_macro_auc01']:.3f}, test {best_val['test_macro_auc01']:.3f}) | "
            f"test optimum is L{int(best_test['layer'])} "
            f"(test {best_test['test_macro_auc01']:.3f})"
        )
        last = subset[subset["layer"] == subset["layer"].max()].iloc[0]
        print(
            f"     last layer L{int(last['layer'])} test {last['test_macro_auc01']:.3f}  "
            f"-> middle beats last: "
            f"{bool(best_test['test_macro_auc01'] > last['test_macro_auc01'])}"
        )


main()
