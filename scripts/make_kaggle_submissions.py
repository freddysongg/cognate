"""A3 -- write Kaggle submission files for the two Phase 1 models.

Format follows sample_submission.csv exactly: the test.csv columns plus a `Prediction`
column of binding probabilities in [0, 1], with row order and IDs untouched.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cognate.baseline_knn import score_by_nearest_positive
from cognate.data import DATA_DIR, load_solutions, load_test, load_train
from cognate.embed import cache_path, load_cache
from cognate.features import build_features
from cognate.metrics import macro_auc01
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, logistic_scores

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "submissions"
MODEL_KEY, LAYER, STRATEGY, RATIO, SEED = "35M", 10, "matched", 5.0, 0


def write_submission(test: pd.DataFrame, scores: np.ndarray, name: str) -> Path:
    template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = test.copy()
    submission["Prediction"] = np.clip(scores, 0.0, 1.0)
    submission = submission[list(template.columns)]

    assert list(submission["ID"]) == list(template["ID"]), "row order must not change"
    assert submission["Prediction"].between(0, 1).all()
    assert submission.notna().all().all()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    positives, test, solutions = load_train(), load_test(), load_solutions()
    y_true = solutions["Label"].to_numpy()

    knn = score_by_nearest_positive(test, positives).score

    split = component_split(positives, validation_fraction=0.2, seed=SEED)
    train = pd.concat(
        [split.train, make_negatives(split.train, STRATEGY, RATIO, SEED)],
        ignore_index=True,
    )
    cache = load_cache(cache_path(MODEL_KEY, ROOT / "data" / "embeddings"))
    model = fit_logistic(
        build_features(train, cache, LAYER), train["Target"].to_numpy(), seed=SEED
    )
    head = logistic_scores(model, build_features(test, cache, LAYER))

    print(f"{'submission':<26} {'rows':>5}  internal macro AUC0.1 (all 20 peptides)")
    for name, scores in [("knn_v1_edit_distance", knn), ("esm2_logistic_matched", head)]:
        path = write_submission(test, scores, name)
        internal = macro_auc01(y_true, scores, solutions["Peptide"]).value
        print(f"{path.name:<26} {len(test):>5}  {internal:.4f}")

    print("\nInternal score is computed over all 3,484 rows / 20 peptides.")
    print("Kaggle scores the Private split (3,217 rows) and the Public split (267 rows)")
    print("separately, so neither returned number should equal the internal one exactly.")
    for usage in ["Private", "Public"]:
        mask = (solutions["Usage"] == usage).to_numpy()
        for name, scores in [("knn", knn), ("head", head)]:
            value = macro_auc01(y_true[mask], scores[mask], solutions["Peptide"][mask])
            print(f"  predicted {usage:<8} {name:<5}: {value.value:.4f} "
                  f"({mask.sum()} rows, {value.n_groups_scored} peptides scored, "
                  f"{value.n_groups_skipped} skipped)")


main()
