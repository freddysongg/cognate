"""T6a -- per-peptide scores against training support and distance to the training set.

Distance is peptide-agnostic on purpose: the nearest training CDR3b regardless of which
peptide it binds. A peptide-specific distance is undefined for the 7 unseen peptides,
which are exactly the points the plot exists to show.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from cognate.baseline_knn import score_by_nearest_positive
from cognate.data import load_solutions, load_train
from cognate.embed import cache_path, load_cache
from cognate.features import build_features
from cognate.metrics import auc01, diagnose_scores
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, logistic_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "t6a_points.json"
MODEL_KEY = "35M"
LAYER = 10
SEED = 0


def nearest_training_distance(test: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """Normalised edit distance from each test CDR3b to the closest training CDR3b."""
    queries = np.array(sorted(set(test["CDR3b"].astype(str))), dtype=object)
    database = np.array(sorted(set(train["CDR3b"].astype(str))), dtype=object)
    similarity = cdist(
        queries, database, scorer=Levenshtein.normalized_similarity, workers=-1
    )
    closest = dict(zip(queries, 1.0 - similarity.max(axis=1)))
    return np.array([closest[s] for s in test["CDR3b"].astype(str)])


def head_scores(test: pd.DataFrame, positives: pd.DataFrame, strategy: str) -> np.ndarray:
    split = component_split(positives, validation_fraction=0.2, seed=SEED)
    train = pd.concat(
        [split.train, make_negatives(split.train, strategy, 5.0, SEED)],
        ignore_index=True,
    )
    cache = load_cache(cache_path(MODEL_KEY))
    model = fit_logistic(
        build_features(train, cache, LAYER), train["Target"].to_numpy(), seed=SEED
    )
    return logistic_scores(model, build_features(test, cache, LAYER))


def main() -> None:
    positives = load_train()
    test = load_solutions()
    y_true = test["Label"].to_numpy()
    support = positives.groupby("Peptide").size()
    train_peptides = set(positives["Peptide"])

    distance = nearest_training_distance(test, positives)
    knn = score_by_nearest_positive(test, positives).score
    head_matched = head_scores(test, positives, "matched")
    head_shuffle = head_scores(test, positives, "shuffle")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        knn_usable = {
            p
            for p in set(test["Peptide"])
            if not diagnose_scores(
                knn[(test["Peptide"] == p).to_numpy()]
            ).is_degenerate
        }

    points = []
    for peptide in sorted(set(test["Peptide"])):
        mask = (test["Peptide"] == peptide).to_numpy()
        positive_mask = mask & (y_true == 1)
        points.append(
            {
                "peptide": peptide,
                "seen": peptide in train_peptides,
                "support": int(support.get(peptide, 0)),
                "n_rows": int(mask.sum()),
                "distance_positives": float(np.median(distance[positive_mask])),
                "distance_all": float(np.median(distance[mask])),
                "knn_auc01": auc01(y_true[mask], knn[mask]),
                "knn_usable": peptide in knn_usable,
                "head_matched_auc01": auc01(y_true[mask], head_matched[mask]),
                "head_shuffle_auc01": auc01(y_true[mask], head_shuffle[mask]),
            }
        )

    OUT.write_text(json.dumps(points, indent=2), encoding="utf-8")
    frame = pd.DataFrame(points)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nk-NN usable peptides: {int(frame['knn_usable'].sum())} / {len(frame)}")
    print(f"excluded (degenerate, constant score): "
          f"{sorted(frame.loc[~frame['knn_usable'], 'peptide'])}")


main()
