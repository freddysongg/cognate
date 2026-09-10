"""Deduplication sensitivity: where does the representation effect die?

`run_dedup_recheck.py` used Levenshtein <= 3, which admits indels and removed 80.8% of eval
TCRs. Liao et al. (arXiv:2606.04994) remove test TCRs within **three amino-acid
substitutions**, which is a different and weaker criterion. This sweeps the substitution
radius 0-3 and reports the Levenshtein-3 column alongside, so the effect's dependence on the
threshold is visible rather than asserted at one point.

**Unequal lengths.** A substitution count is undefined between sequences of different length,
so this treats them as infinitely far apart: an eval CDR3b is compared only against training
CDR3b **of the same length**, and is removed at radius r only if some same-length training
sequence is within r substitutions. Any other choice smuggles indels back in -- padding the
shorter sequence, for instance, scores a length-14/15 pair sharing a prefix as distance 1,
which is an indel counted as a substitution. Same-length-only is the strict reading of
"three amino acid substitutions" and is the conservative one: it removes the fewest rows.

Score vectors are computed once and re-evaluated under each mask. Nothing is refit per
radius; deduplication changes which test rows count, not the models.

Writes `data/dedup_sensitivity.json` and `data/dedup_sensitivity_per_peptide.csv`. Touches no
frozen artifact. Reads existing caches; embeds nothing.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Hamming, Levenshtein
from rapidfuzz.process import cdist
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cognate.baseline_knn import build_database, cosine_similarity, score_by_nearest_positive
from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.metrics import auc01, compare_macro_auc01, evaluate

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = default_cache_dir()
VDJDB_CACHE = CACHE_DIR / "esm2_35M_vdjdb.npz"
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
OUT_JSON = ROOT / "data" / "dedup_sensitivity.json"
OUT_PER_PEPTIDE = ROOT / "data" / "dedup_sensitivity_per_peptide.csv"

MODEL_KEY = "35M"
LAYER = 10
SEED = 0
N_BACKGROUND = 1000
SUBSTITUTION_RADII = (0, 1, 2, 3)
LEVENSHTEIN_RADIUS = 3
UNSTABLE_POSITIVES = 5
NO_SAME_LENGTH_MATCH = 10**6


def nearest_substitutions(evalset: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    """Fewest substitutions turning an eval CDR3b into any *same-length* training CDR3b.

    Sequences with no training counterpart of their length get `NO_SAME_LENGTH_MATCH`, so
    they survive every finite radius. That is the correct reading: no number of
    substitutions converts a 13-mer into a 14-mer.
    """
    by_length: dict[int, list[str]] = defaultdict(list)
    for sequence in sorted(set(positives["CDR3b"].astype(str))):
        by_length[len(sequence)].append(sequence)

    queries = np.array(sorted(set(evalset["CDR3b"].astype(str))), dtype=object)
    closest: dict[str, int] = {}
    for length, group in defaultdict(list, {
        length: [q for q in queries if len(q) == length]
        for length in {len(q) for q in queries}
    }).items():
        database = by_length.get(length)
        if not database:
            closest.update({q: NO_SAME_LENGTH_MATCH for q in group})
            continue
        distances = cdist(
            np.array(group, dtype=object),
            np.array(database, dtype=object),
            scorer=Hamming.distance,
            workers=-1,
        )
        closest.update(dict(zip(group, distances.min(axis=1))))
    return np.array([closest[s] for s in evalset["CDR3b"].astype(str)], dtype=int)


def nearest_levenshtein(evalset: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    queries = np.array(sorted(set(evalset["CDR3b"].astype(str))), dtype=object)
    database = np.array(sorted(set(positives["CDR3b"].astype(str))), dtype=object)
    distances = cdist(queries, database, scorer=Levenshtein.distance, workers=-1)
    closest = dict(zip(queries, distances.min(axis=1)))
    return np.array([closest[s] for s in evalset["CDR3b"].astype(str)], dtype=int)


def fit_per_peptide_svc(evalset, positives, train_cache, eval_cache, seed):
    """Same construction and seed as run_operator_diagnostic.py."""
    database = build_database(positives)
    rng = np.random.default_rng(seed)
    peptides = evalset["Peptide"].astype(str).to_numpy()
    sequences = evalset["CDR3b"].astype(str).to_numpy()
    scores = np.zeros(len(evalset), dtype=float)
    all_train = np.array(sorted(set(positives["CDR3b"].astype(str))), dtype=object)

    for peptide in np.unique(peptides):
        if peptide not in database:
            continue
        reference = database[peptide]
        binders = set(reference.tolist())
        pool = np.array([t for t in all_train if t not in binders], dtype=object)
        background = rng.choice(pool, size=min(N_BACKGROUND, len(pool)), replace=False)
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000),
        ).fit(
            np.concatenate([
                train_cache.lookup(reference, LAYER),
                train_cache.lookup(background, LAYER),
            ]),
            np.concatenate([
                np.ones(len(reference), dtype=int),
                np.zeros(len(background), dtype=int),
            ]),
        )
        rows = np.flatnonzero(peptides == peptide)
        scores[rows] = model.decision_function(eval_cache.lookup(sequences[rows], LAYER))
    return scores


def interval_dict(interval) -> dict:
    return {"point": round(interval.point, 4), "lo": round(interval.lo, 4),
            "hi": round(interval.hi, 4)}


def main() -> None:
    positives = load_train()
    evalset = pd.read_csv(EVAL_CSV)
    y = evalset["Label"].to_numpy()
    peptides = evalset["Peptide"]
    is_seen = peptides.isin(set(positives["Peptide"])).to_numpy()

    substitutions = nearest_substitutions(evalset, positives)
    levenshtein = nearest_levenshtein(evalset, positives)
    finite = substitutions[substitutions < NO_SAME_LENGTH_MATCH]
    print(f"substitution distance to nearest same-length training CDR3b: "
          f"min {finite.min()} median {np.median(finite):.0f} max {finite.max()}; "
          f"{int((substitutions == NO_SAME_LENGTH_MATCH).sum()):,} rows have no same-length "
          f"training sequence")
    print(f"levenshtein distance: min {levenshtein.min()} "
          f"median {np.median(levenshtein):.0f} max {levenshtein.max()}\n")

    train_cache = load_cache(cache_path(MODEL_KEY, CACHE_DIR))
    eval_cache = load_cache(VDJDB_CACHE)
    arms = {
        "edit_retrieval": score_by_nearest_positive(evalset, positives).score,
        "esm_retrieval": score_by_nearest_positive(
            evalset, positives,
            similarity_fn=cosine_similarity(eval_cache, LAYER, train_cache),
        ).score,
        "svc_per_peptide": fit_per_peptide_svc(
            evalset, positives, train_cache, eval_cache, SEED
        ),
    }
    print("regenerated 3 arms; scoring under each mask\n")

    regimes = [("full", np.ones(len(evalset), bool))]
    regimes += [(f"sub_{r}", substitutions > r) for r in SUBSTITUTION_RADII]
    regimes += [(f"lev_{LEVENSHTEIN_RADIUS}", levenshtein > LEVENSHTEIN_RADIUS)]

    results = {}
    for name, keep in regimes:
        mask = is_seen & keep
        seen_rows = int(is_seen.sum())
        tcrs = evalset.loc[is_seen, "CDR3b"].astype(str).to_numpy()
        removed_tcr = pd.DataFrame(
            {"t": tcrs, "gone": ~keep[is_seen]}
        ).groupby("t")["gone"].max()
        per_peptide = []
        for peptide in sorted(set(peptides[is_seen])):
            rows = (peptides == peptide).to_numpy() & keep
            per_peptide.append(int(y[rows].sum()))
        support = np.array(per_peptide)
        scores_block = {}
        for arm, vector in arms.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = evaluate(y, vector, peptides, subset=mask, label="", seed=SEED)
            scores_block[arm] = interval_dict(report.macro_auc01)
            scores_block[f"{arm}_peptides"] = int(report.n_groups_scored)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            effect = compare_macro_auc01(
                y, peptides,
                ("esm_retrieval", arms["esm_retrieval"], mask),
                ("edit_retrieval", arms["edit_retrieval"], mask), seed=SEED,
            )
        results[name] = {
            "rows_kept": int(mask.sum()),
            "rows_removed_pct": round(100 * (1 - mask.sum() / seen_rows), 1),
            "distinct_tcrs_removed_pct": round(100 * removed_tcr.mean(), 1),
            "support_min": int(support.min()),
            "support_median": float(np.median(support)),
            "support_max": int(support.max()),
            "peptides_below_threshold": int((support < UNSTABLE_POSITIVES).sum()),
            "scores": scores_block,
            "representation_effect": {
                "point": round(effect.difference.point, 4),
                "lo": round(effect.difference.lo, 4),
                "hi": round(effect.difference.hi, 4),
                "p": round(effect.p_two_sided, 4),
                "n_peptides": int(effect.n_groups),
                "spans_zero": bool(effect.difference.lo <= 0.0 <= effect.difference.hi),
            },
        }
        r = results[name]
        e = r["representation_effect"]
        print(f"{name:<8} removed {r['rows_removed_pct']:>5.1f}% rows / "
              f"{r['distinct_tcrs_removed_pct']:>5.1f}% TCRs   "
              f"support {r['support_min']}/{r['support_median']:.0f}/{r['support_max']}   "
              f"edit {r['scores']['edit_retrieval']['point']:.4f}  "
              f"esm {r['scores']['esm_retrieval']['point']:.4f}  "
              f"svc {r['scores']['svc_per_peptide']['point']:.4f}   "
              f"effect {e['point']:+.4f} [{e['lo']:+.4f},{e['hi']:+.4f}] "
              f"p={e['p']:.3f}{'  SPANS ZERO' if e['spans_zero'] else ''}")

    rows_out = []
    for peptide in sorted(set(peptides[is_seen])):
        base = (peptides == peptide).to_numpy()
        entry = {"peptide": peptide}
        for name, keep in regimes:
            rows = base & keep
            entry[f"{name}_positives"] = int(y[rows].sum())
            scorable = rows.sum() and 0 < y[rows].sum() < rows.sum()
            for arm, vector in arms.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    entry[f"{name}_{arm}"] = (
                        float(auc01(y[rows], vector[rows])) if scorable else None
                    )
        rows_out.append(entry)
    pd.DataFrame(rows_out).to_csv(OUT_PER_PEPTIDE, index=False)

    OUT_JSON.write_text(json.dumps({
        "config": {
            "model": MODEL_KEY, "layer": LAYER, "seed": SEED,
            "substitution_radii": list(SUBSTITUTION_RADII),
            "levenshtein_radius": LEVENSHTEIN_RADIUS,
            "unequal_length_rule": "substitution distance is undefined across lengths; an "
                                   "eval CDR3b is compared only to training CDR3b of the same "
                                   "length and survives every finite radius if none exists",
            "unstable_positive_threshold": UNSTABLE_POSITIVES,
        },
        "regimes": results,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
