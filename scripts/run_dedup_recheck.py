"""Re-score three arms under a Levenshtein-3 deduplication of the evaluation set.

Our eval set removes IMMREP23 training TCRs by **exact match**: zero eval CDR3b appear
verbatim in `load_train()`. A stricter standard removes every test TCR within a small edit
radius of any training CDR3b, on the argument that a near-duplicate is not an independent
test point for a method whose database is the training set.

This is an evaluation-subset change, not a modelling change. Nothing is retrained on new
data: the same three arms are regenerated at the same seed and config, and the metric is
recomputed over the surviving rows. `edit_retrieval` is the arm with the most to lose, since
its score *is* similarity to the training set.

Writes `data/dedup_recheck.json` and `data/dedup_recheck_per_peptide.csv`. Touches no frozen
artifact. Reads the existing embedding caches; embeds nothing.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from cognate.baseline_knn import cosine_similarity, score_by_nearest_positive
from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.metrics import auc01, compare_macro_auc01, diagnose_scores, evaluate

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = default_cache_dir()
VDJDB_CACHE = CACHE_DIR / "esm2_35M_vdjdb.npz"
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
OUT_JSON = ROOT / "data" / "dedup_recheck.json"
OUT_PER_PEPTIDE = ROOT / "data" / "dedup_recheck_per_peptide.csv"

MODEL_KEY = "35M"
LAYER = 10
SEED = 0
N_BACKGROUND = 1000
DEDUP_RADIUS = 3
UNSTABLE_POSITIVES = 5


def nearest_training_edits(evalset: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    """Absolute Levenshtein distance from each eval CDR3b to the closest training CDR3b.

    Absolute, not normalised: the published threshold is a count of edits, and normalising
    would make the radius depend on sequence length.
    """
    queries = np.array(sorted(set(evalset["CDR3b"].astype(str))), dtype=object)
    database = np.array(sorted(set(positives["CDR3b"].astype(str))), dtype=object)
    distances = cdist(queries, database, scorer=Levenshtein.distance, workers=-1)
    closest = dict(zip(queries, distances.min(axis=1)))
    return np.array([closest[s] for s in evalset["CDR3b"].astype(str)], dtype=int)


def fit_per_peptide_svc(evalset, positives, train_cache, eval_cache, seed):
    """Regenerated verbatim from run_operator_diagnostic.py at the same seed and config.

    Duplicated rather than imported because the diagnostic's copy is frozen provenance for
    `data/operator_diagnostic.json`; importing it would couple two artifacts that must be
    able to move independently.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    from cognate.baseline_knn import build_database

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
        x_fit = np.concatenate(
            [train_cache.lookup(reference, LAYER), train_cache.lookup(background, LAYER)]
        )
        y_fit = np.concatenate(
            [np.ones(len(reference), dtype=int), np.zeros(len(background), dtype=int)]
        )
        rows = np.flatnonzero(peptides == peptide)
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000),
        ).fit(x_fit, y_fit)
        scores[rows] = model.decision_function(eval_cache.lookup(sequences[rows], LAYER))
    return scores


def macro(y, scores, groups, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate(y, scores, groups, subset=mask, label="", seed=SEED)


def interval_dict(interval) -> dict:
    return {
        "point": round(interval.point, 4),
        "lo": round(interval.lo, 4),
        "hi": round(interval.hi, 4),
    }


def main() -> None:
    positives = load_train()
    evalset = pd.read_csv(EVAL_CSV)
    y = evalset["Label"].to_numpy()
    peptides = evalset["Peptide"]
    is_seen = peptides.isin(set(positives["Peptide"])).to_numpy()

    edits = nearest_training_edits(evalset, positives)
    survives = edits > DEDUP_RADIUS
    print(f"eval rows {len(evalset):,}; nearest-training edit distance "
          f"min {edits.min()} median {np.median(edits):.0f} max {edits.max()}")

    # --- how much the radius removes ----------------------------------------------------
    removal = {}
    for name, mask in [("all", np.ones(len(evalset), bool)), ("seen", is_seen), ("unseen", ~is_seen)]:
        tcrs = evalset.loc[mask, "CDR3b"].astype(str)
        near = pd.Series(~survives[mask]).to_numpy()
        distinct = pd.DataFrame({"t": tcrs.to_numpy(), "near": near}).groupby("t")["near"].max()
        removal[name] = {
            "rows": int(mask.sum()),
            "rows_removed": int(near.sum()),
            "rows_removed_pct": round(100 * near.mean(), 1),
            "distinct_tcrs": int(len(distinct)),
            "distinct_tcrs_removed": int(distinct.sum()),
            "distinct_tcrs_removed_pct": round(100 * distinct.mean(), 1),
            "peptides": int(peptides[mask].nunique()),
        }
        r = removal[name]
        print(f"  {name:<7} rows {r['rows']:>7,} removed {r['rows_removed']:>7,} "
              f"({r['rows_removed_pct']:>5.1f}%)   distinct TCRs {r['distinct_tcrs']:>6,} "
              f"removed {r['distinct_tcrs_removed']:>6,} ({r['distinct_tcrs_removed_pct']:>5.1f}%)")

    # --- regenerate the three arms ------------------------------------------------------
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
    print("regenerated 3 arms")

    # --- surviving support, and which peptides become unstable --------------------------
    support = []
    for peptide in sorted(set(peptides[is_seen])):
        rows = (peptides == peptide).to_numpy()
        kept = rows & survives
        support.append({
            "peptide": peptide,
            "positives_before": int(y[rows].sum()),
            "positives_after": int(y[kept].sum()),
            "rows_before": int(rows.sum()),
            "rows_after": int(kept.sum()),
            "scorable_after": bool(kept.sum() and 0 < y[kept].sum() < kept.sum()),
        })
    support_frame = pd.DataFrame(support)
    unstable = support_frame[support_frame.positives_after < UNSTABLE_POSITIVES]
    dropped = support_frame[~support_frame.scorable_after]
    print(f"seen peptides: {len(support_frame)}; "
          f"below {UNSTABLE_POSITIVES} surviving positives: {len(unstable)}; "
          f"no longer scorable: {len(dropped)}")

    # --- scores, full vs surviving ------------------------------------------------------
    scored, comparisons = {}, {}
    print("\n=== macro AUC0.1, seen slice ===")
    for name, scores in arms.items():
        full = macro(y, scores, peptides, is_seen)
        kept = macro(y, scores, peptides, is_seen & survives)
        d = diagnose_scores(scores[is_seen & survives], peptides[is_seen & survives])
        scored[name] = {
            "full": interval_dict(full.macro_auc01),
            "full_peptides": int(full.n_groups_scored),
            "deduped": interval_dict(kept.macro_auc01),
            "deduped_peptides": int(kept.n_groups_scored),
            "deduped_is_degenerate": bool(d.is_degenerate),
            "delta": round(kept.macro_auc01.point - full.macro_auc01.point, 4),
        }
        s = scored[name]
        print(f"  {name:<18} full {s['full']['point']:.4f} "
              f"[{s['full']['lo']:.4f},{s['full']['hi']:.4f}] n={s['full_peptides']}"
              f"   deduped {s['deduped']['point']:.4f} "
              f"[{s['deduped']['lo']:.4f},{s['deduped']['hi']:.4f}] n={s['deduped_peptides']}"
              f"   delta {s['delta']:+.4f}")

    print("\n=== paired, same rows both arms, deduped seen slice ===")
    mask = is_seen & survives
    for key, a, b in [
        ("edit_minus_esm/deduped_seen", "edit_retrieval", "esm_retrieval"),
        ("svc_per_peptide_minus_esm/deduped_seen", "svc_per_peptide", "esm_retrieval"),
        ("svc_per_peptide_minus_edit/deduped_seen", "svc_per_peptide", "edit_retrieval"),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = compare_macro_auc01(y, peptides, (a, arms[a], mask), (b, arms[b], mask), seed=SEED)
        comparisons[key] = {
            "point": round(c.difference.point, 4), "lo": round(c.difference.lo, 4),
            "hi": round(c.difference.hi, 4), "p": round(c.p_two_sided, 4),
            "n_peptides": int(c.n_groups),
            "spans_zero": bool(c.difference.lo <= 0.0 <= c.difference.hi),
        }
        print("  " + str(c))

    print("\n=== subset ablation: deduped minus full, per arm (independent resamples) ===")
    for name, scores in arms.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = compare_macro_auc01(
                y, peptides, (f"{name} deduped", scores, mask),
                (f"{name} full", scores, is_seen), seed=SEED,
            )
        comparisons[f"{name}_deduped_minus_full/seen"] = {
            "point": round(c.difference.point, 4), "lo": round(c.difference.lo, 4),
            "hi": round(c.difference.hi, 4), "p": round(c.p_two_sided, 4),
            "n_peptides": int(c.n_groups),
            "spans_zero": bool(c.difference.lo <= 0.0 <= c.difference.hi),
        }
        print("  " + str(c))

    rows_out = []
    for record in support:
        peptide = record["peptide"]
        rows = (peptides == peptide).to_numpy()
        kept = rows & survives
        entry = dict(record)
        for name, scores in arms.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                entry[f"{name}_auc01_before"] = float(auc01(y[rows], scores[rows]))
                entry[f"{name}_auc01_after"] = (
                    float(auc01(y[kept], scores[kept])) if record["scorable_after"] else None
                )
        rows_out.append(entry)
    pd.DataFrame(rows_out).to_csv(OUT_PER_PEPTIDE, index=False)

    OUT_JSON.write_text(json.dumps({
        "config": {
            "model": MODEL_KEY, "layer": LAYER, "seed": SEED,
            "dedup_radius_levenshtein": DEDUP_RADIUS,
            "unstable_positive_threshold": UNSTABLE_POSITIVES,
            "note": "Removal criterion is Levenshtein <= 3 to any training CDR3b, which "
                    "admits indels. A substitutions-only radius would remove fewer rows.",
        },
        "removal": removal,
        "scores": scored,
        "paired": comparisons,
        "surviving_support": {
            "seen_peptides": int(len(support_frame)),
            "below_threshold": int(len(unstable)),
            "below_threshold_peptides": unstable.peptide.tolist(),
            "no_longer_scorable": int(len(dropped)),
            "no_longer_scorable_peptides": dropped.peptide.tolist(),
            "positives_after_min": int(support_frame.positives_after.min()),
            "positives_after_median": float(support_frame.positives_after.median()),
            "positives_after_max": int(support_frame.positives_after.max()),
        },
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
