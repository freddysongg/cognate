"""Operator diagnostic -- is the parametric arm of the 2x2 losing, or is it not learning?

The 2x2 in docs/findings.md compares an edit-distance retrieval operator against an ESM-2
parametric head and reads the gap as an operator effect. Nagano et al. (Cell Systems 16(1),
2025, Fig. S6) ran the closest published comparison and got the opposite sign: a linear SVC
fitted on ESM-2 features beat ESM-2 nearest-neighbour. Ours loses. Both readings fit our
evidence, because the operator comparison is a difference of two point estimates with no
paired interval.

Two things are conflated in our operator axis and this script separates them:

* **estimator** -- logistic regression versus a linear SVC on the same design matrix.
* **scope** -- one global model fitted across all peptides, versus one model per peptide.

Retrieval is inherently per-peptide: `score_by_nearest_positive` builds a separate database
for each peptide. The global head is not. So `retrieval - global head` charges the operator
axis for a scope change it did not make. SCEPTR's SVC was per-pMHC, which is why it is the
arm that has to be reproduced before the operator effect can be called real.

Arms, all on ESM-2 35M layer 10 so the representation never moves:

    edit_retrieval      normalised Levenshtein, max over the peptide's binders
    esm_retrieval       ESM-2 cosine, same operator            (representation swap only)
    logistic_global     the existing head, interactions blocks (inherited)
    svc_global          linear SVC, same blocks, same rows      (estimator swap only)
    svc_per_peptide     linear SVC per peptide over TCR embeddings, SCEPTR-matched

Deviations from SCEPTR are recorded in the output under `deviations_from_sceptr`. Their
benchmark was 6 pMHCs scored by AUROC on their own splits; ours is 48 seen peptides scored
by macro AUC0.1. This is directional, not a replication.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cognate.baseline_knn import (
    build_database,
    cosine_similarity,
    score_by_nearest_positive,
)
from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.features import build_features
from cognate.metrics import (
    auc01,
    compare_macro_auc01,
    diagnose_scores,
    evaluate,
    macro_auc01,
    report_table,
)
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, logistic_scores

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = default_cache_dir()
VDJDB_CACHE = CACHE_DIR / "esm2_35M_vdjdb.npz"
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
OUT_JSON = ROOT / "data" / "operator_diagnostic.json"
OUT_PER_PEPTIDE = ROOT / "data" / "operator_diagnostic_per_peptide.csv"

MODEL_KEY = "35M"
LAYER = 10
STRATEGY = "matched"
RATIO = 5.0
SEEDS = (0, 1, 2)
HEADLINE_SEED = 0
N_BACKGROUND = 1000
PROBE_TOLERANCE = 0.05
NO_DATABASE_SCORE = 0.0


def _svc() -> object:
    """Standardise then fit a linear SVC with the class imbalance carried in the penalty.

    `dual=False` because the design matrices here are all taller than they are wide, and
    the balanced class weight is what SCEPTR used to handle a reference set far smaller
    than its background pool.
    """
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=5000),
    )


def fit_per_peptide_svc(
    evalset: pd.DataFrame,
    positives: pd.DataFrame,
    train_cache,
    eval_cache,
    seed: int,
    reference_shift: int = 0,
) -> np.ndarray:
    """One linear SVC per peptide: its binders against a shared background pool.

    This is the SCEPTR Fig. S6 arm. The positives are the same per-peptide databases the
    retrieval baseline queries, so swapping this in changes only the operator -- the
    representation, the database and the evaluation set are untouched.

    `reference_shift` is the shortcut probe: rotating each peptide onto another peptide's
    binder set should destroy the signal. A model that still scores is reading something
    other than the reference set.

    A peptide with no training binders gets `NO_DATABASE_SCORE` for every row, the same
    structural fallback retrieval takes. The resulting constant column scores exactly 0.5
    under AUC0.1; that is arithmetic, not measurement.
    """
    database = build_database(positives)
    names = sorted(database)
    rng = np.random.default_rng(seed)

    peptides = evalset["Peptide"].astype(str).to_numpy()
    sequences = evalset["CDR3b"].astype(str).to_numpy()
    scores = np.full(len(evalset), NO_DATABASE_SCORE, dtype=float)

    all_train_tcrs = np.array(
        sorted(set(positives["CDR3b"].astype(str))), dtype=object
    )

    for peptide in np.unique(peptides):
        if peptide not in database:
            continue
        source = names[(names.index(peptide) + reference_shift) % len(names)]
        reference = database[source]
        binders = set(reference.tolist())
        pool = np.array([t for t in all_train_tcrs if t not in binders], dtype=object)
        background = rng.choice(
            pool, size=min(N_BACKGROUND, len(pool)), replace=False
        )

        x_fit = np.concatenate(
            [
                train_cache.lookup(reference, LAYER),
                train_cache.lookup(background, LAYER),
            ]
        )
        y_fit = np.concatenate(
            [np.ones(len(reference), dtype=int), np.zeros(len(background), dtype=int)]
        )
        if len(np.unique(y_fit)) < 2:
            continue

        rows = np.flatnonzero(peptides == peptide)
        model = _svc().fit(x_fit, y_fit)
        scores[rows] = model.decision_function(eval_cache.lookup(sequences[rows], LAYER))

    return scores


def interval_dict(interval) -> dict:
    return {
        "point": round(interval.point, 4),
        "lo": round(interval.lo, 4),
        "hi": round(interval.hi, 4),
    }


def score_block(report) -> dict:
    return {
        "macro_auc01": interval_dict(report.macro_auc01),
        "auroc": interval_dict(report.auroc),
        "auprc": interval_dict(report.auprc),
        "n_rows": int(report.n_rows),
        "n_peptides": int(report.n_groups_scored),
        "is_degenerate": bool(report.diagnostics.is_degenerate),
    }


def paired(y_eval, peptides, name_a, scores_a, name_b, scores_b, mask) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comparison = compare_macro_auc01(
            y_eval,
            peptides,
            (name_a, scores_a, mask),
            (name_b, scores_b, mask),
            seed=HEADLINE_SEED,
        )
    print("  " + str(comparison))
    return {
        "point": round(comparison.difference.point, 4),
        "lo": round(comparison.difference.lo, 4),
        "hi": round(comparison.difference.hi, 4),
        "p": round(comparison.p_two_sided, 4),
        "n_peptides": int(comparison.n_groups),
        "spans_zero": bool(comparison.difference.lo <= 0.0 <= comparison.difference.hi),
    }


def main() -> None:
    positives = load_train()
    evalset = pd.read_csv(EVAL_CSV)
    y_eval = evalset["Label"].to_numpy()
    peptides = evalset["Peptide"]
    is_seen = peptides.isin(set(positives["Peptide"])).to_numpy()

    train_cache = load_cache(cache_path(MODEL_KEY, CACHE_DIR))
    eval_cache = load_cache(VDJDB_CACHE)

    print(f"eval: {len(evalset):,} rows, {peptides.nunique()} peptides "
          f"({is_seen.sum():,} seen / {(~is_seen).sum():,} unseen rows)")
    print(f"cache: {CACHE_DIR}\n")

    # --- retrieval arms, representation swap only ---------------------------------------
    edit_scores = score_by_nearest_positive(evalset, positives).score
    esm_scores = score_by_nearest_positive(
        evalset,
        positives,
        similarity_fn=cosine_similarity(eval_cache, LAYER, train_cache),
    ).score
    print("scored edit_retrieval, esm_retrieval")

    # --- global arms, one model across all peptides -------------------------------------
    split = component_split(positives, validation_fraction=0.2, seed=HEADLINE_SEED)
    x_eval = build_features(evalset, eval_cache, LAYER)

    global_scores: dict[str, dict[int, np.ndarray]] = {"svc_global": {}}
    for seed in SEEDS:
        train = pd.concat(
            [split.train, make_negatives(split.train, STRATEGY, RATIO, seed)],
            ignore_index=True,
        )
        x_train = build_features(train, train_cache, LAYER)
        model = _svc().fit(x_train, train["Target"].to_numpy())
        global_scores["svc_global"][seed] = model.decision_function(x_eval)
        print(f"fitted svc_global seed {seed}")

    # The inherited logistic arm is refit here rather than read from b3_results.json so
    # every arm in the paired comparisons comes from one process and one row ordering.
    global_scores["logistic_global"] = {}
    for seed in SEEDS:
        train = pd.concat(
            [split.train, make_negatives(split.train, STRATEGY, RATIO, seed)],
            ignore_index=True,
        )
        x_train = build_features(train, train_cache, LAYER)
        fitted = fit_logistic(x_train, train["Target"].to_numpy(), seed=seed)
        global_scores["logistic_global"][seed] = logistic_scores(fitted, x_eval)
        print(f"fitted logistic_global seed {seed}")

    # --- per-peptide SVC, the SCEPTR-matched arm ----------------------------------------
    per_peptide: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        per_peptide[seed] = fit_per_peptide_svc(
            evalset, positives, train_cache, eval_cache, seed
        )
        print(f"fitted svc_per_peptide seed {seed}")

    # --- shortcut probes, two-sided on |macro - 0.5| ------------------------------------
    print("\n=== shortcut probes (gate: |macro - 0.5| <= 0.05) ===")
    probes = []
    train0 = pd.concat(
        [split.train, make_negatives(split.train, STRATEGY, RATIO, HEADLINE_SEED)],
        ignore_index=True,
    )
    for blocks in ("peptide_only", "tcr_only"):
        probe_model = _svc().fit(
            build_features(train0, train_cache, LAYER, blocks=blocks),
            train0["Target"].to_numpy(),
        )
        probe_scores = probe_model.decision_function(
            build_features(evalset, eval_cache, LAYER, blocks=blocks)
        )
        value = macro_auc01(y_eval[is_seen], probe_scores[is_seen], peptides[is_seen]).value
        probes.append({"probe": f"svc_global {blocks} / eval seen", "macro_auc01": round(value, 4)})

    shifted = fit_per_peptide_svc(
        evalset, positives, train_cache, eval_cache, HEADLINE_SEED, reference_shift=1
    )
    value = macro_auc01(y_eval[is_seen], shifted[is_seen], peptides[is_seen]).value
    probes.append({"probe": "svc_per_peptide shifted reference / eval seen", "macro_auc01": round(value, 4)})

    for probe in probes:
        probe["deviation"] = round(abs(probe["macro_auc01"] - 0.5), 4)
        probe["pass"] = bool(probe["deviation"] <= PROBE_TOLERANCE)
        print(f"  {probe['probe']:<48} {probe['macro_auc01']:.4f} "
              f"dev {probe['deviation']:.4f} {'PASS' if probe['pass'] else 'FAIL'}")
    probe_verdict = "PASS" if all(p["pass"] for p in probes) else "FAIL"
    print(f"  verdict: {probe_verdict}")
    if probe_verdict == "FAIL":
        print("  probes failed -- scores below are not trustworthy")

    # --- seed spread --------------------------------------------------------------------
    print("\n=== seed spread (seen slice, macro AUC0.1) ===")
    seeded = {
        "svc_global": global_scores["svc_global"],
        "logistic_global": global_scores["logistic_global"],
        "svc_per_peptide": per_peptide,
    }
    seed_spread = {}
    for name, by_seed in seeded.items():
        values = [
            macro_auc01(y_eval[is_seen], s[is_seen], peptides[is_seen]).value
            for s in by_seed.values()
        ]
        seed_spread[name] = {
            "seeds": list(by_seed),
            "values": [round(v, 4) for v in values],
            "mean": round(float(np.mean(values)), 4),
            "spread": round(float(np.max(values) - np.min(values)), 4),
        }
        print(f"  {name:<18} mean {np.mean(values):.4f}  "
              f"spread {np.max(values) - np.min(values):.4f}  "
              f"{[round(v, 4) for v in values]}")

    # --- headline-seed arms -------------------------------------------------------------
    arms = {
        "edit_retrieval": edit_scores,
        "esm_retrieval": esm_scores,
        "logistic_global": global_scores["logistic_global"][HEADLINE_SEED],
        "svc_global": global_scores["svc_global"][HEADLINE_SEED],
        "svc_per_peptide": per_peptide[HEADLINE_SEED],
    }
    slices = [("seen", is_seen), ("unseen", ~is_seen)]

    print("\n=== degeneracy ===")
    degeneracy = {}
    for name, scores in arms.items():
        for slice_name, mask in slices:
            d = diagnose_scores(scores[mask], peptides[mask])
            degeneracy[f"{name}/{slice_name}"] = {
                "is_degenerate": bool(d.is_degenerate),
                "distinct_scores": int(d.n_unique_scores),
                "constant_groups": int(d.n_constant_groups),
            }
            print(f"  {name:<18} {slice_name:<7} "
                  f"{'DEGENERATE' if d.is_degenerate else 'ok':<11} "
                  f"{d.n_unique_scores} distinct")

    print("\n=== scores ===")
    reports, scored = [], {}
    for name, scores in arms.items():
        for slice_name, mask in slices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = evaluate(y_eval, scores, peptides, subset=mask,
                                  label=f"{name} / {slice_name}", seed=HEADLINE_SEED)
            reports.append(report)
            scored[f"{name}/{slice_name}"] = report
    print(report_table(reports).to_string())

    print("\n=== 4.1 the missing paired interval ===")
    comparisons = {
        "esm_retrieval_minus_logistic_global/seen": paired(
            y_eval, peptides, "esm_retrieval seen", esm_scores,
            "logistic_global seen", arms["logistic_global"], is_seen,
        )
    }

    print("\n=== 4.2 SCEPTR arm, and the estimator/scope decomposition ===")
    for key, a, b in [
        ("svc_per_peptide_minus_esm_retrieval/seen", "svc_per_peptide", "esm_retrieval"),
        ("svc_global_minus_logistic_global/seen", "svc_global", "logistic_global"),
        ("svc_per_peptide_minus_svc_global/seen", "svc_per_peptide", "svc_global"),
        ("svc_per_peptide_minus_edit_retrieval/seen", "svc_per_peptide", "edit_retrieval"),
        ("esm_retrieval_minus_edit_retrieval/seen", "esm_retrieval", "edit_retrieval"),
    ]:
        comparisons[key] = paired(
            y_eval, peptides, f"{a} seen", arms[a], f"{b} seen", arms[b], is_seen
        )

    rows = []
    for peptide in sorted(set(peptides)):
        mask = (peptides == peptide).to_numpy()
        row = {
            "peptide": peptide,
            "seen": bool(is_seen[mask][0]),
            "n_rows": int(mask.sum()),
            "n_positive": int(y_eval[mask].sum()),
        }
        for name, scores in arms.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row[f"{name}_auc01"] = float(auc01(y_eval[mask], scores[mask]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_PER_PEPTIDE, index=False)

    OUT_JSON.write_text(json.dumps({
        "config": {
            "model": MODEL_KEY,
            "layer": LAYER,
            "strategy": STRATEGY,
            "ratio": RATIO,
            "seeds": list(SEEDS),
            "headline_seed": HEADLINE_SEED,
            "n_background": N_BACKGROUND,
            "probe_tolerance": PROBE_TOLERANCE,
        },
        "seen": {k: score_block(v) for k, v in scored.items() if k.endswith("/seen")},
        "unseen": {k: score_block(v) for k, v in scored.items() if k.endswith("/unseen")},
        "paired": comparisons,
        "probes": probes,
        "probe_verdict": probe_verdict,
        "seeds": list(SEEDS),
        "seed_spread": seed_spread,
        "degenerate_groups": degeneracy,
        "frozen_contract_changes": [],
        "deviations_from_sceptr": [
            "SCEPTR benchmarked 6 pMHCs by AUROC over reference sets of 1-200 TCRs; this "
            "scores 48 seen peptides by macro AUC0.1 over their full training databases.",
            "SCEPTR used paired alpha-beta TCRs reconstructed to full chains via Stitchr; "
            "this uses CDR3b only, in the IMMREP23 stripped-core convention.",
            "SCEPTR's ESM-2 was T6 8M final layer; this is 35M layer 10, the config the "
            "rest of the project reports.",
            "SCEPTR's negatives were 1000 background TCRs from an unlabelled repertoire; "
            "this samples 1000 from the training positives of other peptides, because the "
            "project has no unlabelled repertoire.",
            "SCEPTR's retrieval arm was nearest-neighbour over the reference set; this is "
            "max-over-database, which is the same operator at top_k=1.",
            "SCEPTR used one shared 1000-TCR background set across all pMHCs, explicitly for "
            "consistency; fit_per_peptide_svc draws a fresh sample per peptide from a shared "
            "RNG, adding per-peptide noise SCEPTR deliberately excluded.",
            "Normalisation is not held fixed across the operator contrast: svc_per_peptide "
            "standardises features per dimension while esm_retrieval L2-normalises, so "
            "svc_per_peptide - esm_retrieval carries a normalisation change as well as an "
            "operator change. Internal to this comparison, not a departure from SCEPTR.",
        ],
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
