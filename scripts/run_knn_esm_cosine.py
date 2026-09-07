"""0.2 -- rerun the k-NN baseline with ESM-2 cosine in place of edit distance.

The Phase 1 comparison confounded two things: the k-NN scored by normalised edit distance
while the logistic head scored from ESM-2 embeddings, so algorithm and representation
differed at once. This script removes the confound by holding the algorithm fixed. Same
database construction, same max-over-database operator, same metric, same evaluation set --
only the similarity function changes.

If ESM-2 cosine scores below edit distance, the claim that follows carries no confound: a
protein language model produces a worse similarity measure for these TCRs than counting
letter differences.

Layers are swept because Session 4 found the final-layer norm collapsing 118.6 -> 7.2, which
makes the last layer a suspect default for a similarity measure.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from cognate.baseline_knn import cosine_similarity, score_by_nearest_positive
from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.metrics import (
    auc01,
    compare_macro_auc01,
    diagnose_scores,
    evaluate,
    report_table,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
OUT_JSON = ROOT / "data" / "knn_esm_cosine.json"
OUT_PER_PEPTIDE = ROOT / "data" / "knn_esm_cosine_per_peptide.csv"
SEED = 0

# (model key, layer index, label). Layer 0 is the embedding layer, so the last hidden layer
# is n_layers - 1: 12 for the 12-layer 35M, 6 for the 6-layer 8M. Layer 10 is the config the
# Phase 1 headline used, kept so the cosine run is comparable to the head it is measured against.
CONFIGS = [
    ("35M", 12, "35M last"),
    ("35M", 10, "35M layer10 (headline)"),
    ("35M", 6, "35M middle"),
    ("8M", 6, "8M last"),
    ("8M", 3, "8M middle"),
]


def vdjdb_cache_path(model_key: str) -> Path:
    return default_cache_dir() / f"esm2_{model_key}_vdjdb.npz"


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


def main() -> None:
    positives = load_train()
    evalset = pd.read_csv(EVAL_CSV)
    y_eval = evalset["Label"].to_numpy()
    peptides = evalset["Peptide"]
    is_seen = peptides.isin(set(positives["Peptide"])).to_numpy()
    slices = [("seen", is_seen), ("unseen", ~is_seen)]

    print(f"eval: {len(evalset):,} rows, {peptides.nunique()} peptides "
          f"({is_seen.sum():,} seen / {(~is_seen).sum():,} unseen rows)")
    print(f"cache dir: {default_cache_dir()}\n")

    scorers = {"edit": score_by_nearest_positive(evalset, positives).score}
    for model_key, layer, label in CONFIGS:
        query_cache = load_cache(vdjdb_cache_path(model_key))
        database_cache = load_cache(cache_path(model_key))
        scorers[label] = score_by_nearest_positive(
            evalset,
            positives,
            similarity_fn=cosine_similarity(query_cache, layer, database_cache),
        ).score
        print(f"scored {label}")

    print("\n=== degeneracy probe ===")
    degeneracy = {}
    for name, scores in scorers.items():
        for slice_name, mask in slices:
            d = diagnose_scores(scores[mask], peptides[mask])
            degeneracy[f"{name}/{slice_name}"] = {
                "is_degenerate": bool(d.is_degenerate),
                "distinct_scores": int(d.n_unique_scores),
                "constant_groups": int(d.n_constant_groups),
            }
            print(f"  {name:<24} {slice_name:<7} "
                  f"{'DEGENERATE' if d.is_degenerate else 'ok':<11} "
                  f"{d.n_unique_scores} distinct")

    print("\n=== scores ===")
    reports, scored = [], {}
    for name, scores in scorers.items():
        for slice_name, mask in slices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = evaluate(y_eval, scores, peptides, subset=mask,
                                  label=f"{name} / {slice_name}", seed=SEED)
            reports.append(report)
            scored[f"{name}/{slice_name}"] = report
    print(report_table(reports).to_string())

    print("\n=== paired: cosine minus edit distance, same peptides both arms ===")
    comparisons = {}
    for name in scorers:
        if name == "edit":
            continue
        for slice_name, mask in slices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comparison = compare_macro_auc01(
                    y_eval, peptides,
                    (f"{name} {slice_name}", scorers[name], mask),
                    (f"edit {slice_name}", scorers["edit"], mask), seed=SEED,
                )
            comparisons[f"{name}_minus_edit/{slice_name}"] = {
                "point": round(comparison.difference.point, 4),
                "lo": round(comparison.difference.lo, 4),
                "hi": round(comparison.difference.hi, 4),
                "p": round(comparison.p_two_sided, 4),
                "n_peptides": int(comparison.n_groups),
            }
            print("  " + str(comparison))

    rows = []
    for peptide in sorted(set(peptides)):
        mask = (peptides == peptide).to_numpy()
        row = {"peptide": peptide, "seen": bool(is_seen[mask][0]),
               "n_rows": int(mask.sum()), "n_positive": int(y_eval[mask].sum())}
        for name, scores in scorers.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row[f"{name}_auc01"] = float(auc01(y_eval[mask], scores[mask]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_PER_PEPTIDE, index=False)

    OUT_JSON.write_text(json.dumps({
        "config": {"seed": SEED, "configs": [
            {"model": m, "layer": l, "label": lab} for m, l, lab in CONFIGS]},
        "scores": {k: score_block(r) for k, r in scored.items()},
        "degeneracy": degeneracy,
        "comparisons": comparisons,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
