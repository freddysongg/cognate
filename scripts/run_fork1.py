"""Fork 1 -- train and score both contrastive variants on the frozen evaluation set.

Training uses IMMREP23 positives only: 11,312 pairs over 808 peptides, no negative sampler
anywhere in the objective. The validation split is component-based, because a peptide-held-out
split leaks ~10.9% of TCRs. The ESM-2 backbone is frozen, so any difference against the
baselines is attributable to the objective rather than to capacity.

**Reporting.** Every variant is trained at three seeds. Point estimates are the mean across
seeds and ``seed_spread`` is max-min, per the work order. Bootstrap CIs are expensive, so they
are computed once on the representative seed -- the seed whose point estimate is nearest the
mean -- and labelled as such in the output. If ``seed_spread`` exceeds the CI width, the
spread is the dominant uncertainty and should be read that way.

**Probes.** The shortcut gate runs on the *learned* representation, using the same methodology
Phase B used on the frozen one: fit a one-sided logistic head on training rows and score the
evaluation set. The probe is the only place a negative sampler appears, and it is a diagnostic,
not part of any training objective -- keeping it identical to Phase B is what makes the numbers
comparable to the recorded 0.5000 / 0.5004 baseline.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.baseline_knn import cosine_similarity, score_by_nearest_positive
from cognate.contrastive import (
    encode_labels,
    fit_tcr_metric,
    fit_two_tower,
    projected_cache,
)
from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.features import build_features
from cognate.metrics import (
    auc01,
    auroc,
    compare_macro_auc01,
    diagnose_scores,
    evaluate,
    macro_auc01,
)
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, logistic_scores

ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
OUT_JSON = ROOT / "data" / "fork1_results.json"
OUT_PER_PEPTIDE = ROOT / "data" / "fork1_per_peptide.csv"

MODEL_KEY = "35M"
LAYER = 10
SEEDS = (0, 1, 2)
VALIDATION_FRACTION = 0.2
PROBE_STRATEGY = "matched"
PROBE_RATIO = 5.0
PROBE_GATE = 0.05
PROJECTED_LAYER = 0


def interval_list(interval) -> list[float]:
    return [round(interval.lo, 4), round(interval.hi, 4)]


def train_one_seed(
    positives: pd.DataFrame,
    train_cache,
    eval_cache,
    evalset: pd.DataFrame,
    seed: int,
) -> dict:
    """Fit 1a and 1b at one seed and score the evaluation set with each."""
    split = component_split(positives, validation_fraction=VALIDATION_FRACTION, seed=seed)
    train_labels = encode_labels(split.train["Peptide"].astype(str).tolist())
    val_labels = encode_labels(split.validation["Peptide"].astype(str).tolist())

    train_tcr = train_cache.lookup(split.train["CDR3b"].astype(str), LAYER)
    val_tcr = train_cache.lookup(split.validation["CDR3b"].astype(str), LAYER)
    train_peptide = train_cache.lookup(split.train["Peptide"].astype(str), LAYER)
    val_peptide = train_cache.lookup(split.validation["Peptide"].astype(str), LAYER)

    metric_head, metric_standardisation, metric_history = fit_tcr_metric(
        train_tcr, train_labels, val_tcr, val_labels, seed=seed
    )
    projected_eval = projected_cache(eval_cache, LAYER, metric_head, metric_standardisation)
    projected_train = projected_cache(train_cache, LAYER, metric_head, metric_standardisation)
    scores_1a = score_by_nearest_positive(
        evalset,
        positives,
        similarity_fn=cosine_similarity(projected_eval, PROJECTED_LAYER, projected_train),
    ).score

    peptide_tower, tcr_tower, tower_standardisation, tower_history = fit_two_tower(
        train_peptide, train_tcr, train_labels,
        val_peptide, val_tcr, val_labels, seed=seed,
    )
    tower_eval_peptides = projected_cache(
        eval_cache, LAYER, peptide_tower, tower_standardisation
    )
    tower_eval_tcrs = projected_cache(eval_cache, LAYER, tcr_tower, tower_standardisation)
    scores_1b = (
        tower_eval_peptides.lookup(evalset["Peptide"].astype(str), PROJECTED_LAYER)
        * tower_eval_tcrs.lookup(evalset["CDR3b"].astype(str), PROJECTED_LAYER)
    ).sum(axis=1)

    print(f"  seed {seed}: 1a {metric_history.n_epochs} epochs "
          f"(best {metric_history.best_epoch}), 1b {tower_history.n_epochs} epochs "
          f"(best {tower_history.best_epoch})")

    return {
        "1a": scores_1a,
        "1b": scores_1b,
        "projected_eval": projected_eval,
        "projected_train": projected_train,
        "tower_eval_peptides": tower_eval_peptides,
        "tower_eval_tcrs": tower_eval_tcrs,
        "tower_train_peptides": projected_cache(
            train_cache, LAYER, peptide_tower, tower_standardisation
        ),
        "tower_train_tcrs": projected_cache(
            train_cache, LAYER, tcr_tower, tower_standardisation
        ),
        "split": split,
    }


def one_sided_probe(
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    train_source,
    eval_source,
    blocks: str,
    seed: int,
) -> float:
    """Phase B's shortcut probe, run against a learned cache instead of the frozen one."""
    x_train = build_features(train_rows, train_source, PROJECTED_LAYER, blocks)
    x_eval = build_features(eval_rows, eval_source, PROJECTED_LAYER, blocks)
    model = fit_logistic(x_train, train_rows["Target"].to_numpy(), seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            macro_auc01(
                eval_rows["Label"].to_numpy(),
                logistic_scores(model, x_eval),
                eval_rows["Peptide"].to_numpy(),
            ).value
        )


def main() -> None:
    positives = load_train()
    evalset = pd.read_csv(EVAL_CSV)
    y_eval = evalset["Label"].to_numpy()
    peptides = evalset["Peptide"]
    is_seen = peptides.isin(set(positives["Peptide"])).to_numpy()
    slices = [("seen", is_seen), ("unseen", ~is_seen)]

    train_cache = load_cache(cache_path(MODEL_KEY))
    eval_cache = load_cache(default_cache_dir() / f"esm2_{MODEL_KEY}_vdjdb.npz")

    print(f"eval: {len(evalset):,} rows, {peptides.nunique()} peptides "
          f"({is_seen.sum():,} seen / {(~is_seen).sum():,} unseen rows)")
    print(f"train: {len(positives):,} positives, {positives['Peptide'].nunique()} peptides, "
          f"no negative sampler\n")

    knn_edit = score_by_nearest_positive(evalset, positives).score

    print("training")
    runs = {
        seed: train_one_seed(positives, train_cache, eval_cache, evalset, seed)
        for seed in SEEDS
    }

    print("\n=== per-seed point estimates (macro AUC0.1) ===")
    per_seed: dict[str, dict[str, list[float]]] = {}
    for variant in ("1a", "1b"):
        per_seed[variant] = {}
        for slice_name, mask in slices:
            values = []
            for seed in SEEDS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    values.append(
                        float(
                            macro_auc01(
                                y_eval[mask], runs[seed][variant][mask],
                                peptides[mask].to_numpy(),
                            ).value
                        )
                    )
            per_seed[variant][slice_name] = values
            spread = max(values) - min(values)
            print(f"  {variant} / {slice_name:<7} "
                  f"{[round(v, 4) for v in values]}  mean {np.mean(values):.4f}  "
                  f"spread {spread:.4f}")

    representative = {}
    for variant in ("1a", "1b"):
        values = per_seed[variant]["seen"]
        representative[variant] = SEEDS[
            int(np.argmin(np.abs(np.array(values) - np.mean(values))))
        ]
        print(f"  representative seed for {variant}: {representative[variant]}")

    print("\n=== shortcut probes on the learned representation ===")
    probes = {}
    for variant in ("1a", "1b"):
        seed = representative[variant]
        split = runs[seed]["split"]
        probe_rows = pd.concat(
            [split.train, make_negatives(split.train, PROBE_STRATEGY, PROBE_RATIO, seed)],
            ignore_index=True,
        )
        if variant == "1a":
            tcr_train, tcr_eval = runs[seed]["projected_train"], runs[seed]["projected_eval"]
            peptide_train, peptide_eval = tcr_train, tcr_eval
        else:
            tcr_train, tcr_eval = runs[seed]["tower_train_tcrs"], runs[seed]["tower_eval_tcrs"]
            peptide_train = runs[seed]["tower_train_peptides"]
            peptide_eval = runs[seed]["tower_eval_peptides"]

        peptide_only = one_sided_probe(
            probe_rows, evalset, peptide_train, peptide_eval, "peptide_only", seed
        )
        tcr_only = one_sided_probe(
            probe_rows, evalset, tcr_train, tcr_eval, "tcr_only", seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if variant == "1a":
                train_scores = score_by_nearest_positive(
                    probe_rows, positives, leave_out_exact_matches=True,
                    similarity_fn=cosine_similarity(tcr_train, PROJECTED_LAYER, tcr_train),
                ).score
            else:
                train_scores = (
                    peptide_train.lookup(probe_rows["Peptide"].astype(str), PROJECTED_LAYER)
                    * tcr_train.lookup(probe_rows["CDR3b"].astype(str), PROJECTED_LAYER)
                ).sum(axis=1)
            target = probe_rows["Target"].to_numpy()
            gap = float(auroc(target, train_scores)) - float(
                macro_auc01(target, train_scores, probe_rows["Peptide"].to_numpy()).value
            )

        worst = max(abs(peptide_only - 0.5), abs(tcr_only - 0.5))
        probes[variant] = {
            "peptide_only": round(peptide_only, 4),
            "tcr_only": round(tcr_only, 4),
            "train_pooled_macro_gap": round(gap, 4),
            "worst_deviation": round(worst, 4),
            "verdict": "PASS" if worst <= PROBE_GATE else "FAIL",
        }
        print(f"  {variant}: peptide_only {peptide_only:.4f}  tcr_only {tcr_only:.4f}  "
              f"train gap {gap:+.4f}  worst |dev| {worst:.4f}  {probes[variant]['verdict']}")

    print("\n=== scores at the representative seed, with CIs ===")
    results = []
    for variant in ("1a", "1b"):
        seed = representative[variant]
        scores = runs[seed][variant]
        block = {"fork": "contrastive", "variant": variant}
        degenerate_groups = 0
        for slice_name, mask in slices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = evaluate(y_eval, scores, peptides, subset=mask,
                                  label=f"{variant} / {slice_name}", seed=seed)
                diagnostics = diagnose_scores(scores[mask], peptides[mask])
            degenerate_groups += int(diagnostics.n_constant_groups)
            values = per_seed[variant][slice_name]
            block[slice_name] = {
                "macro_auc01": round(float(np.mean(values)), 4),
                "ci": interval_list(report.macro_auc01),
                "ci_from_seed": seed,
                "n_peptides": int(report.n_groups_scored),
                "is_degenerate": bool(diagnostics.is_degenerate),
            }
            print(f"  {variant} / {slice_name:<7} mean {np.mean(values):.4f}  "
                  f"CI {interval_list(report.macro_auc01)}  "
                  f"{'DEGENERATE' if diagnostics.is_degenerate else ''}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison = compare_macro_auc01(
                y_eval, peptides,
                (f"{variant} seen", scores, is_seen),
                ("edit k-NN seen", knn_edit, is_seen), seed=seed,
            )
        block["paired_vs_knn_edit"] = {
            "delta": round(comparison.difference.point, 4),
            "ci": [round(comparison.difference.lo, 4), round(comparison.difference.hi, 4)],
            "p": round(comparison.p_two_sided, 4),
            "n_peptides": int(comparison.n_groups),
        }
        print("    " + str(comparison))

        seen_values = per_seed[variant]["seen"]
        ci_width = block["seen"]["ci"][1] - block["seen"]["ci"][0]
        spread = float(max(seen_values) - min(seen_values))
        block["probes"] = probes[variant]
        block["seeds"] = list(SEEDS)
        block["per_seed_seen"] = [round(v, 4) for v in seen_values]
        block["per_seed_unseen"] = [round(v, 4) for v in per_seed[variant]["unseen"]]
        block["seed_spread"] = round(spread, 4)
        block["seed_spread_exceeds_ci_width"] = bool(spread > ci_width)
        block["degenerate_groups"] = degenerate_groups
        block["frozen_contract_changes"] = []
        results.append(block)

    rows = []
    for peptide in sorted(set(peptides)):
        mask = (peptides == peptide).to_numpy()
        row = {"peptide": peptide, "seen": bool(is_seen[mask][0]),
               "n_rows": int(mask.sum()), "n_positive": int(y_eval[mask].sum())}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row["edit_knn_auc01"] = float(auc01(y_eval[mask], knn_edit[mask]))
            for variant in ("1a", "1b"):
                seed = representative[variant]
                row[f"{variant}_auc01"] = float(
                    auc01(y_eval[mask], runs[seed][variant][mask])
                )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_PER_PEPTIDE, index=False)

    OUT_JSON.write_text(json.dumps({
        "config": {"model": MODEL_KEY, "layer": LAYER, "seeds": list(SEEDS),
                   "validation_fraction": VALIDATION_FRACTION,
                   "negative_sampler": None,
                   "probe_strategy": PROBE_STRATEGY, "probe_ratio": PROBE_RATIO,
                   "probe_gate": PROBE_GATE},
        "reference": {"edit_knn_seen": 0.5654, "frozen_esm_cosine_seen_layer10": 0.5358,
                      "logistic_head_seen": 0.5107, "random_seen": 0.5009},
        "results": results,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
