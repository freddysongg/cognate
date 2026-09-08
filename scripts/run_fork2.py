"""Fork 2 -- cross-attention over residues, against its own mean-pool control.

Training uses IMMREP23 with `matched` negatives at ratio 5, the same sampler as the Phase 1
head, so the comparison against 0.511 is like-for-like. The split is component-based, the
backbone is frozen, and both variants are trained at three seeds.

**2a and 2b are reported together or not at all.** 2b is the same architecture with the
cross-attention block removed and nothing else changed, so 2a minus 2b isolates the effect of
attending across residue positions. Without it, a 2a result could not be distinguished from an
incidental difference in training setup.

**The train-side pooled/macro gap is inherited.** Under `matched` negatives, peptide features
alone score 0.5847 pooled on training data while macro AUC0.1 sits at 0.4995, so a train-side
gap near +0.156 is expected and comes from Phase 1's sampler, not from anything here. It is
recorded rather than treated as a defect.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.baseline_knn import score_by_nearest_positive
from cognate.crossattn import (
    RowBatches,
    fit_residue_model,
    pad_residues,
    pick_device,
    score_rows,
)
from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache, load_residue_cache
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
OUT_JSON = ROOT / "data" / "fork2_results.json"
OUT_PER_PEPTIDE = ROOT / "data" / "fork2_per_peptide.csv"

MODEL_KEY = "35M"
LAYER = 10
SEEDS = (0, 1, 2)
VALIDATION_FRACTION = 0.2
STRATEGY = "matched"
RATIO = 5.0
PROBE_GATE = 0.05
VARIANTS = {"2a": True, "2b": False}


def interval_list(interval) -> list[float]:
    return [round(interval.lo, 4), round(interval.hi, 4)]


def make_rows(frame, peptides, tcrs, targets, device) -> RowBatches:
    return RowBatches(
        peptides.rows_for(frame["Peptide"].astype(str).to_numpy()),
        tcrs.rows_for(frame["CDR3b"].astype(str).to_numpy()),
        targets,
        peptides,
        tcrs,
        device,
    )


def one_sided_probe(train_rows, eval_rows, train_cache, eval_cache, blocks, seed) -> float:
    """Phase B's shortcut probe, unchanged, on the representation the model consumes."""
    x_train = build_features(train_rows, train_cache, LAYER, blocks)
    x_eval = build_features(eval_rows, eval_cache, LAYER, blocks)
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
    device = pick_device()

    print(f"eval: {len(evalset):,} rows, {peptides.nunique()} peptides "
          f"({is_seen.sum():,} seen / {(~is_seen).sum():,} unseen rows)")
    print(f"device: {device}, layer {LAYER}, negatives {STRATEGY} x{RATIO}\n")

    residues_train = load_residue_cache(
        default_cache_dir() / f"esm2_{MODEL_KEY}_train_residues.npz"
    )
    residues_eval = load_residue_cache(
        default_cache_dir() / f"esm2_{MODEL_KEY}_eval_residues.npz"
    )
    train_peptides = pad_residues(
        residues_train, sorted(set(positives["Peptide"].astype(str))), LAYER
    )
    train_tcrs = pad_residues(
        residues_train, sorted(set(positives["CDR3b"].astype(str))), LAYER
    )
    eval_peptides = pad_residues(
        residues_eval, sorted(set(evalset["Peptide"].astype(str))), LAYER
    )
    eval_tcrs = pad_residues(
        residues_eval, sorted(set(evalset["CDR3b"].astype(str))), LAYER
    )
    print(f"padded train: peptides {train_peptides.values.shape} "
          f"tcrs {train_tcrs.values.shape}")
    print(f"padded eval:  peptides {eval_peptides.values.shape} "
          f"tcrs {eval_tcrs.values.shape}\n")

    eval_rows = make_rows(evalset, eval_peptides, eval_tcrs, y_eval, device)
    knn_edit = score_by_nearest_positive(evalset, positives).score

    print("training")
    scores: dict[str, dict[int, np.ndarray]] = {v: {} for v in VARIANTS}
    splits = {}
    for seed in SEEDS:
        split = component_split(
            positives, validation_fraction=VALIDATION_FRACTION, seed=seed
        )
        splits[seed] = split
        train_frame = pd.concat(
            [split.train, make_negatives(split.train, STRATEGY, RATIO, seed)],
            ignore_index=True,
        )
        val_frame = pd.concat(
            [split.validation, make_negatives(split.validation, STRATEGY, RATIO, seed)],
            ignore_index=True,
        )
        train_rows = make_rows(
            train_frame, train_peptides, train_tcrs,
            train_frame["Target"].to_numpy(), device,
        )
        val_rows = make_rows(
            val_frame, train_peptides, train_tcrs,
            val_frame["Target"].to_numpy(), device,
        )
        for variant, use_attention in VARIANTS.items():
            model, history = fit_residue_model(
                train_rows, val_rows, val_frame["Peptide"].to_numpy(),
                train_peptides.hidden_size, use_attention=use_attention, seed=seed,
            )
            scores[variant][seed] = score_rows(model, eval_rows)
            print(f"  seed {seed} {variant}: {history.n_epochs} epochs "
                  f"(best {history.best_epoch}), val loss {min(history.val_loss):.4f}, "
                  f"val macro {history.val_macro_auc01[history.best_epoch]:.4f}")

    print("\n=== per-seed point estimates (macro AUC0.1) ===")
    per_seed: dict[str, dict[str, list[float]]] = {}
    for variant in VARIANTS:
        per_seed[variant] = {}
        for slice_name, mask in slices:
            values = []
            for seed in SEEDS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    values.append(float(macro_auc01(
                        y_eval[mask], scores[variant][seed][mask],
                        peptides[mask].to_numpy()).value))
            per_seed[variant][slice_name] = values
            print(f"  {variant} / {slice_name:<7} {[round(v, 4) for v in values]}  "
                  f"mean {np.mean(values):.4f}  spread {max(values) - min(values):.4f}")

    representative = {}
    for variant in VARIANTS:
        values = per_seed[variant]["seen"]
        representative[variant] = SEEDS[
            int(np.argmin(np.abs(np.array(values) - np.mean(values))))
        ]
    print(f"  representative seeds: {representative}")

    print("\n=== 2a minus 2b, the attention ablation ===")
    ablation = {}
    for slice_name, mask in slices:
        a = np.mean(per_seed["2a"][slice_name])
        b = np.mean(per_seed["2b"][slice_name])
        ablation[slice_name] = round(float(a - b), 4)
        print(f"  {slice_name:<7} 2a {a:.4f} - 2b {b:.4f} = {a - b:+.4f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paired_ablation = compare_macro_auc01(
            y_eval, peptides,
            ("2a seen", scores["2a"][representative["2a"]], is_seen),
            ("2b seen", scores["2b"][representative["2b"]], is_seen),
            seed=SEEDS[0],
        )
    print("  paired: " + str(paired_ablation))

    print("\n=== shortcut probes ===")
    pooled_train = load_cache(cache_path(MODEL_KEY))
    pooled_eval = load_cache(default_cache_dir() / f"esm2_{MODEL_KEY}_vdjdb.npz")
    seed = SEEDS[0]
    probe_rows = pd.concat(
        [splits[seed].train, make_negatives(splits[seed].train, STRATEGY, RATIO, seed)],
        ignore_index=True,
    )
    peptide_only = one_sided_probe(
        probe_rows, evalset, pooled_train, pooled_eval, "peptide_only", seed
    )
    tcr_only = one_sided_probe(
        probe_rows, evalset, pooled_train, pooled_eval, "tcr_only", seed
    )
    worst = max(abs(peptide_only - 0.5), abs(tcr_only - 0.5))
    print(f"  peptide_only {peptide_only:.4f}  tcr_only {tcr_only:.4f}  "
          f"worst |dev| {worst:.4f}  {'PASS' if worst <= PROBE_GATE else 'FAIL'}")

    train_gaps = {}
    probe_train_rows = make_rows(
        probe_rows, train_peptides, train_tcrs, probe_rows["Target"].to_numpy(), device
    )
    for variant, use_attention in VARIANTS.items():
        split = splits[seed]
        train_frame = pd.concat(
            [split.train, make_negatives(split.train, STRATEGY, RATIO, seed)],
            ignore_index=True,
        )
        val_frame = pd.concat(
            [split.validation, make_negatives(split.validation, STRATEGY, RATIO, seed)],
            ignore_index=True,
        )
        model, _ = fit_residue_model(
            make_rows(train_frame, train_peptides, train_tcrs,
                      train_frame["Target"].to_numpy(), device),
            make_rows(val_frame, train_peptides, train_tcrs,
                      val_frame["Target"].to_numpy(), device),
            val_frame["Peptide"].to_numpy(), train_peptides.hidden_size,
            use_attention=use_attention, seed=seed,
        )
        train_scores = score_rows(model, probe_train_rows)
        target = probe_rows["Target"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gap = float(auroc(target, train_scores)) - float(
                macro_auc01(target, train_scores, probe_rows["Peptide"].to_numpy()).value
            )
        train_gaps[variant] = round(gap, 4)
        print(f"  {variant} train pooled/macro gap {gap:+.4f}  "
              f"(inherited reference +0.156)")

    print("\n=== scores at the representative seed, with CIs ===")
    results = []
    for variant in VARIANTS:
        seed = representative[variant]
        variant_scores = scores[variant][seed]
        block = {"fork": "crossattn", "variant": variant}
        degenerate_groups = 0
        for slice_name, mask in slices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = evaluate(y_eval, variant_scores, peptides, subset=mask,
                                  label=f"{variant} / {slice_name}", seed=seed)
                diagnostics = diagnose_scores(variant_scores[mask], peptides[mask])
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
                  f"CI {interval_list(report.macro_auc01)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison = compare_macro_auc01(
                y_eval, peptides,
                (f"{variant} seen", variant_scores, is_seen),
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
        spread = float(max(seen_values) - min(seen_values))
        ci_width = block["seen"]["ci"][1] - block["seen"]["ci"][0]
        block["probes"] = {
            "peptide_only": round(peptide_only, 4),
            "tcr_only": round(tcr_only, 4),
            "train_pooled_macro_gap": train_gaps[variant],
            "worst_deviation": round(worst, 4),
            "verdict": "PASS" if worst <= PROBE_GATE else "FAIL",
        }
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
            for variant in VARIANTS:
                row[f"{variant}_auc01"] = float(
                    auc01(y_eval[mask], scores[variant][representative[variant]][mask])
                )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_PER_PEPTIDE, index=False)

    OUT_JSON.write_text(json.dumps({
        "config": {"model": MODEL_KEY, "layer": LAYER, "seeds": list(SEEDS),
                   "validation_fraction": VALIDATION_FRACTION,
                   "negative_strategy": STRATEGY, "negative_ratio": RATIO,
                   "probe_gate": PROBE_GATE},
        "reference": {"edit_knn_seen": 0.5654, "logistic_head_seen": 0.5107,
                      "frozen_esm_cosine_seen_layer10": 0.5364, "random_seen": 0.5009,
                      "inherited_train_gap": 0.156},
        "attention_ablation": {
            "mean_difference": ablation,
            "paired_seen": {
                "delta": round(paired_ablation.difference.point, 4),
                "ci": [round(paired_ablation.difference.lo, 4),
                       round(paired_ablation.difference.hi, 4)],
                "p": round(paired_ablation.p_two_sided, 4),
            },
        },
        "results": results,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
