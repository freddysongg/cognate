"""B2 -- shortcut probes on the Phase B evaluation set. This is a gate, not a result.

Standing procedure since the Session 5 sampler bug (docs/findings.md S4): before scoring
anything on a new evaluation set, check that the set cannot be scored well without actually
modelling TCR-peptide compatibility. A head that does well from one side alone, or a model
whose pooled AUROC vastly exceeds its macro AUC0.1 on its own training data, is reading a
shortcut.

Probes, in order of how badly a failure would matter:

  1. peptide-features-only head   -- expect ~0.5 on the eval set
  2. TCR-features-only head       -- expect ~0.5 on the eval set
  3. TCR frequency (model-free)   -- specific to this construction: negatives reuse TCRs
                                     drawn from other peptides' pools, so a TCR that is
                                     sampled often could be identifiable as a negative
  4. CDR3b length (model-free)    -- the standard confound
  5. train AUROC vs train macro AUC0.1 gap for the full model -- the Session 5 signature

The full model is fitted too, as the reference arm the probes are read against, so its
evaluation-set score does appear in the output. That number is not the B3 result -- B3 is the
paired comparison against the k-NN baseline with bootstrap CIs, per-peptide correlations, and
the held/weakened/reversed verdicts against docs/belief_list.md. None of that runs here.

A probe fails in either direction. Macro AUC0.1 below 0.5 is exactly as exploitable as above --
invert the score -- and the McClish floor is 9/19 ~ 0.474, so the below-chance range is only
0.026 wide and a deviation of 0.012 downward is proportionally much larger than it looks.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cognate.data import load_train
from cognate.embed import cache_path, default_cache_dir, load_cache
from cognate.features import build_features
from cognate.metrics import auroc, macro_auc01
from cognate.negatives import make_negatives
from cognate.split import component_split
from cognate.train_head import fit_logistic, logistic_scores

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = default_cache_dir()
VDJDB_CACHE = CACHE_DIR / "esm2_35M_vdjdb.npz"
EVAL_CSV = ROOT / "data" / "vdjdb_eval.csv"
OUT_JSON = ROOT / "data" / "vdjdb_probes.json"

MODEL_KEY = "35M"
LAYER = 10
STRATEGY = "matched"
RATIO = 5.0
SEED = 0
PASS_BAND = 0.55


def probe_row(name: str, y: np.ndarray, scores: np.ndarray, groups: pd.Series) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro = macro_auc01(y, scores, groups.to_numpy()).value
    return {
        "probe": name,
        "pooled_auroc": round(float(auroc(y, scores)), 4),
        "macro_auc01": round(float(macro), 4),
        "distinct_scores": int(len(np.unique(scores))),
    }


def main() -> None:
    positives = load_train()
    split = component_split(positives, validation_fraction=0.2, seed=SEED)
    train = pd.concat(
        [split.train, make_negatives(split.train, STRATEGY, RATIO, SEED)],
        ignore_index=True,
    )
    y_train = train["Target"].to_numpy()

    evalset = pd.read_csv(EVAL_CSV)
    y_eval = evalset["Label"].to_numpy()
    eval_peptides = evalset["Peptide"]

    train_cache = load_cache(cache_path(MODEL_KEY, CACHE_DIR))
    eval_cache = load_cache(VDJDB_CACHE)

    print(f"train rows {len(train):,} ({y_train.mean():.1%} positive), "
          f"{train['Peptide'].nunique()} peptides")
    print(f"eval rows  {len(evalset):,} ({y_eval.mean():.1%} positive), "
          f"{eval_peptides.nunique()} peptides\n")

    results: list[dict] = []
    for blocks in ("peptide_only", "tcr_only", "interactions"):
        x_train = build_features(train, train_cache, LAYER, blocks=blocks)
        x_eval = build_features(evalset, eval_cache, LAYER, blocks=blocks)
        head = fit_logistic(x_train, y_train, seed=SEED)

        on_train = probe_row(f"{blocks} / train", y_train, logistic_scores(head, x_train),
                             train["Peptide"])
        on_eval = probe_row(f"{blocks} / eval", y_eval, logistic_scores(head, x_eval),
                            eval_peptides)
        results.extend([on_train, on_eval])
        print(f"  fitted {blocks:<13} width={x_train.shape[1]}")

    counts = evalset["CDR3b"].value_counts()
    results.append(probe_row("tcr frequency / eval", y_eval,
                             evalset["CDR3b"].map(counts).to_numpy().astype(float),
                             eval_peptides))
    results.append(probe_row("cdr3b length / eval", y_eval,
                             evalset["CDR3b"].str.len().to_numpy().astype(float),
                             eval_peptides))

    table = pd.DataFrame(results)
    print("\n=== probes ===")
    print(table.to_string(index=False))

    shortcut_probes = table[
        table["probe"].str.endswith("/ eval") & ~table["probe"].str.startswith("interactions")
    ].copy()
    tolerance = PASS_BAND - 0.5
    shortcut_probes["macro_deviation"] = (shortcut_probes["macro_auc01"] - 0.5).abs()
    shortcut_probes["pooled_deviation"] = (shortcut_probes["pooled_auroc"] - 0.5).abs()
    worst = shortcut_probes["macro_deviation"].max()
    failures = shortcut_probes[shortcut_probes["macro_deviation"] > tolerance]

    full_train = table[table["probe"] == "interactions / train"].iloc[0]
    gap = full_train["pooled_auroc"] - full_train["macro_auc01"]

    print("\n=== eval-set shortcut probes, deviation from chance in both directions ===")
    print(shortcut_probes[["probe", "macro_auc01", "macro_deviation",
                           "pooled_auroc", "pooled_deviation"]].to_string(index=False))
    print(f"\nlargest macro deviation: {worst:.4f} (tolerance {tolerance:.4f})")
    print(f"largest pooled deviation: {shortcut_probes['pooled_deviation'].max():.4f}")
    print(f"full model on its own training data: pooled AUROC {full_train['pooled_auroc']:.4f} "
          f"vs macro AUC0.1 {full_train['macro_auc01']:.4f}  (gap {gap:+.4f})")
    verdict = "FAIL" if len(failures) else "PASS"
    print(f"\nB2 verdict: {verdict}")
    if len(failures):
        print(failures.to_string(index=False))

    OUT_JSON.write_text(
        json.dumps(
            {
                "config": {"model": MODEL_KEY, "layer": LAYER, "strategy": STRATEGY,
                           "ratio": RATIO, "seed": SEED, "pass_band": PASS_BAND},
                "probes": results,
                "largest_eval_shortcut_macro_deviation": round(float(worst), 4),
                "largest_eval_shortcut_pooled_deviation": round(
                    float(shortcut_probes["pooled_deviation"].max()), 4
                ),
                "full_model_train_gap": round(float(gap), 4),
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
