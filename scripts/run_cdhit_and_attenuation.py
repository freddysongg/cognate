"""CD-HIT >95% as a sweep regime, and paired intervals for the attenuation itself.

Two gaps the red team left open, closed in one pass because both need the same three
score vectors and the same fixed IMMREP23 reference.

**Part 1 (redteam_curve.md S5).** B2 claims our dedup is judged against "every published
deduplication standard", but Lu et al. (*Nat Methods* 23:248-259) use CD-HIT >95% and no
CD-HIT clustering appears in the sweep. This adds it, run through the real program rather
than a proxy: `cd-hit-2d` compares the evaluation CDR3b against the fixed `load_train()`
CDR3b set and emits the ones no training sequence is >95% identical to. Two databases is
the right tool here -- clustering the union instead makes which member survives depend on
input order, and the sweep needs a mask on the evaluation side against a fixed reference.

**Part 2 (redteam_curve.md S4).** p=0.024 tests the sub-3 gap, not the full->sub-3 change,
and the 47.7% attenuation the writeup leans on has no interval at all. S1 recorded that
cross-radius bootstrap draws are unsynchronised because row counts differ per regime, so
no paired test of the decay was possible. Resolved by drawing rows **once, from the full
row set**, and letting each regime's deterministic mask select from the drawn rows: the
filter is a function of the row, so a replicate that resamples rows induces every regime's
subset simultaneously. Both regimes then share one draw and the difference is paired.
The cost is stated rather than hidden -- a regime's replicate row count becomes binomial
around its fixed count, so the marginal per-regime intervals here are slightly wider than
the saved sweep's, and both are reported side by side.

Scores are computed once and re-evaluated under each mask; nothing is refit per regime.
Writes `data/cdhit_and_attenuation.json` and `data/cdhit_regime_per_peptide.csv`. Touches
no frozen artifact -- in particular `data/dedup_sensitivity.json` is read, never written.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dedup_sensitivity import (  # noqa: E402
    CACHE_DIR,
    EVAL_CSV,
    LAYER,
    MODEL_KEY,
    NO_SAME_LENGTH_MATCH,
    SEED,
    VDJDB_CACHE,
    fit_per_peptide_svc,
    nearest_substitutions,
)

from cognate.baseline_knn import (  # noqa: E402
    cosine_similarity,
    score_by_nearest_positive,
)
from cognate.data import load_train  # noqa: E402
from cognate.embed import cache_path, load_cache  # noqa: E402
from cognate.metrics import (  # noqa: E402
    MAX_FPR,
    auc01,
    compare_macro_auc01,
    evaluate,
    macro_by_group,
)

ROOT = Path(__file__).resolve().parents[1]
SAVED_SWEEP = ROOT / "data" / "dedup_sensitivity.json"
OUT_JSON = ROOT / "data" / "cdhit_and_attenuation.json"
OUT_PER_PEPTIDE = ROOT / "data" / "cdhit_regime_per_peptide.csv"

CDHIT_IDENTITY = 0.95
CDHIT_WORD_LENGTH = 5
CDHIT_MIN_LENGTH = 4
CONTROL_SUBSETS = 2000
CONTROL_SEED = 20260909
ATTENUATION_DRAWS = 20000
ATTENUATION_SEED = 20260909
CONFIDENCE = 0.95
ATTENUATION_REGIMES = ("sub_1", "sub_2", "sub_3")


def find_cdhit_2d() -> str:
    """`cd-hit-2d` from COGNATE_CDHIT_2D or PATH. Absent is fatal, not silently proxied."""
    binary = os.environ.get("COGNATE_CDHIT_2D") or shutil.which("cd-hit-2d")
    if not binary or not Path(binary).exists():
        raise FileNotFoundError(
            "cd-hit-2d not found. Set COGNATE_CDHIT_2D to the binary, or build it: "
            "git clone https://github.com/weizhongli/cdhit && make -C cdhit openmp=no"
        )
    return str(binary)


def cdhit_survivors(
    binary: str, reference: list[str], queries: list[str], *, length_asymmetry: bool
) -> tuple[set[str], set[str], str]:
    """Query sequences no reference sequence clusters with at >95% global identity.

    `length_asymmetry` False relaxes cd-hit-2d's `-s2 1.0` default, which otherwise only
    absorbs a query into a reference sequence at least as long. Returns survivors, the
    sequences cd-hit discarded as too short to cluster at all, and the version banner.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        reference_fasta, query_fasta, out = (
            work / "reference.fa", work / "query.fa", work / "kept.fa",
        )
        reference_fasta.write_text(
            "".join(f">r{i}\n{s}\n" for i, s in enumerate(reference))
        )
        query_fasta.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(queries)))
        command = [
            binary, "-i", str(reference_fasta), "-i2", str(query_fasta), "-o", str(out),
            "-c", str(CDHIT_IDENTITY), "-n", str(CDHIT_WORD_LENGTH),
            "-l", str(CDHIT_MIN_LENGTH), "-d", "0", "-M", "0",
        ]
        if not length_asymmetry:
            command += ["-s2", "0.0", "-S2", "999999"]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        kept = {
            queries[int(line[2:].strip())]
            for line in out.read_text().splitlines()
            if line.startswith(">")
        }
    discarded = {s for s in queries if len(s) <= CDHIT_MIN_LENGTH}
    banner = next(
        line.strip() for line in completed.stdout.splitlines()
        if line.startswith("Program: CD-HIT")
    )
    return kept, discarded, banner


def ranking_layout(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending score order plus the start of each tied block, as sklearn's ROC grid.

    Precomputing this once per peptide turns each bootstrap replicate's partial AUC into
    a handful of numpy reductions over count vectors, which is what makes 20,000 draws
    over 48 peptides and four regimes finish.
    """
    order = np.argsort(-score, kind="mergesort")
    ordered = score[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    return order, starts


def weighted_auc01(
    counts: np.ndarray, y_ordered: np.ndarray, starts: np.ndarray
) -> np.ndarray:
    """McClish-standardised partial AUC at FPR<=0.1 for each row of `counts`.

    `counts` is (n_regimes, n_rows) sample weights over rows already in descending score
    order. Reproduces `sklearn.roc_auc_score(..., max_fpr=0.1)` including its
    interpolation at the cut; verified cell by cell in `check()`.
    """
    positives = np.add.reduceat(counts * y_ordered, starts, axis=1).cumsum(axis=1)
    negatives = np.add.reduceat(counts * (1.0 - y_ordered), starts, axis=1).cumsum(axis=1)
    total_positive, total_negative = positives[:, -1], negatives[:, -1]

    out = np.full(len(counts), np.nan)
    for row in range(len(counts)):
        if total_positive[row] <= 0 or total_negative[row] <= 0:
            continue
        false_rate = np.r_[0.0, negatives[row] / total_negative[row]]
        true_rate = np.r_[0.0, positives[row] / total_positive[row]]
        stop = int(np.searchsorted(false_rate, MAX_FPR, "right"))
        if stop < len(false_rate):
            cut = np.interp(
                MAX_FPR, false_rate[stop - 1: stop + 1], true_rate[stop - 1: stop + 1]
            )
            false_rate = np.r_[false_rate[:stop], MAX_FPR]
            true_rate = np.r_[true_rate[:stop], cut]
        partial = float(np.trapezoid(true_rate, false_rate))
        minimum = 0.5 * MAX_FPR**2
        out[row] = 0.5 * (1.0 + (partial - minimum) / (MAX_FPR - minimum))
    return out


class PeptideBlock:
    """Everything about one peptide that a bootstrap replicate needs, precomputed."""

    def __init__(self, rows: np.ndarray, y: np.ndarray, arms: dict, masks: dict):
        self.rows = rows
        self.y = y[rows].astype(float)
        self.keep = np.stack([masks[name][rows].astype(float) for name in masks])
        self.layout = {}
        for arm, vector in arms.items():
            order, starts = ranking_layout(vector[rows])
            self.layout[arm] = (order, self.y[order], starts)

    def macro_inputs(self, counts: np.ndarray, arm: str) -> np.ndarray:
        order, y_ordered, starts = self.layout[arm]
        return weighted_auc01(counts[:, order], y_ordered, starts)


def percentile_interval(samples: np.ndarray, point: float) -> dict:
    finite = samples[np.isfinite(samples)]
    half = (1.0 - CONFIDENCE) / 2.0
    return {
        "point": float(point),
        "lo": float(np.quantile(finite, half)),
        "hi": float(np.quantile(finite, 1.0 - half)),
        "n_replicates": int(len(finite)),
    }


def two_sided_p(samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    below = int((finite <= 0).sum())
    above = int((finite >= 0).sum())
    return float(min(1.0, 2.0 * min(below, above) / len(finite)))


def sweep_row(y, peptides, sequences, arms, mask, keep_all) -> dict:
    """One row of the sweep table, computed exactly as run_dedup_sensitivity.py does."""
    scores_block = {}
    for arm, vector in arms.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = evaluate(y, vector, peptides, subset=mask, label="", seed=SEED)
        scores_block[arm] = {
            "point": round(report.macro_auc01.point, 4),
            "lo": round(report.macro_auc01.lo, 4),
            "hi": round(report.macro_auc01.hi, 4),
        }
        scores_block[f"{arm}_peptides"] = int(report.n_groups_scored)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effect = compare_macro_auc01(
            y, peptides,
            ("esm_retrieval", arms["esm_retrieval"], mask),
            ("edit_retrieval", arms["edit_retrieval"], mask), seed=SEED,
        )
    support = np.array([
        int(y[((peptides == p).to_numpy()) & mask].sum())
        for p in sorted(set(peptides[keep_all]))
    ])
    removed_tcr = pd.DataFrame(
        {"t": sequences[keep_all], "gone": ~mask[keep_all]}
    ).groupby("t")["gone"].max()
    return {
        "rows_kept": int(mask.sum()),
        "rows_removed_pct": round(100 * (1 - mask.sum() / keep_all.sum()), 2),
        "distinct_tcrs_removed_pct": round(100 * float(removed_tcr.mean()), 2),
        "support_min": int(support.min()),
        "support_median": float(np.median(support)),
        "support_max": int(support.max()),
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


def support_matched_control(blocks, target_counts, rng) -> dict:
    """S1's control, re-run for one regime: does losing this much support alone move the gap?

    Per peptide, draw the regime's exact positive and negative counts without replacement
    from that peptide's full rows, identically for both arms, and score the macro gap.
    """
    edit_means, esm_means = [], []
    for _ in range(CONTROL_SUBSETS):
        edit_values, esm_values = [], []
        for peptide, block in blocks.items():
            n_positive, n_negative = target_counts[peptide]
            positive_rows = np.flatnonzero(block.y == 1.0)
            negative_rows = np.flatnonzero(block.y == 0.0)
            chosen = np.concatenate([
                rng.choice(positive_rows, size=n_positive, replace=False),
                rng.choice(negative_rows, size=n_negative, replace=False),
            ])
            counts = np.zeros((1, len(block.y)))
            counts[0, chosen] = 1.0
            if n_positive == 0 or n_negative == 0:
                continue
            edit_values.append(block.macro_inputs(counts, "edit_retrieval")[0])
            esm_values.append(block.macro_inputs(counts, "esm_retrieval")[0])
        edit_means.append(float(np.mean(edit_values)))
        esm_means.append(float(np.mean(esm_values)))
    edit_means, esm_means = np.array(edit_means), np.array(esm_means)
    differences = esm_means - edit_means
    return {
        "n_subsets": CONTROL_SUBSETS,
        "seed": CONTROL_SEED,
        "random_edit": round(float(edit_means.mean()), 5),
        "random_esm": round(float(esm_means.mean()), 5),
        "random_mean_difference": round(float(differences.mean()), 5),
        "subset_range_lo": round(float(np.quantile(differences, 0.025)), 5),
        "subset_range_hi": round(float(np.quantile(differences, 0.975)), 5),
        "monte_carlo_se": round(float(differences.std(ddof=1) / np.sqrt(len(differences))), 6),
    }


def attenuation_bootstrap(blocks, regime_names, rng) -> dict:
    """Paired interval for the decay itself, one row draw shared by every regime.

    Row positions are drawn once from each peptide's full row set; a regime keeps the drawn
    rows its mask admits. Both arms and all regimes therefore see one draw, so
    `delta_full - delta_regime` is a paired quantity per replicate.
    """
    peptide_names = np.array(sorted(blocks))
    n_regimes = len(regime_names)
    samples = np.full((ATTENUATION_DRAWS, n_regimes), np.nan)
    skipped = np.zeros(n_regimes)
    for draw in range(ATTENUATION_DRAWS):
        drawn = rng.choice(peptide_names, size=len(peptide_names), replace=True)
        totals = np.zeros((n_regimes, 2))
        scored = np.zeros(n_regimes)
        for peptide in drawn:
            block = blocks[peptide]
            positions = rng.integers(0, len(block.y), len(block.y))
            counts = np.bincount(positions, minlength=len(block.y)).astype(float)
            per_regime = block.keep * counts
            edit_values = block.macro_inputs(per_regime, "edit_retrieval")
            esm_values = block.macro_inputs(per_regime, "esm_retrieval")
            usable = np.isfinite(edit_values) & np.isfinite(esm_values)
            totals[usable, 0] += edit_values[usable]
            totals[usable, 1] += esm_values[usable]
            scored += usable
            skipped += ~usable
        with np.errstate(invalid="ignore", divide="ignore"):
            macro = totals / scored[:, None]
        samples[draw] = macro[:, 1] - macro[:, 0]
        if draw and draw % 2000 == 0:
            print(f"  {draw:,}/{ATTENUATION_DRAWS:,} draws", flush=True)
    return {
        "regime_names": list(regime_names),
        "delta_samples": samples,
        "peptide_skip_rate": (skipped / (ATTENUATION_DRAWS * len(peptide_names))).tolist(),
    }


def check(blocks, arms, y, peptides, masks) -> None:
    """The count-based partial AUC must equal sklearn on every real cell, or stop.

    The bootstrap below never calls sklearn -- 7.7M partial AUCs would not finish -- so
    the substitute is worth nothing unless it is shown equal on the data it replaces.
    """
    worst = 0.0
    for name, mask in masks.items():
        index = list(masks).index(name)
        for peptide, block in blocks.items():
            counts = np.zeros((len(masks), len(block.y)))
            counts[index] = block.keep[index]
            rows = block.rows[block.keep[index] > 0]
            if len(rows) == 0 or not (0 < y[rows].sum() < len(rows)):
                continue
            for arm, vector in arms.items():
                mine = block.macro_inputs(counts, arm)[index]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    theirs = auc01(y[rows], vector[rows])
                worst = max(worst, abs(mine - theirs))
    if worst > 1e-9:
        raise AssertionError(f"count-based AUC0.1 disagrees with sklearn by {worst:.3g}")
    print(f"count-based AUC0.1 matches sklearn on every cell (max |diff| {worst:.3g})")


def main() -> None:
    binary = find_cdhit_2d()
    positives = load_train()
    evalset = pd.read_csv(EVAL_CSV)
    y = evalset["Label"].to_numpy()
    peptides = evalset["Peptide"]
    sequences = evalset["CDR3b"].astype(str).to_numpy()
    is_seen = peptides.isin(set(positives["Peptide"])).to_numpy()

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
    saved = json.loads(SAVED_SWEEP.read_text())["regimes"]
    for arm, block in saved["full"]["scores"].items():
        if arm.endswith("_peptides"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            here = macro_by_group(y[is_seen], arms[arm][is_seen],
                                  peptides[is_seen].to_numpy(), auc01).value
        if abs(here - block["point"]) > 5e-5:
            raise AssertionError(f"{arm} regenerates {here:.4f}, sweep saved {block['point']}")
    print("all three arms reproduce the saved full-regime scores\n")

    reference = sorted(set(positives["CDR3b"].astype(str)))
    queries = sorted(set(sequences))
    cdhit_masks = {}
    provenance = {}
    for label, asymmetry in (("cdhit_95", True), ("cdhit_95_relaxed_length", False)):
        kept, discarded, banner = cdhit_survivors(
            binary, reference, queries, length_asymmetry=asymmetry
        )
        survivors = kept | discarded
        cdhit_masks[label] = np.array([s in survivors for s in sequences])
        provenance[label] = {
            "version": banner,
            "identity_threshold": CDHIT_IDENTITY,
            "word_length": CDHIT_WORD_LENGTH,
            "throw_away_length": CDHIT_MIN_LENGTH,
            "length_asymmetry_s2_default": asymmetry,
            "query_sequences": len(queries),
            "reference_sequences": len(reference),
            "clustered_away": len(queries) - len(kept) - len(discarded),
            "too_short_to_cluster": sorted(discarded),
        }
        print(f"{label}: {len(kept):,}/{len(queries):,} unique eval CDR3b survive; "
              f"{len(discarded)} below cd-hit's length floor and retained by hand")

    substitutions = nearest_substitutions(evalset, positives)
    radius_masks = {f"sub_{r}": substitutions > r for r in (1, 2, 3)}
    masks = {"full": np.ones(len(evalset), bool), **radius_masks, **cdhit_masks}

    regimes = {}
    for name in ("full", *cdhit_masks):
        regimes[name] = sweep_row(
            y, peptides, sequences, arms, is_seen & masks[name], is_seen
        )
        row, effect = regimes[name], regimes[name]["representation_effect"]
        print(f"{name:<24} removed {row['rows_removed_pct']:>5.2f}% rows / "
              f"{row['distinct_tcrs_removed_pct']:>5.2f}% TCRs   support "
              f"{row['support_min']}/{row['support_median']:.0f}/{row['support_max']}   "
              f"edit {row['scores']['edit_retrieval']['point']:.4f}  "
              f"esm {row['scores']['esm_retrieval']['point']:.4f}  "
              f"svc {row['scores']['svc_per_peptide']['point']:.4f}   "
              f"{effect['point']:+.4f} [{effect['lo']:+.4f},{effect['hi']:+.4f}] "
              f"p={effect['p']:.4f}")

    seen_peptides = sorted(set(peptides[is_seen]))
    ordered_masks = {name: masks[name] for name in ("full", "sub_1", "sub_2", "sub_3",
                                                    *cdhit_masks)}
    blocks = {
        peptide: PeptideBlock(
            np.flatnonzero(is_seen & (peptides == peptide).to_numpy()), y, arms,
            ordered_masks,
        )
        for peptide in seen_peptides
    }
    check(blocks, {k: arms[k] for k in ("edit_retrieval", "esm_retrieval")},
          y, peptides, ordered_masks)

    controls = {}
    for name in cdhit_masks:
        target = {
            peptide: (
                int(y[blocks[peptide].rows][masks[name][blocks[peptide].rows]].sum()),
                int((~y[blocks[peptide].rows].astype(bool))[
                    masks[name][blocks[peptide].rows]].sum()),
            )
            for peptide in seen_peptides
        }
        controls[name] = support_matched_control(
            blocks, target, np.random.default_rng(CONTROL_SEED)
        )
        controls[name]["observed_difference"] = regimes[name]["representation_effect"]["point"]
        control = controls[name]
        print(f"{name} support control: random edit {control['random_edit']:.5f}  "
              f"random esm {control['random_esm']:.5f}  mean {control['random_mean_difference']:+.5f} "
              f"[{control['subset_range_lo']:+.5f},{control['subset_range_hi']:+.5f}]  "
              f"observed {control['observed_difference']:+.5f}")

    print(f"\npaired attenuation bootstrap, {ATTENUATION_DRAWS:,} synchronised draws")
    attenuation_names = ("full", *ATTENUATION_REGIMES)
    slim = {p: PeptideBlock(blocks[p].rows, y, arms,
                            {k: masks[k] for k in attenuation_names})
            for p in seen_peptides}
    raw = attenuation_bootstrap(
        slim, attenuation_names, np.random.default_rng(ATTENUATION_SEED)
    )
    samples = raw["delta_samples"]
    point = {}
    for index, name in enumerate(attenuation_names):
        mask = is_seen & masks[name]
        values = []
        for peptide in seen_peptides:
            rows = np.flatnonzero(mask & (peptides == peptide).to_numpy())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values.append(
                    auc01(y[rows], arms["esm_retrieval"][rows])
                    - auc01(y[rows], arms["edit_retrieval"][rows])
                )
        point[name] = float(np.mean(values))

    attenuation = {"n_draws": ATTENUATION_DRAWS, "seed": ATTENUATION_SEED,
                   "peptide_skip_rate": dict(zip(attenuation_names,
                                                 raw["peptide_skip_rate"])),
                   "regimes": {}}
    full_samples = samples[:, 0]
    attenuation["delta_full"] = percentile_interval(full_samples, point["full"])
    attenuation["delta_full"]["p"] = two_sided_p(full_samples)
    for index, name in enumerate(attenuation_names[1:], start=1):
        regime_samples = samples[:, index]
        absolute = full_samples - regime_samples
        proportion = 1.0 - regime_samples / full_samples
        attenuation["regimes"][name] = {
            "delta": {**percentile_interval(regime_samples, point[name]),
                      "p": two_sided_p(regime_samples)},
            "attenuation_absolute": percentile_interval(
                absolute, point["full"] - point[name]),
            "attenuation_proportion": percentile_interval(
                proportion, 1.0 - point[name] / point["full"]),
            "attenuation_p": two_sided_p(absolute),
            "replicates_with_delta_full_above_zero": int((full_samples >= 0).sum()),
        }
        block = attenuation["regimes"][name]
        print(f"  full->{name}: delta {point[name]:+.5f}  "
              f"absolute {block['attenuation_absolute']['point']:+.5f} "
              f"[{block['attenuation_absolute']['lo']:+.5f},"
              f"{block['attenuation_absolute']['hi']:+.5f}]  "
              f"proportion {100 * block['attenuation_proportion']['point']:.1f}% "
              f"[{100 * block['attenuation_proportion']['lo']:.1f}%,"
              f"{100 * block['attenuation_proportion']['hi']:.1f}%]  "
              f"p={block['attenuation_p']:.4f}")

    rows_out = []
    for peptide in seen_peptides:
        base = (peptides == peptide).to_numpy()
        entry = {"peptide": peptide}
        for name in ("full", *cdhit_masks):
            rows = base & masks[name] & is_seen
            entry[f"{name}_positives"] = int(y[rows].sum())
            entry[f"{name}_negatives"] = int(rows.sum() - y[rows].sum())
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
            "no_same_length_match_sentinel": NO_SAME_LENGTH_MATCH,
            "cdhit": provenance,
            "cdhit_application": (
                "cd-hit-2d with the fixed IMMREP23 load_train() CDR3b set as db1 and the "
                "distinct evaluation CDR3b as db2; an evaluation sequence is removed when "
                "cd-hit clusters it with any training sequence at >=95% global identity, "
                "which cd-hit defines as identical residues over the length of the shorter "
                "sequence. Peptide-agnostic, matching the substitution-radius regimes."
            ),
        },
        "regimes": regimes,
        "support_matched_control": controls,
        "attenuation": attenuation,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name} and {OUT_PER_PEPTIDE.name}")


if __name__ == "__main__":
    main()
