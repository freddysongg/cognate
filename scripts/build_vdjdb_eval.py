"""Build a wider TCR-epitope evaluation set from VDJdb.

Phase B1. The IMMREP23 test set has 20 peptides (13 seen in training, 7 not), which is the
binding constraint on every correlation in `docs/findings.md`. This script constructs a
replacement evaluation set from VDJdb with 88 peptides, using the same negative-sampling
construction the IMMREP23 organisers used so the two are comparable.

Source: VDJdb release 2026-06-03, `vdjdb.slim.txt`, AGPL-3.0.
Output: `data/vdjdb_eval.csv`, `data/vdjdb_eval_peptides.csv`.

Deterministic: seeded throughout, no network access, no dependence on `data/embeddings`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from cognate import data
from cognate.data import to_immrep_cdr3

REPO_ROOT = Path(__file__).resolve().parents[1]
VDJDB_SLIM = REPO_ROOT / "data" / "vdjdb-2026-06-03" / "vdjdb.slim.txt"
OUT_PAIRS = REPO_ROOT / "data" / "vdjdb_eval.csv"
OUT_PEPTIDES = REPO_ROOT / "data" / "vdjdb_eval_peptides.csv"
OUT_STATS = REPO_ROOT / "data" / "vdjdb_eval_stats.json"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MIN_TCRS_PER_PEPTIDE = 50
MAX_POSITIVES_PER_PEPTIDE = 500
NEGATIVES_PER_POSITIVE = 5
MIN_PEPTIDE_EDIT_DISTANCE = 3
SEED = 0


def load_vdjdb() -> pd.DataFrame:
    raw = pd.read_csv(VDJDB_SLIM, sep="\t", low_memory=False)
    human_trb = raw[
        (raw["species"] == "HomoSapiens")
        & (raw["gene"] == "TRB")
        & (raw["mhc.class"] == "MHCI")
    ].dropna(subset=["cdr3", "antigen.epitope"])

    frame = pd.DataFrame(
        {
            "peptide": human_trb["antigen.epitope"].str.strip().str.upper(),
            "cdr3b": human_trb["cdr3"].map(to_immrep_cdr3),
            "hla": human_trb["mhc.a"].fillna("unknown"),
            "vdjdb_score": human_trb["vdjdb.score"].astype(int),
            "antigen_species": human_trb["antigen.species"].fillna("unknown"),
        }
    )
    valid = frame["cdr3b"].str.fullmatch(f"[{AMINO_ACIDS}]{{4,30}}") & frame[
        "peptide"
    ].str.fullmatch(f"[{AMINO_ACIDS}]{{7,15}}")
    frame = frame[valid]
    return frame.sort_values(["peptide", "cdr3b", "vdjdb_score"]).drop_duplicates(
        ["peptide", "cdr3b"], keep="last"
    )


def build(min_tcrs: int, cap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    vdjdb = load_vdjdb()
    all_cognates = vdjdb.groupby("cdr3b")["peptide"].apply(set).to_dict()

    train = data.load_train()
    train_pairs = set(zip(train["Peptide"].str.upper(), train["CDR3b"].str.upper()))
    train_tcrs = set(train["CDR3b"].str.upper())
    train_peptides = set(train["Peptide"].str.upper())

    stats: dict[str, object] = {
        "vdjdb_release": "2026-06-03",
        "vdjdb_pairs_human_trb_mhci": int(len(vdjdb)),
        "vdjdb_peptides": int(vdjdb["peptide"].nunique()),
        "immrep23_train_pairs_found_verbatim_in_vdjdb": int(
            len(set(zip(vdjdb["peptide"], vdjdb["cdr3b"])) & train_pairs)
        ),
        "immrep23_train_tcrs_found_in_vdjdb": int(len(set(vdjdb["cdr3b"]) & train_tcrs)),
        "immrep23_train_pairs_total": int(len(train_pairs)),
        "immrep23_train_tcrs_total": int(len(train_tcrs)),
    }

    disjoint = vdjdb[~vdjdb["cdr3b"].isin(train_tcrs)]
    stats["pairs_after_removing_training_tcrs"] = int(len(disjoint))

    support = disjoint.groupby("peptide").size()
    kept_peptides = support[support >= min_tcrs].index
    positives = disjoint[disjoint["peptide"].isin(kept_peptides)].copy()
    positives["n_tcr_peptide"] = positives["peptide"].map(support)

    rng = np.random.default_rng(seed)
    positives = positives.sort_values(["peptide", "cdr3b"])
    capped = []
    for _, group in positives.groupby("peptide", sort=True):
        if len(group) > cap:
            group = group.iloc[np.sort(rng.choice(len(group), cap, replace=False))]
        capped.append(group)
    positives = pd.concat(capped, ignore_index=True)
    positives["Label"] = 1

    peptide_pool = sorted(positives["peptide"].unique())
    tcr_pool = positives["cdr3b"].unique()
    distance_ok = {
        p: {
            c
            for c in tcr_pool
            if all(
                Levenshtein.distance(p, cognate) > MIN_PEPTIDE_EDIT_DISTANCE
                for cognate in all_cognates.get(c, ())
            )
        }
        for p in peptide_pool
    }

    negative_rows = []
    for peptide, group in positives.groupby("peptide"):
        eligible = np.array(sorted(distance_ok[peptide]))
        wanted = len(group) * NEGATIVES_PER_POSITIVE
        if len(eligible) < wanted:
            raise RuntimeError(
                f"{peptide}: {len(eligible)} eligible negatives < {wanted} required"
            )
        chosen = eligible[np.sort(rng.choice(len(eligible), wanted, replace=False))]
        negative_rows.append(pd.DataFrame({"peptide": peptide, "cdr3b": chosen}))

    negatives = pd.concat(negative_rows, ignore_index=True)
    negatives["Label"] = 0
    negatives["hla"] = negatives["peptide"].map(
        positives.groupby("peptide")["hla"].agg(lambda s: s.mode().iat[0])
    )
    negatives["vdjdb_score"] = -1
    negatives["antigen_species"] = negatives["peptide"].map(
        positives.groupby("peptide")["antigen_species"].agg(lambda s: s.mode().iat[0])
    )
    negatives["n_tcr_peptide"] = negatives["peptide"].map(
        positives.groupby("peptide")["n_tcr_peptide"].first()
    )

    columns = ["peptide", "cdr3b", "Label", "hla", "vdjdb_score", "antigen_species", "n_tcr_peptide"]
    pairs = pd.concat([positives[columns], negatives[columns]], ignore_index=True)
    pairs = pairs.rename(columns={"peptide": "Peptide", "cdr3b": "CDR3b"})
    pairs["seen_in_immrep23_train"] = pairs["Peptide"].isin(train_peptides)
    pairs = pairs.sort_values(["Peptide", "Label", "CDR3b"], ascending=[True, False, True])
    pairs = pairs.reset_index(drop=True)

    per_peptide = (
        pairs.groupby("Peptide")
        .agg(
            n_positive=("Label", "sum"),
            n_rows=("Label", "size"),
            n_tcr_before_cap=("n_tcr_peptide", "first"),
            seen_in_immrep23_train=("seen_in_immrep23_train", "first"),
            hla=("hla", lambda s: s.mode().iat[0]),
            antigen_species=("antigen_species", lambda s: s.mode().iat[0]),
            n_high_confidence=("vdjdb_score", lambda s: int((s >= 2).sum())),
        )
        .reset_index()
        .sort_values("n_positive", ascending=False)
    )

    stats.update(
        {
            "min_tcrs_per_peptide": min_tcrs,
            "max_positives_per_peptide": cap,
            "negatives_per_positive": NEGATIVES_PER_POSITIVE,
            "seed": seed,
            "n_peptides": int(len(per_peptide)),
            "n_peptides_seen_in_immrep23_train": int(
                per_peptide["seen_in_immrep23_train"].sum()
            ),
            "n_positives": int((pairs["Label"] == 1).sum()),
            "n_negatives": int((pairs["Label"] == 0).sum()),
            "n_rows": int(len(pairs)),
            "positives_per_peptide_min": int(per_peptide["n_positive"].min()),
            "positives_per_peptide_median": int(per_peptide["n_positive"].median()),
            "positives_per_peptide_max": int(per_peptide["n_positive"].max()),
            "n_peptides_with_50plus_high_confidence": int(
                (per_peptide["n_high_confidence"] >= 50).sum()
            ),
            "n_peptides_with_10plus_high_confidence": int(
                (per_peptide["n_high_confidence"] >= 10).sum()
            ),
        }
    )
    return pairs, per_peptide, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-tcrs", type=int, default=MIN_TCRS_PER_PEPTIDE)
    parser.add_argument("--cap", type=int, default=MAX_POSITIVES_PER_PEPTIDE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    pairs, per_peptide, stats = build(args.min_tcrs, args.cap, args.seed)
    pairs.to_csv(OUT_PAIRS, index=False)
    per_peptide.to_csv(OUT_PEPTIDES, index=False)
    OUT_STATS.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
