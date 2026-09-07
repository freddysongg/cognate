"""Regression tests for the Phase B evaluation set (docs/eval_set_construction.md).

The leakage checks are the point of this file. IMMREP23's training set is essentially a subset of
VDJdb -- 82.5% of its pairs appear there verbatim -- so `data/vdjdb_eval.csv` is only a valid
evaluation set for models trained on IMMREP23 because the overlap was explicitly removed. A change
to `scripts/build_vdjdb_eval.py` that reintroduced any of it would not fail loudly; the scores
would simply get better. These assertions are what make that fail loudly instead.

Every check that can be satisfied vacuously is paired with a positive control asserting the same
check fires on deliberately corrupted input, so a green run means the check ran and could have
gone red.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import roc_auc_score

from cognate.data import load_train, to_immrep_cdr3

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = REPO_ROOT / "data" / "vdjdb_eval.csv"
PEPTIDES_CSV = REPO_ROOT / "data" / "vdjdb_eval_peptides.csv"
STATS_JSON = REPO_ROOT / "data" / "vdjdb_eval_stats.json"
VDJDB_SLIM = REPO_ROOT / "data" / "vdjdb-2026-06-03" / "vdjdb.slim.txt"

NEGATIVES_PER_POSITIVE = 5
MIN_PEPTIDE_EDIT_DISTANCE = 3
EXPECTED_POSITIVE_RATE = 1 / (1 + NEGATIVES_PER_POSITIVE)

pytestmark = pytest.mark.skipif(
    not EVAL_CSV.exists(), reason="run scripts/build_vdjdb_eval.py first"
)


@pytest.fixture(scope="module")
def evalset():
    return pd.read_csv(EVAL_CSV)


@pytest.fixture(scope="module")
def per_peptide():
    return pd.read_csv(PEPTIDES_CSV)


@pytest.fixture(scope="module")
def stats():
    return json.loads(STATS_JSON.read_text())


@pytest.fixture(scope="module")
def training_pairs():
    train = load_train()
    return set(zip(train["Peptide"].str.upper(), train["CDR3b"].str.upper()))


@pytest.fixture(scope="module")
def training_tcrs():
    return set(load_train()["CDR3b"].str.upper())


@pytest.fixture(scope="module")
def vdjdb_cognates():
    """Every peptide each human TRB CDR3b is recorded as binding, in IMMREP23 string convention."""
    raw = pd.read_csv(VDJDB_SLIM, sep="\t", low_memory=False)
    human_trb = raw[(raw["species"] == "HomoSapiens") & (raw["gene"] == "TRB")].dropna(
        subset=["cdr3", "antigen.epitope"]
    )
    frame = pd.DataFrame(
        {
            "peptide": human_trb["antigen.epitope"].str.strip().str.upper(),
            "cdr3b": human_trb["cdr3"].map(to_immrep_cdr3),
        }
    )
    return frame.groupby("cdr3b")["peptide"].apply(set).to_dict()


def shared_pairs(frame: pd.DataFrame, reference: set[tuple[str, str]]) -> int:
    return len(set(zip(frame["Peptide"], frame["CDR3b"])) & reference)


def known_binder_pairs(
    frame: pd.DataFrame, cognates: dict[str, set[str]]
) -> list[tuple[str, str]]:
    return [
        (peptide, cdr3b)
        for peptide, cdr3b in zip(frame["Peptide"], frame["CDR3b"])
        if peptide in cognates.get(cdr3b, ())
    ]


def edit_distance_violations(
    frame: pd.DataFrame, cognates: dict[str, set[str]]
) -> list[tuple[str, str]]:
    return [
        (peptide, cdr3b)
        for peptide, cdr3b in zip(frame["Peptide"], frame["CDR3b"])
        if any(
            Levenshtein.distance(peptide, cognate) <= MIN_PEPTIDE_EDIT_DISTANCE
            for cognate in cognates.get(cdr3b, ())
        )
    ]


def a_tcr_with_its_own_cognate(cognates: dict[str, set[str]]) -> pd.DataFrame:
    cdr3b = next(c for c, peptides in sorted(cognates.items()) if peptides)
    return pd.DataFrame([{"Peptide": sorted(cognates[cdr3b])[0], "CDR3b": cdr3b}])


# --- the units bug that nearly produced a leak-free-looking leaky set -------------------------


def test_to_immrep_cdr3_strips_imgt_flanks() -> None:
    assert to_immrep_cdr3("CASSFSGNTGELFF") == "ASSFSGNTGELF"
    assert to_immrep_cdr3("CASSIRSSYEQYW") == "ASSIRSSYEQY"
    assert to_immrep_cdr3("  cassirssyeqyf  ") == "ASSIRSSYEQY"


def test_to_immrep_cdr3_leaves_already_stripped_cores_that_lack_flanks() -> None:
    assert to_immrep_cdr3("ASSARSSYEQY") == "ASSARSSYEQY"


def test_raw_vdjdb_junctions_look_disjoint_from_training_without_normalisation(
    training_tcrs, vdjdb_cognates
) -> None:
    """The failure mode: unnormalised, the two sources share nothing and look cleanly separated."""
    raw = pd.read_csv(VDJDB_SLIM, sep="\t", low_memory=False)
    junctions = set(raw[raw["gene"] == "TRB"]["cdr3"].dropna().str.strip().str.upper())
    assert len(junctions & training_tcrs) == 0
    assert len(set(vdjdb_cognates) & training_tcrs) > 7_000


def test_normalisation_recovers_the_real_training_overlap(training_pairs, training_tcrs) -> None:
    raw = pd.read_csv(VDJDB_SLIM, sep="\t", low_memory=False)
    human_trb = raw[
        (raw["species"] == "HomoSapiens")
        & (raw["gene"] == "TRB")
        & (raw["mhc.class"] == "MHCI")
    ].dropna(subset=["cdr3", "antigen.epitope"])
    pairs = set(
        zip(
            human_trb["antigen.epitope"].str.strip().str.upper(),
            human_trb["cdr3"].map(to_immrep_cdr3),
        )
    )
    assert len(pairs & training_pairs) == 7_680
    assert len(pairs & training_pairs) / len(training_pairs) == pytest.approx(0.825, abs=5e-4)


# --- leakage ----------------------------------------------------------------------------------


def test_no_eval_pair_appears_in_immrep23_training(evalset, training_pairs) -> None:
    assert shared_pairs(evalset, training_pairs) == 0


def test_pair_leakage_check_fires_on_a_planted_training_pair(evalset, training_pairs) -> None:
    planted_peptide, planted_cdr3b = sorted(training_pairs)[0]
    corrupted = pd.concat(
        [evalset, pd.DataFrame([{"Peptide": planted_peptide, "CDR3b": planted_cdr3b}])],
        ignore_index=True,
    )
    assert shared_pairs(corrupted, training_pairs) == 1


def test_no_eval_tcr_appears_in_immrep23_training(evalset, training_tcrs) -> None:
    assert len(set(evalset["CDR3b"]) & training_tcrs) == 0


def test_tcr_leakage_check_fires_on_a_planted_training_tcr(evalset, training_tcrs) -> None:
    planted = sorted(training_tcrs)[0]
    corrupted = pd.concat(
        [evalset, pd.DataFrame([{"Peptide": "AAAAAAAAA", "CDR3b": planted}])],
        ignore_index=True,
    )
    assert len(set(corrupted["CDR3b"]) & training_tcrs) == 1


# --- negatives are actually negative ----------------------------------------------------------


def test_negatives_are_not_known_vdjdb_binders(evalset, vdjdb_cognates) -> None:
    assert known_binder_pairs(evalset[evalset["Label"] == 0], vdjdb_cognates) == []


def test_known_binder_check_fires_on_a_planted_cognate_pair(evalset, vdjdb_cognates) -> None:
    corrupted = pd.concat(
        [evalset[evalset["Label"] == 0], a_tcr_with_its_own_cognate(vdjdb_cognates)],
        ignore_index=True,
    )
    assert len(known_binder_pairs(corrupted, vdjdb_cognates)) == 1


def test_negatives_respect_the_edit_distance_rule(evalset, vdjdb_cognates) -> None:
    assert edit_distance_violations(evalset[evalset["Label"] == 0], vdjdb_cognates) == []


def test_edit_distance_check_fires_on_a_planted_cognate_pair(evalset, vdjdb_cognates) -> None:
    corrupted = pd.concat(
        [evalset[evalset["Label"] == 0], a_tcr_with_its_own_cognate(vdjdb_cognates)],
        ignore_index=True,
    )
    assert len(edit_distance_violations(corrupted, vdjdb_cognates)) == 1


# --- the Session 5 shortcut cannot exist here -------------------------------------------------


def test_positive_rate_is_identical_for_every_peptide(evalset) -> None:
    rates = evalset.groupby("Peptide")["Label"].mean()
    assert rates.min() == pytest.approx(EXPECTED_POSITIVE_RATE, abs=1e-12)
    assert rates.max() == pytest.approx(EXPECTED_POSITIVE_RATE, abs=1e-12)


def test_peptide_identity_alone_scores_exactly_half(evalset) -> None:
    rates = evalset.groupby("Peptide")["Label"].mean()
    assert roc_auc_score(evalset["Label"], evalset["Peptide"].map(rates)) == pytest.approx(
        0.5, abs=1e-12
    )


def test_peptide_identity_probe_detects_an_imbalanced_set() -> None:
    """The same probe on a shuffle-style marginal, which is what it exists to catch."""
    skewed = pd.DataFrame(
        {
            "Peptide": ["A"] * 90 + ["B"] * 10 + ["A"] * 10 + ["B"] * 90,
            "Label": [1] * 100 + [0] * 100,
        }
    )
    rates = skewed.groupby("Peptide")["Label"].mean()
    assert roc_auc_score(skewed["Label"], skewed["Peptide"].map(rates)) > 0.85


# --- composition pinned to docs/eval_set_construction.md --------------------------------------


def test_headline_composition(evalset, per_peptide) -> None:
    assert len(per_peptide) == 88
    assert int(per_peptide["seen_in_immrep23_train"].sum()) == 48
    assert int((~per_peptide["seen_in_immrep23_train"]).sum()) == 40
    assert int((evalset["Label"] == 1).sum()) == 19_443
    assert int((evalset["Label"] == 0).sum()) == 97_215
    assert evalset["CDR3b"].nunique() == 18_760


def test_negative_ratio_holds_per_peptide(evalset) -> None:
    counts = evalset.groupby(["Peptide", "Label"]).size().unstack()
    assert (counts[0] == counts[1] * NEGATIVES_PER_POSITIVE).all()


def test_per_peptide_support_bounds(per_peptide) -> None:
    assert per_peptide["n_positive"].min() == 52
    assert int(per_peptide["n_positive"].median()) == 159
    assert per_peptide["n_positive"].max() == 500
    assert int((per_peptide["n_positive"] == 500).sum()) == 20
    assert int((per_peptide["n_positive"] < 100).sum()) == 34


def test_confidence_filter_cannot_support_the_analysis(per_peptide) -> None:
    """docs/eval_set_construction.md S5: score>=2 collapses the set, so it is not a robustness slice."""
    assert int((per_peptide["n_high_confidence"] >= 50).sum()) == 6
    assert int((per_peptide["n_high_confidence"] >= 10).sum()) == 11
    assert int((per_peptide["n_high_confidence"] == 0).sum()) == 61


def test_hla_concentration(per_peptide) -> None:
    assert int((per_peptide["hla"] == "HLA-A*02:01").sum()) == 30
    assert per_peptide["hla"].nunique() == 19


def test_stats_json_agrees_with_the_written_artefacts(evalset, per_peptide, stats) -> None:
    assert stats["n_rows"] == len(evalset)
    assert stats["n_peptides"] == len(per_peptide)
    assert stats["n_positives"] == int((evalset["Label"] == 1).sum())
    assert stats["n_negatives"] == int((evalset["Label"] == 0).sum())
    assert stats["n_peptides_seen_in_immrep23_train"] == int(
        per_peptide["seen_in_immrep23_train"].sum()
    )
    assert stats["vdjdb_release"] == "2026-06-03"
