"""Pins the headline numbers quoted in docs/findings.md to the real data.

A reference document whose numbers silently rot is worse than no document. Every figure
asserted here appears in findings.md; if one changes, the doc is wrong and this fails.
The '201 CDR3b bind more than one peptide' case is why this file exists -- it was first
recorded as 199, transcribed from a truncated value_counts display.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rapidfuzz.distance import Levenshtein

from cognate.baseline_knn import exact_match_mask, score_by_nearest_positive
from cognate.data import load_solutions, load_train
from cognate.metrics import auc01, compare_macro_auc01, evaluate
from cognate.negatives import make_negatives

AUC01_FLOOR = 9 / 19
TOY_AUC01 = 14 / 19


@pytest.fixture(scope="module")
def train():
    return load_train()


@pytest.fixture(scope="module")
def sol():
    return load_solutions()


def test_row_counts(train, sol) -> None:
    assert len(train) == 11_312
    assert len(sol) == 3_484


def test_label_balance(sol) -> None:
    assert sol["Label"].sum() == 598
    assert sol["Label"].mean() == pytest.approx(0.1716, abs=5e-5)


def test_seen_unseen_split(train, sol) -> None:
    train_peptides, test_peptides = set(train["Peptide"]), set(sol["Peptide"])
    assert len(train_peptides) == 808
    assert len(test_peptides) == 20
    assert len(test_peptides & train_peptides) == 13
    assert len(test_peptides - train_peptides) == 7
    assert sol["Peptide"].isin(train_peptides).mean() == pytest.approx(0.694, abs=5e-4)


def test_training_file_is_positives_only(train) -> None:
    assert (train["Target"] == 1).all()


def test_training_support_spans_two_orders_of_magnitude(train, sol) -> None:
    support = train[train["Peptide"].isin(set(sol["Peptide"]))].groupby("Peptide").size()
    assert support.min() == 6
    assert support.max() == 1_818


def test_verbatim_pair_leakage(train, sol) -> None:
    pairs = set(zip(train["Peptide"], train["CDR3b"]))
    in_train = np.array(
        [(p, c) in pairs for p, c in zip(sol["Peptide"], sol["CDR3b"])]
    )
    assert int(in_train[sol["Label"] == 1].sum()) == 107
    assert int(in_train[sol["Label"] == 0].sum()) == 8


def test_cross_reactive_cdr3b_counts(train, sol) -> None:
    peptides_per_tcr = train.groupby("CDR3b")["Peptide"].nunique()
    assert train["CDR3b"].nunique() == 8_993
    assert int((peptides_per_tcr > 1).sum()) == 201
    assert int(peptides_per_tcr.max()) == 12
    assert int(sol.groupby("CDR3b")["Label"].sum().max()) == 13


def test_test_set_construction_fingerprint(sol) -> None:
    """17 of 20 peptides sit at exactly 5 negatives per positive."""
    counts = sol.groupby("Peptide")["Label"].agg(["size", "sum"])
    ratio = (counts["size"] - counts["sum"]) / counts["sum"]
    assert int((ratio.round(6) == 5.0).sum()) == 17


def test_metric_constants() -> None:
    assert auc01(np.array([0] * 9 + [1]), np.linspace(1, 0, 10)) == pytest.approx(
        AUC01_FLOOR
    )
    toy_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    toy_score = np.array([0.9, 0.5, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert auc01(toy_true, toy_score) == pytest.approx(TOY_AUC01)


def test_random_predictor_noise_floor(train, sol) -> None:
    """The unseen-slice CI width that sets the bar for a meaningful gap."""
    is_seen = sol["Peptide"].isin(set(train["Peptide"])).to_numpy()
    scores = np.random.default_rng(0).random(len(sol))
    report = evaluate(
        sol["Label"].to_numpy(), scores, sol["Peptide"], subset=~is_seen, seed=0
    )
    assert report.macro_auc01.point == pytest.approx(0.504, abs=5e-4)
    assert report.macro_auc01.width == pytest.approx(0.051, abs=5e-4)


@pytest.mark.parametrize(
    ("strategy", "pct_within_3", "median_distance", "peptides_used"),
    [("shuffle", 0.13, 8, 808), ("hard", 17.69, 5, 741)],
)
def test_negative_strategy_distance_profile(
    train, strategy: str, pct_within_3: float, median_distance: int, peptides_used: int
) -> None:
    negatives = make_negatives(train, strategy, ratio=5.0, seed=0)
    assert len(negatives) == 56_560
    assert negatives["Peptide"].nunique() == peptides_used

    true_peptide = train["Peptide"].reindex(negatives.index).to_numpy()
    distances = np.array(
        [Levenshtein.distance(a, b) for a, b in zip(true_peptide, negatives["Peptide"])]
    )
    assert 100 * (distances <= 3).mean() == pytest.approx(pct_within_3, abs=5e-3)
    assert int(np.median(distances)) == median_distance


# --- T4: nearest-neighbour baseline -----------------------------------------------


@pytest.fixture(scope="module")
def knn_setup(train, sol):
    is_seen = sol["Peptide"].isin(set(train["Peptide"])).to_numpy()
    return {
        "y_true": sol["Label"].to_numpy(),
        "peptides": sol["Peptide"],
        "is_seen": is_seen,
        "is_exact": exact_match_mask(sol, train),
        "knn": score_by_nearest_positive(sol, train),
    }


def test_knn_headline_scores(knn_setup) -> None:
    """The four T4 numbers. Acceptance: seen is meaningfully above 0.5."""
    s = knn_setup
    seen = evaluate(
        s["y_true"], s["knn"].score, s["peptides"], subset=s["is_seen"], seed=0
    )
    seen_clean = evaluate(
        s["y_true"],
        s["knn"].score,
        s["peptides"],
        subset=s["is_seen"] & ~s["is_exact"],
        seed=0,
    )
    unseen = evaluate(
        s["y_true"], s["knn"].score, s["peptides"], subset=~s["is_seen"], seed=0
    )

    assert seen.macro_auc01.point == pytest.approx(0.641, abs=5e-4)
    assert seen.macro_auc01.lo > 0.5
    assert seen_clean.macro_auc01.point == pytest.approx(0.591, abs=5e-4)
    assert unseen.macro_auc01.point == pytest.approx(0.5)
    assert unseen.macro_auc01.width == pytest.approx(0.0)


def test_unseen_is_degenerate_not_measured(knn_setup) -> None:
    """Every unseen row takes the default score, so 0.5 is arithmetic."""
    s = knn_setup
    assert (s["knn"].score[~s["is_seen"]] == 0.0).all()
    assert not s["knn"].has_database[~s["is_seen"]].any()


def test_all_exact_matches_score_one(train, sol, knn_setup) -> None:
    s = knn_setup
    assert s["is_exact"].sum() == 115
    assert (s["knn"].score[s["is_exact"]] == 1.0).all()
    assert s["is_exact"][~s["is_seen"]].sum() == 0


def test_lookup_decomposition(knn_setup) -> None:
    """Dropping leaked rows costs 0.051; denying the database the verbatim answer
    costs 0.009. The rows are easy, but the method does not depend on the match."""
    s = knn_setup
    drop_rows = compare_macro_auc01(
        s["y_true"],
        s["peptides"],
        ("seen", s["knn"].score, s["is_seen"]),
        ("seen minus exact", s["knn"].score, s["is_seen"] & ~s["is_exact"]),
        seed=0,
    )
    assert drop_rows.difference.point == pytest.approx(0.051, abs=5e-4)
    assert drop_rows.difference.lo > 0
    assert drop_rows.p_two_sided < 0.05


def test_database_saturation(train, sol, knn_setup) -> None:
    """Bigger databases raise negatives almost perfectly with size, cancelling the gain
    on positives. Training support does not predict per-peptide score."""
    s = knn_setup
    support = train.groupby("Peptide").size()
    rows = []
    for peptide in sorted(set(sol["Peptide"][s["is_seen"]])):
        mask = (sol["Peptide"] == peptide).to_numpy()
        pos = mask & (s["y_true"] == 1)
        neg = mask & (s["y_true"] == 0)
        rows.append(
            (
                np.log10(support[peptide]),
                auc01(s["y_true"][mask], s["knn"].score[mask]),
                s["knn"].score[neg].mean(),
                s["knn"].score[pos].mean() - s["knn"].score[neg].mean(),
            )
        )
    log_support, scores, mean_neg, gap = (np.array(c) for c in zip(*rows))

    assert np.corrcoef(log_support, mean_neg)[0, 1] > 0.95
    assert abs(np.corrcoef(log_support, scores)[0, 1]) < 0.25
    assert np.corrcoef(gap, scores)[0, 1] > 0.9


# --- T5a: embedding inputs and the T5b split hazard --------------------------------


def test_dedup_reduces_embedding_work(train, sol) -> None:
    peptides = set(train["Peptide"]) | set(sol["Peptide"])
    cdr3b = set(train["CDR3b"]) | set(sol["CDR3b"])
    assert len(peptides) == 815
    assert len(cdr3b) == 9_554
    assert peptides & cdr3b == set()
    assert len(peptides | cdr3b) == 10_369


def _components(train) -> dict[tuple[str, str], int]:
    """Connected components of the bipartite peptide <-> CDR3b graph."""
    from collections import defaultdict, deque

    adjacency = defaultdict(set)
    for peptide, tcr in zip(train["Peptide"], train["CDR3b"]):
        adjacency[("p", peptide)].add(("c", tcr))
        adjacency[("c", tcr)].add(("p", peptide))

    component_of: dict[tuple[str, str], int] = {}
    for node in adjacency:
        if node in component_of:
            continue
        queue = deque([node])
        component_of[node] = len(set(component_of.values()))
        identifier = component_of[node]
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in component_of:
                    component_of[neighbour] = identifier
                    queue.append(neighbour)
    return component_of


def test_one_giant_component_dominates(train) -> None:
    """A peptide-clean split is not TCR-clean because 67% of TCRs sit in one component."""
    import collections

    component_of = _components(train)
    sizes = collections.Counter(
        component_of[("c", tcr)] for tcr in set(train["CDR3b"])
    )
    assert len(set(component_of.values())) == 684
    assert sizes.most_common(1)[0][1] == 6_026


def test_component_split_removes_all_leakage(train) -> None:
    component_of = _components(train)
    labelled = train.assign(
        component=[component_of[("p", p)] for p in train["Peptide"]]
    )
    rows_per = labelled.groupby("component").size().sort_values(ascending=False)

    rng = np.random.default_rng(0)
    chosen: list[int] = []
    total = 0
    for component in rng.permutation(rows_per.index[1:]):
        if total >= 0.20 * len(labelled):
            break
        chosen.append(component)
        total += rows_per[component]

    validation = labelled[labelled["component"].isin(chosen)]
    training = labelled[~labelled["component"].isin(chosen)]

    assert 0.18 < len(validation) / len(labelled) < 0.22
    assert set(training["Peptide"]) & set(validation["Peptide"]) == set()
    assert set(training["CDR3b"]) & set(validation["CDR3b"]) == set()


# --- T6a: what predicts per-peptide performance -------------------------------------


def test_baseline_is_independent_of_the_negative_sampler(train, sol) -> None:
    """Both T6a panels sit on the same footing only if this holds. Confirmed, not assumed."""
    from cognate.negatives import make_negatives

    positives_only = score_by_nearest_positive(sol, train).score
    for strategy in ["shuffle", "matched"]:
        augmented = pd.concat(
            [train, make_negatives(train, strategy, 5.0, 0)], ignore_index=True
        )
        assert np.array_equal(
            positives_only, score_by_nearest_positive(sol, augmented).score
        )


def test_baseline_contributes_13_of_20_points(train, sol) -> None:
    """The 7 unseen peptides have an empty database, so their score is constant."""
    from cognate.metrics import diagnose_scores

    knn = score_by_nearest_positive(sol, train).score
    usable = [
        p
        for p in sorted(set(sol["Peptide"]))
        if not diagnose_scores(knn[(sol["Peptide"] == p).to_numpy()]).is_degenerate
    ]
    assert len(usable) == 13
    assert set(usable) == set(sol["Peptide"]) & set(train["Peptide"])


def test_distance_predicts_the_baseline_and_survives_leverage() -> None:
    """The strongest and most robust result in the build: r=-0.88, r2=0.77, and it holds
    when the extreme point (GILGFVFTL, distance 0.000) is removed."""
    points = pd.DataFrame(
        json.loads((Path(__file__).resolve().parents[1] / "data" / "t6a_points.json")
                   .read_text(encoding="utf-8"))
    )
    usable = points[points["knn_usable"]]

    full = np.corrcoef(usable["distance_positives"], usable["knn_auc01"])[0, 1]
    without = usable[usable["peptide"] != "GILGFVFTL"]
    dropped = np.corrcoef(without["distance_positives"], without["knn_auc01"])[0, 1]

    assert full == pytest.approx(-0.878, abs=5e-3)
    assert full**2 > 0.75
    assert dropped < -0.8


def test_support_predicts_the_head_but_not_the_baseline() -> None:
    """The inversion: each model is limited by a different thing."""
    points = pd.DataFrame(
        json.loads((Path(__file__).resolve().parents[1] / "data" / "t6a_points.json")
                   .read_text(encoding="utf-8"))
    )
    seen = points[points["seen"]]
    log_support = np.log10(seen["support"])

    baseline = np.corrcoef(log_support, seen["knn_auc01"])[0, 1]
    head = np.corrcoef(log_support, seen["head_matched_auc01"])[0, 1]
    assert abs(baseline) < 0.25
    assert head > 0.6

    without = seen[seen["peptide"] != "GILGFVFTL"]
    assert np.corrcoef(np.log10(without["support"]), without["head_matched_auc01"])[0, 1] > 0.5


def test_matched_sampler_gain_concentrates_on_near_peptides() -> None:
    points = pd.DataFrame(
        json.loads((Path(__file__).resolve().parents[1] / "data" / "t6a_points.json")
                   .read_text(encoding="utf-8"))
    )
    delta = points["head_matched_auc01"] - points["head_shuffle_auc01"]
    biggest = points.loc[delta.nlargest(3).index, "peptide"].tolist()
    assert set(biggest) == {"GILGFVFTL", "GLCTLVAML", "NLVPMVATV"}
    assert delta[points["seen"]].mean() > 0.03
    assert abs(delta[~points["seen"]].mean()) < 0.01
