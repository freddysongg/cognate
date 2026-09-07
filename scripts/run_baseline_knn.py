"""T4 v1 -- score the nearest-neighbour baseline on the IMMREP23 test set.

Prints the four headline slices (seen/unseen x with/without verbatim training pairs) plus
the leave-one-out variant, which asks a different question: not 'how much does the leaked
subset inflate the score' but 'can the method still rank those rows without the answer key'.
"""

import numpy as np

from cognate.baseline_knn import exact_match_mask, score_by_nearest_positive
from cognate.data import load_solutions, load_train
from cognate.metrics import evaluate, report_table


def main() -> None:
    train = load_train()
    sol = load_solutions()

    y_true = sol["Label"].to_numpy()
    peptides = sol["Peptide"]
    is_seen = sol["Peptide"].isin(set(train["Peptide"])).to_numpy()
    is_exact = exact_match_mask(sol, train)

    knn = score_by_nearest_positive(sol, train)
    held_out = score_by_nearest_positive(sol, train, leave_out_exact_matches=True)

    slices = [
        ("all", knn.score, np.ones(len(sol), dtype=bool)),
        ("seen", knn.score, is_seen),
        ("seen, exact excluded", knn.score, is_seen & ~is_exact),
        ("unseen", knn.score, ~is_seen),
        ("unseen, exact excluded", knn.score, ~is_seen & ~is_exact),
        ("seen, exact held out of db", held_out.score, is_seen),
    ]
    reports = [
        evaluate(y_true, score, peptides, subset=mask, label=label, seed=0)
        for label, score, mask in slices
    ]
    print(report_table(reports).to_string())

    seen_with = reports[1].macro_auc01.point
    seen_without = reports[2].macro_auc01.point
    print(
        f"\nlookup-table decomposition: seen {seen_with:.3f} - "
        f"seen-without-exact {seen_without:.3f} = {seen_with - seen_without:+.3f}"
    )
    print(
        f"leave-one-out variant:      seen {seen_with:.3f} - "
        f"held-out-db {reports[5].macro_auc01.point:.3f} = "
        f"{seen_with - reports[5].macro_auc01.point:+.3f}"
    )

    print("\nper-peptide macro AUC0.1 (seen slice), against training support:")
    support = train.groupby("Peptide").size()
    per_peptide = sorted(
        reports[1].per_group_auc01.items(), key=lambda kv: -support.get(kv[0], 0)
    )
    for peptide, value in per_peptide:
        print(f"  {peptide:<12} support={support.get(peptide, 0):>5}  AUC0.1={value:.3f}")


main()
