"""T6a -- the main figure. Per-peptide AUC0.1 against training support and against
distance to the nearest training example.

Both panels are 20-point scatters, one point per test peptide. The k-NN baseline
contributes only 13 of them: its 7 unseen peptides have an empty database, so their score
is a constant 0.5 produced by arithmetic rather than measurement. Those points are drawn
as excluded markers rather than dropped, and are omitted from the baseline's fit.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "data" / "t6a_points.json"
FIGURE = ROOT / "docs" / "t6a_distance_vs_support.png"

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_SECONDARY = "#5f5e55"
GRID = "#e3e2dc"
EXCLUDED = "#9a998f"

SERIES = {
    "knn": ("#2a78d6", "k-NN baseline", "o"),
    "head_matched": ("#eb6834", "head (matched negatives)", "s"),
    "head_shuffle": ("#1baf7a", "head (shuffle negatives)", "^"),
}
CHANCE = 0.5
FLOOR = 9 / 19
PANEL_A_LABELS = {
    "GILGFVFTL": (-12, 8),
    "RAKFKQLL": (11, -16),
    "QIKVRVDMV": (11, -3),
    "IPSINVHHY": (11, 7),
}
PANEL_B_LABELS = {"GILGFVFTL": (12, -4), "QIKVRVDMV": (12, 2)}


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)


def _reference_lines(axis) -> None:
    axis.axhline(CHANCE, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)
    axis.axhline(FLOOR, color=EXCLUDED, linewidth=0.9, linestyle=":", zorder=1)


def _fit(axis, x, y, colour, log_x: bool = False) -> float:
    """Fit in the space the correlation is computed in, then draw in data space.

    When the axis is logarithmic the fit runs on log10(x) but the line must be plotted
    against the original x values, or it lands at 0.8-3.3 on an axis that runs 6-1818.
    """
    correlation = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    span = np.linspace(min(x), max(x), 50)
    axis.plot(10**span if log_x else span, slope * span + intercept,
              color=colour, linewidth=2, alpha=0.5, zorder=2)
    return correlation


def _scatter(axis, x, y, colour, label, marker="o", seen=None) -> None:
    if seen is None:
        axis.scatter(x, y, s=90, color=colour, edgecolor=SURFACE, linewidth=1.4,
                     marker=marker, label=label, zorder=4)
        return
    seen = np.asarray(seen, dtype=bool)
    axis.scatter(x[seen], y[seen], s=90, color=colour, edgecolor=SURFACE,
                 linewidth=1.4, marker=marker, label=label, zorder=4)
    axis.scatter(x[~seen], y[~seen], s=90, facecolor="none", edgecolor=colour,
                 linewidth=1.8, marker=marker, zorder=4)


def main() -> None:
    points = pd.DataFrame(json.loads(POINTS.read_text(encoding="utf-8")))
    seen = points[points["seen"]].reset_index(drop=True)
    usable = points[points["knn_usable"]].reset_index(drop=True)
    excluded = points[~points["knn_usable"]].reset_index(drop=True)

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=SURFACE)

    # --- Panel A: training support -------------------------------------------------
    _style(left)
    _reference_lines(left)
    log_support = np.log10(seen["support"].to_numpy())
    r_knn = _fit(left, log_support, seen["knn_auc01"].to_numpy(), SERIES["knn"][0],
                 log_x=True)
    r_head = _fit(left, log_support, seen["head_matched_auc01"].to_numpy(),
                  SERIES["head_matched"][0], log_x=True)
    _scatter(left, seen["support"].to_numpy(), seen["knn_auc01"].to_numpy(),
             *SERIES["knn"])
    _scatter(left, seen["support"].to_numpy(), seen["head_matched_auc01"].to_numpy(),
             *SERIES["head_matched"])
    left.set_xlim(4.2, 4200)
    for _, row in seen.iterrows():
        if row["peptide"] in PANEL_A_LABELS:
            left.annotate(
                row["peptide"], (row["support"], row["knn_auc01"]),
                textcoords="offset points", xytext=PANEL_A_LABELS[row["peptide"]],
                fontsize=8, color=INK_SECONDARY, clip_on=False,
            )
    left.set_xscale("log")
    left.set_xlabel("training positives for this peptide (log)", color=INK, fontsize=10)
    left.set_ylabel("per-peptide macro AUC0.1", color=INK, fontsize=10)
    left.set_title(
        f"A · Training support\nk-NN r={r_knn:+.2f} (refuted)   ·   head r={r_head:+.2f} (robust)",
        color=INK, fontsize=11, loc="left", pad=12,
    )
    left.text(0.02, 0.97,
              "13 seen peptides only — the 7 unseen\nhave support 0, off a log axis",
              transform=left.transAxes, ha="left", va="top", fontsize=8.5,
              color=INK_SECONDARY, linespacing=1.4)

    # --- Panel B: distance to nearest training example -----------------------------
    _style(right)
    _reference_lines(right)
    r_knn_d = _fit(right, usable["distance_positives"].to_numpy(),
                   usable["knn_auc01"].to_numpy(), SERIES["knn"][0])
    r_head_d = _fit(right, points["distance_positives"].to_numpy(),
                    points["head_matched_auc01"].to_numpy(), SERIES["head_matched"][0])
    r_shuf_d = _fit(right, points["distance_positives"].to_numpy(),
                    points["head_shuffle_auc01"].to_numpy(), SERIES["head_shuffle"][0])

    right.scatter(excluded["distance_positives"], excluded["knn_auc01"], s=95,
                  marker="x", color=EXCLUDED, linewidth=2, zorder=3,
                  label="k-NN, 7 unseen — EXCLUDED (constant score)")
    _scatter(right, usable["distance_positives"].to_numpy(),
             usable["knn_auc01"].to_numpy(), *SERIES["knn"])
    _scatter(right, points["distance_positives"].to_numpy(),
             points["head_matched_auc01"].to_numpy(), *SERIES["head_matched"],
             seen=points["seen"].to_numpy())
    _scatter(right, points["distance_positives"].to_numpy(),
             points["head_shuffle_auc01"].to_numpy(), *SERIES["head_shuffle"],
             seen=points["seen"].to_numpy())
    right.set_xlim(-0.012, 0.245)

    for _, row in usable.iterrows():
        if row["peptide"] in PANEL_B_LABELS:
            right.annotate(
                row["peptide"], (row["distance_positives"], row["knn_auc01"]),
                textcoords="offset points", xytext=PANEL_B_LABELS[row["peptide"]],
                fontsize=8, color=INK_SECONDARY,
            )
    right.set_xlabel(
        "median normalised edit distance from this peptide's binders\n"
        "to the nearest CDR3b anywhere in training",
        color=INK, fontsize=10,
    )
    right.set_title(
        f"B · Distance to training set\nk-NN r={r_knn_d:+.2f} (robust)   ·   "
        f"head r={r_head_d:+.2f} (leverage-driven)   ·   shuffle r={r_shuf_d:+.2f}",
        color=INK, fontsize=11, loc="left", pad=12,
    )
    right.text(0.985, 0.985,
               "filled = seen peptide,  open = unseen\n"
               "dashed = chance (0.500)\ndotted = metric floor (0.474)",
               transform=right.transAxes, ha="right", va="top", fontsize=8.5,
               color=INK_SECONDARY, linespacing=1.5)

    for axis in (left, right):
        axis.set_ylim(0.425, 0.955)

    handles, labels = [], []
    for axis in (right, left):
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                  fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(
        "T6a · What predicts per-peptide performance? One point per test peptide.",
        color=INK, fontsize=13, x=0.012, ha="left", y=1.0,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(FIGURE, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {FIGURE}")


main()
