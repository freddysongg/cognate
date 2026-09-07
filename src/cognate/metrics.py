"""Scoring for TCR-epitope binding predictions.

The IMMREP23 metric is Macro AUC0.1: partial ROC AUC truncated at 10% false positive
rate, computed independently per peptide and then averaged over peptides. The partial
AUC is McClish-standardised so that a random ranker scores 0.5 and a perfect one 1.0,
which is what ``sklearn.metrics.roc_auc_score(..., max_fpr=0.1)`` computes.

Every metric here can be reported with a bootstrap confidence interval. The headline
number averages over 20 peptides (7 on the unseen slice), so peptide-level sampling
noise is large and needs to be visible.
"""

import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MAX_FPR = 0.1
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0

ResampleMode = Literal["rows", "groups", "both"]
DegenerateAction = Literal["ignore", "warn", "raise"]

TIE_HEAVY_FRACTION = 0.25

Scorer = Callable[[np.ndarray, np.ndarray], float]


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Full ROC AUC. 0.5 = random, 1.0 = perfect."""
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision. Baseline is the positive rate, not 0.5."""
    return float(average_precision_score(y_true, y_score))


def auc01(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Partial ROC AUC up to FPR=0.1, McClish-standardised. 0.5 = random, 1.0 = perfect."""
    return float(roc_auc_score(y_true, y_score, max_fpr=MAX_FPR))


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    lo: float
    hi: float
    n_replicates: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


class DegenerateScoresError(ValueError):
    """Raised when a score column cannot express a ranking."""


@dataclass(frozen=True)
class ScoreDiagnostics:
    """Whether a score column can express a ranking at all.

    A constant score column has AUROC and McClish AUC0.1 of exactly 0.5, identical to a
    perfectly random ranker. Nothing in the metric distinguishes 'the model predicts at
    chance' from 'the model emitted the same number for every row', and the second is
    almost always a bug -- an untrained head, a peptide with no lookup database, an
    embedding join that silently missed. Every report carries this so the two readings
    cannot be confused.
    """

    n_rows: int
    n_unique_scores: int
    n_nonfinite: int
    modal_fraction: float
    constant_groups: tuple[str, ...]

    @property
    def n_constant_groups(self) -> int:
        return len(self.constant_groups)

    @property
    def is_degenerate(self) -> bool:
        return bool(self.n_nonfinite or self.constant_groups)

    @property
    def is_tie_heavy(self) -> bool:
        """Many rows share one score, which caps the achievable ranking."""
        return self.modal_fraction >= TIE_HEAVY_FRACTION

    def __str__(self) -> str:
        parts = [f"{self.n_unique_scores} distinct scores over {self.n_rows} rows"]
        if self.n_nonfinite:
            parts.append(f"{self.n_nonfinite} non-finite")
        if self.constant_groups:
            shown = ", ".join(self.constant_groups[:5])
            more = "..." if self.n_constant_groups > 5 else ""
            parts.append(f"{self.n_constant_groups} constant groups ({shown}{more})")
        if self.is_tie_heavy:
            parts.append(f"{self.modal_fraction:.1%} of rows share one value")
        return "; ".join(parts)


def diagnose_scores(
    y_score: np.ndarray,
    groups: Sequence[str] | np.ndarray | pd.Series | None = None,
) -> ScoreDiagnostics:
    """Report whether ``y_score`` can express a ranking, globally and per group."""
    y_score = np.asarray(y_score, dtype=float)
    finite = np.isfinite(y_score)
    values, counts = np.unique(y_score[finite], return_counts=True)

    groups_arr = (
        np.full(len(y_score), "__all__", dtype=object)
        if groups is None
        else np.asarray(groups, dtype=object)
    )
    constant = tuple(
        name
        for name, idx in _group_indices(groups_arr).items()
        if len(np.unique(y_score[idx])) <= 1
    )
    return ScoreDiagnostics(
        n_rows=len(y_score),
        n_unique_scores=len(values),
        n_nonfinite=int((~finite).sum()),
        modal_fraction=float(counts.max() / len(y_score)) if len(counts) else 1.0,
        constant_groups=constant,
    )


def _handle_degenerate(
    diagnostics: ScoreDiagnostics, label: str, action: DegenerateAction
) -> None:
    if action == "ignore" or not diagnostics.is_degenerate:
        return
    message = (
        f"degenerate scores in slice '{label}': {diagnostics}. "
        "A constant column scores exactly 0.5, which is indistinguishable from a random "
        "ranker -- check the scorer before reading this as a result."
    )
    if action == "raise":
        raise DegenerateScoresError(message)
    warnings.warn(message, stacklevel=3)


@dataclass(frozen=True)
class GroupedScore:
    """A per-group metric averaged over the groups where it was computable."""

    value: float
    per_group: dict[str, float]
    n_groups_skipped: int
    skipped_groups: tuple[str, ...]

    @property
    def n_groups_scored(self) -> int:
        return len(self.per_group)


@dataclass(frozen=True)
class ScoreReport:
    """All three metrics with CIs, plus the per-peptide breakdown behind the macro score."""

    label: str
    n_rows: int
    n_positive: int
    n_groups_scored: int
    n_groups_skipped: int
    skipped_groups: tuple[str, ...]
    auroc: Interval
    auprc: Interval
    macro_auc01: Interval
    per_group_auc01: dict[str, float]
    diagnostics: ScoreDiagnostics

    def __str__(self) -> str:
        skipped = f", {self.n_groups_skipped} skipped" if self.n_groups_skipped else ""
        flag = "  [DEGENERATE SCORES]" if self.diagnostics.is_degenerate else ""
        return (
            f"{self.label}: n={self.n_rows} ({self.n_positive} pos), "
            f"{self.n_groups_scored} peptides{skipped}{flag}\n"
            f"  macro AUC0.1 {self.macro_auc01}\n"
            f"  AUROC        {self.auroc}\n"
            f"  AUPRC        {self.auprc}\n"
            f"  scores       {self.diagnostics}"
        )


def _both_classes_present(y_true: np.ndarray) -> bool:
    return 0 < int(y_true.sum()) < len(y_true)


def _group_indices(groups: np.ndarray) -> dict[str, np.ndarray]:
    """Row indices per group, in first-appearance order."""
    order: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        order.setdefault(str(g), []).append(i)
    return {g: np.asarray(idx, dtype=np.intp) for g, idx in order.items()}


def macro_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Sequence[str] | np.ndarray | pd.Series,
    scorer: Scorer = auc01,
) -> GroupedScore:
    """Compute ``scorer`` independently per group and average over groups.

    Groups with only one class present are skipped and counted; the metric is undefined
    for them, and dropping them is what the IMMREP23 organisers do.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    indices = _group_indices(np.asarray(groups, dtype=object))

    per_group: dict[str, float] = {}
    skipped: list[str] = []
    for name, idx in indices.items():
        if not _both_classes_present(y_true[idx]):
            skipped.append(name)
            continue
        per_group[name] = scorer(y_true[idx], y_score[idx])

    value = float(np.mean(list(per_group.values()))) if per_group else float("nan")
    return GroupedScore(
        value=value,
        per_group=per_group,
        n_groups_skipped=len(skipped),
        skipped_groups=tuple(skipped),
    )


def macro_auc01(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Sequence[str] | np.ndarray | pd.Series,
) -> GroupedScore:
    """The IMMREP23 headline metric: per-peptide AUC0.1, averaged over peptides."""
    return macro_by_group(y_true, y_score, groups, auc01)


def _replicate_group_indices(
    indices: dict[str, np.ndarray],
    mode: ResampleMode,
    n_boot: int,
    rng: np.random.Generator,
) -> Iterator[list[np.ndarray]]:
    """Yield, per bootstrap replicate, one row-index array per drawn group.

    ``groups`` resamples which peptides land in the average, which is the dominant noise
    source when the average is over 20 of them. ``rows`` resamples TCRs within a fixed
    peptide set. ``both`` is the two-level cluster bootstrap and is the default.
    """
    names = list(indices)
    draw_groups = mode in ("groups", "both")
    draw_rows = mode in ("rows", "both")

    for _ in range(n_boot):
        drawn = rng.choice(names, size=len(names), replace=True) if draw_groups else names
        replicate = []
        for name in drawn:
            idx = indices[str(name)]
            replicate.append(rng.choice(idx, size=len(idx), replace=True) if draw_rows else idx)
        yield replicate


def _percentile_interval(
    point: float, samples: list[float], confidence: float
) -> Interval:
    if not samples:
        return Interval(point=point, lo=float("nan"), hi=float("nan"), n_replicates=0)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return Interval(point=point, lo=float(lo), hi=float(hi), n_replicates=len(samples))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_interval(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 5_000,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Pearson r with a bootstrap CI resampling the points themselves.

    Points are peptides, so this is the peptide-level resample the per-peptide correlations in
    docs/findings.md were reported with. Replicates whose resample has no variance in either
    arm are dropped rather than counted as r = 0, which would pull the interval toward zero.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) != len(y_array):
        raise ValueError(f"length mismatch: {len(x_array)} vs {len(y_array)}")

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        picked = rng.choice(len(x_array), size=len(x_array), replace=True)
        replicate = _pearson(x_array[picked], y_array[picked])
        if not np.isnan(replicate):
            samples.append(replicate)
    return _percentile_interval(_pearson(x_array, y_array), samples, confidence)


def correlation_difference_interval(
    x: Sequence[float] | np.ndarray,
    y_a: Sequence[float] | np.ndarray,
    y_b: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 5_000,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> tuple[Interval, float]:
    """Paired bootstrap on ``corr(x, y_a) - corr(x, y_b)`` over the same points.

    Two correlations whose intervals barely overlap are not a tested difference. Comparing
    them by eye is what docs/belief_list.md refused to do at n = 13; this is the test that
    claim's overturn condition names. Points are resampled once per replicate and used for
    both arms, so the pairing is preserved -- the defect that was found in
    ``compare_macro_auc01`` during Phase A.

    Returns the difference interval and a two-sided p-value.
    """
    x_array = np.asarray(x, dtype=float)
    a_array = np.asarray(y_a, dtype=float)
    b_array = np.asarray(y_b, dtype=float)
    if not len(x_array) == len(a_array) == len(b_array):
        raise ValueError(
            f"length mismatch: {len(x_array)}, {len(a_array)}, {len(b_array)}"
        )

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        picked = rng.choice(len(x_array), size=len(x_array), replace=True)
        difference = _pearson(x_array[picked], a_array[picked]) - _pearson(
            x_array[picked], b_array[picked]
        )
        if not np.isnan(difference):
            samples.append(difference)

    point = _pearson(x_array, a_array) - _pearson(x_array, b_array)
    interval = _percentile_interval(point, samples, confidence)
    if not samples:
        return interval, float("nan")
    array = np.asarray(samples)
    tail = min((array <= 0).mean(), (array >= 0).mean())
    return interval, float(min(1.0, 2.0 * tail))


def leave_one_out_correlations(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Pearson r with each point dropped in turn.

    A correlation that survives its interval but collapses when one point is removed is
    leverage, not a relationship -- that is what disqualified 'head performance tracks distance
    to the training set' at n = 13 (r = -0.735, collapsing to -0.341).
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    keep = np.ones(len(x_array), dtype=bool)
    values = []
    for i in range(len(x_array)):
        keep[i] = False
        values.append(_pearson(x_array[keep], y_array[keep]))
        keep[i] = True
    return np.array(values)


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    groups: Sequence[str] | np.ndarray | pd.Series | None = None,
    *,
    subset: np.ndarray | pd.Series | None = None,
    label: str = "all",
    mode: ResampleMode = "both",
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    on_degenerate: DegenerateAction = "warn",
) -> ScoreReport:
    """Score predictions under all three metrics with bootstrap CIs.

    ``groups`` is the per-row peptide. Omit it and every metric is computed globally,
    including AUC0.1, which is then a different quantity from the IMMREP23 headline.

    ``subset`` is a boolean mask applied to all inputs first, so seen/unseen or
    exact-match-excluded slices reuse this function rather than a variant of it.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups_arr = (
        np.full(len(y_true), "__all__", dtype=object)
        if groups is None
        else np.asarray(groups, dtype=object)
    )

    if subset is not None:
        mask = np.asarray(subset, dtype=bool)
        y_true, y_score, groups_arr = y_true[mask], y_score[mask], groups_arr[mask]

    if len(y_true) == 0:
        raise ValueError(f"subset '{label}' selected zero rows")

    diagnostics = diagnose_scores(y_score, groups_arr)
    _handle_degenerate(diagnostics, label, on_degenerate)

    indices = _group_indices(groups_arr)
    point_macro = macro_by_group(y_true, y_score, groups_arr, auc01)
    point_auroc = auroc(y_true, y_score) if _both_classes_present(y_true) else float("nan")
    point_auprc = auprc(y_true, y_score) if _both_classes_present(y_true) else float("nan")

    macro_samples: list[float] = []
    auroc_samples: list[float] = []
    auprc_samples: list[float] = []
    rng = np.random.default_rng(seed)

    for replicate in _replicate_group_indices(indices, mode, n_boot, rng):
        per_group = [
            auc01(y_true[idx], y_score[idx])
            for idx in replicate
            if _both_classes_present(y_true[idx])
        ]
        if per_group:
            macro_samples.append(float(np.mean(per_group)))
        flat = np.concatenate(replicate)
        if _both_classes_present(y_true[flat]):
            auroc_samples.append(auroc(y_true[flat], y_score[flat]))
            auprc_samples.append(auprc(y_true[flat], y_score[flat]))

    return ScoreReport(
        label=label,
        n_rows=len(y_true),
        n_positive=int(y_true.sum()),
        n_groups_scored=point_macro.n_groups_scored,
        n_groups_skipped=point_macro.n_groups_skipped,
        skipped_groups=point_macro.skipped_groups,
        auroc=_percentile_interval(point_auroc, auroc_samples, confidence),
        auprc=_percentile_interval(point_auprc, auprc_samples, confidence),
        macro_auc01=_percentile_interval(point_macro.value, macro_samples, confidence),
        per_group_auc01=point_macro.per_group,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class Comparison:
    """A paired bootstrap difference between two scorings of the same peptides."""

    label_a: str
    label_b: str
    point_a: float
    point_b: float
    difference: Interval
    p_two_sided: float
    n_groups: int

    def __str__(self) -> str:
        return (
            f"{self.label_a} ({self.point_a:.3f}) - {self.label_b} ({self.point_b:.3f}) "
            f"= {self.difference}  p={self.p_two_sided:.3f}  "
            f"over {self.n_groups} peptides"
        )


def compare_macro_auc01(
    y_true: np.ndarray | pd.Series,
    groups: Sequence[str] | np.ndarray | pd.Series,
    a: tuple[str, np.ndarray, np.ndarray | None],
    b: tuple[str, np.ndarray, np.ndarray | None],
    *,
    mode: ResampleMode = "both",
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    """Bootstrap the difference between two scorings, pairing them on the same peptides.

    Comparing two marginal confidence intervals understates significance whenever the two
    scorings share data, which they always do here -- both are computed over overlapping
    slices of one test set. Drawing the same peptides for both arms of each replicate and
    taking the difference inside the replicate removes the shared peptide-selection noise,
    which is the dominant term when the average is over 13 or 20 peptides.

    Each arm is ``(label, score, subset)``; ``subset`` is a boolean row mask or ``None``.
    Only peptides where both arms have both classes present are used, so the pairing is
    well defined.
    """
    y_true = np.asarray(y_true)
    groups_arr = np.asarray(groups, dtype=object)

    label_a, score_a, subset_a = a
    label_b, score_b, subset_b = b
    mask_a = np.ones(len(y_true), dtype=bool) if subset_a is None else np.asarray(subset_a, dtype=bool)
    mask_b = np.ones(len(y_true), dtype=bool) if subset_b is None else np.asarray(subset_b, dtype=bool)

    index_a = _group_indices(np.where(mask_a, groups_arr, None))
    index_b = _group_indices(np.where(mask_b, groups_arr, None))
    common = [
        name
        for name in index_a
        if name in index_b
        and name != "None"
        and _both_classes_present(y_true[index_a[name]])
        and _both_classes_present(y_true[index_b[name]])
    ]
    if not common:
        raise ValueError(
            f"no peptide is scorable in both arms ({label_a}: {len(index_a)} groups, "
            f"{label_b}: {len(index_b)}). Disjoint slices such as seen vs unseen cannot "
            "be paired -- compare their marginal intervals from evaluate() instead."
        )

    point_a = float(np.mean([auc01(y_true[index_a[g]], np.asarray(score_a)[index_a[g]]) for g in common]))
    point_b = float(np.mean([auc01(y_true[index_b[g]], np.asarray(score_b)[index_b[g]]) for g in common]))

    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    rng = np.random.default_rng(seed)
    draw_groups = mode in ("groups", "both")
    draw_rows = mode in ("rows", "both")
    differences: list[float] = []

    for _ in range(n_boot):
        drawn = rng.choice(common, size=len(common), replace=True) if draw_groups else common
        values_a: list[float] = []
        values_b: list[float] = []
        for name in drawn:
            rows_a = index_a[str(name)]
            rows_b = index_b[str(name)]
            if draw_rows:
                # Arms covering the same rows must share the resample, or the row-level
                # pairing this function exists to provide is thrown away and the p-value
                # becomes sensitive to which arm is passed first. Genuinely different row
                # sets (a subset ablation) are resampled independently.
                if np.array_equal(rows_a, rows_b):
                    positions = rng.integers(0, len(rows_a), len(rows_a))
                    rows_a = rows_a[positions]
                    rows_b = rows_b[positions]
                else:
                    rows_a = rng.choice(rows_a, size=len(rows_a), replace=True)
                    rows_b = rng.choice(rows_b, size=len(rows_b), replace=True)
            if _both_classes_present(y_true[rows_a]):
                values_a.append(auc01(y_true[rows_a], score_a[rows_a]))
            if _both_classes_present(y_true[rows_b]):
                values_b.append(auc01(y_true[rows_b], score_b[rows_b]))
        if values_a and values_b:
            differences.append(float(np.mean(values_a) - np.mean(values_b)))

    point = point_a - point_b
    interval = _percentile_interval(point, differences, confidence)
    if differences:
        # Symmetric under swapping the arms: negating every difference swaps the two
        # tail counts and leaves the minimum unchanged.
        below = sum(1 for d in differences if d <= 0)
        above = sum(1 for d in differences if d >= 0)
        p_two_sided = min(1.0, 2.0 * min(below, above) / len(differences))
    else:
        p_two_sided = float("nan")

    return Comparison(
        label_a=label_a,
        label_b=label_b,
        point_a=point_a,
        point_b=point_b,
        difference=interval,
        p_two_sided=p_two_sided,
        n_groups=len(common),
    )


def report_table(reports: Sequence[ScoreReport]) -> pd.DataFrame:
    """One row per report, metrics rendered as 'point [lo, hi]'."""
    return pd.DataFrame(
        [
            {
                "slice": r.label,
                "n": r.n_rows,
                "pos": r.n_positive,
                "peptides": r.n_groups_scored,
                "skipped": r.n_groups_skipped,
                "macro AUC0.1": str(r.macro_auc01),
                "AUROC": str(r.auroc),
                "AUPRC": str(r.auprc),
                "degen": "!" if r.diagnostics.is_degenerate else "",
            }
            for r in reports
        ]
    ).set_index("slice")
