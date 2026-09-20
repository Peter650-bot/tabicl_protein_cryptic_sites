"""Per-protein decision rules for turning scores into calls.

WHY
    Every metric in this project is computed per protein and then averaged, but
    the decision rule is a single global cutoff shared by all 231 test proteins.
    That mismatch is what makes the frozen threshold transfer badly: swept to a
    mean per-protein FPR of 0.05 on the CV folds, it lands at 0.071 on test. A
    rule defined *within* each protein is scale-free, so a per-protein shift in
    the score distribution cannot move its operating point.

THE TRICK
    Both per-protein rules are a MONOTONE PER-PROTEIN TRANSFORM of the score,
    after which the existing global-threshold machinery applies unchanged:
    `evaluate.choose_threshold` still selects the operating point on CV folds
    (exactly, in one pass, with its tie handling intact), and
    `evaluate.per_protein` / `evaluate.summarise` still score it. Nothing here
    re-implements threshold selection or metrics.

    So "flag the top q fraction of each protein" is not a new kind of rule -- it
    is `s >= t` on the within-protein rank. That is why `rank` returns
    `1 - rank_fraction`: it makes "high is better" hold, so a single threshold
    `t` means "top `1 - t` of every protein".

WHAT THIS CANNOT CHANGE
    Per-protein AUC and AUPRC are rank statistics computed within one protein, so
    they are invariant under any strictly monotone per-protein transform. The
    0.862 / 0.441 headline is arithmetically untouched by anything in this
    module; only the threshold-dependent metrics (TPR, FPR, MCC, F1) move.
    `tests/test_operating_point.py` asserts this per protein rather than trusting
    the argument.

TIES
    `rank` uses `method="max"` on the descending order, so residues sharing a
    score share a rank and are admitted or rejected together. A threshold cannot
    split a tie, which is the same assumption `evaluate.choose_threshold` makes.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd


def identity(scores: Sequence[float], groups: Sequence[str]) -> np.ndarray:
    """The incumbent rule: the raw score, one global cutoff for every protein."""
    return np.asarray(scores, dtype=float)


def rank(scores: Sequence[float], groups: Sequence[str]) -> np.ndarray:
    """`1 - (within-protein descending rank) / n`, so high still means cryptic.

    A threshold `t` on this is exactly "flag the top `1 - t` fraction of every
    protein" -- see `top_q_from_threshold`.
    """
    s = pd.Series(np.asarray(scores, dtype=float))
    g = pd.Series(np.asarray(groups))
    if len(s) != len(g):
        raise ValueError("scores and groups must be the same length")
    r = s.groupby(g).rank(method="max", ascending=False)
    n = g.map(g.value_counts())
    return (1.0 - r / n).to_numpy()


def zscore(scores: Sequence[float], groups: Sequence[str]) -> np.ndarray:
    """`(s - mean_i) / sd_i` within each protein.

    A protein whose scores are all equal has `sd == 0` and maps to 0.0 rather
    than NaN, so it is treated as uniformly unremarkable instead of poisoning
    the mean.
    """
    s = pd.Series(np.asarray(scores, dtype=float))
    g = pd.Series(np.asarray(groups))
    if len(s) != len(g):
        raise ValueError("scores and groups must be the same length")
    grp = s.groupby(g)
    centred = s - grp.transform("mean")
    sd = grp.transform("std", ddof=0)
    flat = ~(sd > 0)
    return (centred / sd.where(~flat, 1.0)).where(~flat, 0.0).to_numpy()


#: Rule name -> transform. The key is what gets written into result files.
RULES: Dict[str, Callable[[Sequence[float], Sequence[str]], np.ndarray]] = {
    "global_threshold": identity,
    "rank": rank,
    "zscore": zscore,
}


def transform(rule: str, scores: Sequence[float], groups: Sequence[str]) -> np.ndarray:
    """Apply a named rule's transform, rejecting unknown names loudly."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; valid: {sorted(RULES)}")
    return RULES[rule](scores, groups)


def top_q_from_threshold(threshold: float) -> float:
    """The `q` in "top q of each protein" for a threshold on `rank` output."""
    return 1.0 - threshold


def describe(rule: str, threshold: float) -> str:
    """Human-readable operating point, for tables and logs."""
    if rule == "rank":
        return f"top {100.0 * top_q_from_threshold(threshold):.2f}% per protein"
    if rule == "zscore":
        return f"z >= {threshold:.4f} within protein"
    return f"score >= {threshold:.4f}"
