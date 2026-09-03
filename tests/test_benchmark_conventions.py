"""Lock in the conventions that make our numbers comparable to CryptoBench.

These are not correctness tests of the model; they guard the *comparability*
claims in RESULTS.md. Each one encodes something that was verified once against
the paper or against CryptoBench's own `scripts/Protein.py`, and that would
silently invalidate a published-table comparison if it drifted.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import evaluate  # noqa: E402


def _toy(n=400, prevalence=0.05, seed=0):
    """Two proteins at CryptoBench-like prevalence with a usable ranking."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    score = np.clip(rng.normal(0.3, 0.15, n) + 0.35 * y, 0, 1)
    groups = np.where(np.arange(n) < n // 2, "aaaaA", "bbbbB")
    return y, score, groups


def test_reports_both_positive_class_and_weighted_f1():
    """The published F1 column is a weighted average; ours is positive-class.

    Reporting only one of them invites a 0.40-vs-0.92 comparison that looks like
    a catastrophic deficit and is actually two different statistics.
    """
    y, score, groups = _toy()
    res = evaluate.summarise(y, score, groups, threshold=0.5)
    assert "F1" in res and "F1_weighted" in res
    # At ~5% prevalence the weighted average is dominated by the negatives and
    # must sit far above the positive-class value.
    assert res["F1_weighted"] > res["F1"]


def test_weighted_f1_matches_sklearn_weighted_average():
    y, score, groups = _toy()
    thr = 0.5
    pp = evaluate.per_protein(y, score, groups, thr)
    for gid, expected in zip(pp["apo_id"], pp["F1_weighted"]):
        mask = groups == gid
        got = metrics.f1_score(y[mask], (score[mask] >= thr).astype(int),
                               average="weighted", zero_division=0)
        assert expected == pytest.approx(got)


def test_auprc_uses_trapezoidal_rule_not_average_precision():
    """CryptoBench's `Protein.py` uses `metrics.auc(recall, precision)`.

    `average_precision_score` is the more usual choice and differs from it, so
    matching their estimator is what keeps AUPRC comparable.
    """
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.05).astype(int)
    s = np.clip(rng.normal(0.3, 0.15, 600) + 0.35 * y, 0, 1)
    precision, recall, _ = metrics.precision_recall_curve(y, s)
    assert evaluate._safe_auprc(y, s) == pytest.approx(metrics.auc(recall, precision))


def test_benchmark_rows_all_declare_a_subset():
    """A published number is only meaningful next to its own subset.

    pLM-NN appears three times with AUC 0.86/0.88/0.88 purely because the
    structure set differs; a row without its subset invites picking whichever
    number flatters us.
    """
    for name, row in evaluate.BENCHMARK.items():
        assert "subset" in row, f"{name} has no subset label"
        assert row["subset"] in {"CB-full", "CB-PM", "CB-P2RANK-apo",
                                 "CB-P2RANK-holo"}


def test_holo_row_is_not_in_the_comparison_table():
    """CB-P2RANK-holo scores P2Rank on BOUND structures.

    It is not the apo prediction task and must never appear as a target to beat.
    """
    assert not any(r["subset"] == "CB-P2RANK-holo"
                   for r in evaluate.BENCHMARK.values())
