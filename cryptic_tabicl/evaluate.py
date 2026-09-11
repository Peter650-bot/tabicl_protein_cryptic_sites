"""Evaluation that is directly comparable to the CryptoBench paper.

WHY PER-PROTEIN, THEN AVERAGED
    CryptoBench's own `Protein.py` computes every metric for ONE protein and the
    paper reports the mean across proteins. Pooling all residues instead would
    let a handful of large proteins dominate, and would not be comparable to the
    published table. We report both, but the per-protein mean is the headline.

WHY NOT ACCURACY
    Prevalence is ~6%. Predicting "never cryptic" scores 94% accuracy and is
    useless. AUC and AUPRC are the honest summaries; TPR at a fixed low FPR is
    what actually matters for prioritising residues to test.

THRESHOLD
    CryptoBench fixes pLM-NN's decision threshold at 0.95, chosen on a validation
    set (`scripts/benchmark-example.py`; paper: "A second run using a validation
    set was conducted to determine the decision threshold, which was set at
    0.95"). PocketMiner used 0.7, P2Rank emits binary calls directly -- so the
    threshold is per-method by construction, not a benchmark constant.

    We therefore do the same thing they did, on our own model: SELECT a threshold
    on training folds to hit a target FPR, then apply it unchanged to the test
    set. Applying their 0.95 to our scores would compare calibration, not
    ranking: TabICL's zero-shot probabilities are differently scaled and only
    0.98% of test residues reach 0.95, which collapses TPR to 0.10.

F1 IS WEIGHTED IN THE PUBLISHED TABLE
    The paper computes F1 "using a weighted average" over both classes, which at
    ~5% prevalence is dominated by the negative class -- that is why their F1
    (0.92) tracks their ACC (0.93) rather than their TPR (0.48). Positive-class
    F1 is the informative number, so we report BOTH: `F1` is positive-class,
    `F1_weighted` is the one comparable to the published column.

SUBSETS ARE THE UNIT OF COMPARISON
    The published table has no single leaderboard column. Each method is scored
    on the subset it could actually run, and pLM-NN is re-scored on that same
    subset to give a like-for-like row:
      CB-full        222 test structures (everything)
      CB-PM          the ~162 on which PocketMiner ran
      CB-P2RANK-apo  the 184 single-chain structures (38 multi-chain dropped)
      CB-P2RANK-holo the holo counterparts of CB-P2RANK-apo -- NOT an apo task
    A number is only comparable against the row sharing its subset.

REFERENCE (CryptoBench published table)
    pLM-NN      CB-full        AUC 0.86  AUPRC 0.36  TPR 0.48  FPR 0.05  MCC 0.39
    pLM-NN      CB-PM          AUC 0.88  AUPRC 0.43  TPR 0.52  FPR 0.04  MCC 0.44
    PocketMiner CB-PM          AUC 0.76  AUPRC 0.19  TPR 0.51  FPR 0.16  MCC 0.22
    pLM-NN      CB-P2RANK-apo  AUC 0.88  AUPRC 0.42  TPR 0.51  FPR 0.04  MCC 0.43
    P2Rank      CB-P2RANK-apo  AUC 0.81  AUPRC 0.21  TPR 0.62  FPR 0.14  MCC 0.27
    P2Rank      CB-P2RANK-holo AUC 0.89  AUPRC 0.34  TPR 0.84  FPR 0.15  MCC 0.38
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics

from . import config

# Published numbers, for side-by-side reporting.
BENCHMARK = {
    "pLM-NN (CB-full)":       {"AUC": 0.86, "AUPRC": 0.36, "TPR": 0.48,
                               "FPR": 0.05, "MCC": 0.39, "subset": "CB-full"},
    "pLM-NN (CB-PM)":         {"AUC": 0.88, "AUPRC": 0.43, "TPR": 0.52,
                               "FPR": 0.04, "MCC": 0.44, "subset": "CB-PM"},
    "PocketMiner (CB-PM)":    {"AUC": 0.76, "AUPRC": 0.19, "TPR": 0.51,
                               "FPR": 0.16, "MCC": 0.22, "subset": "CB-PM"},
    "pLM-NN (CB-P2R-apo)":    {"AUC": 0.88, "AUPRC": 0.42, "TPR": 0.51,
                               "FPR": 0.04, "MCC": 0.43, "subset": "CB-P2RANK-apo"},
    "P2Rank (CB-P2R-apo)":    {"AUC": 0.81, "AUPRC": 0.21, "TPR": 0.62,
                               "FPR": 0.14, "MCC": 0.27, "subset": "CB-P2RANK-apo"},
}


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC, or NaN when a protein has only one class present."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(metrics.roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    precision, recall, _ = metrics.precision_recall_curve(y, s)
    return float(metrics.auc(recall, precision))


def mean_per_protein_fpr(
    y: Sequence[int], scores: Sequence[float], groups: Sequence[str],
    threshold: float,
) -> float:
    """The FPR this project reports: per protein, then averaged over proteins."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(scores, dtype=float) >= threshold
    g = np.asarray(groups)
    neg = y == 0
    if not neg.any():
        return float("nan")
    gn, pn = g[neg], pred[neg]
    uniq, inv = np.unique(gn, return_inverse=True)
    fp = np.bincount(inv, weights=pn.astype(float), minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq))
    return float((fp / n).mean())


def choose_threshold(
    y: Sequence[int], scores: Sequence[float], groups: Sequence[str],
    target_fpr: float = config.TARGET_FPR,
) -> float:
    """Lowest threshold whose MEAN PER-PROTEIN FPR does not exceed `target_fpr`.

    Selected on training data only, then frozen.

    WHY PER-PROTEIN AND NOT POOLED
        CryptoBench evaluates one protein at a time and averages the per-protein
        numbers -- `src/H-prediction-evaluation/pocketminer_stats.py` writes an
        "average" row as `sum(avg_fpr) / counter`, over `Protein.get_FPR()` values
        from `src/I-final/Protein.py`. It also emits a pooled "overall" row, but
        the average is the headline, and it is the convention this project matches
        everywhere else (see `per_protein` and `summarise`).

        This function used to call `sklearn.roc_curve`, which pools every residue
        of every protein into one bucket -- so a 1,138-residue protein counted
        twenty times more than a 57-residue one. The threshold was therefore tuned
        against a quantity the project never reports. Measured on the 118-feature
        scores at the old frozen threshold: pooled FPR 0.0658, mean per-protein
        FPR 0.0621. Targeting the reported statistic instead moves the threshold
        0.568 -> 0.636 and TPR 0.591 -> 0.544.

        Note that CryptoBench itself does not sweep to a target FPR at all -- it
        fixes a constant and reports whatever FPR follows. Its own sources do not
        agree on the constant: the paper text says the threshold was set at 0.95
        on a validation set, while `src/I-final/benchmark-example.py` hardcodes
        `DECISION_THRESHOLD = 0.75` and calls that the paper's value (PocketMiner
        was scored at 0.7). The discrepancy does not matter here -- either way it
        is a fixed constant, not an FPR-targeted sweep. Sweeping is this project's
        own choice; the point is only that it must sweep the statistic it then
        publishes. See the module docstring for the threshold rationale.

    HOW
        Exactly, in one pass, without assuming monotonicity. Weight each negative
        residue by `1 / (n_proteins * n_negatives_in_its_protein)`; then the mean
        per-protein FPR at a threshold is precisely the sum of the weights of the
        negatives that threshold admits. Sorting by score descending and taking a
        cumulative sum gives every attainable FPR at once.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(groups)
    if not (len(y) == len(s) == len(g)):
        raise ValueError("y, scores and groups must be the same length")

    neg = y == 0
    if not neg.any():
        # No negatives to constrain the FPR; admit nothing rather than everything.
        return float(np.max(s)) if len(s) else 0.5

    sn, gn = s[neg], g[neg]
    uniq, inv, counts = np.unique(gn, return_inverse=True, return_counts=True)
    w = 1.0 / (len(uniq) * counts[inv])

    order = np.argsort(-sn, kind="stable")
    sn, w = sn[order], w[order]
    cum = np.cumsum(w)

    # A threshold cannot separate negatives that share a score, so only the last
    # index of each tie-group is an attainable operating point.
    tie_end = np.append(np.flatnonzero(np.diff(sn)), len(sn) - 1)
    ok = tie_end[cum[tie_end] <= target_fpr]
    if len(ok) == 0:
        # Even the single highest-scoring negative already breaks the budget.
        return float(np.nextafter(sn[0], np.inf))
    return float(sn[ok[-1]])


def per_protein(
    y_true: Sequence[int], y_score: Sequence[float], groups: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    """One row of metrics per apo structure."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int),
                       "s": np.asarray(y_score, dtype=float),
                       "g": np.asarray(groups)})
    rows = []
    for gid, sub in df.groupby("g", sort=True):
        y, s = sub["y"].to_numpy(), sub["s"].to_numpy()
        pred = (s >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rows.append({
            "apo_id": gid,
            "n_residues": len(y),
            "n_positive": int(y.sum()),
            "AUC": _safe_auc(y, s),
            "AUPRC": _safe_auprc(y, s),
            "ACC": (tp + tn) / len(y),
            "TPR": tp / (tp + fn) if (tp + fn) else float("nan"),
            "FPR": fp / (fp + tn) if (fp + tn) else float("nan"),
            "MCC": float(metrics.matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 or len(np.unique(y)) > 1 else 0.0,
            "F1": float(metrics.f1_score(y, pred, zero_division=0)),
            # The published column is the weighted average over BOTH classes.
            "F1_weighted": float(metrics.f1_score(y, pred, average="weighted",
                                                  zero_division=0)),
        })
    return pd.DataFrame(rows)


def summarise(
    y_true: Sequence[int], y_score: Sequence[float], groups: Sequence[str],
    threshold: float, label: str = "TabICL",
) -> Dict[str, float]:
    """Headline numbers: per-protein means plus pooled AUC/AUPRC."""
    pp = per_protein(y_true, y_score, groups, threshold)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    return {
        "method": label,
        "n_proteins": len(pp),
        "n_residues": len(y),
        "prevalence": float(y.mean()),
        "threshold": threshold,
        "AUC": float(pp["AUC"].mean(skipna=True)),
        "AUPRC": float(pp["AUPRC"].mean(skipna=True)),
        "ACC": float(pp["ACC"].mean()),
        "TPR": float(pp["TPR"].mean(skipna=True)),
        "FPR": float(pp["FPR"].mean(skipna=True)),
        "MCC": float(pp["MCC"].mean()),
        "F1": float(pp["F1"].mean()),
        "F1_weighted": float(pp["F1_weighted"].mean()),
        "AUC_pooled": _safe_auc(y, s),
        "AUPRC_pooled": _safe_auprc(y, s),
    }


def comparison_table(result: Dict[str, float], subset: str = "CB-full") -> str:
    """Render our result against the published benchmark, grouped by subset.

    Rows are grouped by subset because that is the only axis along which the
    published numbers are comparable at all -- pLM-NN appears three times with
    three different AUCs purely because the structure set differs.
    """
    lines = [
        f"{'Method':<24} {'Subset':<15} {'AUC':>6} {'AUPRC':>7} "
        f"{'TPR':>6} {'FPR':>6} {'MCC':>6}",
        "-" * 74,
    ]
    lines.append(
        f"{result['method']:<24} {subset:<15} {result['AUC']:>6.2f} "
        f"{result['AUPRC']:>7.2f} {result['TPR']:>6.2f} {result['FPR']:>6.2f} "
        f"{result['MCC']:>6.2f}   <- ours"
    )
    for name, m in BENCHMARK.items():
        lines.append(
            f"{name:<24} {m['subset']:<15} {m['AUC']:>6.2f} {m['AUPRC']:>7.2f} "
            f"{m['TPR']:>6.2f} {m['FPR']:>6.2f} {m['MCC']:>6.2f}"
        )
    return "\n".join(lines)


def stratified(
    table: pd.DataFrame, y_score: Sequence[float], threshold: float,
) -> pd.DataFrame:
    """Break performance down by whether a residue lines an already-open pocket.

    CryptoBench ships non-cryptic pocket annotations precisely so that
    "found a pocket" can be distinguished from "found a CRYPTIC pocket". If a
    model scores well overall but cannot separate cryptic from non-cryptic
    pocket residues, it has learned pocket detection, not crypticity.
    """
    df = table.copy()
    df["score"] = np.asarray(y_score, dtype=float)
    out = []
    for flag, sub in df.groupby("in_noncryptic_pocket"):
        y = sub[config.LABEL_COL].to_numpy()
        s = sub["score"].to_numpy()
        pred = (s >= threshold).astype(int)
        out.append({
            "stratum": "lines a non-cryptic pocket" if flag else "elsewhere",
            "n_residues": len(y),
            "prevalence": float(y.mean()),
            "AUC": _safe_auc(y, s),
            "AUPRC": _safe_auprc(y, s),
            "TPR": float(((pred == 1) & (y == 1)).sum() / max(y.sum(), 1)),
        })
    return pd.DataFrame(out)
