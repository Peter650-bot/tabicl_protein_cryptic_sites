"""Conventional classifiers on exactly the same table, as a control.

WHY THIS EXISTS
    A good TabICL score on its own is ambiguous: it could mean the in-context
    model is strong, or merely that 34 apo-only structural features carry the
    signal and any reasonable classifier would find it. Those imply different
    follow-ups -- better model vs. better features -- so the comparison is worth
    its few minutes of CPU.

    Everything except the model is held fixed: same parquet, same CryptoBench
    folds, same threshold rule, same per-protein averaging. The only thing that
    varies is what maps features to a score.

    A single-feature baseline (RSA alone) is included as a floor. If the learned
    models do not clearly beat it, the join or the features are broken.

Runs locally on CPU in a few minutes; no GPU, no Modal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C, evaluate, sampling  # noqa: E402
from cryptic_tabicl.model import CATEGORICAL_COLS  # noqa: E402


def _design(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot the categoricals; sklearn has no native categorical handling.

    TabICL does this internally, which is part of why it needs no tuning. Doing
    it explicitly here keeps the feature content identical between the two.
    """
    drop = set(C.NON_FEATURE_COLS) | {"pdb_id"}
    cols = [c for c in train.columns if c not in drop]
    both = pd.concat([train[cols], test[cols]], keys=["tr", "te"])
    both = pd.get_dummies(both, columns=[c for c in CATEGORICAL_COLS if c in cols])
    # Impute on TRAIN statistics only. Using the pooled median would let the
    # test set influence its own preprocessing -- a small leak, but a real one.
    med = both.loc["tr"].median(numeric_only=True)
    both = both.fillna(med).fillna(0.0)
    return both.loc["tr"].to_numpy(float), both.loc["te"].to_numpy(float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=C.BUILD_DIR / "residues.parquet")
    ap.add_argument("--tag", default="", help="suffix for the output json")
    ap.add_argument("--budget", type=int, default=C.CONTEXT_BUDGET_ROWS,
                    help="match TabICL's context budget so the comparison is fair")
    ap.add_argument("--full-train", action="store_true",
                    help="also fit on all 304k training rows, which TabICL cannot do")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    table = pd.read_parquet(args.table)
    n_feat = len([c for c in table.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])
    print(f"table={args.table.name}  features={n_feat}")
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    y_te = test[C.LABEL_COL].to_numpy()
    g_te = test[C.GROUP_COL].to_numpy()

    # Same subsampled context TabICL sees.
    ctx, plan = sampling.build_context(train, budget=args.budget)
    print(plan, "\n")

    results = []

    # --- floor: a single feature, no learning -----------------------------
    # Buried residues are more likely to be cryptic, so the sign is negative.
    res = evaluate.summarise(y_te, -test["rsa"].to_numpy(), g_te,
                             threshold=-0.5, label="RSA alone (floor)")
    results.append(res)
    print(f"RSA alone            AUC {res['AUC']:.3f}  AUPRC {res['AUPRC']:.3f}")

    fits = [("context", ctx)]
    if args.full_train:
        fits.append(("full-train", train))

    for tag, fit_on in fits:
        X_tr, X_te = _design(fit_on, test)
        y_tr = fit_on[C.LABEL_COL].to_numpy()
        g_tr = fit_on[C.GROUP_COL].to_numpy()

        scaler = StandardScaler().fit(X_tr)
        Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

        models = {
            f"LogisticRegression ({tag})": (
                LogisticRegression(max_iter=2000, class_weight="balanced"),
                Xs_tr, Xs_te),
            f"RandomForest ({tag})": (
                RandomForestClassifier(n_estimators=300, min_samples_leaf=4,
                                       class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=C.RANDOM_SEED),
                X_tr, X_te),
        }
        for name, (clf, a, b) in models.items():
            t0 = time.time()
            clf.fit(a, y_tr)
            s_tr = clf.predict_proba(a)[:, 1]
            s_te = clf.predict_proba(b)[:, 1]
            # threshold picked on the fitting data only, never on test
            thr = evaluate.choose_threshold(y_tr, s_tr, g_tr)
            res = evaluate.summarise(y_te, s_te, g_te, thr, label=name)
            results.append(res)
            print(f"{name:28s} AUC {res['AUC']:.3f}  AUPRC {res['AUPRC']:.3f}  "
                  f"TPR {res['TPR']:.3f}  FPR {res['FPR']:.3f}  ({time.time()-t0:.0f}s)")

    out = C.BUILD_DIR / f"baselines{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    print("\n" + "=" * 60)
    print(f"{'Method':<30} {'AUC':>6} {'AUPRC':>7}")
    print("-" * 60)
    for r in sorted(results, key=lambda r: -r["AUC"]):
        print(f"{r['method']:<30} {r['AUC']:>6.3f} {r['AUPRC']:>7.3f}")
    for name, m in evaluate.BENCHMARK.items():
        print(f"{name:<30} {m['AUC']:>6.2f} {m['AUPRC']:>7.2f}   (published)")


if __name__ == "__main__":
    main()
