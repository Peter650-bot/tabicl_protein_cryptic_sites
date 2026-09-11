"""How much of a headline number is the 50k context draw?

WHY
    TabICL sees a 50,000-row subsample of the 304k training rows, not the whole
    thing. `sampling.build_context` is seeded (config.RANDOM_SEED = 0) and the
    headline runs never override it, so reruns reproduce exactly -- but the
    seed is arbitrary. A different draw is a different number, and none of the
    reported metrics carry an error bar.

    Note WHAT varies: every positive is always taken (`pos = train[y == 1]` in
    full). Only the negative sample moves. So this measures the sensitivity of
    the result to which ~46k negatives out of ~289k happen to be drawn.

    This probe uses LogisticRegression, not TabICL -- the sklearn fit is CPU
    seconds where TabICL is GPU minutes. It measures the variance the CONTEXT
    DRAW injects into a linear model on the same rows and features. Treat it as
    an order-of-magnitude estimate for TabICL, not a substitute: re-run
    deploy/tabicl_fanout.py over seeds for the real thing.

Run:  python scripts/context_seed_variance.py --seeds 20
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
from scripts.baseline import _design  # noqa: E402

METRICS = ["AUC", "AUPRC", "F1_weighted", "MCC", "TPR", "FPR"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=C.BUILD_DIR / "residues_cons.parquet")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=50_000)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    table = pd.read_parquet(args.table)
    n_feat = len([c for c in table.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])
    train, test = table[table.split == "train"], table[table.split == "test"]
    y_te, g_te = test[C.LABEL_COL].to_numpy(), test[C.GROUP_COL].to_numpy()
    print(f"table={args.table.name}  features={n_feat}  "
          f"train={len(train):,}  test={len(test):,}  budget={args.budget:,}\n")

    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        ctx, plan = sampling.build_context(train, budget=args.budget, seed=seed)
        X_tr, X_te = _design(ctx, test)
        y_tr = ctx[C.LABEL_COL].to_numpy()
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X_tr), y_tr)
        s_tr = clf.predict_proba(sc.transform(X_tr))[:, 1]
        s_te = clf.predict_proba(sc.transform(X_te))[:, 1]
        # on the context, never on test
        thr = evaluate.choose_threshold(y_tr, s_tr, ctx[C.GROUP_COL].to_numpy())
        r = evaluate.summarise(y_te, s_te, g_te, thr, label=f"seed {seed}")
        r["seed"] = seed
        r["n_negative"] = plan.n_negative
        rows.append(r)
        print(f"  seed {seed:>2}  AUC {r['AUC']:.4f}  AUPRC {r['AUPRC']:.4f}  "
              f"F1w {r['F1_weighted']:.4f}  MCC {r['MCC']:.4f}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    print(f"\nEvery draw keeps all {rows[0]['n_residues'] and plan.n_positive:,} positives; "
          f"only the {plan.n_negative:,} negatives differ.\n")

    stat = pd.DataFrame({
        "mean": df[METRICS].mean(), "sd": df[METRICS].std(ddof=1),
        "min": df[METRICS].min(), "max": df[METRICS].max(),
    })
    stat["range"] = stat["max"] - stat["min"]
    stat["seed 0"] = df.set_index("seed").loc[0, METRICS]
    print(stat.to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\nOver {args.seeds} draws, the context alone moves "
          f"AUC by {stat.loc['AUC', 'range']:.4f} and AUPRC by "
          f"{stat.loc['AUPRC', 'range']:.4f} peak-to-peak "
          f"(sd {stat.loc['AUC','sd']:.4f} / {stat.loc['AUPRC','sd']:.4f}).")
    print("Any reported difference smaller than ~2 sd is inside the noise of a single draw.")

    out = C.BUILD_DIR / "context_seed_variance.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
