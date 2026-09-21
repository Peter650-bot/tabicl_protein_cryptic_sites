"""Score every feature arm with a cheap sklearn model, on a CV fold, for free.

WHAT THIS IS FOR
    A TabICL screen costs about $0.11 per arm per fold. This costs nothing and
    runs in a minute, so a feature idea can be looked at before it is paid for.

WHAT IT IS NOT
    It is not the screen, and it does not decide anything. The model ladder
    (RESULTS.md section 3) has TabICL ahead of logistic regression by
    0.027-0.061 AUPRC at every feature count, and the gap itself MOVES with the
    feature set -- which is precisely the quantity a proxy cannot see. A linear
    model also has the least possible use for neighbour-pooled columns, because
    they are near-collinear with the parents it already has; TabICL is not
    linear and need not agree.

    So: a large proxy gain is a reason to look, a proxy null is not a reason to
    stop, and neither is a result. The pre-registered arms in
    `deploy/feature_grid.py` are screened on TabICL whatever this says.

NO TEST SET
    Context comes from the train folds other than `--held`, and scoring is on
    `--held`. The test split is never read.
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

from cryptic_tabicl import config as C  # noqa: E402
from cryptic_tabicl import evaluate, feature_sets as FS, sampling  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path,
                    default=C.BUILD_DIR / "residues_derived.parquet")
    ap.add_argument("--held", default="train-fold-0")
    ap.add_argument("--budget", type=int, default=C.CONTEXT_BUDGET_ROWS)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--out", type=Path,
                    default=C.PROJECT_ROOT / "feature_sweep" / "sklearn_proxy.json")
    args = ap.parse_args()

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    table = pd.read_parquet(args.table)
    train = table[table["split"] == "train"]
    ctx_src, target = train[train["fold"] != args.held], train[train["fold"] == args.held]
    ctx, plan = sampling.build_context(ctx_src, budget=args.budget,
                                       seed=C.RANDOM_SEED)
    print(f"held={args.held}  target {len(target):,} rows\n{plan}\n")

    supervised = sorted(c for c in table.columns
                        if c.startswith(FS.SUPERVISED_PREFIX)
                        and c.endswith(f"__fit_{args.held}"))
    arms = args.arms or [a for a in FS.FEATURE_SETS
                         if a not in FS.USES_SUPERVISED_ESM or supervised]

    y_tr = ctx[C.LABEL_COL].to_numpy()
    y_te = target[C.LABEL_COL].to_numpy()
    groups = target[C.GROUP_COL].to_numpy()

    rows = {}
    print(f"{'arm':<24} {'feat':>5} {'AUPRC':>8} {'AUC':>8}")
    print("-" * 48)
    for arm in arms:
        want = supervised if arm in FS.USES_SUPERVISED_ESM else ()
        cols = FS.resolve(arm, table, want)
        # A linear model cannot take the two categoricals; they are constant
        # across arms, so dropping them changes no comparison.
        num = [c for c in cols if table[c].dtype.kind in "fiu"]
        pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                             LogisticRegression(max_iter=2000))
        t0 = time.time()
        pipe.fit(ctx[num].to_numpy(float), y_tr)
        s = pipe.predict_proba(target[num].to_numpy(float))[:, 1]
        m = evaluate.summarise(y_te, s, groups, 0.5, label=arm)
        rows[arm] = {"AUPRC": m["AUPRC"], "AUC": m["AUC"],
                     "n_features": len(cols), "n_numeric": len(num),
                     "seconds": time.time() - t0}
        print(f"{arm:<24} {len(cols):>5} {m['AUPRC']:>8.4f} {m['AUC']:>8.4f}")

    base = rows.get("control", {}).get("AUPRC")
    if base is not None:
        print(f"\ndelta vs control (proxy only, not the screen):")
        for arm in arms:
            if arm != "control":
                print(f"  {arm:<24} {rows[arm]['AUPRC'] - base:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": "LogisticRegression", "held": args.held, "seed": C.RANDOM_SEED,
         "budget": args.budget, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
