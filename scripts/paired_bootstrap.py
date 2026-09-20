"""Paired bootstrap CI on the difference between two decision rules.

WHY PAIRED
    Both rules score the SAME 231 test proteins from the SAME scores; only the
    threshold rule differs. Comparing two independent intervals from
    `bootstrap_test.py` would throw that pairing away and badly overstate the
    uncertainty, because most of the variance is "which proteins are in the test
    set", which is common to both arms and cancels in the difference.

    So: resample proteins with replacement, recompute BOTH rules' per-protein
    means on the same resample, and take a percentile interval on the paired
    difference.

WHAT IT CANNOT TELL YOU
    AUC and AUPRC are identical across rules by construction, so their intervals
    are exactly zero. That is a correctness check, not a finding.

USAGE
    python scripts/paired_bootstrap.py                       # secondary vs headline
    python scripts/paired_bootstrap.py --a rank --b zscore
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config, evaluate, operating_point as op

OUT_DIR = config.PROJECT_ROOT / "operating_point"
FROZEN = OUT_DIR / "frozen.json"
TEST_RESULT = config.BUILD_DIR / "modal_result_esm64.json"
METRICS = ("TPR", "FPR", "MCC", "F1", "F1_weighted", "AUC", "AUPRC")


def _per_protein(rule: str, thr: float, y, s, g) -> pd.DataFrame:
    return evaluate.per_protein(y, op.transform(rule, s, g), g, thr).set_index("apo_id")


def main(a: str, b: str, n_boot: int, seed: int) -> dict:
    frozen = json.loads(FROZEN.read_text())
    block = json.loads(TEST_RESULT.read_text())["scores"]
    y = np.asarray(block["y"], dtype=int)
    s = np.asarray(block["score"], dtype=float)
    g = np.asarray(block["apo_id"])

    ta = frozen["rules"][a]["frozen_threshold"]
    tb = frozen["rules"][b]["frozen_threshold"]
    pa, pb = _per_protein(a, ta, y, s, g), _per_protein(b, tb, y, s, g)
    assert list(pa.index) == list(pb.index)

    A = pa[list(METRICS)].to_numpy(dtype=float)
    B = pb[list(METRICS)].to_numpy(dtype=float)
    n = len(A)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    # nan-aware means (AUC/AUPRC are NaN for single-class proteins)
    def boot(M):
        sel = M[idx]                      # (n_boot, n, n_metrics)
        return np.nanmean(sel, axis=1)    # (n_boot, n_metrics)

    dist = boot(B) - boot(A)
    point = np.nanmean(B, axis=0) - np.nanmean(A, axis=0)
    lo, hi = np.percentile(dist, [2.5, 97.5], axis=0)

    print(f"paired bootstrap, {n_boot} resamples of {n} proteins, seed {seed}")
    print(f"  A = {a}  (threshold {ta:.4f}, {op.describe(a, ta)})")
    print(f"  B = {b}  (threshold {tb:.4f}, {op.describe(b, tb)})")
    print(f"\n{'metric':<12} {'A':>8} {'B':>8} {'B - A':>9} {'95% CI':>22} {'':>4}")
    rows = {}
    for i, m in enumerate(METRICS):
        excl = (lo[i] > 0) or (hi[i] < 0)
        mark = "*" if excl else ""
        print(f"{m:<12} {np.nanmean(A[:, i]):8.4f} {np.nanmean(B[:, i]):8.4f} "
              f"{point[i]:+9.4f}   [{lo[i]:+.4f}, {hi[i]:+.4f}] {mark:>3}")
        rows[m] = {"a": float(np.nanmean(A[:, i])), "b": float(np.nanmean(B[:, i])),
                   "delta": float(point[i]), "ci_low": float(lo[i]),
                   "ci_high": float(hi[i]), "excludes_zero": bool(excl)}
    print("\n* = 95% CI excludes zero. AUC/AUPRC are identical by construction; "
          "their zero interval is a correctness check, not a finding.")

    out = {"a": a, "b": b, "n_boot": n_boot, "seed": seed, "n_proteins": int(n),
           "metrics": rows}
    (OUT_DIR / "paired_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'paired_bootstrap.json'}")
    return out


if __name__ == "__main__":
    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=frozen.get("headline_rule", "global_threshold"))
    ap.add_argument("--b", default=frozen.get("secondary_rule", "zscore"))
    ap.add_argument("--b-resamples", dest="n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.a, args.b, args.n_boot, args.seed)
