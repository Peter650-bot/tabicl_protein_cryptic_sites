"""Per-stratum AUC noise across the stored context-bootstrap seeds.

WHY
    § 5.1 of RESULTS.md compares the "lines a non-cryptic pocket" stratum
    against the bulk surface. That stratum is 4.4 % of the test set, so a
    difference read off a single context draw can be pure resampling noise.
    deploy/context_bootstrap.py already stores every seed's per-residue
    scores, so the error bar costs nothing but a re-read.

Run:  python scripts/stratum_seed_noise.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=Path,
                    default=C.BUILD_DIR / "context_bootstrap.json")
    args = ap.parse_args()

    d = json.loads(args.bootstrap.read_text())
    y = np.asarray(d["identity"]["y"], dtype=int)
    flag = np.asarray(d["identity"]["in_noncryptic_pocket"], dtype=int).astype(bool)
    seeds = sorted(d["scores_by_seed"], key=int)
    print(f"{len(seeds)} seeds  |  bulk {int((~flag).sum()):,} residues "
          f"({y[~flag].mean():.3%} positive)  |  hard {int(flag.sum()):,} "
          f"({y[flag].mean():.3%} positive)\n")

    rows = []
    for s in seeds:
        sc = np.asarray(d["scores_by_seed"][s], dtype=float)
        bulk, hard = roc_auc_score(y[~flag], sc[~flag]), roc_auc_score(y[flag], sc[flag])
        rows.append({"seed": int(s), "bulk surface": bulk,
                     "lines a non-cryptic pocket": hard, "gap": bulk - hard})

    df = pd.DataFrame(rows).set_index("seed")
    stat = pd.DataFrame({"mean": df.mean(), "sd": df.std(ddof=1),
                         "min": df.min(), "max": df.max()})
    stat["seed 0"] = df.loc[0] if 0 in df.index else np.nan
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(stat.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n| stratum | mean | sd | min – max |\n|---|---|---|---|")
    for name in ("bulk surface", "lines a non-cryptic pocket", "gap"):
        r = stat.loc[name]
        print(f"| {name} | {r['mean']:.4f} | {r['sd']:.4f} | "
              f"{r['min']:.4f} – {r['max']:.4f} |")

    ratio = stat.loc["lines a non-cryptic pocket", "sd"] / stat.loc["bulk surface", "sd"]
    print(f"\nThe hard stratum's AUC is {ratio:.1f}x noisier under a change of "
          f"context draw than the bulk surface's.")


if __name__ == "__main__":
    main()
