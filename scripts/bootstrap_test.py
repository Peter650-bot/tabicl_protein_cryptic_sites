"""Bootstrap CIs over the TEST SET — free, because the scores are already stored.

WHY THIS IS A DIFFERENT QUESTION FROM THE CONTEXT DRAW
    Two independent uncertainties sit under every headline number:

      test-set    would this ordering hold on another 231 apo structures?
      context     how much did the arbitrary 50k negative draw move it?

    Only the first is a bootstrap in the statistical sense, and it is the one a
    comparative claim ("we beat pLM-NN") actually rests on. It costs nothing:
    every reported metric is a mean over per-protein values, so resampling the
    231 proteins with replacement and re-averaging the precomputed per-protein
    table is exact -- no re-scoring, no GPU.

    The context draw needs GPU replicates (scripts/context_seed_variance.py
    estimates it with a linear stand-in; deploy/ does it properly for TabICL).

Run:  python scripts/bootstrap_test.py --b 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from scripts.f1_by_subset import single_chain_pdbs  # noqa: E402

VARIANTS = [("modal_result_esm64.json", "TabICLv2 + ESM (118 feat)"),
            ("modal_result_esm0.json",  "TabICLv2, no pLM (54 feat)")]
METRICS = ["AUC", "AUPRC", "TPR", "FPR", "MCC", "F1_weighted"]
PUBLISHED = {           # pLM-NN, for "does their point estimate sit inside ours?"
    "CB-full":       {"AUC": .86, "AUPRC": .36, "TPR": .48, "FPR": .05, "MCC": .39, "F1_weighted": .92},
    "CB-P2RANK-apo": {"AUC": .88, "AUPRC": .42, "TPR": .51, "FPR": .04, "MCC": .43, "F1_weighted": .93},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    single = single_chain_pdbs()
    out = []

    for fname, label in VARIANTS:
        blob = json.loads((C.BUILD_DIR / fname).read_text())
        df = pd.DataFrame(blob["scores"])
        thr = blob["threshold"]
        df["pdb"] = df["apo_id"].str[:4].str.lower()

        for subset in ("CB-full", "CB-P2RANK-apo"):
            sub = df if subset == "CB-full" else df[df["pdb"].isin(single)]
            pp = evaluate.per_protein(sub["y"], sub["score"], sub["apo_id"], thr)
            M = pp[METRICS].to_numpy(float)                 # (n_proteins, n_metrics)
            n = len(M)
            idx = rng.integers(0, n, size=(args.b, n))
            # nanmean: AUC/AUPRC are NaN for single-class proteins, as in summarise()
            draws = np.nanmean(M[idx], axis=1)              # (B, n_metrics)
            point = np.nanmean(M, axis=0)
            lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)

            print(f"\n{label} — {subset}  (n = {n} proteins, B = {args.b:,})")
            print(f"  {'metric':<12} {'point':>7} {'95% CI':>18} {'half-width':>11}   pLM-NN")
            for j, m in enumerate(METRICS):
                pub = PUBLISHED[subset][m]
                inside = "inside" if lo[j] <= pub <= hi[j] else ("BELOW CI" if pub < lo[j] else "ABOVE CI")
                print(f"  {m:<12} {point[j]:>7.3f} {'[' + f'{lo[j]:.3f}, {hi[j]:.3f}' + ']':>18} "
                      f"{(hi[j]-lo[j])/2:>11.3f}   {pub:.2f} {inside}")
                out.append(dict(variant=label, subset=subset, metric=m, n_proteins=n,
                                point=float(point[j]), lo=float(lo[j]), hi=float(hi[j]),
                                plmnn=pub, plmnn_inside_ci=bool(lo[j] <= pub <= hi[j])))

    (C.BUILD_DIR / "bootstrap_test.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {C.BUILD_DIR / 'bootstrap_test.json'}")
    print("\nCI is over TEST PROTEINS only. It does not include the context draw,")
    print("and it is not a CI on the DIFFERENCE vs pLM-NN — only their point")
    print("estimate is published, so the best available check is containment.")


if __name__ == "__main__":
    main()
