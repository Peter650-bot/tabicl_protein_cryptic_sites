"""Per-subset metrics for our two headline variants, including the F1 column
in the form the CryptoBench paper publishes it.

WHY
    The published table's "F1 Score" is a **weighted average over both
    classes** (paper: F1 "computed using a weighted average"). At 5.28 %
    prevalence that is dominated by the negative class, which is why pLM-NN's
    published F1 (0.92) tracks its ACC (0.93) rather than its TPR (0.48).
    Our positive-class F1 is ~0.40 and is NOT comparable to that column.
    This script emits both, so the figure can plot the comparable one and the
    write-up can quote the informative one.

    CB-P2RANK-apo is reconstructed the way the paper defines it: the
    single-chain test structures (222 test entries - 38 multi-chain = 184),
    where "single-chain" means every residue in the entry's apo pocket
    selections lies on one chain.

    CB-PM is NOT reconstructible -- it is the subset PocketMiner happened to
    run on, a property of that tool's run rather than of the dataset.

Run:  python scripts/f1_by_subset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C, evaluate  # noqa: E402

VARIANTS = [
    ("modal_result_esm64.json", "TabICLv2 + ESM (118 feat)"),
    ("modal_result_esm0.json",  "TabICLv2, no pLM (54 feat)"),
]

# Published CryptoBench table (external/CryptoBench/README.md). F1 is weighted.
PUBLISHED = [
    ("pLM-NN",      "CB-full",        dict(AUC=.86, AUPRC=.36, ACC=.93, FPR=.05, TPR=.48, MCC=.39, F1=.92)),
    ("pLM-NN",      "CB-PM",          dict(AUC=.88, AUPRC=.43, ACC=.93, FPR=.04, TPR=.52, MCC=.44, F1=.93)),
    ("PocketMiner", "CB-PM",          dict(AUC=.76, AUPRC=.19, ACC=.82, FPR=.16, TPR=.51, MCC=.22, F1=.78)),
    ("pLM-NN",      "CB-P2RANK-apo",  dict(AUC=.88, AUPRC=.42, ACC=.93, FPR=.04, TPR=.51, MCC=.43, F1=.93)),
    ("P2Rank",      "CB-P2RANK-apo",  dict(AUC=.81, AUPRC=.21, ACC=.85, FPR=.14, TPR=.62, MCC=.27, F1=.81)),
    ("P2Rank",      "CB-P2RANK-holo", dict(AUC=.89, AUPRC=.34, ACC=.85, FPR=.15, TPR=.84, MCC=.38, F1=.81)),
]


def single_chain_pdbs() -> set[str]:
    """CB-P2RANK-apo: test entries whose pocket selections touch exactly one chain."""
    test = json.loads((C.FOLDS_DIR / "test.json").read_text())
    return {
        pdb.lower()
        for pdb, entries in test.items()
        if len({res.split("_")[0] for e in entries for res in e["apo_pocket_selection"]}) == 1
    }


def main() -> None:
    single = single_chain_pdbs()
    print(f"CB-P2RANK-apo reconstruction: {len(single)} single-chain test entries "
          f"({222 - len(single)} multi-chain dropped)\n")

    rows = []
    for fname, label in VARIANTS:
        blob = json.loads((C.BUILD_DIR / fname).read_text())
        df = pd.DataFrame(blob["scores"])
        thr = blob["threshold"]
        df["pdb"] = df["apo_id"].str[:4].str.lower()

        for subset, mask in (("CB-full", slice(None)),
                             ("CB-P2RANK-apo", df["pdb"].isin(single))):
            sub = df if subset == "CB-full" else df[mask]
            r = evaluate.summarise(sub["y"], sub["score"], sub["apo_id"], thr, label)
            r["subset"] = subset
            rows.append(r)

    out = pd.DataFrame(rows)[["method", "subset", "n_proteins", "AUC", "AUPRC",
                              "ACC", "TPR", "FPR", "MCC", "F1_weighted", "F1"]]
    pd.set_option("display.width", 160)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nPublished (external/CryptoBench/README.md) — F1 is the weighted column:")
    pub = pd.DataFrame([{"method": m, "subset": s, **v} for m, s, v in PUBLISHED])
    print(pub.to_string(index=False))

    print("\nOur F1_weighted is the column comparable to their F1.")
    print("Our positive-class F1 (last column) is NOT — do not place it beside theirs.")


if __name__ == "__main__":
    main()
