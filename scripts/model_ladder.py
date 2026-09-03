"""Every model configuration we ran on CB-full, in one tidy table.

WHY
    The headline comparison (scripts/compare_variants.py, figures/) is against
    the published baselines, and those exist on CB-full / CB-PM /
    CB-P2RANK-apo. Our own sklearn controls exist on CB-full ONLY -- the stored
    baseline artifacts keep summary metrics, not per-residue scores, so they
    cannot be re-scored on a subset. This script assembles them so the model
    families can be compared to each other and to TabICL at equal footing.

    The comparison that matters is WITHIN a condition (same features, same
    negatives): that isolates model choice. Across conditions the features and
    the negative sampler move, and those moves are much larger.

Run:  python scripts/model_ladder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C  # noqa: E402

# build file -> (n_features, sampler era of its "(context)" rows)
BASELINES = [
    ("baselines.json",       34,  "old"),
    ("baselines_ens.json",   48,  "old"),
    ("baselines_cons.json",  54,  "fixed"),
    ("baselines_esm16.json", 70,  "fixed"),
    ("baselines_esm32.json", 86,  "fixed"),
    ("baselines_esm64.json", 118, "fixed"),
]
# TabICL runs: build file -> (n_features, sampler era)
TABICL = [
    ("modal_result_tierA.json",     34,  "old"),
    ("modal_result_ensemble.json",  48,  "fixed"),
    ("modal_result_esm0.json",      54,  "fixed"),
    ("modal_result_esm16.json",     70,  "fixed"),
    ("modal_result_esm32.json",     86,  "fixed"),
    ("modal_result_esm64.json",     118, "fixed"),
]
# Rows quoted in RESULTS.md that have no stored artifact under build/.
NOT_STORED = [
    "TabICLv2 zero-shot, 48 feat, old sampler 50k  (RESULTS.md: AUC 0.839 / AUPRC 0.323)",
    "LogisticRegression, 48 feat, fixed sampler 50k (RESULTS.md: AUC 0.848 / AUPRC 0.422)",
    "TabICLv2 zero-shot x5 contexts, 48 feat, old sampler (RESULTS.md: AUC 0.840 / AUPRC 0.327)",
]

METRICS = ("AUC", "AUPRC", "TPR", "FPR", "MCC", "F1_weighted")


def rows() -> list[dict]:
    out = []
    for fname, nfeat, era in BASELINES:
        for r in json.loads((C.BUILD_DIR / fname).read_text()):
            m = r["method"]
            if m.startswith("RSA"):
                if nfeat == 118:          # identical in every file; keep one
                    out.append(dict(family="RSA alone (floor)", n_features=1,
                                    negatives="—", condition="floor",
                                    **{k: r.get(k) for k in METRICS}))
                continue
            family = "LogisticRegression" if m.startswith("Logistic") else "RandomForest"
            if "(context)" in m:
                neg, cond = f"50k context · {era} sampler", f"{nfeat}f · 50k {era}"
            else:
                neg, cond = "full 304k train", f"{nfeat}f · full 304k"
            out.append(dict(family=family, n_features=nfeat, negatives=neg,
                            condition=cond, **{k: r.get(k) for k in METRICS}))

    for fname, nfeat, era in TABICL:
        blob = json.loads((C.BUILD_DIR / fname).read_text())
        r = blob["result"]
        out.append(dict(family="TabICLv2 zero-shot", n_features=nfeat,
                        negatives=f"50k context · {era} sampler",
                        condition=f"{nfeat}f · 50k {era}",
                        **{k: r.get(k) for k in METRICS}))
    return out


def main() -> None:
    import pandas as pd
    df = pd.DataFrame(rows())
    df = df.sort_values(["n_features", "condition", "family"]).reset_index(drop=True)

    pd.set_option("display.width", 170)
    print(df.to_string(index=False,
                       float_format=lambda v: f"{v:.3f}" if v == v else "—"))

    print("\nBy condition (the within-condition spread is the model effect):")
    piv = df[df.family != "RSA alone (floor)"].pivot_table(
        index="condition", columns="family", values="AUPRC")
    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
    print(piv.to_string(float_format=lambda v: f"{v:.3f}" if v == v else "—"))

    out = C.BUILD_DIR / "model_ladder.json"
    out.write_text(json.dumps(df.where(df.notna(), None).to_dict("records"), indent=1))
    print(f"\nwrote {out}")
    print("\nQuoted in RESULTS.md but NOT stored under build/ (absent from the figure):")
    for n in NOT_STORED:
        print("  -", n)


if __name__ == "__main__":
    main()
