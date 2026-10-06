"""Step 7 of finetuned_esm/PLAN.md: the pre-registered late-fusion blends, scored once.

Everything here is fixed by finetuned_esm/frozen.json (committed in 245448a before
this script existed). The file's hash is checked first, so a later edit to the
rule cannot be scored under the pre-registered name.

    rank (primary)    0.5 * within-protein percentile rank of each score
    prob (secondary)  0.5 * tabicl + 0.5 * s2p

Nothing is fitted. AUC/AUPRC are the result; threshold metrics use the CV-frozen
"top 7.74 % per protein" rule and are labelled approximate.

Run: python scripts/s2p_blend.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryptic_tabicl import config as C, evaluate, operating_point as op  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402
from s2p_evaluate import load, pooled  # noqa: E402

FROZEN = Path("finetuned_esm/frozen.json")
FROZEN_SHA256 = "9aa0ebe8630b5b1f81e630075a01aa8005ae51fbe69f0a02bb14c0cfc8e2622f"
N_BOOT, N_BOOT_POOLED, SEED = 10_000, 2_000, 0
ARMS = ("s2p", "tabicl", "rank", "prob")
APPROX = ("TPR", "FPR", "MCC", "F1_weighted")


def add_blends(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("apo_id")
    # rankdata(method='average') / n  ==  pandas rank(method='average', pct=True)
    df["rank"] = 0.5 * g["tabicl"].rank(method="average", pct=True) \
        + 0.5 * g["s2p"].rank(method="average", pct=True)
    df["prob"] = 0.5 * df["tabicl"] + 0.5 * df["s2p"]
    return df


def ci(x) -> list:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def score_subset(df: pd.DataFrame, rank_thr: float, rng) -> dict:
    y, g = df["y"].to_numpy(), df["apo_id"].to_numpy()
    pp, approx = {}, {}
    for a in ARMS:
        s = df[a].to_numpy()
        pp[a] = evaluate.per_protein(y, s, g, 0.5).set_index("apo_id")[["AUC", "AUPRC"]]
        top = evaluate.per_protein(y, op.transform("rank", s, g), g, rank_thr)
        approx[a] = {m: float(top[m].mean(skipna=True)) for m in APPROX}
    prots = pp["s2p"].index
    assert all(list(pp[a].index) == list(prots) for a in ARMS)

    n = len(prots)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = {a: np.nanmean(pp[a].to_numpy(float)[idx], axis=1) for a in ARMS}
    point = {a: np.nanmean(pp[a].to_numpy(float), axis=0) for a in ARMS}

    rows_of = {p: np.flatnonzero(g == p) for p in prots}
    plist = np.array(list(rows_of))
    pboot = {a: np.empty((N_BOOT_POOLED, 2)) for a in ARMS}
    score_of = {a: df[a].to_numpy() for a in ARMS}
    for b in range(N_BOOT_POOLED):
        r = np.concatenate([rows_of[p] for p in rng.choice(plist, n)])
        for a in ARMS:
            pboot[a][b] = pooled(y[r], score_of[a][r])
    ppoint = {a: pooled(y, df[a].to_numpy()) for a in ARMS}

    out = {"n_proteins": n, "per_protein": {}, "pooled": {}, "approximate_threshold_metrics": approx}
    for kind, pt, bt in (("per_protein", point, boot), ("pooled", ppoint, pboot)):
        for j, m in enumerate(("AUC", "AUPRC")):
            block = {a: {"value": float(pt[a][j]), "ci": ci(bt[a][:, j])} for a in ARMS}
            for blend in ("rank", "prob"):
                for ref in ("s2p", "tabicl"):
                    d = bt[blend][:, j] - bt[ref][:, j]
                    block[f"{blend}_minus_{ref}"] = {"diff": float(pt[blend][j] - pt[ref][j]),
                                                     "ci": ci(d)}
            out[kind][m] = block
    return out


def main() -> None:
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        raise SystemExit(f"frozen.json changed since pre-registration ({digest})")
    rank_thr = json.loads((C.PROJECT_ROOT / "operating_point/frozen.json").read_text()
                          )["rules"]["rank"]["frozen_threshold"]

    rng = np.random.default_rng(SEED)
    df = add_blends(load())
    single = single_chain_pdbs()
    res = {"frozen_sha256": digest, "rank_rule_threshold": rank_thr}
    for subset, sub in (("CB-full", df),
                        ("CB-P2RANK-apo", df[df["apo_id"].str[:4].str.lower().isin(single)])):
        res[subset] = score_subset(sub.reset_index(drop=True), rank_thr, rng)

    prim = res["CB-full"]["per_protein"]["AUPRC"]["rank_minus_s2p"]
    sec = res["CB-P2RANK-apo"]["pooled"]["AUPRC"]["prob_minus_s2p"]
    res["decision"] = {
        "primary (rank - s2p, per-protein AUPRC, CB-full)": {**prim, "passes": prim["ci"][0] > 0},
        "secondary (prob - s2p, pooled AUPRC, CB-P2RANK-apo)": {**sec, "passes": sec["ci"][0] > 0},
    }
    Path("finetuned_esm/step7_result.json").write_text(json.dumps(res, indent=1))

    for subset in ("CB-full", "CB-P2RANK-apo"):
        r = res[subset]
        print(f"\n== {subset} ({r['n_proteins']} proteins)")
        print(f"{'':16s}" + "".join(f"{a:>22s}" for a in ARMS))
        for kind in ("per_protein", "pooled"):
            for m in ("AUC", "AUPRC"):
                b = r[kind][m]
                print(f"{kind+' '+m:16s}" + "".join(
                    f"   {b[a]['value']:.3f} [{b[a]['ci'][0]:.3f},{b[a]['ci'][1]:.3f}]" for a in ARMS))
                for k in ("rank_minus_s2p", "rank_minus_tabicl", "prob_minus_s2p", "prob_minus_tabicl"):
                    d = b[k]
                    star = " *" if d["ci"][0] > 0 or d["ci"][1] < 0 else ""
                    print(f"{'':18s}{k:20s} {d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]{star}")
        print("approximate threshold metrics (top 7.74 % per protein):")
        for a in ARMS:
            v = r["approximate_threshold_metrics"][a]
            print(f"  {a:7s} " + "  ".join(f"{m} {v[m]:.3f}" for m in APPROX))
    print("\nDECISION:", json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
