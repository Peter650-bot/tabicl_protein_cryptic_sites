"""Step 8f of finetuned_esm/PLAN.md: the stacked arms, reported exactly as frozen_step8.json says.

ORDER OF OPERATIONS IS THE PROTOCOL
    1. Thresholds come from the CV stage alone (per fold at mean per-protein FPR
       0.05, frozen = median of four), computed and written BEFORE a test score
       is read.
    2. Test: both arms and the three references on the same 231 proteins,
       paired bootstrap, the four pre-registered claims evaluated by their
       pre-registered criteria.

Run: python scripts/s2p_stack_report.py --run-id st1                 # step 8
     python scripts/s2p_stack_report.py --run-id st2 --step 9        # step 9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402
from s2p_evaluate import pooled  # noqa: E402

FOLDS = [f"train-fold-{h}" for h in range(4)]
N_BOOT, N_BOOT_POOLED, SEED = 10_000, 2_000, 0
THR_METRICS = ("TPR", "FPR", "MCC", "F1", "F1_weighted")
# Everything a step's pre-registration fixes. Claims: name -> (arm, reference); the
# criterion is always a paired 95 % CI on per-protein AUPRC over CB-full above zero.
STEPS = {
    8: {"frozen": "finetuned_esm/frozen_step8.json",
        "sha256": "205e34cb4f2e77bd9a6f78fa8b84343a673183e2b4b63caa8fe8ec4c6f0ac8e2",
        "arms": ("s2p_117", "s2p_53"), "table": "residues_s2p.parquet",
        "component": ("s2p_oof", "plain_alone"), "extra": {},
        "claims": {"primary: s2p_117 - plain_alone": ("s2p_117", "plain_alone"),
                   "secondary_1: s2p_117 - tabicl_117": ("s2p_117", "tabicl_117"),
                   "secondary_2: s2p_117 - fixed_blend": ("s2p_117", "fixed_blend"),
                   "secondary_3: s2p_117 - s2p_53": ("s2p_117", "s2p_53")}},
    9: {"frozen": "finetuned_esm/frozen_step9.json",
        "sha256": "c61241d2746aeeabf59783c74c40dd88ec13ad84f717d117ad1e3fd83049d656",
        "arms": ("s2pfull_117", "s2pfull_53"), "table": "residues_s2pfull.parquet",
        "component": ("s2pfull_oof", "full_alone"),
        "extra": {"s2p_117": "build/s2p_stack/st1/s2p_117_final.json"},
        "claims": {"primary: s2pfull_117 - full_alone": ("s2pfull_117", "full_alone"),
                   "secondary_1: s2pfull_117 - tabicl_117": ("s2pfull_117", "tabicl_117"),
                   "secondary_2: s2pfull_117 - fixed_blend": ("s2pfull_117", "fixed_blend"),
                   "secondary_3: s2pfull_117 - s2pfull_53": ("s2pfull_117", "s2pfull_53"),
                   "secondary_4: s2pfull_117 - s2p_117": ("s2pfull_117", "s2p_117")}},
}
CFG = STEPS[8]
ARMS = CFG["arms"]
CLAIMS = CFG["claims"]


def cv_thresholds(run_dir: Path) -> dict:
    out = {}
    for arm in ARMS:
        per_fold, metrics = [], []
        for fold in FOLDS:
            r = json.loads((run_dir / f"{arm}_{fold}.json").read_text())
            y, s, g = np.array(r["y"]), np.array(r["score"]), np.array(r["apo_id"])
            t = evaluate.choose_threshold(y, s, g)
            m = evaluate.summarise(y, s, g, t, f"{arm}/{fold}")
            per_fold.append(float(t))
            metrics.append({k: m[k] for k in ("AUC", "AUPRC", "TPR", "FPR", "MCC",
                                              "AUC_pooled", "AUPRC_pooled")} | {"fold": fold})
        out[arm] = {"fold_thresholds": per_fold, "frozen_threshold": float(np.median(per_fold)),
                    "cv": metrics}
    return out


def load_test(run_dir: Path) -> pd.DataFrame:
    blob = json.loads((C.BUILD_DIR / "modal_result_esm64.json").read_text())
    df = pd.DataFrame({k: blob["scores"][k] for k in ("apo_id", "chain", "seq_id", "y")})
    df["tabicl_117"] = blob["scores"]["score"]
    for arm in ARMS:
        r = json.loads((run_dir / f"{arm}_final.json").read_text())
        for c in ("apo_id", "chain", "seq_id", "y"):
            assert list(r[c]) == df[c].tolist(), f"{arm}: test rows differ on {c}"
        df[arm] = r["score"]
    col, ref = CFG["component"]
    stack = pd.read_parquet(C.BUILD_DIR / CFG["table"],
                            columns=["apo_id", "chain", "seq_id", "split", C.LABEL_COL, col])
    test = stack[stack["split"] == "test"].reset_index(drop=True)
    for c in ("apo_id", "chain", "seq_id"):
        assert (test[c].to_numpy() == df[c].to_numpy()).all(), f"component rows differ on {c}"
    df[ref] = test[col].to_numpy()
    for name, path in CFG["extra"].items():
        r = json.loads(Path(path).read_text())
        assert list(r["apo_id"]) == df["apo_id"].tolist() and list(r["seq_id"]) == df["seq_id"].tolist()
        df[name] = r["score"]
    g = df.groupby("apo_id")
    df["fixed_blend"] = 0.5 * g["tabicl_117"].rank(method="average", pct=True) \
        + 0.5 * g[ref].rank(method="average", pct=True)
    return df


def paired(df: pd.DataFrame, names: list, rng) -> dict:
    y, g = df["y"].to_numpy(), df["apo_id"].to_numpy()
    pp = {n: evaluate.per_protein(y, df[n].to_numpy(), g, 0.5).set_index("apo_id")[["AUC", "AUPRC"]]
          for n in names}
    prots = pp[names[0]].index
    n = len(prots)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = {k: np.nanmean(v.to_numpy(float)[idx], axis=1) for k, v in pp.items()}
    point = {k: np.nanmean(v.to_numpy(float), axis=0) for k, v in pp.items()}
    rows_of = {p: np.flatnonzero(g == p) for p in prots}
    plist = np.array(list(rows_of))
    arr = {k: df[k].to_numpy() for k in names}
    pb = {k: np.empty((N_BOOT_POOLED, 2)) for k in names}
    for b in range(N_BOOT_POOLED):
        r = np.concatenate([rows_of[p] for p in rng.choice(plist, n)])
        for k in names:
            pb[k][b] = pooled(y[r], arr[k][r])
    ppt = {k: pooled(y, arr[k]) for k in names}
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    out = {"n_proteins": n}
    for kind, pt, bt in (("per_protein", point, boot), ("pooled", ppt, pb)):
        for j, m in enumerate(("AUC", "AUPRC")):
            block = {k: {"value": float(pt[k][j]), "ci": ci(bt[k][:, j])} for k in names}
            for claim, (a, ref) in CLAIMS.items():
                block[f"{a} - {ref}"] = {"diff": float(pt[a][j] - pt[ref][j]),
                                         "ci": ci(bt[a][:, j] - bt[ref][:, j])}
            out[f"{kind}_{m}"] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--step", type=int, choices=sorted(STEPS), default=8)
    a = ap.parse_args()
    global CFG, ARMS, CLAIMS
    CFG = STEPS[a.step]
    ARMS, CLAIMS = CFG["arms"], CFG["claims"]
    if hashlib.sha256(Path(CFG["frozen"]).read_bytes()).hexdigest() != CFG["sha256"]:
        raise SystemExit(f"{CFG['frozen']} changed since pre-registration")
    run_dir = C.BUILD_DIR / "s2p_stack" / a.run_id

    # ---- 1. thresholds from CV only, written before test is read
    thr = cv_thresholds(run_dir)
    Path(f"finetuned_esm/step{a.step}_thresholds.json").write_text(json.dumps(thr, indent=1))
    for arm, t in thr.items():
        print(f"{arm}: fold thresholds {[round(x, 4) for x in t['fold_thresholds']]} "
              f"-> frozen {t['frozen_threshold']:.4f}")
        for m in t["cv"]:
            print(f"   {m['fold']}: AUC {m['AUC']:.3f} AUPRC {m['AUPRC']:.3f} "
                  f"FPR {m['FPR']:.4f} TPR {m['TPR']:.3f}")

    # ---- 2. test, once
    rng = np.random.default_rng(SEED)
    df = load_test(run_dir)
    names = list(ARMS) + ["tabicl_117", CFG["component"][1], "fixed_blend"] + list(CFG["extra"])
    single = single_chain_pdbs()
    res = {"run_id": a.run_id, "step": a.step, "frozen_sha256": CFG["sha256"], "thresholds": thr}
    for subset, sub in (("CB-full", df), ("CB-P2RANK-apo", df[df["apo_id"].str[:4].str.lower().isin(single)])):
        sub = sub.reset_index(drop=True)
        res[subset] = paired(sub, names, rng)
        y, g = sub["y"].to_numpy(), sub["apo_id"].to_numpy()
        res[subset]["threshold_metrics"] = {
            arm: {k: v for k, v in evaluate.summarise(
                y, sub[arm].to_numpy(), g, thr[arm]["frozen_threshold"], arm).items()
                if k in THR_METRICS} for arm in ARMS}
        res[subset]["threshold_metrics"]["tabicl_117 (published 0.5706)"] = {
            k: v for k, v in evaluate.summarise(y, sub["tabicl_117"].to_numpy(), g, 0.5705558061599731,
                                                "tabicl").items() if k in THR_METRICS}

    full = res["CB-full"]["per_protein_AUPRC"]
    res["claims"] = {c: {**full[f"{a_} - {r_}"], "passes": full[f"{a_} - {r_}"]["ci"][0] > 0}
                     for c, (a_, r_) in CLAIMS.items()}
    Path(f"finetuned_esm/step{a.step}_result.json").write_text(json.dumps(res, indent=1))

    for subset in ("CB-full", "CB-P2RANK-apo"):
        r = res[subset]
        print(f"\n== {subset} ({r['n_proteins']} proteins)")
        print(f"{'':20s}" + "".join(f"{n:>14s}" for n in names))
        for key in ("per_protein_AUC", "per_protein_AUPRC", "pooled_AUC", "pooled_AUPRC"):
            print(f"{key:20s}" + "".join(f"{r[key][n]['value']:>14.3f}" for n in names))
            for c, (a_, r_) in CLAIMS.items():
                d = r[key][f"{a_} - {r_}"]
                star = " *" if d["ci"][0] > 0 or d["ci"][1] < 0 else ""
                print(f"{'':22s}{a_+' - '+r_:30s} {d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]{star}")
        print("threshold metrics (stacked arms at their CV-frozen thresholds):")
        for k, v in r["threshold_metrics"].items():
            print(f"   {k:30s} " + "  ".join(f"{m} {v[m]:.3f}" for m in THR_METRICS))
    print("\nCLAIMS (per-protein AUPRC, CB-full, paired 95 % CI > 0):")
    for c, v in res["claims"].items():
        print(f"  {c:40s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  "
              f"{'PASSES' if v['passes'] else 'does not pass'}")


if __name__ == "__main__":
    main()
