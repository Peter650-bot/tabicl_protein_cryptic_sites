"""Step 10 of finetuned_esm/PLAN.md: TabICL on the fine-tuned embedding, reported as
frozen_step10.json says.

ORDER OF OPERATIONS IS THE PROTOCOL
    1. Thresholds from the CV stage alone: per fold model k, the out-of-fold scores of
       its four protein sub-folds give a threshold at mean per-protein FPR 0.05; the
       frozen threshold is the median of the four. Written BEFORE a test score is read.
    2. Test: each arm's score is the mean of its four cross-fitted predictions
       (k = 0..3); the component reference `ft_alone` is the mean of the four fold
       models' own scores. Paired bootstrap; the pre-registered claims by their
       pre-registered criterion.

Run: python scripts/ftemb_report.py --run-id fe1
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
import s2p_stack_report as R  # noqa: E402
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

FROZEN = "finetuned_esm/frozen_step10.json"
FROZEN_SHA256 = "707f56c46d301f30699dbdb3dacd3dfa3c1c4c1e8446561e30c0333e7c1b6d19"
ARMS = ("ft117", "ft64only", "score118", "control")
KS = range(4)
CLAIMS = {"primary: ft117 - control": ("ft117", "control"),
          "secondary_1: ft117 - score118": ("ft117", "score118"),
          "secondary_2: ft64only - ft_alone": ("ft64only", "ft_alone")}
CONTEXT = {"score118 - control": ("score118", "control"),
           "ft117 - s2pfull_117": ("ft117", "s2pfull_117"),
           "score118 - s2pfull_117": ("score118", "s2pfull_117"),
           "control - tabicl_117": ("control", "tabicl_117")}
THR_METRICS = R.THR_METRICS


def cv_thresholds(run_dir: Path) -> dict:
    out = {}
    for arm in ARMS:
        per_k, metrics = [], []
        for k in KS:
            rs = [json.loads((run_dir / f"{arm}_k{k}_sub{s}.json").read_text()) for s in range(4)]
            y = np.concatenate([r["y"] for r in rs])
            s_ = np.concatenate([r["score"] for r in rs])
            g = np.concatenate([r["apo_id"] for r in rs])
            assert len(set(g)) == sum(len(set(r["apo_id"])) for r in rs), "sub-folds overlap"
            t = evaluate.choose_threshold(y, s_, g)
            m = evaluate.summarise(y, s_, g, t, f"{arm}/k{k}")
            per_k.append(float(t))
            metrics.append({x: m[x] for x in ("AUC", "AUPRC", "TPR", "FPR", "MCC",
                                              "AUC_pooled", "AUPRC_pooled")} | {"k": k})
        out[arm] = {"k_thresholds": per_k, "frozen_threshold": float(np.median(per_k)), "cv": metrics}
    return out


def load_test(run_dir: Path) -> pd.DataFrame:
    blob = json.loads((C.BUILD_DIR / "modal_result_esm64.json").read_text())
    df = pd.DataFrame({k: blob["scores"][k] for k in ("apo_id", "chain", "seq_id", "y")})
    df["tabicl_117"] = blob["scores"]["score"]
    for arm in ARMS:
        per_k = []
        for k in KS:
            r = json.loads((run_dir / f"{arm}_k{k}_test.json").read_text())
            for c in ("apo_id", "chain", "seq_id", "y"):
                assert list(r[c]) == df[c].tolist(), f"{arm} k{k}: test rows differ on {c}"
            per_k.append(np.asarray(r["score"]))
            df[f"{arm}_k{k}"] = r["score"]
        df[arm] = np.mean(per_k, axis=0)
    t = pd.read_parquet(C.BUILD_DIR / "residues_ftemb.parquet",
                        columns=["apo_id", "chain", "seq_id", "split"] + [f"ftk{k}_score" for k in KS])
    t = t[t["split"] == "test"].reset_index(drop=True)
    for c in ("apo_id", "chain", "seq_id"):
        assert (t[c].to_numpy() == df[c].to_numpy()).all(), f"component rows differ on {c}"
    df["ft_alone"] = t[[f"ftk{k}_score" for k in KS]].to_numpy().mean(axis=1)
    r = json.loads((C.BUILD_DIR / "s2p_stack" / "st2" / "s2pfull_117_final.json").read_text())
    assert list(r["apo_id"]) == df["apo_id"].tolist() and list(r["seq_id"]) == df["seq_id"].tolist()
    df["s2pfull_117"] = r["score"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cv-only", action="store_true")
    a = ap.parse_args()
    sha = hashlib.sha256(Path(FROZEN).read_bytes()).hexdigest()
    if FROZEN_SHA256 is None or sha != FROZEN_SHA256:
        raise SystemExit(f"{FROZEN} is not pinned or changed since pre-registration ({sha})")
    run_dir = C.BUILD_DIR / "ftemb_stack" / a.run_id

    # ---- 1. thresholds from CV only, written before test is read
    thr = cv_thresholds(run_dir)
    Path("finetuned_esm/step10_thresholds.json").write_text(json.dumps(thr, indent=1))
    for arm, t in thr.items():
        print(f"{arm}: k thresholds {[round(x, 4) for x in t['k_thresholds']]} "
              f"-> frozen {t['frozen_threshold']:.4f}")
        for m in t["cv"]:
            print(f"   k{m['k']}: AUC {m['AUC']:.3f} AUPRC {m['AUPRC']:.3f} "
                  f"FPR {m['FPR']:.4f} TPR {m['TPR']:.3f}")
    if a.cv_only:
        return

    # ---- 2. test, once
    rng = np.random.default_rng(R.SEED)
    df = load_test(run_dir)
    names = list(ARMS) + ["ft_alone", "tabicl_117", "s2pfull_117"]
    R.CLAIMS = CLAIMS | CONTEXT
    single = single_chain_pdbs()
    res = {"run_id": a.run_id, "step": 10, "frozen_sha256": sha, "thresholds": thr}
    for subset, sub in (("CB-full", df),
                        ("CB-P2RANK-apo", df[df["apo_id"].str[:4].str.lower().isin(single)])):
        sub = sub.reset_index(drop=True)
        res[subset] = R.paired(sub, names, rng)
        y, g = sub["y"].to_numpy(), sub["apo_id"].to_numpy()
        res[subset]["threshold_metrics"] = {
            arm: {k: v for k, v in evaluate.summarise(
                y, sub[arm].to_numpy(), g, thr[arm]["frozen_threshold"], arm).items()
                if k in THR_METRICS} for arm in ARMS}
        # Each single-context prediction alone: how much the 4-way average buys.
        res[subset]["per_k_per_protein_AUPRC"] = {
            arm: [float(evaluate.per_protein(y, sub[f"{arm}_k{k}"].to_numpy(), g, 0.5)["AUPRC"].mean())
                  for k in KS] for arm in ARMS}

    full = res["CB-full"]["per_protein_AUPRC"]
    res["claims"] = {c: {**full[f"{a_} - {r_}"], "passes": full[f"{a_} - {r_}"]["ci"][0] > 0}
                     for c, (a_, r_) in CLAIMS.items()}
    Path("finetuned_esm/step10_result.json").write_text(json.dumps(res, indent=1))

    for subset in ("CB-full", "CB-P2RANK-apo"):
        r = res[subset]
        print(f"\n== {subset} ({r['n_proteins']} proteins)")
        print(f"{'':20s}" + "".join(f"{n:>13s}" for n in names))
        for key in ("per_protein_AUC", "per_protein_AUPRC", "pooled_AUC", "pooled_AUPRC"):
            print(f"{key:20s}" + "".join(f"{r[key][n]['value']:>13.3f}" for n in names))
            for c, (a_, r_) in (CLAIMS | CONTEXT).items():
                d = r[key][f"{a_} - {r_}"]
                star = " *" if d["ci"][0] > 0 or d["ci"][1] < 0 else ""
                print(f"{'':22s}{a_ + ' - ' + r_:30s} {d['diff']:+.3f} "
                      f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]{star}")
        print("single-context per-protein AUPRC, k=0..3:")
        for arm, v in r["per_k_per_protein_AUPRC"].items():
            print(f"   {arm:10s} " + " ".join(f"{x:.3f}" for x in v))
        print("threshold metrics (CV-frozen thresholds):")
        for k, v in r["threshold_metrics"].items():
            print(f"   {k:10s} " + "  ".join(f"{m} {v[m]:.3f}" for m in THR_METRICS))
    print("\nCLAIMS (per-protein AUPRC, CB-full, paired 95 % CI > 0):")
    for c, v in res["claims"].items():
        print(f"  {c:36s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  "
              f"{'PASSES' if v['passes'] else 'does not pass'}")


if __name__ == "__main__":
    main()
