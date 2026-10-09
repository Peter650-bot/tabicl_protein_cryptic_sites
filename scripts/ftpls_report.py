"""Step 11 of finetuned_esm/PLAN.md: PLS instead of PCA, reported as frozen_step11.json says.

ORDER OF OPERATIONS IS THE PROTOCOL
    1. Thresholds (d = 64 arms only) from the CV stage alone, step 10's rule, written
       BEFORE a test score is read.
    2. Test: each arm's score is the mean of its four cross-fitted predictions (k = 0..3).
       Step 10's arms come from run fe1 unchanged. Paired bootstrap; the two claims by
       their pre-registered criterion; the dimension sweep is descriptive.

Run: python scripts/ftpls_report.py --run-id fp1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftemb_report as FR  # noqa: E402
import s2p_stack_report as R  # noqa: E402
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from cryptic_tabicl.feature_sets import PLS_DIMS  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

FROZEN = "finetuned_esm/frozen_step11.json"
FROZEN_SHA256 = "aacccb7befe54b7fb9727bc0267e27cc7be8b53b9394fe981a2504156adbe48f"
STEP10_RUN = "fe1"
CV_ARMS = ("pls64_53", "pls64_only")
PLS_ARMS = tuple(f"pls{d}_{b}" for d in PLS_DIMS for b in ("53", "only"))
CLAIMS = {"primary: pls64_53 - ft117": ("pls64_53", "ft117"),
          "secondary: pls64_only - ft64only": ("pls64_only", "ft64only")}
CONTEXT = {"pls64_53 - score118": ("pls64_53", "score118"),
           "pls64_53 - control": ("pls64_53", "control"),
           "pls64_53 - pls64_only": ("pls64_53", "pls64_only")}
KS = range(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    sha = hashlib.sha256(Path(FROZEN).read_bytes()).hexdigest()
    if sha != FROZEN_SHA256:
        raise SystemExit(f"{FROZEN} changed since pre-registration ({sha})")
    run_dir = C.BUILD_DIR / "ftpls_stack" / a.run_id

    # ---- 0. the determinism gate: step 10's ft117 k=0 through this harness, bit for bit
    g_new = json.loads((run_dir / "ft117_k0_test.json").read_text())
    g_old = json.loads((C.BUILD_DIR / "ftemb_stack" / STEP10_RUN / "ft117_k0_test.json").read_text())
    gate = float(np.abs(np.asarray(g_new["score"]) - np.asarray(g_old["score"])).max())
    assert g_new["apo_id"] == g_old["apo_id"] and gate == 0.0, f"determinism gate failed ({gate})"
    manifests = {k: json.loads((run_dir / f"pls_manifest_k{k}.json").read_text()) for k in KS}

    # ---- 1. thresholds from CV only, written before test is read
    step10_arms, FR.ARMS = FR.ARMS, CV_ARMS
    thr = FR.cv_thresholds(run_dir)
    FR.ARMS = step10_arms
    Path("finetuned_esm/step11_thresholds.json").write_text(json.dumps(thr, indent=1))
    for arm, t in thr.items():
        print(f"{arm}: k thresholds {[round(x, 4) for x in t['k_thresholds']]} "
              f"-> frozen {t['frozen_threshold']:.4f}")

    # ---- 2. test, once
    df = FR.load_test(C.BUILD_DIR / "ftemb_stack" / STEP10_RUN)
    for arm in PLS_ARMS:
        per_k = []
        for k in KS:
            r = json.loads((run_dir / f"{arm}_k{k}_test.json").read_text())
            for c in ("apo_id", "chain", "seq_id", "y"):
                assert list(r[c]) == df[c].tolist(), f"{arm} k{k}: test rows differ on {c}"
            per_k.append(np.asarray(r["score"]))
        df[arm] = np.mean(per_k, axis=0)

    rng = np.random.default_rng(R.SEED)
    names = list(PLS_ARMS) + ["ft117", "ft64only", "score118", "control", "ft_alone"]
    R.CLAIMS = CLAIMS | CONTEXT
    single = single_chain_pdbs()
    res = {"run_id": a.run_id, "step": 11, "frozen_sha256": sha, "determinism_gate_max_abs": gate,
           "pls_manifests": manifests, "thresholds": thr}
    for subset, sub in (("CB-full", df),
                        ("CB-P2RANK-apo", df[df["apo_id"].str[:4].str.lower().isin(single)])):
        sub = sub.reset_index(drop=True)
        res[subset] = R.paired(sub, names, rng)
        y, g = sub["y"].to_numpy(), sub["apo_id"].to_numpy()
        res[subset]["threshold_metrics"] = {
            arm: {k: v for k, v in evaluate.summarise(
                y, sub[arm].to_numpy(), g, thr[arm]["frozen_threshold"], arm).items()
                if k in R.THR_METRICS} for arm in CV_ARMS}
        res[subset]["sweep"] = {
            b: {str(d): {key: res[subset][key][f"pls{d}_{b}"]
                         for key in ("per_protein_AUPRC", "pooled_AUPRC", "per_protein_AUC", "pooled_AUC")}
                for d in PLS_DIMS} for b in ("53", "only")}

    full = res["CB-full"]["per_protein_AUPRC"]
    res["claims"] = {c: {**full[f"{a_} - {r_}"], "passes": full[f"{a_} - {r_}"]["ci"][0] > 0}
                     for c, (a_, r_) in CLAIMS.items()}
    Path("finetuned_esm/step11_result.json").write_text(json.dumps(res, indent=1))

    for subset in ("CB-full", "CB-P2RANK-apo"):
        r = res[subset]
        print(f"\n== {subset} ({r['n_proteins']} proteins)")
        print(f"{'d':>5s} | {'pp AUPRC 53+d':>14s} {'only d':>8s} | {'pooled 53+d':>12s} {'only d':>8s}")
        for d in PLS_DIMS:
            print(f"{d:>5d} | {r['per_protein_AUPRC'][f'pls{d}_53']['value']:>14.3f} "
                  f"{r['per_protein_AUPRC'][f'pls{d}_only']['value']:>8.3f} | "
                  f"{r['pooled_AUPRC'][f'pls{d}_53']['value']:>12.3f} "
                  f"{r['pooled_AUPRC'][f'pls{d}_only']['value']:>8.3f}")
        for ref in ("ft117", "ft64only", "score118", "control", "ft_alone"):
            print(f"{ref:>9s}: pp AUPRC {r['per_protein_AUPRC'][ref]['value']:.3f}  "
                  f"pooled {r['pooled_AUPRC'][ref]['value']:.3f}")
        for key in ("per_protein_AUPRC", "pooled_AUPRC", "per_protein_AUC", "pooled_AUC"):
            for c, (a_, r_) in (CLAIMS | CONTEXT).items():
                d = r[key][f"{a_} - {r_}"]
                star = " *" if d["ci"][0] > 0 or d["ci"][1] < 0 else ""
                print(f"  {key:18s} {a_ + ' - ' + r_:26s} {d['diff']:+.3f} "
                      f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]{star}")
        for k, v in r["threshold_metrics"].items():
            print(f"   {k:10s} " + "  ".join(f"{m} {v[m]:.3f}" for m in R.THR_METRICS))
    print("\nCLAIMS (per-protein AUPRC, CB-full, paired 95 % CI > 0):")
    for c, v in res["claims"].items():
        print(f"  {c:36s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  "
              f"{'PASSES' if v['passes'] else 'does not pass'}")


if __name__ == "__main__":
    main()
