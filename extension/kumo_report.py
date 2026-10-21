"""Kumo Tabular vs PAM (ft117, run fe1): gates, then the result as frozen.json says.

TWO MODES
    --gates   Steps 2-3 of extension/PLAN.md, before pre-registration: the harness check
              (sdm's TabICLv2 must reproduce fe1 on k=0 / test) and the determinism and
              cost pilot (two Kumo runs of k=0 / sub0). Writes extension/gates.json.
    default   Steps 5-6, after pre-registration. ORDER OF OPERATIONS IS THE PROTOCOL:
              1. Thresholds from the CV stage alone, exactly as scripts/ftemb_report.py
                 derives PAM's: per fold context k, the out-of-fold scores of its four
                 protein sub-folds give a threshold at mean per-protein FPR 0.05; the
                 frozen threshold is the median of the four. Written BEFORE a test score
                 is read.
              2. Test: Kumo's score is the mean of its four cross-fitted predictions, as
                 PAM's is. Paired bootstrap against PAM, row for row; the pre-registered
                 claim by its pre-registered two-sided criterion.

Run: python extension/kumo_report.py --gates --run-id ku1
     python extension/kumo_report.py --run-id ku1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import s2p_stack_report as R  # noqa: E402
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

EXT = ROOT / "extension"
FROZEN = EXT / "frozen.json"
FROZEN_SHA256 = "3b1f78fcb578b271cdbfc8856f4c2d1c669e4707cedb8748c3269313a8fc0693"
PAM_RUN = C.BUILD_DIR / "ftemb_stack" / "fe1"
PAM_THRESHOLDS = ROOT / "finetuned_esm" / "step10_thresholds.json"
PAM_RESULT = ROOT / "finetuned_esm" / "step10_result.json"
KS = range(4)
CLAIMS = {"primary: kumo117 - PAM": ("kumo117", "PAM")}
CONTEXT = {"rank_blend - PAM": ("rank_blend", "PAM"),
           "rank_blend - kumo117": ("rank_blend", "kumo117")}
THR_METRICS = R.THR_METRICS

# Gate tolerances, fixed before any gate ran (extension/PLAN.md steps 2-3).
HARNESS_AUPRC_TOL, HARNESS_MIN_SPEARMAN = 0.01, 0.97
DETERMINISM_TOL = 1e-3
A100_80GB_USD_PER_HOUR = 2.50              # Modal list price at the time of writing
COST_CEILING_USD = 30.0


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_pp_auprc(r: dict) -> float:
    return float(evaluate.per_protein(np.asarray(r["y"]), np.asarray(r["score"]),
                                      np.asarray(r["apo_id"]), 0.5)["AUPRC"].mean())


def same_rows(a: dict, b: dict) -> bool:
    return all(list(a[c]) == list(b[c]) for c in ("apo_id", "chain", "seq_id", "icode", "y"))


def gates(run_dir: Path) -> dict:
    out = {"run_id": run_dir.name}

    # ---- step 2: harness check, sdm TabICLv2 vs tabicl 2.1.1 on the same context
    mine, ref = read(run_dir / "ft117_k0_test_tabiclv2.json"), read(PAM_RUN / "ft117_k0_test.json")
    assert same_rows(mine, ref), "harness: test rows differ from fe1"
    a, b = mean_pp_auprc(mine), mean_pp_auprc(ref)
    rho = float(pd.Series(mine["score"]).corr(pd.Series(ref["score"]), method="spearman"))
    out["harness"] = {
        "sdm_tabiclv2_pp_auprc": a, "fe1_tabicl_pp_auprc": b, "diff": a - b, "spearman": rho,
        "n_context": [mine["n_context"], ref["n_context"]],
        "features_identical": mine["features"] == ref["features"],
        "passed": bool(abs(a - b) <= HARNESS_AUPRC_TOL and rho > HARNESS_MIN_SPEARMAN
                       and mine["n_context"] == ref["n_context"]
                       and mine["features"] == ref["features"])}

    # ---- step 3: determinism and the cost pilot
    r0, r1 = read(run_dir / "ft117_k0_sub0_rep0.json"), read(run_dir / "ft117_k0_sub0_rep1.json")
    ref = read(PAM_RUN / "ft117_k0_sub0.json")
    assert same_rows(r0, r1) and same_rows(r0, ref), "gate: sub0 rows differ"
    dmax = float(np.max(np.abs(np.asarray(r0["score"]) - np.asarray(r1["score"]))))
    out["determinism"] = {
        "max_abs_diff": dmax, "tolerance": DETERMINISM_TOL,
        "n_context": [r0["n_context"], ref["n_context"]],
        "n_features": r0["n_features"], "n_context_columns": r0["config"]["n_context_columns"],
        "features_identical_to_fe1": r0["features"] == ref["features"],
        "kumo_pp_auprc_k0_sub0": mean_pp_auprc(r0), "fe1_pp_auprc_k0_sub0": mean_pp_auprc(ref),
        "passed": bool(dmax <= DETERMINISM_TOL and r0["n_context"] == ref["n_context"]
                       and r0["n_features"] == 117 and r0["config"]["n_context_columns"] == 117
                       and r0["features"] == ref["features"])}

    # Cost: fit time scaled by context rows squared (the ICL transformer's context
    # self-attention), predict time linearly by query rows; fe1 supplies the sizes.
    fit_s = np.mean([r["seconds_fit"] for r in (r0, r1)])
    pred_s = np.mean([r["seconds"] - r["seconds_fit"] for r in (r0, r1)])
    n_ctx, n_q = r0["n_context"], len(r0["score"])
    secs = 0.0
    for k in KS:
        for q in [f"sub{s}" for s in range(4)] + ["test"]:
            f = read(PAM_RUN / f"ft117_k{k}_{q}.json")
            secs += fit_s * (f["n_context"] / n_ctx) ** 2 + pred_s * len(f["score"]) / n_q
    usd = secs / 3600 * A100_80GB_USD_PER_HOUR
    out["cost_pilot"] = {"seconds_fit": fit_s, "seconds_predict": pred_s,
                         "peak_gpu_gb": max(r0["peak_gpu_gb"], r1["peak_gpu_gb"]),
                         "gpu": r0["gpu"], "projected_seconds_20_jobs": secs,
                         "projected_usd_20_jobs": usd, "ceiling_usd": COST_CEILING_USD,
                         "passed": bool(usd <= COST_CEILING_USD)}
    out["passed"] = all(out[g]["passed"] for g in ("harness", "determinism", "cost_pilot"))
    return out


def cv_thresholds(run_dir: Path) -> dict:
    per_k, metrics = [], []
    for k in KS:
        rs = [read(run_dir / f"ft117_k{k}_sub{s}.json") for s in range(4)]
        for s, r in enumerate(rs):
            assert r["backbone"] == "kumo" and r["config"]["n_context_columns"] == 117
            assert same_rows(r, read(PAM_RUN / f"ft117_k{k}_sub{s}.json")), f"k{k} sub{s} rows differ"
        y = np.concatenate([r["y"] for r in rs])
        s_ = np.concatenate([r["score"] for r in rs])
        g = np.concatenate([r["apo_id"] for r in rs])
        assert len(set(g)) == sum(len(set(r["apo_id"])) for r in rs), "sub-folds overlap"
        t = evaluate.choose_threshold(y, s_, g)
        m = evaluate.summarise(y, s_, g, t, f"kumo117/k{k}")
        per_k.append(float(t))
        metrics.append({x: m[x] for x in ("AUC", "AUPRC", "TPR", "FPR", "MCC",
                                          "AUC_pooled", "AUPRC_pooled")} | {"k": k})
    return {"kumo117": {"k_thresholds": per_k, "frozen_threshold": float(np.median(per_k)),
                        "cv": metrics}}


def load_test(run_dir: Path) -> pd.DataFrame:
    df = None
    for arm, d in (("PAM", PAM_RUN), ("kumo117", run_dir)):
        per_k = []
        for k in KS:
            r = read(d / f"ft117_k{k}_test.json")
            if df is None:
                df = pd.DataFrame({c: r[c] for c in ("apo_id", "chain", "seq_id", "icode", "y")})
            for c in ("apo_id", "chain", "seq_id", "icode", "y"):
                assert list(r[c]) == df[c].tolist(), f"{arm} k{k}: test rows differ on {c}"
            if arm == "kumo117":
                assert r["backbone"] == "kumo" and r["config"]["n_context_columns"] == 117
            df[f"{arm}_k{k}"] = r["score"]
            per_k.append(np.asarray(r["score"]))
        df[arm] = np.mean(per_k, axis=0)
    g = df.groupby("apo_id")
    df["rank_blend"] = 0.5 * g["PAM"].rank(method="average", pct=True) \
        + 0.5 * g["kumo117"].rank(method="average", pct=True)
    return df


def verdict(ci: list) -> str:
    return "better" if ci[0] > 0 else "worse" if ci[1] < 0 else "tie"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gates", action="store_true")
    a = ap.parse_args()
    run_dir = C.BUILD_DIR / "kumo_stack" / a.run_id

    if a.gates:
        res = gates(run_dir)
        (EXT / "gates.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return

    sha = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    if FROZEN_SHA256 is None or sha != FROZEN_SHA256:
        raise SystemExit(f"{FROZEN} is not pinned or changed since pre-registration ({sha})")
    if not read(EXT / "gates.json")["passed"]:
        raise SystemExit("extension/gates.json does not record passed gates")

    # ---- 1. thresholds from CV only, written before test is read
    thr = cv_thresholds(run_dir)
    (EXT / "kumo_thresholds.json").write_text(json.dumps(thr, indent=1))
    pam_thr = read(PAM_THRESHOLDS)["ft117"]
    thr["PAM"] = {k: pam_thr[k] for k in ("k_thresholds", "frozen_threshold", "cv")}
    for arm in ("kumo117", "PAM"):
        t = thr[arm]
        print(f"{arm}: k thresholds {[round(x, 4) for x in t['k_thresholds']]} "
              f"-> frozen {t['frozen_threshold']:.4f}")
        for m in t["cv"]:
            print(f"   k{m['k']}: AUC {m['AUC']:.3f} AUPRC {m['AUPRC']:.3f} "
                  f"FPR {m['FPR']:.4f} TPR {m['TPR']:.3f}")

    # ---- 2. test, once
    rng = np.random.default_rng(R.SEED)
    df = load_test(run_dir)
    names = ["kumo117", "PAM", "rank_blend"]
    R.CLAIMS = CLAIMS | CONTEXT
    single = single_chain_pdbs()
    res = {"run_id": a.run_id, "frozen_sha256": sha, "thresholds": thr}
    for subset, sub in (("CB-full", df),
                        ("CB-P2RANK-apo", df[df["apo_id"].str[:4].str.lower().isin(single)])):
        sub = sub.reset_index(drop=True)
        res[subset] = R.paired(sub, names, rng)
        y, g = sub["y"].to_numpy(), sub["apo_id"].to_numpy()
        res[subset]["threshold_metrics"] = {
            arm: {k: v for k, v in evaluate.summarise(
                y, sub[arm].to_numpy(), g, thr[arm]["frozen_threshold"], arm).items()
                if k in THR_METRICS} for arm in ("kumo117", "PAM")}
        res[subset]["per_k_per_protein_AUPRC"] = {
            arm: [float(evaluate.per_protein(y, sub[f"{arm}_k{k}"].to_numpy(), g, 0.5)["AUPRC"].mean())
                  for k in KS] for arm in ("kumo117", "PAM")}

    # The PAM column must be PAM: same rows, same averaging, same number as step 10.
    pam_ref = read(PAM_RESULT)["CB-full"]["per_protein_AUPRC"]["ft117"]["value"]
    pam_now = res["CB-full"]["per_protein_AUPRC"]["PAM"]["value"]
    assert abs(pam_now - pam_ref) < 1e-12, f"PAM reproduces as {pam_now}, step 10 says {pam_ref}"

    full = res["CB-full"]["per_protein_AUPRC"]
    res["claims"] = {c: {**full[f"{a_} - {r_}"], "verdict": verdict(full[f"{a_} - {r_}"]["ci"])}
                     for c, (a_, r_) in CLAIMS.items()}
    (EXT / "kumo_result.json").write_text(json.dumps(res, indent=1))

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
    print("\nCLAIM (per-protein AUPRC, CB-full, paired 95 % CI, two-sided):")
    for c, v in res["claims"].items():
        print(f"  {c:30s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  {v['verdict']}")


if __name__ == "__main__":
    main()
