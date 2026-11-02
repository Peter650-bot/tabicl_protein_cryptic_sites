"""Step 13 of finetuned_esm/PLAN.md: PAM's feature-family ablation.

ARMS (nested, all on PAM's protocol: fold-k contexts, seed 0, CV-frozen threshold,
test = mean of the four cross-fitted predictions)
    fam33   33 geometry and physico-chemistry features
    fam47   + 14 ensemble pocket features
    fam53   + 6 conservation features (no protein language model)
    ft69    + the first 16 PCs of the fine-tuned (full recipe, clean, fold) ESM-2 3B embedding
    ft85    + the first 32
    ft117   + all 64 = PAM (run fe1, reused)

ORDER OF OPERATIONS
    1. Thresholds from the CV sub-folds alone (as step 10), written before test is read.
       ft117's is recomputed from fe1 and must equal step10_thresholds.json.
    2. Test, scored on CryptoBench's evaluation unit (scripts/published_comparison.py):
       CB-full (262 chains) and the 184 single-chain structures, per protein and pooled,
       with paired 95 % CIs for each step (arm - previous arm) and for arm - PAM,
       resampling whole structures (10,000 draws, seed 0).

Run: python scripts/ftemb_ablation_report.py --run-id ab1 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_comparison as F  # noqa: E402
import published_comparison as P  # noqa: E402
from cryptic_tabicl import config as C, evaluate  # noqa: E402

ARMS = ("fam33", "fam47", "fam53", "ft69", "ft85", "ft117")
N_FEATURES = dict(zip(ARMS, (33, 47, 53, 69, 85, 117)))
KS = range(4)
SUBSETS = ("CB-full", "CB-P2RANK-184")
METRICS = ("AUC", "AUPRC", "ACC", "TPR", "FPR", "MCC", "F1_weighted")
N_BOOT, SEED = 10_000, 0
ROWS = ("apo_id", "chain", "seq_id", "y")


def run_dir(arm: str, run_id: str) -> Path:
    return C.BUILD_DIR / "ftemb_stack" / ("fe1" if arm == "ft117" else run_id)


def cv_thresholds(run_id: str) -> dict:
    out = {}
    for arm in ARMS:
        d, per_k, cv = run_dir(arm, run_id), [], []
        for k in KS:
            rs = [json.loads((d / f"{arm}_k{k}_sub{s}.json").read_text()) for s in range(4)]
            assert all(r["n_features"] == N_FEATURES[arm] for r in rs), arm
            y = np.concatenate([r["y"] for r in rs])
            s_ = np.concatenate([r["score"] for r in rs])
            g = np.concatenate([r["apo_id"] for r in rs])
            t = evaluate.choose_threshold(y, s_, g)
            m = evaluate.summarise(y, s_, g, t, f"{arm}/k{k}")
            per_k.append(float(t))
            cv.append({x: float(m[x]) for x in ("AUC", "AUPRC", "TPR", "FPR")} | {"k": k})
        out[arm] = {"k_thresholds": per_k, "frozen_threshold": float(np.median(per_k)), "cv": cv}
    ref = json.loads(Path("finetuned_esm/step10_thresholds.json").read_text())["ft117"]["frozen_threshold"]
    assert abs(out["ft117"]["frozen_threshold"] - ref) < 1e-12, "ft117 threshold differs from step 10"
    return out


def load_test(run_id: str, thr: dict) -> pd.DataFrame:
    blob = json.loads((C.BUILD_DIR / "modal_result_esm64.json").read_text())["scores"]
    df = pd.DataFrame({c: blob[c] for c in ROWS})
    for arm in ARMS:
        per_k = []
        for k in KS:
            r = json.loads((run_dir(arm, run_id) / f"{arm}_k{k}_test.json").read_text())
            for c in ROWS:
                assert list(r[c]) == df[c].tolist(), f"{arm} k{k}: test rows differ on {c}"
            per_k.append(np.asarray(r["score"]))
        df[arm] = np.mean(per_k, axis=0)
        df[f"{arm}_call"] = (df[arm] >= thr[arm]["frozen_threshold"]).astype(int)
    return df


_BOOT: dict = {}                                     # read by forked workers, never pickled


def _draw_chunk(picks: list) -> dict:
    s = _BOOT
    out = {kind: {a: np.empty((len(picks), len(METRICS))) for a in ARMS} for kind in ("per_protein", "pooled")}
    for i, pick in enumerate(picks):
        r = np.concatenate([s["rows"][p] for p in pick])
        for a in ARMS:
            pr = np.concatenate([s["pp_rows"][a][p] for p in pick])
            out["per_protein"][a][i] = s["tabs"][a].iloc[pr][list(METRICS)].mean(skipna=True).to_numpy()
            v = F.pooled_all(s["y"][r], s["score"][a][r], s["call"][a][r])
            out["pooled"][a][i] = [v[m] for m in METRICS]
    return out


def bootstrap(sub: pd.DataFrame, tabs: dict, rng, workers: int = 1) -> dict:
    """Per-protein and pooled bootstrap draws for every arm, same structure resample.

    The resamples are all drawn here, in order, before the work is split, so the
    result does not depend on `workers`."""
    import multiprocessing as mp

    cl = sub["pdb"].to_numpy()
    structs = np.unique(cl)
    unit_cl = {a: sub.groupby("unit")["pdb"].first().loc[t.index].to_numpy() for a, t in tabs.items()}
    _BOOT.update(y=sub["y"].to_numpy(), tabs=tabs,
                 rows={p: np.flatnonzero(cl == p) for p in structs},
                 pp_rows={a: {p: np.flatnonzero(unit_cl[a] == p) for p in structs} for a in tabs},
                 score={a: sub[a].to_numpy() for a in ARMS},
                 call={a: sub[f"{a}_call"].to_numpy() for a in ARMS})
    picks = [structs[rng.integers(0, len(structs), len(structs))] for _ in range(N_BOOT)]
    n = workers * 8 if workers > 1 else 1
    idx = [np.arange(N_BOOT)[i::n] for i in range(n)]
    chunks = [[picks[j] for j in ix] for ix in idx]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_draw_chunk, chunks)
    else:
        parts = [_draw_chunk(chunks[0])]
    order = np.argsort(np.concatenate(idx))
    return {kind: {a: np.concatenate([p[kind][a] for p in parts])[order] for a in ARMS}
            for kind in ("per_protein", "pooled")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=1, help="bootstrap processes; result is the same")
    a = ap.parse_args()

    thr = cv_thresholds(a.run_id)
    Path("finetuned_esm/step13_thresholds.json").write_text(json.dumps(thr, indent=1))
    for arm, t in thr.items():
        print(f"{arm:6s} k thresholds {[round(x, 4) for x in t['k_thresholds']]} -> {t['frozen_threshold']:.4f}")

    df = load_test(a.run_id, thr)
    u = P.cryptobench_units(df)
    masks = P.subsets(u)
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    res = {"run_id": a.run_id, "arms": N_FEATURES, "thresholds": thr, "n_boot": N_BOOT,
           "bootstrap": "paired, whole structures resampled, seed 0"}
    for name in SUBSETS:
        sub = u[masks[name]].reset_index(drop=True)
        y, g = sub["y"].to_numpy(), sub["unit"].to_numpy()
        tabs = {arm: F.per_protein_table(y, sub[arm].to_numpy(), sub[f"{arm}_call"].to_numpy(), g)
                for arm in ARMS}
        point = {"per_protein": {arm: tabs[arm][list(METRICS)].mean(skipna=True).to_dict() for arm in ARMS},
                 "pooled": {arm: F.pooled_all(y, sub[arm].to_numpy(), sub[f"{arm}_call"].to_numpy())
                            for arm in ARMS}}
        draws = bootstrap(sub, tabs, np.random.default_rng(SEED), a.workers)
        r = {"n_units": int(sub["unit"].nunique()), "n_structures": int(sub["pdb"].nunique()),
             "n_residues": int(len(y)), "positives": int(y.sum())}
        pairs = {f"{b} - {p}": (b, p) for p, b in zip(ARMS, ARMS[1:])} | \
                {f"{x} - ft117": (x, "ft117") for x in ARMS[:-1]}
        for kind in ("per_protein", "pooled"):
            r[kind] = {arm: {m: {"value": float(point[kind][arm][m]),
                                 "ci": ci(draws[kind][arm][:, i])} for i, m in enumerate(METRICS)}
                       for arm in ARMS}
            r[f"{kind}_diff"] = {
                lab: {m: {"diff": float(point[kind][x][m] - point[kind][y_][m]),
                          "ci": ci(draws[kind][x][:, i] - draws[kind][y_][:, i])}
                      for i, m in enumerate(METRICS)} for lab, (x, y_) in pairs.items()}
        res[name] = r
        print(f"\n== {name}: {r['n_units']} units ({r['n_structures']} structures), "
              f"{r['n_residues']:,} residues")
        for kind in ("per_protein", "pooled"):
            print(f"  {kind}{'':10s}" + "".join(f"{m:>12s}" for m in METRICS))
            for arm in ARMS:
                print(f"  {arm:6s} ({N_FEATURES[arm]:3d})     " +
                      "".join(f"{r[kind][arm][m]['value']:>12.3f}" for m in METRICS))
            for lab, d in r[f"{kind}_diff"].items():
                print(f"  {lab:18s}" + "".join(
                    f"{d[m]['diff']:+9.3f}{'*' if d[m]['ci'][0] > 0 or d[m]['ci'][1] < 0 else ' ':3s}"
                    for m in METRICS))
    Path("finetuned_esm/step13_ablation.json").write_text(json.dumps(res, indent=1))
    print("\nwrote finetuned_esm/step13_ablation.json")


if __name__ == "__main__":
    main()
