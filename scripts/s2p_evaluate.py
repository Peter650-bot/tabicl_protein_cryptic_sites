"""Step 5 of finetuned_esm/PLAN.md: the published fine-tuned ESM-2, scored our way.

WHAT IS COMPARED
    s2p     Škrhák et al.'s released model (`build/s2p_scores_keyed.parquet`),
            verified to reproduce their published pooled numbers (step 4).
    TabICL  our zero-shot TabICLv2, 117 features (`build/modal_result_esm64.json`).
    Both on the same 231 test proteins, residue for residue (row order verified).

WHICH METRICS ARE THE RESULT
    AUC and AUPRC, per protein (this project's convention) and pooled (theirs).
    They need no threshold, so nothing about either model's operating point
    enters. Threshold metrics are printed for completeness only, each model at
    its own point: TabICL at its CV-frozen 0.5706; s2p at the 0.7 its authors
    use. Our CV-fold threshold protocol cannot be applied to s2p -- it was
    trained on those folds, so its CV scores are in-sample.

UNCERTAINTY
    Proteins are resampled with replacement. Per-protein means: 10,000 resamples
    of the stored per-protein table (exact, cheap). Pooled AUC/AUPRC need the
    residues re-pooled on every resample, so they use 2,000. Both arms always
    see the same resample, so the differences are paired.

Run: python scripts/s2p_evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

S2P_THRESHOLD = 0.7
N_BOOT, N_BOOT_POOLED, SEED = 10_000, 2_000, 0
PP_METRICS = ["AUC", "AUPRC", "TPR", "FPR", "MCC", "F1_weighted"]


def load() -> pd.DataFrame:
    blob = json.loads((C.BUILD_DIR / "modal_result_esm64.json").read_text())
    tab = pd.DataFrame(blob["scores"])
    keyed = pd.read_parquet(C.BUILD_DIR / "s2p_scores_keyed.parquet")
    test = keyed[keyed["split"] == "test"].reset_index(drop=True)
    for c in ("apo_id", "chain", "seq_id"):
        assert (tab[c].to_numpy() == test[c].to_numpy()).all(), f"row order differs on {c}"
    assert (tab["y"].to_numpy() == test[C.LABEL_COL].to_numpy()).all(), "labels differ"
    test["tabicl"] = tab["score"].to_numpy()
    test["tabicl_threshold"] = blob["threshold"]
    return test.rename(columns={C.LABEL_COL: "y", "s2p_score": "s2p"})


def pooled(y, s) -> tuple[float, float]:
    prec, rec, _ = metrics.precision_recall_curve(y, s)
    return metrics.roc_auc_score(y, s), metrics.auc(rec, prec)


def compare(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    thr_tab = float(df["tabicl_threshold"].iloc[0])
    y, g = df["y"].to_numpy(), df["apo_id"].to_numpy()
    arms = {"s2p": (df["s2p"].to_numpy(), S2P_THRESHOLD),
            "tabicl": (df["tabicl"].to_numpy(), thr_tab)}

    out, pp = {}, {}
    for name, (s, thr) in arms.items():
        out[name] = evaluate.summarise(y, s, g, thr, name)
        pp[name] = evaluate.per_protein(y, s, g, thr).set_index("apo_id")[PP_METRICS]
    assert list(pp["s2p"].index) == list(pp["tabicl"].index)

    # ---- per-protein means: CIs and paired differences
    A, B = pp["tabicl"].to_numpy(float), pp["s2p"].to_numpy(float)
    n = len(A)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    mA, mB = np.nanmean(A[idx], axis=1), np.nanmean(B[idx], axis=1)
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5], axis=0).T.ravel()]  # noqa: E731
    per_protein = {}
    for i, m in enumerate(PP_METRICS):
        lo, hi = np.percentile(mB[:, i] - mA[:, i], [2.5, 97.5])
        d = pp["s2p"][m] - pp["tabicl"][m]
        per_protein[m] = {
            "s2p": float(np.nanmean(B[:, i])), "s2p_ci": ci(mB[:, i]),
            "tabicl": float(np.nanmean(A[:, i])), "tabicl_ci": ci(mA[:, i]),
            "diff_s2p_minus_tabicl": float(np.nanmean(B[:, i]) - np.nanmean(A[:, i])),
            "diff_ci": [float(lo), float(hi)],
            "proteins_s2p_better": int((d > 0).sum()), "proteins_s2p_worse": int((d < 0).sum()),
        }

    # ---- pooled AUC / AUPRC: re-pool residues on every resample
    rows_of = {p: np.flatnonzero(g == p) for p in pp["s2p"].index}
    prots = np.array(list(rows_of))
    sT, sS = arms["tabicl"][0], arms["s2p"][0]
    dist = np.empty((N_BOOT_POOLED, 4))
    for b in range(N_BOOT_POOLED):
        r = np.concatenate([rows_of[p] for p in rng.choice(prots, len(prots))])
        dist[b] = (*pooled(y[r], sS[r]), *pooled(y[r], sT[r]))
    pS, pT = pooled(y, sS), pooled(y, sT)
    pooled_out = {}
    for j, m in enumerate(("AUC", "AUPRC")):
        lo, hi = np.percentile(dist[:, j] - dist[:, j + 2], [2.5, 97.5])
        pooled_out[m] = {
            "s2p": float(pS[j]), "s2p_ci": ci(dist[:, j]),
            "tabicl": float(pT[j]), "tabicl_ci": ci(dist[:, j + 2]),
            "diff_s2p_minus_tabicl": float(pS[j] - pT[j]), "diff_ci": [float(lo), float(hi)],
        }
    return {"n_proteins": n, "n_residues": int(len(y)), "summaries": out,
            "per_protein": per_protein, "pooled": pooled_out}


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load()
    single = single_chain_pdbs()
    res = {"n_boot": N_BOOT, "n_boot_pooled": N_BOOT_POOLED, "seed": SEED,
           "s2p_threshold": S2P_THRESHOLD}
    for subset, sub in (("CB-full", df),
                        ("CB-P2RANK-apo", df[df["apo_id"].str[:4].str.lower().isin(single)])):
        res[subset] = compare(sub.reset_index(drop=True), rng)

    Path("finetuned_esm/step5_result.json").write_text(json.dumps(res, indent=1))
    for subset in ("CB-full", "CB-P2RANK-apo"):
        r = res[subset]
        print(f"\n== {subset}  ({r['n_proteins']} proteins)")
        print(f"{'':22s} {'s2p':>22s} {'TabICL-117':>22s} {'s2p - TabICL':>26s}")
        for kind, block in (("per-protein", r["per_protein"]), ("pooled", r["pooled"])):
            for m, v in block.items():
                if kind == "pooled" or m in ("AUC", "AUPRC"):
                    print(f"{kind+' '+m:22s} {v['s2p']:.3f} [{v['s2p_ci'][0]:.3f},{v['s2p_ci'][1]:.3f}]"
                          f"   {v['tabicl']:.3f} [{v['tabicl_ci'][0]:.3f},{v['tabicl_ci'][1]:.3f}]"
                          f"   {v['diff_s2p_minus_tabicl']:+.3f} [{v['diff_ci'][0]:+.3f},{v['diff_ci'][1]:+.3f}]")
        print("threshold metrics (each at its own point; not the result):")
        for m in ("TPR", "FPR", "MCC", "F1_weighted"):
            v = r["per_protein"][m]
            print(f"  per-protein {m:12s} s2p@0.7 {v['s2p']:.3f}   TabICL@frozen {v['tabicl']:.3f}")
        v = r["per_protein"]["AUPRC"]
        print(f"proteins where s2p AUPRC is higher / lower: {v['proteins_s2p_better']} / {v['proteins_s2p_worse']}")


if __name__ == "__main__":
    main()
