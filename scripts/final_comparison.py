"""The final comparison, on Škrhák et al.'s own test set (CB-P2RANK-apo, 184 proteins).

MODELS (test scores already computed; nothing is re-run)
    released     their fine-tuned ESM-2 3B, multitask + LIGYSIS (cbs-model.pt), step 4/5
    stack9       step 9: TabICLv2 117 + out-of-fold full-recipe score  (s2pfull_117)
    ft117        step 10: TabICLv2, 53 features + 64 PCs of the fine-tuned embedding
    blend        step 9: fixed 0.5/0.5 within-protein rank blend of TabICLv2-117 and the
                 clean full-recipe score

GATE BEFORE ANYTHING IS REPORTED (two links; see main())
    `released` on these 184 proteins must reproduce their published pooled numbers
    (Hidden paper: AUC 0.894, AUPRC 0.475, MCC 0.472; seq2pocket notebook: 0.8935 /
    0.4752 / 0.4724, F1w 0.939, ACC 0.9351) to within 0.0005, and every other model must
    reproduce its recorded step-9/10 result on this subset.

OPERATING POINTS (threshold metrics), each model at its own, none chosen on test except theirs
    released  sigmoid > 0.7      their own threshold (swept on their test set)
    stack9    >= 0.5843          step 9 CV-frozen (step9_thresholds.json)
    ft117     >= 0.3407          step 10 CV-frozen (step10_thresholds.json)
    blend     top 7.74 % of each protein -- TabICL's CV-frozen rank rule; approximate for a
              blend, as frozen.json pre-registered

METRICS
    pooled (their convention): AUC, AUPRC, MCC, F1 weighted, ACC, TPR, FPR
    per protein (CryptoBench / this project): the same seven, averaged over proteins
    95 % CIs and paired differences vs `released`: 2,000 protein resamples, seed 0.

Run: python scripts/final_comparison.py   ->  finetuned_esm/final_comparison.json
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
from cryptic_tabicl import config as C, evaluate, operating_point as op  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

N_BOOT, SEED = 2_000, 0
MODELS = ("released", "stack9", "ft117", "blend")
METRICS = ("AUC", "AUPRC", "MCC", "F1_weighted", "ACC", "TPR", "FPR")
PUBLISHED_PAPER = {"AUC": 0.894, "AUPRC": 0.475, "MCC": 0.472}           # Hidden, ACM BCB 2025
PUBLISHED_NOTEBOOK = {"AUC": 0.8935, "AUPRC": 0.4752, "MCC": 0.4724,      # seq2pocket notebook
                      "F1_weighted": 0.9390, "ACC": 0.9351}
RECORDED = {  # this subset, from step9_result.json / step10_result.json
    "stack9": {"pp_AUPRC": 0.517, "pooled_AUC": 0.907, "pooled_AUPRC": 0.476},
    "ft117": {"pp_AUPRC": 0.520, "pooled_AUC": 0.906, "pooled_AUPRC": 0.488},
    "blend": {"pp_AUPRC": 0.510, "pooled_AUC": 0.890, "pooled_AUPRC": 0.413},
}


def load() -> pd.DataFrame:
    blob = json.loads((C.BUILD_DIR / "modal_result_esm64.json").read_text())
    df = pd.DataFrame({k: blob["scores"][k] for k in ("apo_id", "chain", "seq_id", "y")})
    df["tabicl"] = blob["scores"]["score"]

    def attach(name, rows, col):
        for c in ("apo_id", "chain", "seq_id"):
            assert (np.asarray(rows[c]) == df[c].to_numpy()).all(), f"{name}: rows differ on {c}"
        df[name] = np.asarray(col, dtype=float)

    keyed = pd.read_parquet(C.BUILD_DIR / "s2p_scores_keyed.parquet")
    keyed = keyed[keyed["split"] == "test"].reset_index(drop=True)
    assert (keyed[C.LABEL_COL].to_numpy() == df["y"].to_numpy()).all()
    attach("released", keyed, keyed["s2p_score"])
    r = json.loads((C.BUILD_DIR / "s2p_stack" / "st2" / "s2pfull_117_final.json").read_text())
    attach("stack9", r, r["score"])
    per_k = []
    for k in range(4):
        r = json.loads((C.BUILD_DIR / "ftemb_stack" / "fe1" / f"ft117_k{k}_test.json").read_text())
        attach(f"ft117_k{k}", r, r["score"])
        per_k.append(df[f"ft117_k{k}"].to_numpy())
    df["ft117"] = np.mean(per_k, axis=0)
    t = pd.read_parquet(C.BUILD_DIR / "residues_s2pfull.parquet",
                        columns=["apo_id", "chain", "seq_id", "split", "s2pfull_oof"])
    t = t[t["split"] == "test"].reset_index(drop=True)
    attach("full_alone", t, t["s2pfull_oof"])
    g = df.groupby("apo_id")
    df["blend"] = 0.5 * g["tabicl"].rank(method="average", pct=True) \
        + 0.5 * g["full_alone"].rank(method="average", pct=True)
    single = single_chain_pdbs()
    return df[df["apo_id"].str[:4].str.lower().isin(single)].reset_index(drop=True)


def calls(df: pd.DataFrame) -> tuple[dict, dict]:
    """Binary predictions at each model's own operating point, and a description of it."""
    g = df["apo_id"].to_numpy()
    t9 = json.loads(Path("finetuned_esm/step9_thresholds.json").read_text())["s2pfull_117"]["frozen_threshold"]
    t10 = json.loads(Path("finetuned_esm/step10_thresholds.json").read_text())["ft117"]["frozen_threshold"]
    rank_thr = json.loads((C.PROJECT_ROOT / "operating_point/frozen.json").read_text()
                          )["rules"]["rank"]["frozen_threshold"]
    pred = {"released": df["released"].to_numpy() > 0.7,
            "stack9": df["stack9"].to_numpy() >= t9,
            "ft117": df["ft117"].to_numpy() >= t10,
            "blend": op.transform("rank", df["blend"].to_numpy(), g) >= rank_thr}
    desc = {"released": "score > 0.7 (theirs; chosen on their test set)",
            "stack9": f"score >= {t9:.4f} (step 9, CV-frozen)",
            "ft117": f"score >= {t10:.4f} (step 10, CV-frozen)",
            "blend": f"top {100 * op.top_q_from_threshold(rank_thr):.2f} % of each protein "
                     "(TabICL's CV-frozen rank rule; approximate for a blend)"}
    return {k: v.astype(int) for k, v in pred.items()}, desc


def threshold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    return {"MCC": metrics.matthews_corrcoef(y, p),
            "F1_weighted": metrics.f1_score(y, p, average="weighted"),
            "ACC": (tp + tn) / len(y), "TPR": tp / max(tp + fn, 1), "FPR": fp / max(fp + tn, 1)}


def pooled_all(y, s, p) -> dict:
    prec, rec, _ = metrics.precision_recall_curve(y, s)
    return {"AUC": metrics.roc_auc_score(y, s), "AUPRC": metrics.auc(rec, prec)} | threshold_metrics(y, p)


def per_protein_table(y, s, p, g) -> pd.DataFrame:
    rows = []
    for prot in pd.unique(g):
        m = g == prot
        yy, ss, pp = y[m], s[m], p[m]
        prec, rec, _ = metrics.precision_recall_curve(yy, ss)
        rows.append({"apo_id": prot, "AUC": metrics.roc_auc_score(yy, ss),
                     "AUPRC": metrics.auc(rec, prec)} | threshold_metrics(yy, pp))
    return pd.DataFrame(rows).set_index("apo_id")


def main() -> None:
    df = load()
    y, g = df["y"].to_numpy(), df["apo_id"].to_numpy()
    pred, desc = calls(df)
    n_prot = len(pd.unique(g))
    assert n_prot == 184, n_prot

    point = {"pooled": {}, "per_protein": {}}
    pp_tab = {}
    for m in MODELS:
        s = df[m].to_numpy()
        point["pooled"][m] = pooled_all(y, s, pred[m])
        pp_tab[m] = per_protein_table(y, s, pred[m], g)
        point["per_protein"][m] = pp_tab[m].mean(skipna=True).to_dict()

    # ---- gate: their model reproduces their paper; ours reproduce their recorded results
    # Their test file has 5 residues our table lacks (55,572 vs 55,567 rows; 3,158 vs 3,157
    # positives), so the gate is two links: (a) their model on THEIR file (step 4, already
    # computed) rounds to every published number; (b) on these 184 proteins it is within
    # 0.0005 of (a) -- the residue difference, nothing else.
    rel = point["pooled"]["released"]
    a = json.loads(Path("finetuned_esm/step4_reproduction.json").read_text())["A_their_file_their_labels"]
    gate = {"paper_vs_their_file": {k: {"published": v, "their_file": round(a[k], 5),
                                        "ok": round(a[k], 3) == v} for k, v in PUBLISHED_PAPER.items()},
            "notebook_vs_their_file": {k: {"published": v, "their_file": round(a[k], 5),
                                           "ok": abs(a[k] - v) <= 0.0005}
                                       for k, v in PUBLISHED_NOTEBOOK.items()},
            "these_184_vs_their_file": {k: {"their_file": round(a[k], 5), "these_184": round(rel[k], 5),
                                            "ok": abs(rel[k] - a[k]) <= 0.0005}
                                        for k in PUBLISHED_NOTEBOOK},
            "recorded": {m: {k: {"recorded": v,
                                 "ours": round(point["per_protein"][m]["AUPRC"] if k == "pp_AUPRC"
                                               else point["pooled"][m][k.split("_")[1]], 4)}
                             for k, v in r.items()} for m, r in RECORDED.items()},
            "n_residues": int(len(y)), "positives": int(y.sum())}
    for m, r in gate["recorded"].items():
        for k, v in r.items():
            v["ok"] = abs(v["ours"] - v["recorded"]) <= 0.0005
    oks = [v["ok"] for b in ("paper_vs_their_file", "notebook_vs_their_file", "these_184_vs_their_file")
           for v in gate[b].values()] + \
          [v["ok"] for r in gate["recorded"].values() for v in r.values()]
    print(json.dumps(gate, indent=1))
    if not all(oks):
        raise SystemExit("GATE FAILED: a model does not reproduce its published/recorded number")

    # ---- bootstrap: same protein resample for every model; paired differences vs released
    rng = np.random.default_rng(SEED)
    prots = np.array(pd.unique(g))
    rows_of = {p: np.flatnonzero(g == p) for p in prots}
    boot = {kind: {m: {k: np.empty(N_BOOT) for k in METRICS} for m in MODELS}
            for kind in ("pooled", "per_protein")}
    pp_arr = {m: pp_tab[m].loc[prots] for m in MODELS}
    for b in range(N_BOOT):
        pick = rng.choice(len(prots), len(prots))
        r = np.concatenate([rows_of[p] for p in prots[pick]])
        for m in MODELS:
            v = pooled_all(y[r], df[m].to_numpy()[r], pred[m][r])
            w = pp_arr[m].iloc[pick].mean(skipna=True)
            for k in METRICS:
                boot["pooled"][m][k][b] = v[k]
                boot["per_protein"][m][k][b] = w[k]
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    res = {"subset": "CB-P2RANK-apo (Škrhák et al.'s test set)", "n_proteins": n_prot,
           "n_residues": int(len(y)), "operating_points": desc, "gate": gate, "n_boot": N_BOOT}
    for kind in ("pooled", "per_protein"):
        res[kind] = {}
        for m in MODELS:
            res[kind][m] = {k: {"value": float(point[kind][m][k]), "ci": ci(boot[kind][m][k])}
                            for k in METRICS}
        res[kind]["diff_vs_released"] = {
            m: {k: {"diff": float(point[kind][m][k] - point[kind]["released"][k]),
                    "ci": ci(boot[kind][m][k] - boot[kind]["released"][k])} for k in METRICS}
            for m in MODELS if m != "released"}
    Path("finetuned_esm/final_comparison.json").write_text(json.dumps(res, indent=1))

    for kind in ("pooled", "per_protein"):
        print(f"\n== {kind} ({n_prot} proteins, {len(y):,} residues)")
        print(f"{'':12s}" + "".join(f"{k:>13s}" for k in METRICS))
        for m in MODELS:
            print(f"{m:12s}" + "".join(f"{res[kind][m][k]['value']:>13.4f}" for k in METRICS))
        for m, d in res[kind]["diff_vs_released"].items():
            print(f"{m + '-rel':12s}" + "".join(
                f"{d[k]['diff']:+7.3f}{'*' if d[k]['ci'][0] > 0 or d[k]['ci'][1] < 0 else ' ':5s} "
                for k in METRICS))


if __name__ == "__main__":
    main()
