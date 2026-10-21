"""PAM against Škrhák et al.'s released fine-tuned ESM-2, on three subsets of the test set.

SUBSETS
    CB-full          all 231 test proteins
    CB-P2RANK-apo    the 184 single-chain structures (their papers' test set)
    multi-chain      CB-full minus CB-P2RANK-apo: the 47 structures their papers leave out

MODELS (test scores already computed; nothing is re-run)
    ft117      PAM: TabICLv2, 53 features + 64 PCs of the fine-tuned embedding (step 10)
    released   their fine-tuned ESM-2 3B, multitask + LIGYSIS (cbs-model.pt), step 4/5

Operating points, metrics and bootstrap are scripts/final_comparison.py's: PAM at its
CV-frozen 0.3407, theirs at their own 0.7; seven metrics, pooled and per protein; 95 % CIs
and paired PAM − released differences from 2,000 protein resamples, seed 0.

Run: python scripts/readme_comparison.py   ->  finetuned_esm/readme_comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_comparison as F  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

MODELS = ("ft117", "released")


def subsets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    single = df["apo_id"].str[:4].str.lower().isin(single_chain_pdbs()).to_numpy()
    return {"CB-full": np.ones(len(df), bool), "CB-P2RANK-apo": single, "multi-chain": ~single}


def score(df: pd.DataFrame, pred: dict) -> dict:
    y, g = df["y"].to_numpy(), df["apo_id"].to_numpy()
    point = {"pooled": {}, "per_protein": {}}
    pp_tab = {}
    for m in MODELS:
        s = df[m].to_numpy()
        point["pooled"][m] = F.pooled_all(y, s, pred[m])
        pp_tab[m] = F.per_protein_table(y, s, pred[m], g)
        point["per_protein"][m] = pp_tab[m].mean(skipna=True).to_dict()

    rng = np.random.default_rng(F.SEED)
    prots = np.array(pd.unique(g))
    rows_of = {p: np.flatnonzero(g == p) for p in prots}
    pp_arr = {m: pp_tab[m].loc[prots] for m in MODELS}
    boot = {kind: {m: {k: np.empty(F.N_BOOT) for k in F.METRICS} for m in MODELS}
            for kind in ("pooled", "per_protein")}
    for b in range(F.N_BOOT):
        pick = rng.choice(len(prots), len(prots))
        r = np.concatenate([rows_of[p] for p in prots[pick]])
        for m in MODELS:
            v = F.pooled_all(y[r], df[m].to_numpy()[r], pred[m][r])
            w = pp_arr[m].iloc[pick].mean(skipna=True)
            for k in F.METRICS:
                boot["pooled"][m][k][b] = v[k]
                boot["per_protein"][m][k][b] = w[k]
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    res = {"n_proteins": len(prots), "n_residues": int(len(y)), "positives": int(y.sum())}
    for kind in ("pooled", "per_protein"):
        res[kind] = {m: {k: {"value": float(point[kind][m][k]), "ci": ci(boot[kind][m][k])}
                         for k in F.METRICS} for m in MODELS}
        res[kind]["ft117 - released"] = {
            k: {"diff": float(point[kind]["ft117"][k] - point[kind]["released"][k]),
                "ci": ci(boot[kind]["ft117"][k] - boot[kind]["released"][k])} for k in F.METRICS}
    return res


def main() -> None:
    df = F.load(single_chain_only=False)
    pred, desc = F.calls(df)
    out = {"operating_points": {m: desc[m] for m in MODELS}, "n_boot": F.N_BOOT}
    for name, mask in subsets(df).items():
        sub = df[mask].reset_index(drop=True)
        out[name] = score(sub, {m: pred[m][mask] for m in MODELS})
        r = out[name]
        print(f"\n== {name}: {r['n_proteins']} proteins, {r['n_residues']:,} residues")
        for kind in ("per_protein", "pooled"):
            print(f"  {kind:12s}" + "".join(f"{k:>13s}" for k in F.METRICS))
            for m in MODELS:
                print(f"  {m:12s}" + "".join(f"{r[kind][m][k]['value']:>13.4f}" for k in F.METRICS))
            d = r[kind]["ft117 - released"]
            print(f"  {'PAM - rel':12s}" + "".join(
                f"{d[k]['diff']:+7.3f}{'*' if d[k]['ci'][0] > 0 or d[k]['ci'][1] < 0 else ' ':6s}"
                for k in F.METRICS))
    assert out["CB-full"]["n_proteins"] == 231 and out["CB-P2RANK-apo"]["n_proteins"] == 184
    Path("finetuned_esm/readme_comparison.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
