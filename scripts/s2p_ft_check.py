"""Gates 8c and 8d of finetuned_esm/PLAN.md, read from a fine-tune job's scores.

8c  model `all` (their setup): their rebuilt test file + their labels, pooled, must be
    within ±0.015 AUC and ±0.03 AUPRC of their published 0.888 / 0.446.
8d  fold model h: the proteins of fold h must score like unseen proteins, not like
    memorised ones. Every job scores every chain, so in-sample folds sit beside the
    held-out one in the same table for comparison.

Usage: python scripts/s2p_ft_check.py --run-id s2p_ft_20261006 --name all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from s2p_evaluate import pooled  # noqa: E402

KEY = ["apo_id", "chain", "seq_id", "icode"]
TARGET = {"AUC": 0.888, "AUPRC": 0.446}
TOL = {"AUC": 0.015, "AUPRC": 0.03}


def keyed_scores(path: Path) -> pd.DataFrame:
    """Join a job's (apo_id, chain, pos) scores onto our table rows."""
    scores = pd.read_parquet(path)
    inputs = pd.read_parquet(C.BUILD_DIR / "s2p_inputs.parquet")
    assert len(scores) == len(inputs), "job did not score every input residue"
    k = inputs.merge(scores, on=["apo_id", "chain", "pos"], how="left", validate="1:1")
    assert k["score"].notna().all()
    k = k.drop_duplicates(KEY, keep=False)   # 4z0y HIS/HS8 pair; neither is in our table
    table = pd.read_parquet(C.BUILD_DIR / "residues_esm64.parquet",
                            columns=KEY + ["split", "fold", C.LABEL_COL])
    out = table.merge(k[KEY + ["score"]], on=KEY, how="left", validate="1:1")
    assert out["score"].notna().all(), "a table residue has no score"
    return out, scores


def their_test_pooled(scores: pd.DataFrame) -> dict:
    rows = []
    for line in (C.BUILD_DIR / "s2p_their_test.txt").read_text().splitlines():
        pdb, chain, _, lab, seq = line.split(";")
        pos = {int(x[1:]) for x in lab.split()} if lab else set()
        rows += [(pdb + chain, chain, i, int(i in pos)) for i in range(len(seq))]
    t = pd.DataFrame(rows, columns=["apo_id", "chain", "pos", "y"]).merge(
        scores, on=["apo_id", "chain", "pos"], how="left", validate="1:1")
    auc, auprc = pooled(t["y"].to_numpy(), t["score"].to_numpy())
    return {"AUC": float(auc), "AUPRC": float(auprc)}


def by_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(multichain=df["apo_id"].str.contains("-"))
    rows = []
    for (fold, mc), g in df.groupby(["fold", "multichain"]):
        pp = evaluate.per_protein(g[C.LABEL_COL], g["score"], g["apo_id"], 0.5)
        auc, auprc = pooled(g[C.LABEL_COL].to_numpy(), g["score"].to_numpy())
        rows.append({"fold": fold, "multichain": bool(mc), "n": g["apo_id"].nunique(),
                     "pp_AUC": pp["AUC"].mean(), "pp_AUPRC": pp["AUPRC"].mean(),
                     "pooled_AUC": auc, "pooled_AUPRC": auprc})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    path = C.BUILD_DIR / "s2p_ft" / a.run_id / f"scores_{a.name}.parquet"
    ours, scores = keyed_scores(path)
    res = {"run_id": a.run_id, "name": a.name, "by_fold": by_group(ours).round(4).to_dict("records")}
    if a.name == "all":
        r = their_test_pooled(scores)
        res["gate_8c"] = {"their_test_pooled": r, "target": TARGET, "tolerance": TOL,
                          "passes": all(abs(r[m] - TARGET[m]) <= TOL[m] for m in TARGET)}
    out = Path(f"finetuned_esm/step8_check_{a.name}.json")
    out.write_text(json.dumps(res, indent=1, default=float))
    print(pd.DataFrame(res["by_fold"]).to_string(index=False))
    if "gate_8c" in res:
        print("\n8c:", json.dumps(res["gate_8c"], indent=1))


if __name__ == "__main__":
    main()
