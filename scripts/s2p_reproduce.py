"""Step 4 gate of finetuned_esm/PLAN.md: does their model, as we run it, give their numbers?

A. Their rebuilt test file, their labels, their metric code path (everything pooled
   into one vector; AUPRC as `metrics.auc(recall, precision)`; prediction is
   `sigmoid > 0.7`; F1 weighted). Target: the seq2pocket notebook output
   `Accuracy: 93.51% | AUC: 0.8935, MCC: 0.4724, F1: 0.9390, AUPRC: 0.4752`.
B. The same scores joined to OUR table on author keys, our labels, CB-P2RANK-apo.

Also writes build/s2p_scores_keyed.parquet: every row of our table with its score,
joined one-to-one (asserted), for every later step.
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
from cryptic_tabicl import config as C  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

RUN_ID = "s2p_20261006"
KEY = ["apo_id", "chain", "seq_id", "icode"]
PUBLISHED = {"AUC": 0.8935, "AUPRC": 0.4752, "MCC": 0.4724, "F1_weighted": 0.9390,
             "ACC": 0.9351}
THEIR_THRESHOLD = 0.7


def pooled(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    fpr, tpr, _ = metrics.roc_curve(y, s)
    prec, rec, _ = metrics.precision_recall_curve(y, s)
    pred = (s > thr).astype(int)
    return {"AUC": metrics.auc(fpr, tpr), "AUPRC": metrics.auc(rec, prec),
            "MCC": metrics.matthews_corrcoef(y, pred),
            "F1_weighted": metrics.f1_score(y, pred, average="weighted"),
            "ACC": float((pred == y).mean()), "n": int(len(y)), "positives": int(y.sum())}


def main() -> None:
    scores = pd.read_parquet(C.BUILD_DIR / f"{RUN_ID}.parquet")
    inputs = pd.read_parquet(C.BUILD_DIR / "s2p_inputs.parquet")
    assert len(scores) == len(inputs), "score rows != input rows"
    keyed = inputs.merge(scores, on=["apo_id", "chain", "pos"], how="left", validate="1:1")
    assert keyed["s2p_score"].notna().all(), "an input residue has no score"

    # ---- A. their file, their labels
    rows = []
    for line in (C.BUILD_DIR / "s2p_their_test.txt").read_text().splitlines():
        pdb, chain, _, lab, seq = line.split(";")
        pos = {int(x[1:]) for x in lab.split()} if lab else set()
        rows += [(pdb + chain, chain, i, int(i in pos)) for i in range(len(seq))]
    their = pd.DataFrame(rows, columns=["apo_id", "chain", "pos", "y"]).merge(
        scores, on=["apo_id", "chain", "pos"], how="left", validate="1:1")
    assert their["s2p_score"].notna().all()
    a = pooled(their["y"].to_numpy(), their["s2p_score"].to_numpy(), THEIR_THRESHOLD)
    a["delta_vs_published"] = {k: round(a[k] - v, 4) for k, v in PUBLISHED.items()}

    # ---- B. our table, our labels, via author keys
    table = pd.read_parquet(C.BUILD_DIR / "residues_esm64.parquet",
                            columns=KEY + ["pdb_id", "split", C.LABEL_COL])
    # Their sequences hold 2 duplicated author keys (4z0y B/D 252, modelled as
    # both HIS and HS8). Dropping them is only safe if neither is a residue we
    # evaluate, so that is asserted rather than assumed.
    dup = keyed[keyed.duplicated(KEY, keep=False)]
    assert dup[KEY].merge(table[KEY], on=KEY).empty, "a duplicated key is in our table"
    keyed = keyed.drop(dup.index)
    ours = table.merge(keyed[KEY + ["s2p_score"]], on=KEY, how="left", validate="1:1")
    assert ours["s2p_score"].notna().all(), "a table residue has no score"
    ours.to_parquet(C.BUILD_DIR / "s2p_scores_keyed.parquet", index=False)
    single = single_chain_pdbs()
    sub = ours[(ours["split"] == "test") & ours["pdb_id"].str.lower().isin(single)]
    b = pooled(sub[C.LABEL_COL].to_numpy(), sub["s2p_score"].to_numpy(), THEIR_THRESHOLD)
    b["n_proteins"] = int(sub["apo_id"].nunique())

    out = {"run_id": RUN_ID, "published": PUBLISHED,
           "A_their_file_their_labels": a, "B_our_table_CB_P2RANK_apo": b}
    Path("finetuned_esm/step4_reproduction.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
