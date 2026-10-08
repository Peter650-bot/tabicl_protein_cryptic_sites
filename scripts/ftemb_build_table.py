"""Step 10 of finetuned_esm/PLAN.md: the fine-tuned model's embedding as table columns.

    block k  (ftk{k}_pc00..63, ftk{k}_score)  <- fold model cleantest_k
             filled on train-fold-k rows and test rows only, NaN elsewhere

Model k never trained on fold k (nor on LIGYSIS homologs of fold k or of test), so
both of the row sets it fills are rows it never saw. Its 64 PCs were fit on fold-k
chains only (`deploy/s2p_finetune_modal.py::_embed_and_reduce`). Blocks of different
k are different embedding spaces and are never mixed: `deploy/ftemb_stack.py` renames
block k to `ft_pc..`/`ft_score` for jobs whose context is fold k and drops the rest.

The base is `residues_esm64.parquet` -- the published 117 in the published order --
with row order untouched, written to build/residues_ftemb.parquet and staged.

Run: python scripts/ftemb_build_table.py --run-id s2p_full_cleantest_20261008
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryptic_tabicl import config as C  # noqa: E402
from s2p_evaluate import pooled  # noqa: E402

STAGE_DIR = Path.home() / "cryptobench_stage" / "tables"
KEY = ["apo_id", "chain", "seq_id", "icode"]
N_PCS = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s2p_full_cleantest_20261008")
    a = ap.parse_args()
    table = pd.read_parquet(C.BUILD_DIR / "residues_esm64.parquet")
    key_index = pd.MultiIndex.from_frame(table[KEY])
    assert key_index.is_unique
    inputs = pd.read_parquet(C.BUILD_DIR / "s2p_inputs.parquet",
                             columns=["apo_id", "chain", "pos", "seq_id", "icode"])
    fold = table["fold"].to_numpy()
    y = table[C.LABEL_COL].to_numpy()

    report, blocks = {}, {}
    for k in range(4):
        emb = pd.read_parquet(C.BUILD_DIR / "s2p_ft" / a.run_id / f"ftemb_cleantest_{k}.parquet")
        e = inputs.merge(emb, on=["apo_id", "chain", "pos"], how="inner", validate="1:1")
        e = e.drop_duplicates(KEY, keep=False)      # 4z0y HIS/HS8 pair; neither is in our table
        lookup = e.set_index(KEY)
        rows = np.flatnonzero((fold == f"train-fold-{k}") | (fold == "test"))
        sub = lookup.reindex(key_index[rows])
        cols = [f"ftpc_{i:02d}" for i in range(N_PCS)] + ["score"]
        assert not sub[cols].isna().any().any(), f"block {k}: a fold-{k}/test row has no embedding"
        for c in cols:
            name = f"ftk{k}_pc{c[-2:]}" if c != "score" else f"ftk{k}_score"
            v = np.full(len(table), np.nan, dtype=np.float32)
            v[rows] = sub[c].to_numpy(np.float32)
            blocks[name] = v
        rep = {}
        for part in (f"train-fold-{k}", "test"):
            r = np.flatnonzero(fold == part)
            auc, auprc = pooled(y[r], blocks[f"ftk{k}_score"][r])
            rep[part] = {"rows": len(r), "score_pooled_AUC": round(float(auc), 4),
                         "score_pooled_AUPRC": round(float(auprc), 4)}
        report[f"cleantest_{k}"] = rep

    table = pd.concat([table, pd.DataFrame(blocks)], axis=1)
    out = C.BUILD_DIR / "residues_ftemb.parquet"
    table.to_parquet(out, index=False)
    stage = STAGE_DIR / out.name
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, stage)
    report["_file"] = {"rows": len(table), "columns": table.shape[1], "run_id": a.run_id,
                       "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "staged": str(stage)}
    Path("finetuned_esm/step10_ftemb_table.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
