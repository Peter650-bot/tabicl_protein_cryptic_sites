"""Step 11 of finetuned_esm/PLAN.md: which stored hidden-state row is which table row.

The PLS projection is fit on Modal, where the fine-tuned models' hidden states live
(`hidden_cleantest_k.npy`, ~0.8 GB each; this host has 5 GB of RAM and a slow link).
This writes the small map it needs: for every train-fold-k and test row of
`residues_ftemb.parquet`, the row of `hidden_cleantest_k.npy` holding its residue.

Row r of `ftemb_cleantest_k.parquet` is row r of the hidden-state file
(`_embed_and_reduce` writes both from one index), so the map is step 10's own join
(`scripts/ftemb_build_table.py`) with the row number carried through; it is checked
by requiring the PCs it points at to equal the table's `ftk{k}_pc` columns.

Run: python scripts/ftpls_build_index.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C  # noqa: E402

STAGE_DIR = Path.home() / "cryptobench_stage" / "tables"
KEY = ["apo_id", "chain", "seq_id", "icode"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s2p_full_cleantest_20261008")
    a = ap.parse_args()
    pcs = [f"ftk{{k}}_pc{i:02d}" for i in range(64)]
    table = pd.read_parquet(C.BUILD_DIR / "residues_ftemb.parquet",
                            columns=KEY + ["fold"] + [c.format(k=k) for k in range(4) for c in pcs])
    key_index = pd.MultiIndex.from_frame(table[KEY])
    inputs = pd.read_parquet(C.BUILD_DIR / "s2p_inputs.parquet",
                             columns=["apo_id", "chain", "pos", "seq_id", "icode"])
    fold = table["fold"].to_numpy()
    out = []
    for k in range(4):
        emb = pd.read_parquet(C.BUILD_DIR / "s2p_ft" / a.run_id / f"ftemb_cleantest_{k}.parquet")
        emb["hrow"] = np.arange(len(emb))
        e = inputs.merge(emb, on=["apo_id", "chain", "pos"], how="inner", validate="1:1")
        e = e.drop_duplicates(KEY, keep=False)
        rows = np.flatnonzero((fold == f"train-fold-{k}") | (fold == "test"))
        sub = e.set_index(KEY).reindex(key_index[rows])
        assert sub["hrow"].notna().all(), f"k{k}: a fold-{k}/test row has no hidden state"
        hrow = sub["hrow"].to_numpy(np.int64)
        got = emb[[f"ftpc_{i:02d}" for i in range(64)]].to_numpy(np.float32)[hrow]
        want = table.loc[rows, [c.format(k=k) for c in pcs]].to_numpy(np.float32)
        assert np.array_equal(got, want), f"k{k}: hidden rows do not point at the table's PCs"
        out.append(pd.DataFrame({"k": k, "table_row": rows.astype(np.int32),
                                 "hrow": hrow.astype(np.int32)}))
        print(f"k{k}: {len(rows):,} rows mapped, PCs identical")
    out = pd.concat(out, ignore_index=True)
    path = C.BUILD_DIR / "ftpls_rows.parquet"
    out.to_parquet(path, index=False)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, STAGE_DIR / path.name)
    print(f"wrote {path} ({len(out):,} rows) and staged it")


if __name__ == "__main__":
    main()
