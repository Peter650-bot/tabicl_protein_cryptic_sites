"""Step 8 of finetuned_esm/PLAN.md: the out-of-fold score as a column of our table.

    train-fold-h rows  <- fold model h   (trained on the other three folds)
    test rows          <- model `all`    (trained on all four; never saw test)

Every value therefore comes from a model that never trained on that row. The
table is `residues_esm64.parquet` -- the published 117 columns in the published
order -- with `s2p_oof` appended and the row order untouched, written to
build/residues_s2p.parquet and staged for Modal.

Step 9 adds a second column, `s2pfull_oof`, from the clean full-recipe models
(`clean_0..3`, `clean_all`): it reads the step-8 table and writes a NEW file,
build/residues_s2pfull.parquet, so the step-8 table (whose sha256 frozen_step8.json
records) is never modified. Existing columns and row order are kept.

Run: python scripts/s2p_build_stack_table.py                                    # step 8
     python scripts/s2p_build_stack_table.py --run-id s2p_full_20261006 \
            --column s2pfull_oof --prefix clean_                                  # step 9
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
from s2p_ft_check import keyed_scores  # noqa: E402

SOURCE = {f"train-fold-{h}": str(h) for h in range(4)} | {"test": "all"}
STAGE_DIR = Path.home() / "cryptobench_stage" / "tables"
KEY = ["apo_id", "chain", "seq_id", "icode"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s2p_ft_20261006")
    ap.add_argument("--column", default="s2p_oof")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    step8 = a.column == "s2p_oof"
    src = C.BUILD_DIR / ("residues_esm64.parquet" if step8 else "residues_s2p.parquet")
    out = C.BUILD_DIR / ("residues_s2p.parquet" if step8 else "residues_s2pfull.parquet")
    table = pd.read_parquet(src)
    assert set(table["fold"].unique()) == set(SOURCE), sorted(table["fold"].unique())
    key_index = pd.MultiIndex.from_frame(table[KEY])
    assert key_index.is_unique

    oof = np.full(len(table), np.nan)
    report = {}
    for fold, model in SOURCE.items():
        keyed, _ = keyed_scores(C.BUILD_DIR / "s2p_ft" / a.run_id / f"scores_{a.prefix}{model}.parquet")
        lookup = pd.Series(keyed["score"].to_numpy(), index=pd.MultiIndex.from_frame(keyed[KEY]))
        rows = np.flatnonzero(table["fold"].to_numpy() == fold)
        oof[rows] = lookup.reindex(key_index[rows]).to_numpy()
        y = table[C.LABEL_COL].to_numpy()[rows]
        auc, auprc = pooled(y, oof[rows])
        report[fold] = {"model": a.prefix + model, "rows": len(rows), "pooled_AUC": round(float(auc), 4),
                        "pooled_AUPRC": round(float(auprc), 4)}
    assert not np.isnan(oof).any(), "a row has no out-of-fold score"
    table[a.column] = oof.astype(np.float32)
    table.to_parquet(out, index=False)
    stage = STAGE_DIR / out.name
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, stage)
    report["_file"] = {"rows": len(table), "columns": table.shape[1],
                       "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                       "staged": str(stage)}
    tag = "step8" if step8 else "step9"
    Path(f"finetuned_esm/{tag}_stack_table.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
