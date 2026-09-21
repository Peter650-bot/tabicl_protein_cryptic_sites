"""Emit one residue-centroid coordinate table, so neighbourhood pooling is free.

WHY THIS EXISTS
    Every neighbour-pooled feature in this repo builds its own KD-tree inside
    its own structure loop -- `features._add_neighbourhood_means`,
    `ensemble_features._add_neighbourhood` and `conservation.add_neighbourhood`
    are three near-identical copies. That works for families computed from the
    structure, and fails for families that are not: the 64 ESM principal
    components arrive from a GPU job that never sees a coordinate, which is
    exactly why they are the one family with no spatial pooling at all.

    Writing the centroids once turns pooling into a pandas join. Any later
    family can be pooled at any radius with any statistic, at zero structural
    cost.

CENTROIDS, NOT CA
    `features.py:271` passes residue CENTROIDS to its KD-tree, not alpha
    carbons. Anything built here has to agree with the published `_nb8` columns,
    so it uses `ResidueRecord.centroid` -- the mean over every atom of the
    residue, the same quantity, computed the same way.

KEYS
    The join key is (apo_id, chain, seq_id, icode), with icode part of the
    identity. 3bv9 chain B has ten residues at position 60; dropping the
    insertion code collapses them (RESULTS.md section 7).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C  # noqa: E402
from cryptic_tabicl import dataset as ds_mod  # noqa: E402
from cryptic_tabicl.structures import (  # noqa: E402
    fetch_cif, load_structure, residue_records,
)


def _coords_one(apo_id: str, pdb_id: str, chains: Sequence[str]) -> List[Dict]:
    """Centroid of every residue this apo structure contributes to the table."""
    st = load_structure(fetch_cif(pdb_id))
    rows: List[Dict] = []
    for rec in residue_records(st, chains):
        cx, cy, cz = rec.centroid
        rows.append({
            "apo_id": apo_id,
            "chain": rec.chain,
            "seq_id": rec.seq_id,
            "icode": rec.icode,
            "cx": float(cx), "cy": float(cy), "cz": float(cz),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path,
                    default=C.BUILD_DIR / "residues_esm64.parquet",
                    help="the built table whose rows must all be covered")
    ap.add_argument("--out", type=Path, default=C.BUILD_DIR / "residue_coords.parquet")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    base = pd.read_parquet(args.table, columns=["apo_id", "chain", "seq_id", "icode"])
    base["icode"] = base["icode"].fillna("").astype(str)
    targets = sorted(base["apo_id"].unique())

    ds = ds_mod.load_json(C.DATASET_JSON)
    chains = ds_mod.apo_chains(ds)
    pdb_of = ds_mod.apo_pdb_ids(ds)

    print(f"centroids for {len(targets)} apo structures on {args.workers} workers ...")
    rows: List[Dict] = []
    failed: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_coords_one, aid, pdb_of[aid], sorted(chains[aid])): aid
            for aid in targets
        }
        for i, fut in enumerate(as_completed(futs), 1):
            aid = futs[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001 - one bad entry must not kill the run
                failed[aid] = f"{type(exc).__name__}: {exc}"
            if i % 100 == 0:
                print(f"  {i}/{len(targets)} ({len(failed)} failed)", flush=True)

    coords = pd.DataFrame(rows)
    coords["icode"] = coords["icode"].fillna("").astype(str)

    keys = ["apo_id", "chain", "seq_id", "icode"]
    if coords.duplicated(keys).any():
        raise SystemExit("duplicate residue keys; pooling would fan out")

    # The gate that matters: every row of the built table must get a centroid.
    # A partial coordinate table would silently make some residues' neighbour
    # pools smaller than others', which is a bias, not a gap.
    merged = base.merge(coords, on=keys, how="left", validate="1:1")
    missing = int(merged["cx"].isna().sum())
    print(f"\n{len(coords):,} centroids / {coords['apo_id'].nunique()} proteins")
    print(f"table rows without a centroid: {missing:,}")
    for aid, err in list(failed.items())[:10]:
        print(f"  failed {aid}: {err}")
    if missing:
        raise SystemExit(f"{missing} rows have no centroid; refusing to write a partial table")

    coords.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(coords):,} rows)")


if __name__ == "__main__":
    main()
