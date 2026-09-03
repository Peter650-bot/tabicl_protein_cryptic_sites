"""Compute ANM-ensemble pocket features for every apo structure, on Modal.

WHY REMOTE
    ~1,135 proteins x 25 fpocket invocations is ~28,000 runs. Serially on the
    4-core development box that is most of a day; fanned out over Modal
    containers it is minutes. fpocket is also a compiled binary that isn't
    installed locally, and bioconda provides it cleanly in a container.

    This is CPU work, not GPU -- fpocket and the ANM eigendecomposition are both
    CPU-bound, so no accelerator is requested.

STRUCTURES
    mmCIFs are mounted from the local cache built by `build_table`, rather than
    re-fetched from RCSB inside 30-odd containers. That keeps the run
    reproducible and avoids hammering a public service.

OUTPUT
    One parquet of per-residue ensemble features keyed by (apo_id, chain,
    seq_id), for `scripts/merge_ensemble.py` to join onto the main table. It is
    kept separate so the expensive step runs once and can be re-joined freely.
"""
from __future__ import annotations

import os
import sys

import modal

STRUCTURE_CACHE = os.path.expanduser("~/cryptobench_stage/structures")
if modal.is_local() and not os.path.isdir(STRUCTURE_CACHE):
    raise SystemExit(
        f"missing structure cache at {STRUCTURE_CACHE}\n"
        "Run `python -m cryptic_tabicl.build_table` first; it populates the cache."
    )

image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("fpocket", channels=["conda-forge", "bioconda"])
    .pip_install("gemmi", "numpy<2", "scipy", "pandas", "pyarrow")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
    .add_local_dir(STRUCTURE_CACHE, "/structures")
)

app = modal.App("cryptobench-ensemble")


@app.function(image=image, timeout=5400, cpu=2.0, max_containers=40, retries=1)
def shard(payload: dict) -> list:
    """Ensemble features for one batch of apo structures."""
    sys.path.insert(0, "/app")
    import time
    from pathlib import Path

    from cryptic_tabicl.ensemble_features import compute_ensemble_features
    from cryptic_tabicl.structures import load_structure, residue_records

    n_conf = payload["n_conformations"]
    target_rmsd = payload.get("target_rmsd", 1.5)
    out = []
    for item in payload["items"]:
        apo_id, pdb_id, chains = item["apo_id"], item["pdb_id"], item["chains"]
        t0 = time.time()
        try:
            st = load_structure(Path(f"/structures/{pdb_id.lower()}.cif.gz"))
            recs = residue_records(st, chains)
            if not recs:
                continue
            feats = compute_ensemble_features(recs, n_conformations=n_conf,
                                              target_rmsd=target_rmsd,
                                              seed=abs(hash(apo_id)) % (2 ** 31))
            for rec, f in zip(recs, feats):
                row = {"apo_id": apo_id, "chain": rec.chain, "seq_id": rec.seq_id}
                row.update(f)
                out.append(row)
            print(f"{apo_id}: {len(recs)} residues in {time.time()-t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad entry must not kill a shard
            print(f"{apo_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
    return out


@app.local_entrypoint()
def main(n_conformations: int = 24, batch_size: int = 30, limit: int = 0,
         target_rmsd: float = 1.5, tag: str = ""):
    import json
    import time

    sys.path.insert(0, ".")
    import pandas as pd
    from cryptic_tabicl import config as C, dataset as D

    ds = D.load_json(C.DATASET_JSON)
    chains = D.apo_chains(ds)
    pdb_of = D.apo_pdb_ids(ds)

    targets = sorted(chains)
    if limit:
        targets = targets[:limit]

    items = [{"apo_id": a, "pdb_id": pdb_of[a], "chains": sorted(chains[a])}
             for a in targets]
    batches = [{"items": items[i:i + batch_size],
                "n_conformations": n_conformations,
                "target_rmsd": target_rmsd}
               for i in range(0, len(items), batch_size)]

    print(f"{len(items)} apo structures -> {len(batches)} batches "
          f"x {n_conformations} conformations @ {target_rmsd} A RMSD "
          f"(~{len(items) * (n_conformations + 1):,} fpocket runs)")

    t0 = time.time()
    rows = []
    for i, result in enumerate(shard.map(batches), 1):
        rows.extend(result)
        print(f"  batch {i}/{len(batches)}  ({len(rows):,} rows)", flush=True)

    df = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    out = C.BUILD_DIR / f"ensemble_features{suffix}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(df):,} rows, {time.time()-t0:.0f}s)")
    if not df.empty:
        cols = [c for c in df.columns if c.startswith("ens_")]
        print(df[cols].describe().T[["mean", "std", "min", "max"]].to_string())
