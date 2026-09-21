"""Replace the unsupervised ESM basis with a supervised one, fold-nested.

WHY
    The 64 ESM columns are PCA components: directions of maximum VARIANCE in
    ESM-2-3B's 2,560-dimensional space. Variance in a protein language model is
    dominated by what the model mostly encodes -- residue identity, local motif,
    protein family -- and none of that is crypticity. Measured on the train
    split, 24 of the 64 components have univariate |AUC - 0.5| <= 0.02, and the
    block of components 32-63 is worth +0.007 AUPRC end to end against
    components 16-31's +0.023 (RESULTS.md section 3). The basis is spending
    budget on directions the task does not use.

    Partial least squares asks the other question: which directions of ESM space
    covary with the LABEL. Same linear family as PCA, same cost to apply, one
    changed objective. 32 supervised components replace 64 unsupervised ones, so
    the arm that uses them is SMALLER than the control -- 85 features against
    117.

WHY THIS RUNS ON MODAL AND NOT LOCALLY
    The raw fp16 vectors already live in the `cryptobench-esm` volume, so no
    re-embedding is needed. Bringing them down instead would mean moving 1.95 GB
    over a link that measured ~8 MB/min, to hold a 3.9 GB float32 matrix on a
    host with 5 GB of RAM. The reduce step for the unsupervised basis is on
    Modal for the same reason (`esm_modal.py:reduce_pca`); this is its
    supervised sibling and stays next to it. CPU only.

THIS IS THE ONE LABEL-USING STEP IN THE FEATURE WORK PACKAGE
    Everything in `cryptic_tabicl/derive.py` is a function of features and
    geometry, and `tests/test_no_leakage.py` proves it by permuting the label.
    This file cannot make that claim, so it makes a narrower one instead: each
    projection is fit WITHOUT the rows it will be evaluated on.

    That means one projection per nesting, not one projection:

        train-fold-0 : fit on train minus fold 0, used when fold 0 is held
        ...            (and likewise for folds 1, 2, 3)
        final        : fit on all of train, used only for the single test run

    Every block is written with a manifest recording which fold it excluded and
    a digest of the exact apo_ids it saw. `deploy/feature_sweep.py` looks its
    own held fold up in that manifest and refuses to run on a mismatch.

    Note in passing what this makes visible about the EXISTING columns: the
    unsupervised PCA in `esm_modal.py:200-205` is fit on all of `train`,
    including whichever fold a CV run holds out. It uses no labels, so it is not
    leakage in the sense that matters, and it is identical across every arm so
    it cannot bias a comparison -- but it is not fold-nested either, and that is
    worth saying out loud rather than leaving implied.

FIT ROWS ARE THE CONTEXT ROWS
    The projection is fit on exactly the rows `sampling.build_context` would
    draw for that nesting, at the same seed. The directions are then optimised
    for the data the model actually conditions on, and the fit set is provably a
    subset of the context, so it cannot contain a held-out or test residue.

USAGE
    modal run deploy/esm_supervised_modal.py --run-id run_388b16f423cd
"""
from __future__ import annotations

import os
import sys

import modal

INDEX_LOCAL = os.path.expanduser("~/cryptobench_stage/tables/residue_index.parquet")
if modal.is_local() and not os.path.exists(INDEX_LOCAL):
    raise SystemExit(
        f"missing {INDEX_LOCAL}\n\nStage it first:\n"
        "    python -c \"import pandas as pd; "
        "pd.read_parquet('build/residues_esm64.parquet', columns=["
        "'apo_id','chain','seq_id','icode','is_cryptic','in_noncryptic_pocket',"
        "'fold','split']).assign(icode=lambda d: d.icode.fillna('').astype(str))"
        ".to_parquet('build/residue_index.parquet', index=False)\"\n"
        f"    cp build/residue_index.parquet {os.path.dirname(INDEX_LOCAL)}/"
    )

N_COMPONENTS = 32
PREFIX = "esm_sup"
KEYS = ["apo_id", "chain", "seq_id", "icode"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("scikit-learn", "pandas", "pyarrow", "numpy<2")
    .add_local_file(INDEX_LOCAL, "/data/index.parquet")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
)

app = modal.App("cryptobench-esm-supervised")
esm_vol = modal.Volume.from_name("cryptobench-esm")


def block_cols(held: str | None, n: int = N_COMPONENTS) -> list[str]:
    """Column names for one nesting. `held=None` is the all-of-train block."""
    return [f"{PREFIX}{i:02d}__fit_{held or 'final'}" for i in range(n)]


# 16 GiB is sized for the 3.9 GB float32 matrix plus PLS's float64 working
# copy of the 50,000-row fit block. The projection is applied in chunks below
# for the same reason: `PLSRegression.transform` centres and scales its input,
# which on the full matrix would materialise a second copy of it.
@app.function(image=image, timeout=7200, cpu=4.0, memory=16384,
              volumes={"/vol": esm_vol})
def fit_supervised(run_id: str, n_components: int = N_COMPONENTS,
                   budget: int = 50_000) -> tuple[bytes, dict]:
    sys.path.insert(0, "/app")
    import glob
    import hashlib
    import io
    import time

    import numpy as np
    import pandas as pd
    from sklearn.cross_decomposition import PLSRegression

    from cryptic_tabicl import config as C, sampling

    esm_vol.reload()
    files = sorted(glob.glob(f"/vol/{run_id}/emb_*.npz"))
    if not files:
        raise RuntimeError(f"no shards under /vol/{run_id}")
    print(f"reading {len(files)} shards from {run_id}", flush=True)

    embs, meta = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        if len(z["seq_id"]) == 0:
            continue
        embs.append(z["emb"])
        meta.append(pd.DataFrame({
            "apo_id": z["apo_id"].astype(str), "chain": z["chain"].astype(str),
            "seq_id": z["seq_id"].astype(int), "icode": z["icode"].astype(str)}))
    X = np.vstack(embs).astype(np.float32)
    del embs
    m = pd.concat(meta, ignore_index=True)
    m["icode"] = m["icode"].fillna("").astype(str)
    print(f"{X.shape[0]:,} residues x {X.shape[1]} dims", flush=True)

    # Two shard runs live in this volume and mixing them once produced a table
    # belonging to no single run (RESULTS.md section 9.5). Assert, do not trust.
    if m.duplicated(KEYS).any():
        raise RuntimeError("duplicate residue keys across shards -- runs mixed")

    table = pd.read_parquet("/data/index.parquet")
    table["icode"] = table["icode"].fillna("").astype(str)
    pos = m.reset_index().set_index(KEYS)["index"]
    idx = pos.reindex(pd.MultiIndex.from_frame(table[KEYS]))
    if idx.isna().any():
        raise RuntimeError(f"{int(idx.isna().sum())} table rows have no embedding")
    row_of_table = idx.to_numpy(dtype=int)

    folds = sorted(table.loc[table["split"] == "train", "fold"].dropna().unique())
    nestings = list(folds) + [None]
    print(f"nestings: {[n or 'final' for n in nestings]}", flush=True)

    out = table[KEYS].copy()
    manifest: dict = {}
    for held in nestings:
        name = held or "final"
        t0 = time.time()
        train = table[table["split"] == "train"]
        fit_src = train if held is None else train[train["fold"] != held]
        ctx, plan = sampling.build_context(fit_src, budget=budget,
                                           seed=C.RANDOM_SEED)

        # The assertions that make the nesting real rather than declared.
        if held is not None:
            assert (ctx["fold"] != held).all(), f"{name}: held fold reached the fit"
        assert (ctx["split"] == "train").all(), f"{name}: a test row reached the fit"

        rows = pos.reindex(pd.MultiIndex.from_frame(ctx[KEYS])).to_numpy(dtype=int)
        y = ctx[C.LABEL_COL].to_numpy(float)
        print(f"[{name}] {plan}", flush=True)

        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(X[rows], y)

        Z = np.empty((len(X), n_components), dtype=np.float32)
        for lo in range(0, len(X), 25_000):
            hi = min(lo + 25_000, len(X))
            Z[lo:hi] = pls.transform(X[lo:hi]).astype(np.float32)
        for i, col in enumerate(block_cols(held, n_components)):
            out[col] = Z[row_of_table, i]
        del Z

        apo = sorted(ctx[C.GROUP_COL].unique())
        manifest[name] = {
            "held": held,
            "n_fit_rows": int(len(rows)),
            "n_fit_proteins": len(apo),
            "n_fit_positives": int(y.sum()),
            "n_components": n_components,
            "columns": block_cols(held, n_components),
            "fit_apo_ids_sha256": hashlib.sha256("\n".join(apo).encode()).hexdigest(),
            "seed": C.RANDOM_SEED,
            "budget": budget,
            "seconds": time.time() - t0,
        }
        print(f"[{name}] done in {manifest[name]['seconds']:.0f}s", flush=True)

    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    return buf.getvalue(), {
        "source_run": run_id, "n_shards": len(files),
        "model": "facebook/esm2_t36_3B_UR50D", "method": "PLSRegression",
        "blocks": manifest,
    }


@app.local_entrypoint()
def main(run_id: str = "run_388b16f423cd", n_components: int = N_COMPONENTS,
         budget: int = 50_000):
    import json
    import time

    sys.path.insert(0, ".")
    from cryptic_tabicl import config as C

    t0 = time.time()
    blob, manifest = fit_supervised.remote(run_id, n_components, budget)

    out = C.BUILD_DIR / "esm_supervised.parquet"
    out.write_bytes(blob)
    (C.BUILD_DIR / "esm_supervised_manifest.json").write_text(
        json.dumps(manifest, indent=2))

    print(f"\n{time.time()-t0:.0f}s wall")
    for name, b in manifest["blocks"].items():
        print(f"  {name:>14}: fit on {b['n_fit_rows']:,} rows / "
              f"{b['n_fit_proteins']} proteins, held={b['held']}")
    print(f"\nwrote {out} ({len(blob)/1e6:.0f} MB)")
    print(f"wrote {C.BUILD_DIR / 'esm_supervised_manifest.json'}")
