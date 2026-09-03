"""ESM-2 per-residue embeddings for every CryptoBench apo structure, on Modal.

WHY THIS RUN EXISTS
    pLM-NN -- the published benchmark we are being measured against -- is a
    network over ESM-2-3B embeddings. Everything we have used so far is
    hand-built geometry, conservation and ensemble-pocket numbers. Adding ESM
    tests whether the language model carries signal our features do not, or
    whether it mostly re-encodes the conservation we already fetched.

WHICH MODEL
    facebook/esm2_t36_3B_UR50D -- the same one pLM-NN used, 2560 dims. A 650M
    variant would be 5x cheaper and probably nearly as good, but then a null
    result would be ambiguous between "embeddings do not add anything" and "we
    used a smaller model than they did".

PCA IS FIT ON TRAIN ROWS ONLY
    2560 raw dims would swamp 54 hand-built features and blow past TabICL's
    context budget, so they are reduced to 64. Fitting that projection on all
    rows would let test-set structure leak into the basis -- unsupervised, but
    still information from held-out proteins. The fit therefore sees train rows
    only, and the components are ordered, so taking the first k columns gives a
    free sweep over k without recomputing anything.

TWO STAGES, ONE VOLUME
    Raw embeddings are ~2 GB (380,930 x 2560 x fp16). Passing that through
    function return values would be slow and fragile, so shards write npz files
    to a Volume and a single reducer reads them back, fits the PCA and returns
    only the 64-dim table.

USAGE
    modal run deploy/esm_modal.py --limit 8      # pilot: check cost and shapes
    modal run deploy/esm_modal.py                # full
"""
from __future__ import annotations

import os
import sys

import modal

STRUCTURE_CACHE = os.path.expanduser("~/cryptobench_stage/structures")
if modal.is_local() and not os.path.isdir(STRUCTURE_CACHE):
    raise SystemExit(f"missing structure cache at {STRUCTURE_CACHE}")

MODEL_ID = "facebook/esm2_t36_3B_UR50D"
EMBED_DIM = 2560
N_COMPONENTS = 64
GPU = "A10G"
GPU_HOURLY_USD = 1.10

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.46.3", "gemmi", "numpy<2", "scipy",
                 "pandas", "pyarrow", "scikit-learn", "huggingface_hub")
    # Bake the ~11 GB checkpoint into the image. Downloading it per cold
    # container would dominate the run and is exactly how the TabICL run first
    # failed -- the client timed out during the download, before any compute.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download as d; "
        f"d(repo_id='{MODEL_ID}', allow_patterns=['*.json','*.txt','*.safetensors'])\""
    )
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
    .add_local_dir(STRUCTURE_CACHE, "/structures")
)

vol = modal.Volume.from_name("cryptobench-esm", create_if_missing=True)
app = modal.App("cryptobench-esm")


@app.function(image=image, gpu=GPU, timeout=5400, max_containers=8, retries=1,
              volumes={"/vol": vol})
def embed_shard(payload: dict) -> dict:
    """Embed one batch of apo structures; write raw fp16 vectors to the Volume."""
    sys.path.insert(0, "/app")
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from transformers import AutoTokenizer, EsmModel

    from cryptic_tabicl.esm_embed import chain_sequences, chunk_spans
    from cryptic_tabicl.structures import load_structure, residue_records

    batch_id = payload["batch_id"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()

    ids, vecs, failures = [], [], []
    t0 = time.time()
    for item in payload["items"]:
        apo_id, pdb_id, chains = item["apo_id"], item["pdb_id"], item["chains"]
        try:
            st = load_structure(Path(f"/structures/{pdb_id.lower()}.cif.gz"))
            recs = residue_records(st, chains)
            if not recs:
                failures.append((apo_id, "no residues"))
                continue

            for grp in chain_sequences(recs):
                seq, keys = grp["seq"], grp["keys"]
                pieces = []
                for start, end, keep_start, keep_end in chunk_spans(len(seq)):
                    enc = tok(seq[start:end], return_tensors="pt",
                              add_special_tokens=True)
                    with torch.no_grad():
                        out = model(**{k: v.cuda() for k, v in enc.items()})
                    # Strip BOS/EOS so row i is residue i of the chunk.
                    h = out.last_hidden_state[0, 1:-1].float().cpu().numpy()
                    assert h.shape[0] == end - start, (
                        f"{apo_id}/{grp['chain']}: tokeniser returned "
                        f"{h.shape[0]} rows for {end - start} residues")
                    pieces.append(h[keep_start - start: keep_end - start])

                emb = np.concatenate(pieces, axis=0)
                assert len(emb) == len(keys), "embedding/residue count mismatch"
                for (seq_id, icode, res_ix), v in zip(keys, emb):
                    ids.append((apo_id, grp["chain"], int(seq_id), icode, int(res_ix)))
                    vecs.append(v.astype(np.float16))
        except Exception as exc:  # noqa: BLE001 - one bad entry must not kill a shard
            failures.append((apo_id, f"{type(exc).__name__}: {exc}"))
            print(f"{apo_id}: FAILED {type(exc).__name__}: {exc}", flush=True)

    arr = (np.vstack(vecs) if vecs
           else np.zeros((0, EMBED_DIM), dtype=np.float16))
    path = f"/vol/emb_{batch_id:04d}.npz"
    np.savez(path, emb=arr,
             apo_id=np.array([i[0] for i in ids]),
             chain=np.array([i[1] for i in ids]),
             seq_id=np.array([i[2] for i in ids], dtype=np.int32),
             icode=np.array([i[3] for i in ids]),
             res_ix=np.array([i[4] for i in ids], dtype=np.int32))
    vol.commit()
    print(f"batch {batch_id}: {len(ids):,} residues in {time.time()-t0:.0f}s "
          f"({len(failures)} failures)", flush=True)
    return {"batch_id": batch_id, "n_rows": len(ids), "failures": failures,
            "seconds": time.time() - t0}


@app.function(image=image, timeout=5400, cpu=8.0, memory=65536,
              volumes={"/vol": vol})
def reduce_pca(split_of: dict, n_components: int = N_COMPONENTS) -> bytes:
    """Concatenate every shard, fit PCA on TRAIN rows only, return 64 dims."""
    import glob
    import io

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    vol.reload()
    files = sorted(glob.glob("/vol/emb_*.npz"))
    print(f"reading {len(files)} shard files", flush=True)

    embs, meta = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        if len(z["seq_id"]) == 0:
            continue
        embs.append(z["emb"])
        meta.append(pd.DataFrame({
            "apo_id": z["apo_id"].astype(str), "chain": z["chain"].astype(str),
            "seq_id": z["seq_id"].astype(int), "icode": z["icode"].astype(str),
            "res_ix": z["res_ix"].astype(int)}))

    X = np.vstack(embs).astype(np.float32)
    m = pd.concat(meta, ignore_index=True)
    print(f"total {X.shape[0]:,} residues x {X.shape[1]} dims", flush=True)

    is_train = m["apo_id"].map(split_of).eq("train").to_numpy()
    print(f"PCA fit on {is_train.sum():,} train rows "
          f"({(~is_train).sum():,} held out of the fit)", flush=True)

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X[is_train])
    Z = pca.transform(X).astype(np.float32)
    print(f"explained variance: first 8 {pca.explained_variance_ratio_[:8].round(3)}"
          f"  cumulative {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    out = m.copy()
    for i in range(n_components):
        out[f"esm_pc{i:02d}"] = Z[:, i]
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    return buf.getvalue()


@app.local_entrypoint()
def main(batch_size: int = 40, limit: int = 0, n_components: int = N_COMPONENTS):
    import json
    import time

    sys.path.insert(0, ".")
    from cryptic_tabicl import config as C, dataset as D

    ds = D.load_json(C.DATASET_JSON)
    chains = D.apo_chains(ds)
    pdb_of = D.apo_pdb_ids(ds)

    # Take the split straight off the built table rather than re-deriving it.
    # The PCA basis must be fit on exactly the rows the evaluation calls
    # "train"; re-deriving invites the two definitions drifting apart, and a
    # drift that leaked test proteins into the basis would be invisible.
    import pandas as pd

    built = pd.read_parquet(C.BUILD_DIR / "residues_cons.parquet",
                            columns=[C.GROUP_COL, "split"])
    split_of = dict(built.drop_duplicates(C.GROUP_COL)
                    .set_index(C.GROUP_COL)["split"].astype(str))
    missing = [a for a in chains if a not in split_of]
    if missing:
        print(f"note: {len(missing)} apo structures absent from the built table "
              f"(quarantined upstream); they are embedded but excluded from the "
              f"PCA fit, e.g. {missing[:3]}")

    targets = sorted(chains)
    if limit:
        targets = targets[:limit]
    items = [{"apo_id": a, "pdb_id": pdb_of[a], "chains": sorted(chains[a])}
             for a in targets]
    batches = [{"batch_id": i // batch_size, "items": items[i:i + batch_size]}
               for i in range(0, len(items), batch_size)]

    print(f"{len(items)} apo structures -> {len(batches)} batches on {GPU}")
    t0 = time.time()
    total, all_failures, gpu_seconds = 0, [], 0.0
    for r in embed_shard.map(batches):
        total += r["n_rows"]
        gpu_seconds += r["seconds"]
        all_failures.extend(r["failures"])
        print(f"  {total:,} residues embedded", flush=True)

    print(f"\nembedding done: {total:,} residues, {len(all_failures)} failures, "
          f"{time.time()-t0:.0f}s wall / {gpu_seconds/3600:.2f} GPU-h "
          f"(~${gpu_seconds/3600*GPU_HOURLY_USD:.2f})")
    for apo_id, msg in all_failures[:10]:
        print(f"  FAILED {apo_id}: {msg}")

    blob = reduce_pca.remote(split_of, n_components)
    out = C.BUILD_DIR / "esm_features.parquet"
    out.write_bytes(blob)
    print(f"\nwrote {out} ({len(blob)/1e6:.0f} MB)")
    (C.BUILD_DIR / "esm_manifest.json").write_text(
        json.dumps({"model": MODEL_ID, "n_components": n_components,
                    "n_rows": total, "failures": all_failures,
                    "gpu_hours": gpu_seconds / 3600}, indent=1))
