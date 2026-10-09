"""TabICL on a PLS projection of the fine-tuned ESM-2's embedding (PLAN.md step 11).

Step 10 (`deploy/ftemb_stack.py`) gave TabICL 64 PCA components of fold model k's
last hidden state. This replaces PCA with PLS on the label, and changes nothing else:
the same stored hidden states (`cryptobench-s2p` volume, nothing re-embedded), the same
cross-fitted design (context = train fold k, query = test, mean over k = 0..3), the
same `select()` and job body, tabicl 2.1.1, seed 0, budget 50k.

PLS USES LABELS, SO ITS FIT ROWS ARE THE CONTEXT'S SOURCE ROWS, NEVER A QUERY ROW
    fit `k{k}_test`   : every table row of train fold k; 128 components; applied to
                        fold k + test. The d-dim arms take the first d (PLS components
                        are nested, so this equals a d-component fit).
    fit `k{k}_sub{s}` : train fold k minus protein sub-fold s; 64 components; applied to
                        fold k. Used only by the CV jobs that choose thresholds.
    Each fit's manifest lists its proteins; a job refuses to run if a query protein is
    among them or a context protein is not.

MAPPING GATE (in `fit_pls`)
    The hidden-state rows gathered for the table rows (`build/ftpls_rows.parquet`,
    `scripts/ftpls_build_index.py`) must reproduce step 10's PCA columns: PCA(64,
    random_state=0) refit on the fold-k chains, |corr| >= 0.999 per component.

STAGES
    pls    fit_pls for k = 0..3 (CPU)
    gate   ft117 k=0 test through this harness   must equal fe1/ft117_k0_test.json
    cv     pls64_53, pls64_only x k=0..3 x 4 sub-folds (thresholds only)
    test   pls{d}_53, pls{d}_only x d in PLS_DIMS x k=0..3

USAGE
    modal deploy deploy/ftpls_stack.py
    modal run deploy/ftpls_stack.py --stage pls  --run-id fp1
    modal run deploy/ftpls_stack.py --stage gate --run-id fp1      (then cv, test)
    modal run deploy/ftpls_stack.py --collect    --run-id fp1      # build/ftpls_stack/fp1/
"""
from __future__ import annotations

import os
import sys

import modal

TABLE_NAME = "residues_ftemb.parquet"
STAGE = os.path.expanduser("~/cryptobench_stage/tables")
TABLE_LOCAL, ROWS_LOCAL = f"{STAGE}/{TABLE_NAME}", f"{STAGE}/ftpls_rows.parquet"
if modal.is_local():
    for f, how in ((TABLE_LOCAL, "scripts/ftemb_build_table.py"),
                   (ROWS_LOCAL, "scripts/ftpls_build_index.py")):
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}: run {how}")

GPU = "T4"
CKPT = "tabicl-classifier-v2-20260212.ckpt"
SEED, BUDGET = 0, 50_000
SRC_RUN = "s2p_full_cleantest_20261008"
PLS_DIMS = (2, 4, 8, 16, 32, 64, 128)
N_PLS, N_PLS_CV = 128, 64
KS = [0, 1, 2, 3]
CV_ARMS = ["pls64_53", "pls64_only"]
STAGES = {
    "gate": [{"name": "ft117", "k": 0, "query": "test", "pls": None}],
    "cv": [{"name": n, "k": k, "query": f"sub{s}", "pls": True}
           for n in CV_ARMS for k in KS for s in range(4)],
    "test": [{"name": f"pls{d}_{b}", "k": k, "query": "test", "pls": True}
             for d in PLS_DIMS for b in ("53", "only") for k in KS],
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("tabicl==2.1.1", "scikit-learn", "pandas", "pyarrow", "numpy<2")
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"d(repo_id='jingang/TabICL', filename='{CKPT}')\"")
    .add_local_file(TABLE_LOCAL, "/data/table.parquet")
    .add_local_file(ROWS_LOCAL, "/data/rows.parquet")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
)

app = modal.App("cryptobench-ftpls-stack")
results_vol = modal.Volume.from_name("cryptobench-ftemb-stack-results", create_if_missing=True)
s2p_vol = modal.Volume.from_name("cryptobench-s2p")


@app.function(image=image, cpu=8.0, memory=32768, timeout=3 * 3600,
              volumes={"/results": results_vol, "/s2p": s2p_vol})
def fit_pls(k: int, pls_run: str) -> dict:
    sys.path.insert(0, "/app")
    import hashlib
    import json
    import time

    import numpy as np
    import pandas as pd
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    from cryptic_tabicl import config as C, sampling

    t0 = time.time()
    rm = pd.read_parquet("/data/rows.parquet")
    rm = rm[rm["k"] == k].reset_index(drop=True)
    pcs = [f"ftk{k}_pc{i:02d}" for i in range(64)]
    table = pd.read_parquet("/data/table.parquet", columns=["apo_id", "fold", C.LABEL_COL] + pcs)
    t = table.iloc[rm["table_row"].to_numpy()].reset_index(drop=True)
    H = np.load(f"/s2p/ft/{SRC_RUN}/hidden_cleantest_{k}.npy")          # fp16, ~0.8 GB
    hidx = pd.read_parquet(f"/s2p/ft/{SRC_RUN}/hidden_index_cleantest_{k}.parquet")
    assert len(hidx) == len(H)
    X = H[rm["hrow"].to_numpy()].astype(np.float32)
    fold_k = (t["fold"] == f"train-fold-{k}").to_numpy()
    y = t[C.LABEL_COL].to_numpy(float)
    print(f"k{k}: {len(X):,} rows ({fold_k.sum():,} fold {k}) x {X.shape[1]}", flush=True)

    # ---- mapping gate: these rows reproduce step 10's PCA columns
    fit_pca = hidx["apo_id"].isin(set(t.loc[fold_k, "apo_id"])).to_numpy()
    pca = PCA(n_components=64, random_state=0).fit(H[fit_pca].astype(np.float32))
    Zp = pca.transform(X)
    want = t[pcs].to_numpy(np.float64)
    corr = [abs(np.corrcoef(Zp[:, i], want[:, i])[0, 1]) for i in range(64)]
    assert min(corr) >= 0.999, f"k{k}: gathered rows do not reproduce step 10's PCs ({min(corr):.4f})"
    gate = {"pca_min_abs_corr": float(min(corr)),
            "pca_max_abs_diff": float(np.abs(Zp - want).max()), "n_pca_fit": int(fit_pca.sum())}
    print(f"k{k}: mapping gate {gate}", flush=True)
    del Zp, want

    out_dir = f"/results/pls/{pls_run}"
    os.makedirs(out_dir, exist_ok=True)
    sub = sampling.protein_subfolds(t.loc[fold_k, "apo_id"], n=4, seed=SEED)
    s_of = t["apo_id"].map(sub).to_numpy()            # NaN on test rows
    fits = {"test": (fold_k, np.ones(len(t), bool), N_PLS)}
    for s in range(4):
        fits[f"sub{s}"] = (fold_k & (s_of != s), fold_k, N_PLS_CV)
    manifest = {"k": k, "source_run": SRC_RUN, "gate": gate, "fits": {}}
    for name, (fit, apply, n) in fits.items():
        t1 = time.time()
        pls = PLSRegression(n_components=n, scale=True).fit(X[fit], y[fit])
        rows = np.flatnonzero(apply)
        Z = np.empty((len(rows), n), dtype=np.float32)
        for lo in range(0, len(rows), 20_000):
            Z[lo:lo + 20_000] = pls.transform(X[rows[lo:lo + 20_000]])
        cols = [f"ft_pls{i:03d}" for i in range(n)]
        df = pd.DataFrame(Z, columns=cols)
        df.insert(0, "table_row", rm["table_row"].to_numpy()[rows])
        df.to_parquet(f"{out_dir}/k{k}_{name}.parquet", index=False)
        held = (~fold_k) if name == "test" else (fold_k & (s_of == int(name[3:])))
        z_held = Z[np.searchsorted(rows, np.flatnonzero(held))]
        apos = sorted(t.loc[fit, "apo_id"].unique())
        manifest["fits"][name] = {
            "n_components": n, "n_fit_rows": int(fit.sum()), "n_fit_positives": int(y[fit].sum()),
            "fit_apo_ids": apos, "fit_apo_ids_sha256": hashlib.sha256("\n".join(apos).encode()).hexdigest(),
            "n_applied_rows": int(len(rows)),
            "held_AUC_first4": [float(roc_auc_score(y[held], z_held[:, i])) for i in range(4)],
            "seconds": time.time() - t1}
        assert not set(apos) & set(t.loc[held, "apo_id"]), f"k{k} {name}: a held protein was fit"
        print(f"k{k} {name}: fit {fit.sum():,} rows, n={n}, held AUC of c0-3 "
              f"{[round(v, 3) for v in manifest['fits'][name]['held_AUC_first4']]} "
              f"in {time.time() - t1:.0f}s", flush=True)
    manifest["seconds"] = time.time() - t0
    with open(f"{out_dir}/k{k}_manifest.json", "w") as fh:
        json.dump(manifest, fh)
    results_vol.commit()
    return {kk: v for kk, v in manifest.items() if kk != "fits"}


def select(table, job: dict):
    """Step 10's `deploy/ftemb_stack.py::select`, verbatim (the gate stage checks it):
    (context source, query) rows, block k renamed to the generic `ft_*` names."""
    from cryptic_tabicl import feature_sets as FS, sampling

    k, query = job["k"], job["query"]
    blocks = [c for c in table.columns if c.startswith(FS.FT_BLOCK_PREFIX)]
    mine = {f"ftk{k}_pc{i:02d}": f"ft_pc{i:02d}" for i in range(64)} | {f"ftk{k}_score": "ft_score"}
    table = table.drop(columns=[c for c in blocks if c not in mine]).rename(columns=mine)
    fold_k = table[table["fold"] == f"train-fold-{k}"]
    if query == "test":
        return fold_k, table[table["split"] == "test"], table
    sub = sampling.protein_subfolds(fold_k["apo_id"], n=4, seed=SEED)
    s = fold_k["apo_id"].map(sub).to_numpy()
    held = int(query.removeprefix("sub"))
    return fold_k[s != held], fold_k[s == held], table


def job_tag(job: dict) -> str:
    import hashlib
    import json

    payload = {k: v for k, v in job.items() if k != "run_id"}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{job['name']}_k{job['k']}_{job['query']}_{digest[:8]}"


def attach_pls(table, job: dict):
    """Table with fit `k{k}_{query}`'s PLS columns filled on the rows it was applied to
    (NaN elsewhere), and that fit's protein list."""
    import json

    import numpy as np
    import pandas as pd

    base = f"/results/pls/{job['pls']}/k{job['k']}"
    z = pd.read_parquet(f"{base}_{job['query']}.parquet")
    fit_apos = set(json.load(open(f"{base}_manifest.json"))["fits"][job["query"]]["fit_apo_ids"])
    cols = [c for c in z.columns if c.startswith("ft_pls")]
    arr = np.full((len(table), len(cols)), np.nan, dtype=np.float32)
    arr[z["table_row"].to_numpy()] = z[cols].to_numpy(np.float32)
    return pd.concat([table, pd.DataFrame(arr, columns=cols, index=table.index)], axis=1), fit_apos


@app.function(image=image, gpu=GPU, timeout=7200, max_containers=10, retries=1,
              volumes={"/results": results_vol})
def score_one(job: dict) -> dict:
    """One (arm, k, query) job: step 10's body, plus the PLS columns of the matching fit."""
    sys.path.insert(0, "/app")
    import json as _json
    import time

    import numpy as np

    tag = job_tag(job)
    run_dir = f"/results/{job['run_id']}"
    cached = f"{run_dir}/{tag}.json"
    results_vol.reload()
    if os.path.exists(cached):
        print(f"{tag}: cached", flush=True)
        return _json.load(open(cached))

    import pandas as pd
    from cryptic_tabicl import config as C, feature_sets as FS, model, sampling

    table = pd.read_parquet("/data/table.parquet")
    if job["pls"]:
        table, fit_apos = attach_pls(table, job)
    ctx_src, target, table = select(table, job)
    if job["pls"]:
        assert set(ctx_src[C.GROUP_COL]) <= fit_apos, "a context protein is outside the PLS fit"
        assert not set(target[C.GROUP_COL]) & fit_apos, "a query protein was in the PLS fit"
    cols = FS.resolve(job["name"], table)
    wanted = set(cols) | {c for c in C.NON_FEATURE_COLS if c in table.columns}
    keep = [c for c in table.columns if c in wanted]
    ctx_src, target = ctx_src[keep], target[keep]
    n_feat = len([c for c in keep if c not in C.NON_FEATURE_COLS])
    assert n_feat == len(cols), f"{n_feat} features reached the model, wanted {len(cols)}"
    used = [c for c in cols if c.startswith("ft_")]
    assert not ctx_src[used].isna().any().any() and not target[used].isna().any().any(), \
        "a context or query row has no embedding from this fold model"

    ctx, plan = sampling.build_context(ctx_src, budget=job["budget"], seed=job["seed"])
    print(f"{tag} features={n_feat} query={len(target):,} rows\n{plan}", flush=True)
    t0 = time.time()
    clf = model.CrypticSitePredictor().fit(ctx)
    scores = clf.predict_by_protein(target)
    out = {**{k: job[k] for k in ("name", "k", "query")}, "n_features": n_feat, "features": cols,
           "n_context": len(ctx), "n_context_proteins": int(ctx[C.GROUP_COL].nunique()),
           "seconds": time.time() - t0, "config": clf.config(),
           "apo_id": target[C.GROUP_COL].tolist(), "chain": target["chain"].tolist(),
           "seq_id": [int(s) for s in target["seq_id"]],
           "icode": target["icode"].astype(str).tolist(),
           "y": target[C.LABEL_COL].astype(int).tolist(),
           "score": [float(s) for s in np.asarray(scores)]}
    os.makedirs(run_dir, exist_ok=True)
    with open(cached, "w") as fh:
        _json.dump(out, fh)
    results_vol.commit()
    print(f"{tag}: {len(target):,} rows in {out['seconds']:.0f}s", flush=True)
    return out


@app.function(image=image, timeout=1800, volumes={"/results": results_vol})
def fetch_results(run_id: str) -> tuple[list, list]:
    import glob
    import json

    results_vol.reload()
    scores = [json.load(open(f)) for f in sorted(glob.glob(f"/results/{run_id}/*.json"))]
    manifests = []
    for f in sorted(glob.glob(f"/results/pls/{run_id}/k*_manifest.json")):
        m = json.load(open(f))
        for v in m["fits"].values():
            v.pop("fit_apo_ids")
        manifests.append(m)
    return scores, manifests


@app.local_entrypoint()
def main(stage: str = "", run_id: str = "", collect: bool = False):
    import json
    from pathlib import Path

    assert run_id, "--run-id is required"
    if collect:
        out = Path("build/ftpls_stack") / run_id
        out.mkdir(parents=True, exist_ok=True)
        scores, manifests = fetch_results.remote(run_id)
        for m in manifests:
            (out / f"pls_manifest_k{m['k']}.json").write_text(json.dumps(m, indent=1))
        for r in scores:
            name = f"{r['name']}_k{r['k']}_{r['query']}.json"
            (out / name).write_text(json.dumps(r))
        print(f"wrote {len(manifests)} PLS manifests and {len(scores)} score files to {out}")
        return
    if stage == "pls":
        f = modal.Function.from_name("cryptobench-ftpls-stack", "fit_pls")
        for k in KS:
            print(f"spawned fit_pls k={k}: {f.spawn(k, run_id).object_id}")
        return
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {sorted(STAGES) + ['pls']}")
    jobs = [{**j, "pls": run_id if j["pls"] else None, "budget": BUDGET,
             "seed": SEED, "run_id": run_id, "table": TABLE_NAME} for j in STAGES[stage]]
    tags = [job_tag(j) for j in jobs]
    assert len(set(tags)) == len(tags), "job tags collide"
    f = modal.Function.from_name("cryptobench-ftpls-stack", "score_one")
    for j in jobs:
        print(f"spawned {j['name']} k={j['k']} {j['query']}: {f.spawn(j).object_id}")
