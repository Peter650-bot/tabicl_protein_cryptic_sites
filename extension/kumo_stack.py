"""NVIDIA Kumo Tabular on PAM's features, contexts and protocol (extension/PLAN.md).

WHAT CHANGES AGAINST PAM (run fe1, deploy/ftemb_stack.py)
    Only the classifier: `KumoTabularPredictor` (extension/kumo_model.py) replaces
    `CrypticSitePredictor`. The table, `select`, `sampling.build_context` at seed 0, the
    ft117 column set and the job/result schema are fe1's, so every Kumo job sees exactly
    the rows and columns its fe1 twin saw, and the comparison is paired. The GPU changes
    too (A100-80GB, not T4): Kumo's model card lists Ampere/Hopper/Blackwell only.

    `select` is a verbatim copy of deploy/ftemb_stack.py's, as deploy/ftpls_stack.py
    already does; extension/tests/test_kumo_extension.py fails if the two ever diverge.

STAGES
    smoke    breast-cancer quick-start with Kumo large; versions and checkpoint sha256
    harness  sdm's TabICLv2 (PAM's checkpoint, 8 estimators) through this code, k=0 / test
    gate     Kumo on k=0 / sub0, twice (rep 0 and 1): determinism and the cost pilot
    cv       Kumo x k=0..3 x 4 protein sub-folds of fold k (thresholds only)
    test     Kumo x k=0..3, query = test               (the result, once)

USAGE
    modal run    extension/kumo_stack.py --stage smoke             # writes extension/smoke.json
    modal deploy extension/kumo_stack.py
    modal run    extension/kumo_stack.py --stage harness --run-id ku1
    modal run    extension/kumo_stack.py --stage gate    --run-id ku1
    modal run    extension/kumo_stack.py --collect       --run-id ku1   # build/kumo_stack/ku1/
    python extension/kumo_report.py --gates --run-id ku1
    ... pre-register (extension/frozen.json), then --stage cv and --stage test, collect,
    python extension/kumo_report.py --run-id ku1
"""
from __future__ import annotations

import os
import sys

import modal

TABLE_NAME = "residues_ftemb.parquet"
TABLE_LOCAL = os.path.expanduser(f"~/cryptobench_stage/tables/{TABLE_NAME}")
if modal.is_local() and not os.path.exists(TABLE_LOCAL):
    raise SystemExit(f"missing {TABLE_LOCAL}: run scripts/ftemb_build_table.py")

GPU = "A100-80GB"
SEED, BUDGET = 0, 50_000
KS = [0, 1, 2, 3]
KUMO = {"backbone": "kumo", "size": "large", "num_estimators": 16}
TABICL = {"backbone": "tabiclv2", "size": None, "num_estimators": 8}
STAGES = {"harness": [{"name": "ft117", "k": 0, "query": "test", **TABICL}],
          "gate": [{"name": "ft117", "k": 0, "query": "sub0", "rep": r, **KUMO} for r in (0, 1)],
          "cv": [{"name": "ft117", "k": k, "query": f"sub{s}", **KUMO} for k in KS for s in range(4)],
          "test": [{"name": "ft117", "k": k, "query": "test", **KUMO} for k in KS]}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("structured-data-models==0.2.1", "scikit-learn",
                 "pandas==2.3.3", "numpy==2.2.6", "pyarrow==25.0.1")  # = the local env
    # Bake both checkpoints into the image: Kumo pins its HF revision (v1.0.1) in code.
    .run_commands(
        "python -c \"import sdm; "
        "sdm.models.KumoTabular(task='classification', size='large', device='cpu'); "
        "sdm.models.TabICLv2(task='classification', device='cpu')\"")
    .add_local_file(TABLE_LOCAL, "/data/table.parquet")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
    .add_local_file("extension/kumo_model.py", "/app/extension/kumo_model.py")
)

app = modal.App("cryptobench-kumo-stack")
results_vol = modal.Volume.from_name("cryptobench-kumo-stack-results", create_if_missing=True)


def job_tag(job: dict) -> str:
    import hashlib
    import json

    payload = {k: v for k, v in job.items() if k != "run_id"}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{job['name']}_{job['backbone']}_k{job['k']}_{job['query']}_{digest[:8]}"


def result_name(r: dict) -> str:
    """Local file name: fe1's `{arm}_k{k}_{query}.json` for Kumo, suffixed otherwise."""
    name = f"{r['name']}_k{r['k']}_{r['query']}"
    if r["backbone"] != "kumo":
        name += f"_{r['backbone']}"
    if "rep" in r:
        name += f"_rep{r['rep']}"
    return name + ".json"


def select(table, job: dict):
    """(context source, query) rows for one job, with block k renamed to the generic
    `ft_*` names and every other block dropped. Pure; imported by the tests."""
    from cryptic_tabicl import feature_sets as FS, sampling

    k, query = job["k"], job["query"]
    blocks = [c for c in table.columns if c.startswith(FS.FT_BLOCK_PREFIX)]
    if k is None:                                    # the standard pipeline (gate)
        table = table.drop(columns=blocks)
        train = table[table["split"] == "train"]
        return train[train["fold"] != query], train[train["fold"] == query], table
    mine = {f"ftk{k}_pc{i:02d}": f"ft_pc{i:02d}" for i in range(64)} | {f"ftk{k}_score": "ft_score"}
    table = table.drop(columns=[c for c in blocks if c not in mine]).rename(columns=mine)
    fold_k = table[table["fold"] == f"train-fold-{k}"]
    if query == "test":
        return fold_k, table[table["split"] == "test"], table
    sub = sampling.protein_subfolds(fold_k["apo_id"], n=4, seed=SEED)
    s = fold_k["apo_id"].map(sub).to_numpy()
    held = int(query.removeprefix("sub"))
    return fold_k[s != held], fold_k[s == held], table


@app.function(image=image, gpu=GPU, timeout=600)
def smoke() -> dict:
    """The model card's quick-start, plus everything needed to pin the environment."""
    import hashlib
    import time

    import numpy as np
    import sdm
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    df = load_breast_cancer(as_frame=True).frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    table = sdm.TableTensor.from_pandas(
        df=df, stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}), device="cuda")
    model = sdm.models.KumoTabular(task="classification", size="large", device="cuda").eval()
    t0 = time.time()
    with torch.amp.autocast("cuda", torch.float16):
        out = model(x_context=table[:300].drop_columns("target"), y_context=table[:300, "target"],
                    x_query=table[300:].drop_columns("target"), num_estimators=16,
                    generator=torch.Generator("cuda").manual_seed(0))
    names = list(out.columns[sdm.Stype.numerical])
    p = out.numerical[..., names.index("1")].float().cpu().numpy()
    y = df["target"].to_numpy()[300:]
    kumo_ckpt = hf_hub_download("nvidia/Kumo-Tabular", "large/classifier.pt", revision="v1.0.1",
                                local_files_only=True)
    tabicl_ckpt = hf_hub_download("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt",
                                  local_files_only=True)
    return {"sdm": sdm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "output_columns": names,
            "n_params": int(sum(x.numel() for x in model.parameters())),
            "auc": float(roc_auc_score(y, p)), "n_query": int(len(y)),
            "seconds": time.time() - t0,
            "kumo_large_classifier_sha256": sha(kumo_ckpt),
            "tabicl_classifier_sha256": sha(tabicl_ckpt)}


@app.function(image=image, gpu=GPU, timeout=7200, max_containers=10, retries=1,
              volumes={"/results": results_vol})
def score_one(job: dict) -> dict:
    """One (k, query) job. Raw scores only; thresholds are chosen locally."""
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
    import torch
    from cryptic_tabicl import config as C, feature_sets as FS, sampling
    from extension.kumo_model import KumoTabularPredictor

    ctx_src, target, table = select(pd.read_parquet("/data/table.parquet"), job)
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
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    clf = KumoTabularPredictor(backbone=job["backbone"], size=job["size"] or "large",
                               num_estimators=job["num_estimators"], seed=job["seed"]).fit(ctx)
    t_fit = time.time() - t0
    scores = clf.predict_by_protein(target)
    out = {**{k: job[k] for k in ("name", "k", "query", "seed", "backbone") if k in job},
           **({"rep": job["rep"]} if "rep" in job else {}),
           "n_features": n_feat, "features": cols,
           "n_context": len(ctx), "n_context_proteins": int(ctx[C.GROUP_COL].nunique()),
           "seconds": time.time() - t0, "seconds_fit": t_fit,
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
           "gpu": torch.cuda.get_device_name(0), "config": clf.config(),
           "apo_id": target[C.GROUP_COL].tolist(), "chain": target["chain"].tolist(),
           "seq_id": [int(s) for s in target["seq_id"]],
           "icode": target["icode"].astype(str).tolist(),
           "y": target[C.LABEL_COL].astype(int).tolist(),
           "score": [float(s) for s in np.asarray(scores)]}
    os.makedirs(run_dir, exist_ok=True)
    with open(cached, "w") as fh:
        _json.dump(out, fh)
    results_vol.commit()
    print(f"{tag}: {len(target):,} rows in {out['seconds']:.0f}s, "
          f"peak {out['peak_gpu_gb']:.1f} GB", flush=True)
    return out


@app.function(image=image, timeout=1800, volumes={"/results": results_vol})
def fetch_results(run_id: str) -> list:
    import glob
    import json

    results_vol.reload()
    return [json.load(open(f)) for f in sorted(glob.glob(f"/results/{run_id}/*.json"))]


@app.local_entrypoint()
def main(stage: str = "", run_id: str = "", collect: bool = False):
    import json
    from pathlib import Path

    if stage == "smoke":
        res = smoke.remote()
        Path("extension/smoke.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    assert run_id, "--run-id is required"
    if collect:
        out = Path("build/kumo_stack") / run_id
        out.mkdir(parents=True, exist_ok=True)
        for r in fetch_results.remote(run_id):
            (out / result_name(r)).write_text(json.dumps(r))
            print(f"wrote {out / result_name(r)} ({len(r['score']):,} scores, {r['seconds']:.0f}s)")
        return
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {sorted(STAGES) + ['smoke']}")
    jobs = [{"budget": BUDGET, "seed": SEED, **j, "run_id": run_id, "table": TABLE_NAME}
            for j in STAGES[stage]]
    tags = [job_tag(j) for j in jobs]
    assert len(set(tags)) == len(tags), "job tags collide"
    f = modal.Function.from_name("cryptobench-kumo-stack", "score_one")
    for j in jobs:
        print(f"spawned {j['backbone']} k={j['k']} {j['query']}: {f.spawn(j).object_id}")
