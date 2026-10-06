"""TabICL with the fine-tuned ESM-2's out-of-fold score as a feature (PLAN.md step 8f).

WHY A SIBLING OF feature_sweep.py AND NOT A RE-USE
    `feature_sweep.py` is the record of a finished screen, staged on its own
    table. This job runs on `residues_s2p.parquet` (the published 117 columns in
    published order, plus `s2p_oof`), with its own app and volume, so neither
    record can be overwritten. The job body is the same: one arm's columns, in
    table order, the same `sampling.build_context` at the same seed, the same
    `CrypticSitePredictor`, tabicl 2.1.1 as in the headline run.

THE DETERMINISM GATE COMES FIRST
    `--stage gate` scores the CONTROL arm on train-fold-0 from this table. It must
    equal build/cv_scores/k64_train-fold-0.json exactly (max |diff| 0.0, same
    row order), as the feature sweep's control did. That proves the extra column
    is invisible to the control and that this harness is the headline pipeline.

STAGES
    gate   control x train-fold-0                       (harness check)
    cv     s2p_117, s2p_53 x four train folds            (thresholds, CV report)
    test   s2p_117, s2p_53 x test, once                  (the result)

USAGE
    modal deploy deploy/s2p_stack.py
    modal run deploy/s2p_stack.py --stage gate --run-id st1
    modal run deploy/s2p_stack.py --stage cv   --run-id st1
    modal run deploy/s2p_stack.py --stage test --run-id st1
    modal run deploy/s2p_stack.py --collect    --run-id st1     # writes build/s2p_stack/st1/
"""
from __future__ import annotations

import os
import sys

import modal

TABLE_LOCAL = os.path.expanduser("~/cryptobench_stage/tables/residues_s2p.parquet")
if modal.is_local() and not os.path.exists(TABLE_LOCAL):
    raise SystemExit(f"missing {TABLE_LOCAL}: run scripts/s2p_build_stack_table.py")

GPU = "T4"
GPU_HOURLY_USD = 0.59
CKPT = "tabicl-classifier-v2-20260212.ckpt"
SEED, BUDGET = 0, 50_000
FOLDS = [f"train-fold-{h}" for h in range(4)]
STAGES = {"gate": (["control"], ["train-fold-0"]),
          "cv": (["s2p_117", "s2p_53"], FOLDS),
          "test": (["s2p_117", "s2p_53"], [None])}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("tabicl==2.1.1", "scikit-learn", "pandas", "pyarrow", "numpy<2")
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"d(repo_id='jingang/TabICL', filename='{CKPT}')\"")
    .add_local_file(TABLE_LOCAL, "/data/table.parquet")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
)

app = modal.App("cryptobench-s2p-stack")
results_vol = modal.Volume.from_name("cryptobench-s2p-stack-results", create_if_missing=True)


def job_tag(job: dict) -> str:
    import hashlib
    import json

    payload = {k: v for k, v in job.items() if k != "run_id"}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{job['name']}_{job['held'] or 'final'}_{digest[:8]}"


@app.function(image=image, gpu=GPU, timeout=7200, max_containers=8, retries=1,
              volumes={"/results": results_vol})
def score_one(job: dict) -> dict:
    """One (arm, fold) job. Raw scores only; thresholds are chosen locally."""
    sys.path.insert(0, "/app")
    import json as _json
    import time

    tag = job_tag(job)
    run_dir = f"/results/{job['run_id']}"
    cached = f"{run_dir}/{tag}.json"
    results_vol.reload()
    if os.path.exists(cached):
        print(f"{tag}: cached", flush=True)
        return _json.load(open(cached))

    import pandas as pd
    from cryptic_tabicl import config as C, feature_sets as FS, model, sampling

    held = job["held"]
    table = pd.read_parquet("/data/table.parquet")
    cols = FS.resolve(job["name"], table)
    # Exactly this arm's columns in TABLE order (TabICL groups features by
    # position), plus the bookkeeping columns feature_matrix drops anyway.
    wanted = set(cols) | {c for c in C.NON_FEATURE_COLS if c in table.columns}
    table = table[[c for c in table.columns if c in wanted]]
    n_feat = len([c for c in table.columns if c not in C.NON_FEATURE_COLS])
    assert n_feat == len(cols), f"{n_feat} features reached the model, wanted {len(cols)}"

    train = table[table["split"] == "train"]
    if held is None:
        ctx_src, target = train, table[table["split"] == "test"]
    else:
        ctx_src, target = train[train["fold"] != held], train[train["fold"] == held]
    ctx, plan = sampling.build_context(ctx_src, budget=job["budget"], seed=job["seed"])
    print(f"{job['name']} held={held} features={n_feat}\n{plan}", flush=True)

    t0 = time.time()
    clf = model.CrypticSitePredictor().fit(ctx)
    scores = clf.predict_by_protein(target)
    out = {"name": job["name"], "held": held, "n_features": n_feat, "features": cols,
           "seconds": time.time() - t0, "config": clf.config(),
           "apo_id": target[C.GROUP_COL].tolist(), "chain": target["chain"].tolist(),
           "seq_id": [int(s) for s in target["seq_id"]],
           "icode": target["icode"].astype(str).tolist(),
           "y": target[C.LABEL_COL].astype(int).tolist(),
           "score": [float(s) for s in scores]}
    os.makedirs(run_dir, exist_ok=True)
    with open(cached, "w") as fh:
        _json.dump(out, fh)
    results_vol.commit()
    print(f"{tag}: {len(target):,} rows in {out['seconds']:.0f}s", flush=True)
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

    assert run_id, "--run-id is required"
    if collect:
        out = Path("build/s2p_stack") / run_id
        out.mkdir(parents=True, exist_ok=True)
        for r in fetch_results.remote(run_id):
            name = f"{r['name']}_{r['held'] or 'final'}.json"
            (out / name).write_text(json.dumps(r))
            print(f"wrote {out / name} ({len(r['score']):,} scores, {r['seconds']:.0f}s)")
        return
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {sorted(STAGES)}")
    names, folds = STAGES[stage]
    jobs = [{"name": n, "held": h, "budget": BUDGET, "seed": SEED, "run_id": run_id}
            for n in names for h in folds]
    tags = [job_tag(j) for j in jobs]
    assert len(set(tags)) == len(tags), "job tags collide"
    # Spawned on the deployed app: this host drops long client streams.
    f = modal.Function.from_name("cryptobench-s2p-stack", "score_one")
    for j in jobs:
        print(f"spawned {j['name']} held={j['held']}: {f.spawn(j).object_id}")
