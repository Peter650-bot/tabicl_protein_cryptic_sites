"""TabICL on the fine-tuned ESM-2's own embedding, cross-fitted (PLAN.md step 10).

WHY THE CONTEXT IS ONE FOLD
    Fold model k (`cleantest_k`) never trained on train fold k, nor on LIGYSIS
    homologs of fold k or of test. Its embeddings are honest on exactly those rows,
    and each fold model has its own embedding space, so a job that uses block k
    takes its context from fold k and its query from test (or, for thresholds,
    from a protein sub-fold of fold k). The four test predictions, k = 0..3, are
    averaged in `scripts/ftemb_report.py`. Every arm, the control included, uses
    these same contexts, so the comparison is paired.

    The job body is `deploy/s2p_stack.py`'s: one arm's columns in table order, the
    same `sampling.build_context` at the same seed, the same `CrypticSitePredictor`,
    tabicl 2.1.1.

STAGES
    gate   control, standard context, train-fold-0       must equal k64_train-fold-0.json
    cv     4 arms x k=0..3 x 4 protein sub-folds of fold k (thresholds only)
    test   4 arms x k=0..3, query = test                 (the result, once)
    seeds  ft117 only, context-draw seeds 1..11 x k=0..3 x (4 sub-folds + test): the
           RESULTS.md 4.6 study for the best model. Only `build_context`'s seed varies;
           the sub-fold split stays at SEED, so each seed reruns the whole protocol
           (thresholds included) on new negatives. Seed 0 is run `fe1`.
    abgate   ft117 k=0 test on the redeployed harness; must match fe1 (step 13)
    ablation 5 nested family arms (33, 47, 53, 69, 85) x k=0..3 x (4 sub-folds + test)

USAGE
    modal deploy deploy/ftemb_stack.py
    modal run deploy/ftemb_stack.py --stage gate --run-id fe1
    modal run deploy/ftemb_stack.py --stage cv   --run-id fe1
    modal run deploy/ftemb_stack.py --stage test --run-id fe1
    modal run deploy/ftemb_stack.py --collect    --run-id fe1     # writes build/ftemb_stack/fe1/
"""
from __future__ import annotations

import os
import sys

import modal

TABLE_NAME = "residues_ftemb.parquet"
TABLE_LOCAL = os.path.expanduser(f"~/cryptobench_stage/tables/{TABLE_NAME}")
if modal.is_local() and not os.path.exists(TABLE_LOCAL):
    raise SystemExit(f"missing {TABLE_LOCAL}: run scripts/ftemb_build_table.py")

GPU = "T4"
CKPT = "tabicl-classifier-v2-20260212.ckpt"
SEED, BUDGET = 0, 50_000
ARMS = ["control", "ft117", "ft64only", "score118"]
KS = [0, 1, 2, 3]
STAGES = {"gate": [{"name": "control", "k": None, "query": "train-fold-0"}],
          "cv": [{"name": n, "k": k, "query": f"sub{s}"} for n in ARMS for k in KS for s in range(4)],
          "test": [{"name": n, "k": k, "query": "test"} for n in ARMS for k in KS]}
N_SEEDS = 12
STAGES["seeds"] = [{"name": "ft117", "k": k, "query": q, "seed": s}
                   for s in range(1, N_SEEDS) for k in KS
                   for q in [f"sub{i}" for i in range(4)] + ["test"]]
# Step 13, PAM's feature-family ablation. ft117 itself is fe1 (seed 0), reused; `abgate`
# reruns one of its jobs on the redeployed harness and must match fe1.
ABLATION_ARMS = ["fam33", "fam47", "fam53", "ft69", "ft85"]
STAGES["abgate"] = [{"name": "ft117", "k": 0, "query": "test"}]
STAGES["ablation"] = [{"name": n, "k": k, "query": q} for n in ABLATION_ARMS for k in KS
                      for q in [f"sub{i}" for i in range(4)] + ["test"]]

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

app = modal.App("cryptobench-ftemb-stack")
results_vol = modal.Volume.from_name("cryptobench-ftemb-stack-results", create_if_missing=True)


def job_tag(job: dict) -> str:
    import hashlib
    import json

    payload = {k: v for k, v in job.items() if k != "run_id"}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{job['name']}_k{job['k']}_{job['query']}_{digest[:8]}"


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


@app.function(image=image, gpu=GPU, timeout=7200, max_containers=10, retries=1,
              volumes={"/results": results_vol})
def score_one(job: dict) -> dict:
    """One (arm, k, query) job. Raw scores only; thresholds are chosen locally."""
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
    t0 = time.time()
    clf = model.CrypticSitePredictor().fit(ctx)
    scores = clf.predict_by_protein(target)
    out = {**{k: job[k] for k in ("name", "k", "query", "seed")}, "n_features": n_feat, "features": cols,
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
        out = Path("build/ftemb_stack") / run_id
        out.mkdir(parents=True, exist_ok=True)
        for r in fetch_results.remote(run_id):
            seed = r.get("seed", SEED)              # fe1 results predate the field
            name = f"{r['name']}_k{r['k']}_{r['query']}" + (f"_s{seed}" if seed != SEED else "") + ".json"
            (out / name).write_text(json.dumps(r))
            print(f"wrote {out / name} ({len(r['score']):,} scores, {r['seconds']:.0f}s)")
        return
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {sorted(STAGES)}")
    jobs = [{"budget": BUDGET, "seed": SEED, **j, "run_id": run_id, "table": TABLE_NAME}
            for j in STAGES[stage]]
    tags = [job_tag(j) for j in jobs]
    assert len(set(tags)) == len(tags), "job tags collide"
    f = modal.Function.from_name("cryptobench-ftemb-stack", "score_one")
    for j in jobs:
        print(f"spawned {j['name']} k={j['k']} {j['query']}: {f.spawn(j).object_id}")
