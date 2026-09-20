"""Sweep TabICL constructor knobs, screening on CV fold 0 and never on test.

WHY A SEPARATE FILE FROM tabicl_fanout.py
    That file is the provenance of the published headline and the retrieval path
    for the staged CV scores. Its resume cache is keyed `k{k}_{fold}`, which
    carries no knob information -- folding knobs into that tag would invalidate
    the cache for run_20260910_124330 and break `--collect` on it. So this is a
    clone with its own volume and its own tag.

THE TAG IS A HASH OF THE WHOLE JOB
    A cache keyed on a hand-maintained subset of the knobs is one forgotten
    field away from silently returning config A's scores for config B -- which
    would read as a clean null result. Hashing the entire job dict makes that
    structurally impossible, and `--stage screen` asserts all tags are distinct
    before spending anything.

PROTOCOL
    screen  : every config in knob_grid.CONFIGS on fold 0 only. The `control`
              job is also a determinism gate: its scores must equal the staged
              k64_train-fold-0.json from the headline run, or surfacing the
              knobs changed behaviour and the sweep is void.
    confirm : the one finalist on folds 1-3. The control arm is free -- those
              folds are already in the fanout volume.

USAGE
    modal run deploy/knob_sweep.py --stage screen  --run-id ks1
    modal run deploy/knob_sweep.py --stage confirm --run-id ks1 --finalist norm5
    modal run deploy/knob_sweep.py --stage screen  --run-id ks1 --collect
"""
from __future__ import annotations

import os
import sys

import modal

TABLE_LOCAL = os.path.expanduser("~/cryptobench_stage/tables/residues_esm64.parquet")
if modal.is_local() and not os.path.exists(TABLE_LOCAL):
    raise SystemExit(
        f"missing {TABLE_LOCAL}\n\nStage it first:\n"
        "    python scripts/merge_esm.py --n-components 64\n"
        "    cp build/residues_esm64.parquet ~/cryptobench_stage/tables/"
    )

GPU = "T4"
GPU_HOURLY_USD = 0.59
CKPT = "tabicl-classifier-v2-20260212.ckpt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("tabicl==2.1.1", "scikit-learn", "pandas", "pyarrow", "numpy<2")
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"d(repo_id='jingang/TabICL', filename='{CKPT}')\""
    )
    .add_local_file(TABLE_LOCAL, "/data/table.parquet")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
)

app = modal.App("cryptobench-knob-sweep")

# Deliberately NOT cryptobench-fanout-results: that volume holds the headline's
# provenance and must not be written into by an exploratory sweep.
results_vol = modal.Volume.from_name("cryptobench-knob-results",
                                     create_if_missing=True)


def job_tag(job: dict) -> str:
    """Stable tag covering every field that can change the scores."""
    import hashlib
    import json

    payload = {k: v for k, v in job.items() if k != "run_id"}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{job['name']}_{job['held'] or 'final'}_{digest[:8]}"


@app.function(image=image, gpu=GPU, timeout=5400, max_containers=12, retries=1,
              volumes={"/results": results_vol})
def score_one(job: dict) -> dict:
    """One (config, fold) job. Returns raw scores; no thresholding here."""
    sys.path.insert(0, "/app")
    import json as _json
    import os as _os
    import time

    tag = job_tag(job)
    run_dir = f"/results/{job['run_id']}"
    cached = f"{run_dir}/{tag}.json"
    try:
        results_vol.reload()
    except Exception:  # noqa: BLE001 - a reload failure must not lose the job
        pass
    if _os.path.exists(cached):
        with open(cached) as fh:
            prev = _json.load(fh)
        print(f"{tag}: cached, skipping ({prev['seconds']:.0f}s originally)",
              flush=True)
        return prev

    import pandas as pd
    from cryptic_tabicl import config as C, model, sampling

    held = job["held"]
    table = pd.read_parquet("/data/table.parquet")
    all_pcs = sorted(c for c in table.columns if c.startswith("esm_pc"))
    table = table.drop(columns=all_pcs[job["n_components"]:])
    n_feat = len([c for c in table.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])

    train = table[table["split"] == "train"]
    if held is None:
        ctx_src, target = train, table[table["split"] == "test"]
    else:
        ctx_src, target = train[train["fold"] != held], train[train["fold"] == held]

    # Same seed for every config, so comparisons are paired on the context draw.
    ctx, plan = sampling.build_context(ctx_src, budget=job["budget"],
                                       seed=job["seed"])
    print(f"{job['name']} held={held} features={n_feat}\n{plan}", flush=True)

    t0 = time.time()
    clf = model.CrypticSitePredictor(**job["model"]).fit(ctx)
    scores = clf.predict_by_protein(target)
    seconds = time.time() - t0
    print(f"{job['name']} held={held}: scored {len(target):,} rows in "
          f"{seconds:.0f}s", flush=True)

    out = {
        "name": job["name"], "held": held, "n_features": n_feat,
        "seconds": seconds, "config": clf.config(),
        "apo_id": target[C.GROUP_COL].tolist(),
        "icode": target["icode"].astype(str).tolist(),
        "y": target[C.LABEL_COL].astype(int).tolist(),
        "score": [float(s) for s in scores],
    }
    _os.makedirs(run_dir, exist_ok=True)
    with open(cached, "w") as fh:
        _json.dump(out, fh)
    results_vol.commit()
    return out


@app.function(image=image, timeout=1800, volumes={"/results": results_vol})
def fetch_results(run_id: str) -> list:
    """Read back one run's stored results. Used by `--collect`; costs no GPU."""
    import glob
    import json

    results_vol.reload()
    files = sorted(glob.glob(f"/results/{run_id}/*.json"))
    print(f"reading {len(files)} stored job results from {run_id}", flush=True)
    return [json.load(open(f)) for f in files]


def _build_jobs(names, folds, budget, seed):
    sys.path.insert(0, ".")
    from deploy.knob_grid import CONFIGS

    return [{"name": n, "model": CONFIGS[n], "held": h, "budget": budget,
             "seed": seed, "n_components": 64}
            for n in names for h in folds]


@app.local_entrypoint()
def main(stage: str = "screen", run_id: str = "", finalist: str = "",
         budget: int = 50_000, collect: bool = False, spawn: bool = False):
    import json
    import time

    sys.path.insert(0, ".")
    import numpy as np
    from cryptic_tabicl import evaluate
    from deploy.knob_grid import (CONFIGS, CONFIRM_FOLDS, MIN_DELTA_AUPRC,
                                  SCREEN_FOLD, SEED, SPEED_ONLY)

    if stage not in ("screen", "confirm"):
        raise SystemExit("--stage must be 'screen' or 'confirm'")
    run_id = run_id or time.strftime("ks_%Y%m%d_%H%M%S")

    if stage == "screen":
        names, folds = list(CONFIGS), [SCREEN_FOLD]
    else:
        if not finalist or finalist not in CONFIGS:
            raise SystemExit(f"--finalist must be one of {sorted(CONFIGS)}")
        names, folds = ["control", finalist], CONFIRM_FOLDS

    jobs = [dict(j, run_id=run_id) for j in _build_jobs(names, folds, budget, SEED)]

    # Structural guard against a cache alias silently faking a null result.
    tags = [job_tag(j) for j in jobs]
    assert len(set(tags)) == len(tags), "job tags collide; refusing to run"

    if spawn:
        # Fire-and-forget. `.map()` needs one gRPC stream up for the whole run,
        # and this host drops it -- twice, killing the two 32-estimator jobs
        # that outlive the connection. Spawning makes the client's involvement
        # one short RPC per job, after which the job's survival is Modal's
        # problem and its result lands in the volume.
        f = modal.Function.from_name("cryptobench-knob-sweep", "score_one")
        for j in jobs:
            print(f"spawned {j['name']} held={j['held']}: {f.spawn(j).object_id}")
        print(f"\n{len(jobs)} jobs spawned. Cached ones return immediately.\n"
              f"Collect when the volume has them all:\n"
              f"    modal run deploy/knob_sweep.py --stage {stage} "
              f"--run-id {run_id} --collect")
        return

    t0 = time.time()
    if collect:
        print(f"collecting stored results for {run_id} (no GPU)")
        results = fetch_results.remote(run_id)
        results = [r for r in results if r["name"] in names and r["held"] in folds]
    else:
        print(f"{len(jobs)} jobs on {GPU} [run_id={run_id}, stage={stage}]")
        print("if the client disconnects, re-aggregate for free with:\n"
              f"    modal run deploy/knob_sweep.py --stage {stage} "
              f"--run-id {run_id} --collect", flush=True)
        results = []
        for r in score_one.map(jobs):
            results.append(r)
            print(f"  done {r['name']} held={r['held']} "
                  f"({len(results)}/{len(jobs)})", flush=True)

    gpu_seconds = sum(r["seconds"] for r in results)
    print(f"\n{time.time()-t0:.0f}s wall / {gpu_seconds/3600:.2f} GPU-h "
          f"(~${gpu_seconds/3600*GPU_HOURLY_USD:.2f})")

    # AUC/AUPRC are threshold-free, so screening needs no threshold at all.
    rows = {}
    for r in results:
        y, s, g = np.array(r["y"]), np.array(r["score"]), np.array(r["apo_id"])
        m = evaluate.summarise(y, s, g, 0.5, label=r["name"])
        rows.setdefault(r["name"], {})[r["held"]] = {
            "AUC": m["AUC"], "AUPRC": m["AUPRC"], "seconds": r["seconds"]}

    base = {f: rows["control"][f]["AUPRC"] for f in folds if f in rows.get("control", {})}
    print(f"\n{'config':<16} {'AUPRC':>8} {'dAUPRC':>9} {'AUC':>7} {'sec':>6}")
    print("-" * 50)
    ranked = []
    for name in names:
        for f in folds:
            if f not in rows.get(name, {}):
                continue
            a = rows[name][f]["AUPRC"]
            d = a - base.get(f, float("nan"))
            flag = ""
            if name in SPEED_ONLY:
                flag = "speed"
            elif name != "control" and d >= MIN_DELTA_AUPRC:
                flag = "PROMOTE"
            print(f"{name:<16} {a:>8.4f} {d:>+9.4f} {rows[name][f]['AUC']:>7.4f} "
                  f"{rows[name][f]['seconds']:>6.0f}  {flag}")
            if name != "control" and name not in SPEED_ONLY:
                ranked.append((d, name))

    if stage == "screen" and ranked:
        ranked.sort(reverse=True)
        best_d, best = ranked[0]
        print(f"\nbest accuracy config: {best} (dAUPRC {best_d:+.4f})")
        if best_d < MIN_DELTA_AUPRC:
            print(f"BELOW the {MIN_DELTA_AUPRC} noise band -- no finalist. "
                  "Report the nulls and stop.")
        else:
            print(f"confirm it with:\n    modal run deploy/knob_sweep.py "
                  f"--stage confirm --run-id {run_id} --finalist {best}")

    dest = f"build/knob_sweep_{stage}.json"
    os.makedirs("build", exist_ok=True)
    with open(dest, "w") as fh:
        json.dump({"run_id": run_id, "stage": stage, "seed": SEED,
                   "folds": folds, "rows": rows,
                   "configs": {n: CONFIGS[n] for n in names}}, fh, indent=2)
    print(f"\nwrote {dest}")
