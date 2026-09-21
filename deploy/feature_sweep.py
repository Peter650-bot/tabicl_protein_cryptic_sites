"""Screen feature sets on CV fold 0, confirm one finalist, score test once.

WHY A SEPARATE FILE FROM knob_sweep.py AND tabicl_fanout.py
    `tabicl_fanout.py` is the provenance of the published headline and the
    retrieval path for the staged CV scores; its resume cache is keyed
    `k{k}_{fold}`, which carries no feature-set information at all, so reusing
    it would alias every arm onto the control's stored scores and read as a
    clean sweep of nulls. `knob_sweep.py` is the record of a finished
    experiment. Both are left alone.

THE TAG IS A HASH OF THE WHOLE JOB
    Same guard as the knob sweep, and it matters more here: the arms differ only
    in a column list, so a tag that forgot the feature set would produce ten
    identical results and look like a decisive null. `main()` asserts all tags
    are distinct before spending anything.

THE SUPERVISED BLOCK IS CHECKED AGAINST THE FOLD IT HOLDS OUT
    `scripts/esm_supervised.py` writes one PLS block per nesting, each fitted
    without the fold it will be screened on, plus a manifest recording which
    fold each block excluded. A job that wants a supervised block looks up its
    own held fold in that manifest and refuses to run on a mismatch. This is
    the only place in the feature work package where a label has touched an
    input, so it is the only place that needs this check -- and a check that
    only runs when someone remembers to look is not a check.

USAGE
    modal deploy deploy/feature_sweep.py          # this host drops .map() streams
    modal run deploy/feature_sweep.py --stage screen  --run-id fs1 --spawn
    modal run deploy/feature_sweep.py --stage screen  --run-id fs1 --collect
    modal run deploy/feature_sweep.py --stage confirm --run-id fs1 --finalist esmnb --spawn
    modal run deploy/feature_sweep.py --stage test    --run-id fs1 --finalist esmnb --spawn
"""
from __future__ import annotations

import os
import sys

import modal

STAGE_DIR = os.path.expanduser("~/cryptobench_stage/tables")
TABLE_LOCAL = f"{STAGE_DIR}/residues_derived.parquet"
MANIFEST_LOCAL = f"{STAGE_DIR}/esm_supervised_manifest.json"
MANIFEST_IN_IMAGE = "/data/esm_supervised_manifest.json"

if modal.is_local() and not os.path.exists(TABLE_LOCAL):
    raise SystemExit(
        f"missing {TABLE_LOCAL}\n\nStage it first:\n"
        "    python scripts/build_coords.py\n"
        "    python scripts/build_derived.py\n"
        "    modal run deploy/esm_supervised_modal.py\n"
        "    python scripts/merge_derived.py --extra build/derived_features.parquet "
        "build/esm_supervised.parquet\n"
        f"    cp build/residues_derived.parquet {STAGE_DIR}/\n"
        f"    cp build/esm_supervised_manifest.json {STAGE_DIR}/"
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
    .add_local_file(MANIFEST_LOCAL, "/data/esm_supervised_manifest.json")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
)

app = modal.App("cryptobench-feature-sweep")

# Deliberately NOT the fanout or knob volume: those hold finished records.
results_vol = modal.Volume.from_name("cryptobench-feature-results",
                                     create_if_missing=True)


def job_tag(job: dict) -> str:
    """Stable tag covering every field that can change the scores."""
    import hashlib
    import json

    payload = {k: v for k, v in job.items() if k != "run_id"}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{job['name']}_{job['held'] or 'final'}_{digest[:8]}"


@app.function(image=image, gpu=GPU, timeout=7200, max_containers=12, retries=1,
              volumes={"/results": results_vol})
def score_one(job: dict) -> dict:
    """One (feature set, fold) job. Returns raw scores; no thresholding here."""
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
    from cryptic_tabicl import config as C, feature_sets as FS, model, sampling

    held = job["held"]
    table = pd.read_parquet("/data/table.parquet")

    supervised = (FS.supervised_block(held, MANIFEST_IN_IMAGE)
                  if job["needs_supervised"] else [])
    cols = FS.resolve(job["name"], table, supervised)

    # Hand the model exactly this arm's columns, plus the bookkeeping columns
    # `feature_matrix` is going to drop anyway. Anything else in the table --
    # another arm's family, another nesting's block -- must not be visible.
    #
    # ORDER IS PRESERVED, and that is not cosmetic. TabICL's default
    # `feat_shuffle_method="latin"` assigns features to groups by POSITION, so
    # reordering the columns changes the scores. Selecting in table order keeps
    # the control arm's 117 columns in exactly the order the published run fed
    # them, which is what makes the determinism gate meaningful rather than
    # approximate.
    wanted = set(cols) | {c for c in C.NON_FEATURE_COLS if c in table.columns}
    table = table[[c for c in table.columns if c in wanted]]
    if supervised:
        table = table.rename(columns={c: f"{FS.SUPERVISED_PREFIX}{i:02d}"
                                      for i, c in enumerate(supervised)})
    n_feat = len([c for c in table.columns if c not in C.NON_FEATURE_COLS])
    assert n_feat == len(cols), f"{n_feat} features reached the model, wanted {len(cols)}"

    train = table[table["split"] == "train"]
    if held is None:
        ctx_src, target = train, table[table["split"] == "test"]
    else:
        ctx_src, target = train[train["fold"] != held], train[train["fold"] == held]

    # Same seed for every arm, so comparisons are paired on the context draw.
    ctx, plan = sampling.build_context(ctx_src, budget=job["budget"],
                                       seed=job["seed"])
    print(f"{job['name']} held={held} features={n_feat}\n{plan}", flush=True)

    t0 = time.time()
    clf = model.CrypticSitePredictor().fit(ctx)
    scores = clf.predict_by_protein(target)
    seconds = time.time() - t0
    print(f"{job['name']} held={held}: scored {len(target):,} rows in "
          f"{seconds:.0f}s", flush=True)

    out = {
        "name": job["name"], "held": held, "n_features": n_feat,
        "features": cols, "supervised": supervised,
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


def _build_jobs(names, folds, budget, seed, needs):
    return [{"name": n, "held": h, "budget": budget, "seed": seed,
             "needs_supervised": n in needs}
            for n in names for h in folds]


@app.local_entrypoint()
def main(stage: str = "screen", run_id: str = "", finalist: str = "",
         budget: int = 50_000, collect: bool = False, spawn: bool = False,
         skip: str = ""):
    import json
    import time

    sys.path.insert(0, ".")
    import numpy as np
    from cryptic_tabicl import evaluate
    from deploy.feature_grid import (ARMS, CONFIRM_FOLDS, MIN_DELTA_AUPRC,
                                     NEEDS_SUPERVISED, SCREEN_FOLD, SEED)

    if stage not in ("screen", "confirm", "test"):
        raise SystemExit("--stage must be 'screen', 'confirm' or 'test'")
    run_id = run_id or time.strftime("fs_%Y%m%d_%H%M%S")
    dropped = [s for s in skip.split(",") if s]

    if stage == "screen":
        names = [n for n in ARMS if n not in dropped]
        folds = [SCREEN_FOLD]
    elif stage == "confirm":
        if not finalist or finalist not in ARMS:
            raise SystemExit(f"--finalist must be one of {sorted(ARMS)}")
        names, folds = [finalist], CONFIRM_FOLDS
    else:
        if not finalist or finalist not in ARMS:
            raise SystemExit(f"--finalist must be one of {sorted(ARMS)}")
        # The control's test scores are already published, in
        # build/modal_result_esm64.json, and the fold-0 determinism gate proves
        # this harness reproduces them. Re-running it would buy a number we
        # already have, at the cost of a job. `scripts/paired_bootstrap.py`
        # takes the stored scores as the other arm.
        names, folds = [finalist], [None]

    jobs = [dict(j, run_id=run_id)
            for j in _build_jobs(names, folds, budget, SEED, NEEDS_SUPERVISED)]

    # Structural guard against a cache alias silently faking a null result.
    tags = [job_tag(j) for j in jobs]
    assert len(set(tags)) == len(tags), "job tags collide; refusing to run"

    if spawn:
        # Fire-and-forget. `.map()` needs one gRPC stream up for the whole run
        # and this host drops it; the two longest jobs of the knob sweep died
        # that way, twice. Spawning reduces the client's involvement to one
        # short RPC per job.
        f = modal.Function.from_name("cryptobench-feature-sweep", "score_one")
        for j in jobs:
            print(f"spawned {j['name']} held={j['held']}: {f.spawn(j).object_id}")
        print(f"\n{len(jobs)} jobs spawned. Cached ones return immediately.\n"
              f"Collect when the volume has them all:\n"
              f"    modal run deploy/feature_sweep.py --stage {stage} "
              f"--run-id {run_id} --collect"
              + (f" --finalist {finalist}" if finalist else ""))
        return

    t0 = time.time()
    if collect:
        print(f"collecting stored results for {run_id} (no GPU)")
        results = fetch_results.remote(run_id)
        results = [r for r in results if r["name"] in names and r["held"] in folds]
    else:
        print(f"{len(jobs)} jobs on {GPU} [run_id={run_id}, stage={stage}]")
        print("if the client disconnects, re-aggregate for free with:\n"
              f"    modal run deploy/feature_sweep.py --stage {stage} "
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
            "AUC": m["AUC"], "AUPRC": m["AUPRC"], "seconds": r["seconds"],
            "n_features": r["n_features"]}

    # No control arm in the confirm and test stages: their baselines are the
    # stored CV scores and the published test scores respectively.
    base = {f: rows["control"][f]["AUPRC"]
            for f in folds if f in rows.get("control", {})}
    if "control" not in rows:
        print("\n(no control arm in this stage; deltas are against the stored "
              "scores, computed separately)")
    print(f"\n{'arm':<24} {'feat':>5} {'AUPRC':>8} {'dAUPRC':>9} {'AUC':>7} "
          f"{'sec':>6} {'prior':>8}")
    print("-" * 76)
    ranked = []
    for name in names:
        for f in folds:
            if f not in rows.get(name, {}):
                continue
            a = rows[name][f]["AUPRC"]
            d = a - base.get(f, float("nan"))
            flag = "PROMOTE" if (name != "control" and d >= MIN_DELTA_AUPRC) else ""
            print(f"{name:<24} {rows[name][f]['n_features']:>5} {a:>8.4f} "
                  f"{d:>+9.4f} {rows[name][f]['AUC']:>7.4f} "
                  f"{rows[name][f]['seconds']:>6.0f} {ARMS[name][0]:>8}  {flag}")
            if name != "control":
                ranked.append((d, name))

    if stage == "screen" and ranked:
        ranked.sort(reverse=True)
        best_d, best = ranked[0]
        print(f"\nbest arm: {best} (dAUPRC {best_d:+.4f})")
        if best_d < MIN_DELTA_AUPRC:
            print(f"BELOW the {MIN_DELTA_AUPRC} bar -- no finalist. "
                  "Report the nulls and do not score test.")
        else:
            print(f"confirm it with:\n    modal run deploy/feature_sweep.py "
                  f"--stage confirm --run-id {run_id} --finalist {best} --spawn")

    dest = f"build/feature_sweep_{stage}.json"
    os.makedirs("build", exist_ok=True)
    with open(dest, "w") as fh:
        json.dump({"run_id": run_id, "stage": stage, "seed": SEED,
                   "folds": [f or "final" for f in folds], "rows": rows,
                   "bar": MIN_DELTA_AUPRC,
                   "priors": {n: ARMS[n][0] for n in names},
                   "rationale": {n: ARMS[n][1] for n in names}}, fh, indent=2)
    print(f"\nwrote {dest}")
