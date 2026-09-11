"""How much does a headline number move if only the 50k context draw changes?

WHAT VARIES AND WHAT DOES NOT
    TabICL is an in-context learner: the 50,000-row context IS the model. That
    context is a subsample of the 304,294 training rows, and `build_context`
    takes a seed. Every headline run uses the default (config.RANDOM_SEED = 0)
    and never overrides it, so the reported numbers are ONE draw.

    This sweeps `seed` and holds every selection rule fixed:

      varies    which 33,718 negatives are drawn from the pool of 288,012,
                and the row order of the context
      fixed     all 16,282 positives (taken in full, every time), the 50k
                budget, the 120-negatives-per-protein cap, the natural
                negative proportions (HARD_NEGATIVE_SHARE = 0), the feature
                columns, the train/test split, and the threshold PROCEDURE

    Two draws share only ~12% of their negatives, so these are genuinely
    different contexts rather than perturbations of one.

THE THRESHOLD IS RE-CHOSEN PER SEED
    Each seed runs the full protocol: 4 CV jobs (one held-out train fold each)
    to pick a decision threshold at the target FPR, median over the four, then
    one test job scored with that threshold. Reusing seed 0's threshold would
    hide exactly the variance that TPR / FPR / MCC / F1 suffer from -- which is
    the largest component in the CPU stand-in (scripts/context_seed_variance.py).

    5 GPU jobs per seed. Nothing about the test set touches the threshold.

WHAT THIS DOES AND DOES NOT ANSWER
    Answers: is a difference between two of OUR variants bigger than the noise
    of the context draw? All seeds share one test set, so test-set uncertainty
    largely cancels in such a paired comparison.
    Does not answer: anything about the comparison to pLM-NN. Every seed sits
    the same 231-protein test set; resampling contexts says nothing about
    whether that test set is representative. See RESULTS.md section 2.1.

USAGE
    modal run deploy/context_bootstrap.py --seeds 12

    On a flaky connection, prefer the fire-and-forget path. `.map()` below needs
    one gRPC stream to stay up for the whole run, and a drop cancels every job
    still in flight (they only reach the volume on completion). `--detach` does
    NOT cover a fan-out -- it keeps the last triggered function alive, not 60 of
    them. Instead deploy once and spawn, so the client only makes one short RPC
    per job:

        modal deploy deploy/context_bootstrap.py
        python - <<'EOF'
        import modal, pandas as pd
        RUN = "my_run_id"
        tb = pd.read_parquet(TABLE_LOCAL, columns=["split", "fold"])
        folds = sorted(tb.loc[tb.split == "train", "fold"].dropna().unique())
        f = modal.Function.from_name("cryptobench-context-bootstrap", "score_one")
        for s in range(12):
            for h in list(folds) + [None]:
                f.spawn({"seed": s, "held": h, "budget": 50_000,
                         "n_estimators": 8, "n_components": 64, "run_id": RUN})
        EOF

    Then wait for 5 x seeds files to appear in the volume and aggregate for free:

        modal volume ls cryptobench-bootstrap-results <run_id>
        modal run deploy/context_bootstrap.py --collect --run-id <run_id> --seeds 12
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

app = modal.App("cryptobench-context-bootstrap")

# Each job's scores are ALSO written here, keyed by (run_id, seed, fold).
#
# WHY: aggregation happens in the local entrypoint, so a client-side gRPC drop
# throws away every GPU-second already spent. This is a 60-job run and it has
# died that way on this host -- `AttributeError: 'Connection' object has no
# attribute '_transport'` at job 10 of 60. Persisting each job makes the GPU
# work durable; `--collect` replays the aggregation from the volume for free.
# Same pattern as deploy/tabicl_fanout.py.
results_vol = modal.Volume.from_name("cryptobench-bootstrap-results",
                                     create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=5400, max_containers=20, retries=1,
              volumes={"/results": results_vol})
def score_one(job: dict) -> dict:
    """One (seed, fold) job. Returns raw scores; thresholding happens client-side."""
    sys.path.insert(0, "/app")
    import time

    _TAG = f"s{job['seed']}_{job['held'] or 'final'}"
    # ---- resume: a job already in the volume is not recomputed ------------
    # This is what makes the run pausable. Stop the app at any point, re-run
    # later with the SAME --run-id, and only the missing jobs cost GPU time.
    # The guard is the file's existence, so it is safe to re-enter repeatedly.
    import json as _json
    import os as _os
    _run_dir = f"/results/{job['run_id']}"
    _cached = f"{_run_dir}/{_TAG}.json"
    try:
        results_vol.reload()
    except Exception:  # noqa: BLE001 - a reload failure must not lose the job
        pass
    if _os.path.exists(_cached):
        with open(_cached) as _fh:
            _prev = _json.load(_fh)
        print(f"{_TAG}: cached, skipping ({_prev['seconds']:.0f}s originally)",
              flush=True)
        return _prev



    import pandas as pd
    from cryptic_tabicl import config as C, model, sampling

    seed, held, k = job["seed"], job["held"], job["n_components"]
    table = pd.read_parquet("/data/table.parquet")

    all_pcs = sorted(c for c in table.columns if c.startswith("esm_pc"))
    table = table.drop(columns=all_pcs[k:])
    n_feat = len([c for c in table.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])

    train = table[table["split"] == "train"]
    if held is None:
        ctx_src, target = train, table[table["split"] == "test"]
    else:
        ctx_src, target = train[train["fold"] != held], train[train["fold"] == held]

    # THE ONE THING THAT VARIES ACROSS JOBS OF THE SAME (k, held).
    ctx, plan = sampling.build_context(ctx_src, budget=job["budget"], seed=seed)
    print(f"seed={seed} held={held} features={n_feat}\n{plan}", flush=True)

    t0 = time.time()
    clf = model.CrypticSitePredictor(n_estimators=job["n_estimators"]).fit(ctx)
    scores = clf.predict_by_protein(target)
    print(f"seed={seed} held={held}: scored {len(target):,} rows in "
          f"{time.time()-t0:.0f}s", flush=True)

    out = {
        "seed": seed, "held": held, "n_features": n_feat,
        "seconds": time.time() - t0,
        "n_positive": plan.n_positive, "n_negative": plan.n_negative,
        "apo_id": target[C.GROUP_COL].tolist(),
        "y": target[C.LABEL_COL].astype(int).tolist(),
        "score": [float(s) for s in scores],
        "in_noncryptic_pocket": target["in_noncryptic_pocket"].astype(int).tolist(),
    }

    # Persist before returning, so the GPU-seconds survive a client disconnect.
    import json
    import os
    # Same path the resume check reads, derived from the same _TAG so the two
    # cannot drift apart.
    os.makedirs(_run_dir, exist_ok=True)
    with open(_cached, "w") as fh:
        json.dump(out, fh)
    results_vol.commit()
    return out


@app.function(image=image, timeout=1800, volumes={"/results": results_vol})
def fetch_results(run_id: str) -> list:
    """Read back one run's stored job results. Used by `--collect`."""
    import glob
    import json

    results_vol.reload()
    files = sorted(glob.glob(f"/results/{run_id}/*.json"))
    print(f"reading {len(files)} stored job results from {run_id}", flush=True)
    return [json.load(open(f)) for f in files]


@app.local_entrypoint()
def main(seeds: int = 12, budget: int = 50_000, n_estimators: int = 8,
         n_components: int = 64, run_id: str = "", collect: bool = False):
    import json
    import time

    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd
    from cryptic_tabicl import config as C, evaluate

    METRICS = ["AUC", "AUPRC", "TPR", "FPR", "MCC", "F1", "F1_weighted", "ACC"]

    table = pd.read_parquet(TABLE_LOCAL, columns=["split", "fold"])
    folds = sorted(table.loc[table["split"] == "train", "fold"].dropna().unique())

    run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    jobs = [{"seed": s, "held": h, "budget": budget, "n_estimators": n_estimators,
             "n_components": n_components, "run_id": run_id}
            for s in range(seeds) for h in list(folds) + [None]]

    t0 = time.time()
    if collect:
        # Re-aggregate a run whose client died. Costs nothing: no GPU is used.
        print(f"collecting stored results for {run_id}")
        results = fetch_results.remote(run_id)
        gpu_seconds = sum(r["seconds"] for r in results)
        missing = {(s, h) for s in range(seeds) for h in list(folds) + [None]} - {
            (r["seed"], r["held"]) for r in results}
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(jobs)} jobs are missing from the "
                f"volume, e.g. {sorted(missing, key=str)[:3]}. Re-run without "
                "--collect.")
    else:
        print(f"{len(jobs)} jobs ({seeds} seeds x {len(folds)+1} runs) "
              f"across up to 20 {GPU} containers  [run_id={run_id}]")
        print("if the client disconnects, re-aggregate for free with:\n"
              f"    modal run deploy/context_bootstrap.py --collect "
              f"--run-id {run_id} --seeds {seeds}", flush=True)
        results, gpu_seconds = [], 0.0
        for r in score_one.map(jobs):
            results.append(r)
            gpu_seconds += r["seconds"]
            print(f"  done seed={r['seed']} held={r['held']} "
                  f"({len(results)}/{len(jobs)})", flush=True)

    wall = time.time() - t0
    print(f"\nall jobs finished in {wall:.0f}s wall / {gpu_seconds/3600:.2f} GPU-h "
          f"(~${gpu_seconds/3600*GPU_HOURLY_USD:.2f})")

    per_seed, score_by_seed, ident = [], {}, None
    for s in range(seeds):
        cv = [r for r in results if r["seed"] == s and r["held"] is not None]
        fin = next(r for r in results if r["seed"] == s and r["held"] is None)

        thresholds = [evaluate.choose_threshold(np.array(r["y"]), np.array(r["score"]),
                                                np.array(r["apo_id"]))
                      for r in sorted(cv, key=lambda r: str(r["held"]))]
        threshold = float(np.median(thresholds))

        y, sc, g = np.array(fin["y"]), np.array(fin["score"]), np.array(fin["apo_id"])
        res = evaluate.summarise(y, sc, g, threshold, label=f"seed {s}")
        res["seed"] = s
        res["n_negative"] = fin["n_negative"]
        res["n_positive"] = fin["n_positive"]
        per_seed.append(res)
        score_by_seed[str(s)] = [round(float(v), 6) for v in fin["score"]]
        if ident is None:
            ident = {"apo_id": fin["apo_id"], "y": fin["y"],
                     "in_noncryptic_pocket": fin["in_noncryptic_pocket"]}
        print(f"  seed {s:>2}  thr {threshold:.4f}  AUC {res['AUC']:.4f}  "
              f"AUPRC {res['AUPRC']:.4f}  F1w {res['F1_weighted']:.4f}  "
              f"MCC {res['MCC']:.4f}")

    df = pd.DataFrame(per_seed)
    stat = pd.DataFrame({
        "mean": df[METRICS].mean(),
        "sd": df[METRICS].std(ddof=1),
        "min": df[METRICS].min(),
        "max": df[METRICS].max(),
        "lo95": df[METRICS].quantile(0.025),
        "hi95": df[METRICS].quantile(0.975),
        "seed 0": df.set_index("seed").loc[0, METRICS],
    })
    stat["range"] = stat["max"] - stat["min"]
    print("\n" + stat.to_string(float_format=lambda v: f"{v:.4f}"))

    out = {
        "config": {"seeds": seeds, "budget": budget, "n_components": n_components,
                   "n_features": results[0]["n_features"],
                   "n_estimators": n_estimators, "gpu": GPU,
                   "gpu_hours": gpu_seconds / 3600,
                   "usd": gpu_seconds / 3600 * GPU_HOURLY_USD,
                   "wall_seconds": wall},
        "per_seed": per_seed,
        "summary": stat.to_dict(orient="index"),
        "identity": ident,
        "scores_by_seed": score_by_seed,
    }
    dest = "build/context_bootstrap.json"
    with open(dest, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {dest}")
