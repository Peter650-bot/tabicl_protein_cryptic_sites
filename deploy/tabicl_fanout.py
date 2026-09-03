"""TabICL over several feature sets, fanned out across Modal containers.

WHY THIS EXISTS ALONGSIDE evaluate_modal.py
    `evaluate_modal.py` runs one variant per invocation and does its 4 CV folds
    plus the final test run serially inside a single container. That was fine
    when there was one feature set. Sweeping k over {0, 16, 32, 64} that way is
    20 sequential GPU jobs behind one queue.

    The 5 jobs of a variant are independent: the CV folds only exist to pick a
    decision threshold, and the final run's SCORES do not depend on that
    threshold -- only the metrics computed from them do. So all 20 can run at
    once and the threshold can be applied afterwards, on the client. Same
    arithmetic, same frozen-threshold discipline, ~20x less wall-clock.

ONE TABLE, NOT FOUR
    residues_esm64 already contains the 54 hand-built features plus 64 ordered
    principal components, and PCA components are ordered -- so `n_components=k`
    is a column trim, not a different fit. Shipping esm16/esm32/esm64 as three
    files uploaded 510 MB of overlapping data; this ships 198 MB once. It is
    also read from local disk rather than the OneDrive-backed mount, which is
    where the previous run actually spent its time.

THE THRESHOLD IS STILL CHOSEN ON TRAIN ONLY
    Each CV job returns the held-out fold's scores. The client picks a threshold
    per fold at the target FPR, takes the median, and only then applies it to
    the test scores. Nothing about the test set touches the threshold.

USAGE
    modal run deploy/tabicl_fanout.py --variants 0,16,32,64
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

app = modal.App("cryptobench-tabicl-fanout")


@app.function(image=image, gpu=GPU, timeout=5400, max_containers=20, retries=1)
def score_one(job: dict) -> dict:
    """One (feature set, fold) job. Returns raw scores; no thresholding here.

    `held` is a train fold name for a CV job, or None for the final run that
    scores the test split.
    """
    sys.path.insert(0, "/app")
    import time

    import pandas as pd
    from cryptic_tabicl import config as C, model, sampling

    k, held = job["n_components"], job["held"]
    table = pd.read_parquet("/data/table.parquet")

    # Column trim: keep the hand-built features and the first k components.
    all_pcs = sorted(c for c in table.columns if c.startswith("esm_pc"))
    drop = all_pcs[k:]
    table = table.drop(columns=drop)
    n_feat = len([c for c in table.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])

    train = table[table["split"] == "train"]
    if held is None:
        ctx_src, target = train, table[table["split"] == "test"]
    else:
        ctx_src, target = train[train["fold"] != held], train[train["fold"] == held]

    ctx, plan = sampling.build_context(ctx_src, budget=job["budget"])
    print(f"k={k} held={held} features={n_feat}\n{plan}", flush=True)

    t0 = time.time()
    clf = model.CrypticSitePredictor(n_estimators=job["n_estimators"]).fit(ctx)
    scores = clf.predict_by_protein(target)
    print(f"k={k} held={held}: scored {len(target):,} rows in "
          f"{time.time()-t0:.0f}s", flush=True)

    return {
        "n_components": k, "held": held, "n_features": n_feat,
        "seconds": time.time() - t0,
        "apo_id": target[C.GROUP_COL].tolist(),
        "chain": target["chain"].tolist(),
        "seq_id": target["seq_id"].tolist(),
        "y": target[C.LABEL_COL].astype(int).tolist(),
        "score": [float(s) for s in scores],
        "in_noncryptic_pocket": target["in_noncryptic_pocket"].astype(int).tolist(),
    }


@app.local_entrypoint()
def main(variants: str = "0,16,32,64", budget: int = 50_000,
         n_estimators: int = 8):
    import json
    import time

    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd
    from cryptic_tabicl import config as C, evaluate

    ks = [int(x) for x in variants.split(",")]
    table = pd.read_parquet(TABLE_LOCAL, columns=["split", "fold"])
    folds = sorted(table.loc[table["split"] == "train", "fold"].dropna().unique())

    jobs = [{"n_components": k, "held": h, "budget": budget,
             "n_estimators": n_estimators}
            for k in ks for h in list(folds) + [None]]
    print(f"{len(jobs)} jobs ({len(ks)} feature sets x {len(folds)+1} runs) "
          f"across up to 20 {GPU} containers")

    t0 = time.time()
    results, gpu_seconds = [], 0.0
    for r in score_one.map(jobs):
        results.append(r)
        gpu_seconds += r["seconds"]
        print(f"  done k={r['n_components']} held={r['held']} "
              f"({len(results)}/{len(jobs)})", flush=True)

    print(f"\nall jobs finished in {time.time()-t0:.0f}s wall / "
          f"{gpu_seconds/3600:.2f} GPU-h (~${gpu_seconds/3600*GPU_HOURLY_USD:.2f})")

    summary = []
    for k in ks:
        cv = [r for r in results if r["n_components"] == k and r["held"] is not None]
        fin = next(r for r in results if r["n_components"] == k and r["held"] is None)

        # Threshold from the train folds only, exactly as the serial script did.
        thresholds, cv_rows = [], []
        for r in sorted(cv, key=lambda r: str(r["held"])):
            y, s = np.array(r["y"]), np.array(r["score"])
            thr = evaluate.choose_threshold(y, s)
            thresholds.append(thr)
            cv_rows.append(evaluate.summarise(y, s, np.array(r["apo_id"]), thr,
                                              label=f"cv/{r['held']}"))
        threshold = float(np.median(thresholds))

        y, s, g = (np.array(fin["y"]), np.array(fin["score"]),
                   np.array(fin["apo_id"]))
        res = evaluate.summarise(y, s, g, threshold,
                                 label=f"TabICLv2 zero-shot (esm{k})")
        strat = evaluate.stratified(
            pd.DataFrame({C.LABEL_COL: y,
                          "in_noncryptic_pocket": fin["in_noncryptic_pocket"]}),
            s, threshold)

        out = {"result": res, "cv": cv_rows, "threshold": threshold,
               "n_features": fin["n_features"],
               "stratified": strat.to_dict(orient="records"),
               "per_protein": evaluate.per_protein(y, s, g, threshold)
               .to_dict(orient="records"),
               "scores": {"apo_id": fin["apo_id"], "chain": fin["chain"],
                          "seq_id": fin["seq_id"], "y": fin["y"],
                          "score": fin["score"]}}
        dest = f"build/modal_result_esm{k}.json"
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=2)
        summary.append((k, fin["n_features"], res))
        print(f"\nwrote {dest}")

    print(f"\n{'k':>4} {'feat':>5} {'AUC':>6} {'AUPRC':>7} {'TPR':>6} "
          f"{'FPR':>6} {'MCC':>6}")
    print("-" * 44)
    for k, nf, r in summary:
        print(f"{k:>4} {nf:>5} {r['AUC']:>6.3f} {r['AUPRC']:>7.3f} "
              f"{r['TPR']:>6.3f} {r['FPR']:>6.3f} {r['MCC']:>6.3f}")
