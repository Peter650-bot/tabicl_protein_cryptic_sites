"""Fine-tuned TabICLv2 on CryptoBench -- a ceiling measurement beside zero-shot.

WHAT CHANGES AND WHAT DOES NOT
    Everything in the zero-shot protocol is kept: the 117-feature table, the
    50k context drawn by `sampling.build_context` with the same seed, the four
    CryptoBench train folds used only to pick a threshold at mean per-protein
    FPR 0.05, and the untouched test fold. The only difference is the weights
    that context is pushed through: `tabicl[finetune]`'s
    `FinetunedTabICLClassifier` first adapts the pretrained checkpoint to this
    table, and the resulting `best.ckpt` is loaded back into the same
    `CrypticSitePredictor` via `model_path`.

    Each job ALSO scores the same context with the pretrained checkpoint, in
    the same container and library version. That paired control is what the
    fine-tuned number should be read against: tabicl is 2.2.0 here (the
    zero-shot headline ran 2.1.1), so a delta against the old JSON alone would
    mix fine-tuning with a library bump.

NO LEAKAGE INTO THE WEIGHTS
    One fine-tune per job, and each job only ever sees its own training pool:
      CV job (held = train fold h)  pool = the other three train folds
      final job (held = None)       pool = all four train folds
    Inside the pool, 8% of PROTEINS are carved off as the early-stopping set.
    The library's default is a random 10% of ROWS, which would put residues of
    the same protein on both sides and reward memorising proteins. The weights
    of the job that scores fold h therefore never saw fold h, and the weights
    that score the test set never saw the test set.

    The fine-tuning data is the pool's own 50k `build_context` draw (all
    positives, proportional negatives), so fine-tuning sees the same class
    balance that inference will.

USAGE
    # memory/time pilot: one epoch on a real-size context, small val set
    modal run deploy/finetune_modal.py --smoke

    # the run: deploy once, spawn five jobs, poll, collect (see
    # memory: client streams die on this host, so no long-lived .map())
    modal deploy deploy/finetune_modal.py
    modal run deploy/finetune_modal.py --spawn --run-id ft_YYYYMMDD
    modal volume ls cryptobench-finetune-results ft_YYYYMMDD
    modal run deploy/finetune_modal.py --collect --run-id ft_YYYYMMDD

    `--collect` writes everything to fine_tuned_tabicl/. No existing result
    file is touched.
"""
from __future__ import annotations

import os
import sys

import modal

TABLE_LOCAL = os.path.expanduser("~/cryptobench_stage/tables/residues_esm64.parquet")
if modal.is_local() and not os.path.exists(TABLE_LOCAL):
    raise SystemExit(
        f"missing {TABLE_LOCAL}\n\nStage it first:\n"
        "    cp build/residues_esm64.parquet ~/cryptobench_stage/tables/"
    )

# A 10k-row training chunk x 117 features does not fit in 22 GB (L4 OOMs in
# the column embedder's backward-carrying forward), so 80 GB it is.
GPU = "A100-80GB"
GPU_HOURLY_USD = 2.50
CKPT = "tabicl-classifier-v2-20260212.ckpt"
TABICL_VERSION = "2.2.0"
OUT_DIR = "fine_tuned_tabicl"

# Fine-tuning hyperparameters: the library defaults, except where noted.
FT_PARAMS = dict(
    epochs=30,
    learning_rate=1e-5,
    n_estimators_finetune=2,
    n_estimators_validation=2,
    n_estimators_inference=8,
    max_data_size=10_000,
    finetune_ctx_query_ratio=0.2,
    early_stopping=True,
    patience=8,
    eval_metric="roc_auc",
    random_state=0,          # project seed (library default 42)
    time_limit=30 * 60,      # cost cap per job (A100-80GB)
)
# ~70 proteins / ~23k rows. Validation runs after every epoch, so this is the
# main per-epoch cost; 8% keeps it comparable to the training steps.
VAL_PROTEIN_SHARE = 0.08

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(f"tabicl[finetune]=={TABICL_VERSION}", "scikit-learn", "pandas",
                 "pyarrow", "numpy<2")
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"d(repo_id='jingang/TabICL', filename='{CKPT}')\""
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(TABLE_LOCAL, "/data/table.parquet")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
)

app = modal.App("cryptobench-tabicl-finetune")
results_vol = modal.Volume.from_name("cryptobench-finetune-results",
                                     create_if_missing=True)


def _split_val_proteins(pool, share: float, seed: int):
    """Protein-disjoint early-stopping split of a training pool."""
    import numpy as np
    from cryptic_tabicl import config as C

    prots = np.array(sorted(pool[C.GROUP_COL].unique()))
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(prots) * share)))
    val_prots = set(rng.choice(prots, size=n_val, replace=False))
    is_val = pool[C.GROUP_COL].isin(val_prots)
    return pool[~is_val], pool[is_val]


@app.function(image=image, gpu=GPU, timeout=4 * 3600, retries=0,
              volumes={"/results": results_vol})
def finetune_one(job: dict) -> dict:
    """Fine-tune on one pool, then score its target with both checkpoints."""
    sys.path.insert(0, "/app")
    import json
    import logging
    import time

    import numpy as np
    import pandas as pd
    import torch
    from tabicl import FinetunedTabICLClassifier
    from cryptic_tabicl import config as C, model, sampling

    held, smoke = job["held"], job.get("smoke", False)
    tag = ("smoke_" if smoke else "") + (held or "final")
    run_dir = f"/results/{job['run_id']}"
    cached = f"{run_dir}/{tag}.json"
    results_vol.reload()
    if os.path.exists(cached) and not smoke:
        print(f"{tag}: cached, skipping", flush=True)
        with open(cached) as fh:
            return json.load(fh)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    table = pd.read_parquet("/data/table.parquet")
    n_feat = len([c for c in table.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])
    train = table[table["split"] == "train"]
    if held is None:
        pool, target = train, table[table["split"] == "test"]
    else:
        pool, target = train[train["fold"] != held], train[train["fold"] == held]

    # ---- 1. fine-tune -------------------------------------------------------
    ft_src, val = _split_val_proteins(pool, VAL_PROTEIN_SHARE, C.RANDOM_SEED)
    ft_ctx, ft_plan = sampling.build_context(ft_src, budget=job["budget"])
    params = dict(FT_PARAMS)
    if smoke:
        keep = list(dict.fromkeys(val[C.GROUP_COL]))[:20]
        val = val[val[C.GROUP_COL].isin(keep)]
        params.update(epochs=1, early_stopping=False)
        target = target[target[C.GROUP_COL].isin(
            list(dict.fromkeys(target[C.GROUP_COL]))[:10])]
    print(f"[{tag}] features={n_feat}  pool {pool[C.GROUP_COL].nunique()} prot"
          f"  -> fine-tune {ft_src[C.GROUP_COL].nunique()} / early-stop "
          f"{val[C.GROUP_COL].nunique()} prot ({len(val):,} rows)\n{ft_plan}",
          flush=True)

    history, baseline = [], {}

    # The pretrained model's early-stopping score is only emitted as a log
    # line ("Baseline val roc_auc: 0.xxxx"), so capture it from there.
    class _BaselineGrab(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if msg.startswith("Baseline val"):
                baseline["val_auc"] = float(msg.rsplit(":", 1)[1])

    logging.getLogger("tabicl._finetune.base").addHandler(_BaselineGrab())

    class _Recorder:
        def setup(self, config):
            pass

        def log_step(self, metrics, step):
            pass

        def log_epoch(self, metrics, step):
            history.append({"step": step, **{k: float(v) for k, v in metrics.items()}})
            print(f"[{tag}] {history[-1]}", flush=True)

        def finish(self):
            pass

    class _FT(FinetunedTabICLClassifier):
        def _make_experiment_logger(self):
            return _Recorder()

    ckpt_dir = f"{run_dir}/ckpt_{tag}"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ft = _FT(verbose=True, **params)
    ft.fit(model.feature_matrix(ft_ctx), ft_ctx[C.LABEL_COL].to_numpy(),
           X_val=model.feature_matrix(val), y_val=val[C.LABEL_COL].to_numpy(),
           output_dir=ckpt_dir)
    ft_seconds = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{tag}] fine-tune done in {ft_seconds:.0f}s, peak {peak_gb:.1f} GB, "
          f"best val {ft._best_metric_:.4f}", flush=True)
    ft_best = ft._best_metric_
    best_ckpt = f"{ckpt_dir}/best.ckpt"
    del ft
    torch.cuda.empty_cache()
    results_vol.commit()

    # ---- 2. score the target with the zero-shot protocol, both checkpoints --
    ctx, plan = sampling.build_context(pool, budget=job["budget"])
    print(f"[{tag}] scoring context\n{plan}", flush=True)
    scores = {}
    for name, path in (("pretrained", None), ("finetuned", best_ckpt)):
        t1 = time.time()
        clf = model.CrypticSitePredictor(n_estimators=job["n_estimators"],
                                         model_path=path).fit(ctx)
        scores[name] = [float(s) for s in clf.predict_by_protein(target)]
        print(f"[{tag}] {name}: scored {len(target):,} rows in "
              f"{time.time()-t1:.0f}s", flush=True)
        del clf
        torch.cuda.empty_cache()

    out = {
        "held": held, "n_features": n_feat, "tabicl": TABICL_VERSION,
        "ft_params": params, "ft_seconds": ft_seconds,
        "seconds": time.time() - t0, "peak_gpu_gb": peak_gb,
        "baseline_val_auc": baseline.get("val_auc", float("nan")),
        "best_val_auc": float(ft_best),
        "history": history,
        "n_ft_proteins": int(ft_src[C.GROUP_COL].nunique()),
        "n_val_proteins": int(val[C.GROUP_COL].nunique()),
        "apo_id": target[C.GROUP_COL].tolist(),
        "chain": target["chain"].tolist(),
        "seq_id": target["seq_id"].tolist(),
        "icode": target["icode"].astype(str).tolist(),
        "y": target[C.LABEL_COL].astype(int).tolist(),
        "in_noncryptic_pocket": target["in_noncryptic_pocket"].astype(int).tolist(),
        "score_pretrained": scores["pretrained"],
        "score_finetuned": scores["finetuned"],
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(cached, "w") as fh:
        json.dump(out, fh)
    results_vol.commit()
    return out


@app.function(image=image, timeout=1800, volumes={"/results": results_vol})
def fetch_results(run_id: str) -> list:
    import glob
    import json

    results_vol.reload()
    files = sorted(f for f in glob.glob(f"/results/{run_id}/*.json")
                   if "/smoke_" not in f)
    return [json.load(open(f)) for f in files]


def _folds():
    import pandas as pd
    tb = pd.read_parquet(TABLE_LOCAL, columns=["split", "fold"])
    return sorted(tb.loc[tb["split"] == "train", "fold"].dropna().unique())


def _single_chain_pdbs(C) -> set:
    """CB-P2RANK-apo, reconstructed exactly as scripts/f1_by_subset.py does."""
    import json
    test = json.loads((C.FOLDS_DIR / "test.json").read_text())
    return {pdb.lower() for pdb, entries in test.items()
            if len({r.split("_")[0] for e in entries
                    for r in e["apo_pocket_selection"]}) == 1}


def _collect(results: list, run_id: str) -> None:
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    sys.path.insert(0, ".")
    from cryptic_tabicl import config as C, evaluate

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)
    cv = sorted((r for r in results if r["held"] is not None),
                key=lambda r: r["held"])
    fin = next(r for r in results if r["held"] is None)
    single = _single_chain_pdbs(C)

    summary, blobs = [], {}
    for name, label in (("finetuned", "TabICLv2 fine-tuned (esm64)"),
                        ("pretrained", f"TabICLv2 zero-shot, tabicl {TABICL_VERSION} (esm64)")):
        key = f"score_{name}"
        thresholds, cv_rows = [], []
        for r in cv:
            y, s, g = np.array(r["y"]), np.array(r[key]), np.array(r["apo_id"])
            thr = evaluate.choose_threshold(y, s, g)
            thresholds.append(thr)
            cv_rows.append(evaluate.summarise(y, s, g, thr, label=f"cv/{r['held']}"))
        threshold = float(np.median(thresholds))

        y, s, g = np.array(fin["y"]), np.array(fin[key]), np.array(fin["apo_id"])
        res = evaluate.summarise(y, s, g, threshold, label=label)
        pdb = pd.Series(g).str[:4].str.lower()
        m = pdb.isin(single).to_numpy()
        res_p2 = evaluate.summarise(y[m], s[m], g[m], threshold, label=label)
        strat = evaluate.stratified(
            pd.DataFrame({C.LABEL_COL: y,
                          "in_noncryptic_pocket": fin["in_noncryptic_pocket"]}),
            s, threshold)
        blobs[name] = {
            "result": res, "result_cb_p2rank_apo": res_p2, "cv": cv_rows,
            "threshold": threshold, "cv_thresholds": thresholds,
            "n_features": fin["n_features"], "tabicl": TABICL_VERSION,
            "stratified": strat.to_dict(orient="records"),
            "per_protein": evaluate.per_protein(y, s, g, threshold)
            .to_dict(orient="records"),
            "scores": {"apo_id": fin["apo_id"], "chain": fin["chain"],
                       "seq_id": fin["seq_id"], "icode": fin["icode"],
                       "y": fin["y"], "score": fin[key]},
        }
        for subset, r in (("CB-full", res), ("CB-P2RANK-apo", res_p2)):
            summary.append({"model": name, "subset": subset,
                            **{k: r[k] for k in ("n_proteins", "AUC", "AUPRC",
                                                 "TPR", "FPR", "MCC", "F1",
                                                 "F1_weighted")}})

    (out_dir / "result_finetuned_esm64.json").write_text(
        json.dumps(blobs["finetuned"], indent=2))
    (out_dir / "result_pretrained_control_esm64.json").write_text(
        json.dumps(blobs["pretrained"], indent=2))
    training = [{k: r[k] for k in ("held", "ft_seconds", "seconds", "peak_gpu_gb",
                                   "baseline_val_auc", "best_val_auc",
                                   "n_ft_proteins",
                                   "n_val_proteins", "ft_params", "history")}
                for r in cv + [fin]]
    (out_dir / "training.json").write_text(json.dumps(
        {"run_id": run_id, "gpu": GPU, "jobs": training}, indent=2))
    table = pd.DataFrame(summary)
    table.to_csv(out_dir / "summary.csv", index=False)

    gpu_h = sum(r["seconds"] for r in results) / 3600
    pd.set_option("display.width", 160)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nthresholds: finetuned {blobs['finetuned']['threshold']:.4f}  "
          f"pretrained {blobs['pretrained']['threshold']:.4f}")
    print(f"{gpu_h:.2f} GPU-h (~${gpu_h * GPU_HOURLY_USD:.2f})")
    print(f"wrote {out_dir}/")


@app.local_entrypoint()
def main(smoke: bool = False, spawn: bool = False, collect: bool = False,
         run_id: str = "", held: str = "all", budget: int = 50_000,
         n_estimators: int = 8):
    import json
    import time

    run_id = run_id or time.strftime("ft_%Y%m%d_%H%M%S")
    base = {"budget": budget, "n_estimators": n_estimators, "run_id": run_id}
    if smoke:
        out = finetune_one.remote({**base, "held": None, "smoke": True})
        print(json.dumps({k: out[k] for k in ("ft_seconds", "seconds",
                                              "peak_gpu_gb", "history")}, indent=2))
        return
    folds = _folds()
    if spawn:
        f = modal.Function.from_name("cryptobench-tabicl-finetune", "finetune_one")
        # --held final | <fold name> | all -- lets one job be timed before
        # paying for the other four.
        todo = {"all": list(folds) + [None], "final": [None]}.get(held, [held])
        for h in todo:
            call = f.spawn({**base, "held": h})
            print(f"spawned held={h}: {call.object_id}")
        print(f"\nrun_id={run_id}\npoll:    modal volume ls "
              f"cryptobench-finetune-results {run_id}\ncollect: modal run "
              f"deploy/finetune_modal.py --collect --run-id {run_id}")
        return
    if collect:
        results = fetch_results.remote(run_id)
        have = {r["held"] for r in results}
        missing = (set(folds) | {None}) - have
        if missing:
            raise SystemExit(f"missing jobs: {sorted(missing, key=str)}")
        _collect(results, run_id)
        return
    raise SystemExit("pass one of --smoke, --spawn, --collect")
