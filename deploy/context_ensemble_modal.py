"""Context ensembling: average TabICL over K independent context subsamples.

THE PROBLEM THIS ATTACKS
    TabICL is in-context, so its labelled table must fit in one forward pass.
    With 304k training residues and a ~50k budget, roughly seven eighths of the
    available negatives are discarded on any single run. Measured against an
    otherwise-identical comparison, that ceiling is expensive:

        logistic regression, 50k context   AUPRC 0.333
        logistic regression, full 304k     AUPRC 0.417      <- +0.084
        TabICLv2,            50k context   AUPRC 0.323

    At matched context TabICL and logistic regression are within 0.01. The gap
    is the context ceiling, not the model. So rather than accept one arbitrary
    subsample, draw K of them and average.

WHY THIS SHOULD WORK
    Every positive appears in every context (they always fit), so the K contexts
    differ only in WHICH negatives they contain. Each therefore places the
    decision boundary using a different slice of the negative distribution, and
    averaging approximates the boundary you would get from seeing all of it --
    the standard bagging argument, applied to the context rather than to the
    training set.

WHAT IS HELD FIXED
    TabICL's own `random_state` is constant across contexts. TabICL already
    ensembles internally (feature shuffling over `n_estimators`), and letting
    that vary too would confound "context ensembling helps" with "more model
    ensembling helps". Only the context seed moves.

COMBINATION RULE
    Contexts differ in negative composition, so their probability outputs need
    not be calibrated alike. Mean-of-probabilities, mean-of-logits and mean-rank
    are all computed; which one to prefer is an empirical question, not one to
    settle by assertion.

THE K CURVE
    Metrics are reported for every prefix K'=1..K using a running average. This
    is free -- the per-context scores are already computed -- and it answers
    whether the gain has saturated or more contexts would still pay.
"""
from __future__ import annotations

import json
import os

import modal

TABLE_ENS = "build/residues_ens.parquet"
if modal.is_local() and not os.path.exists(TABLE_ENS):
    raise SystemExit(
        f"missing {TABLE_ENS}\n\nBuild it first:\n"
        "    python -m cryptic_tabicl.build_table --out build/residues.parquet\n"
        "    python scripts/merge_ensemble.py"
    )

GPU = "T4"
CKPT = "tabicl-classifier-v2-20260212.ckpt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("tabicl==2.1.1", "scikit-learn", "pandas", "pyarrow", "numpy<2")
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download as d; "
        f"d(repo_id='jingang/TabICL', filename='{CKPT}')\""
    )
    .add_local_file(TABLE_ENS, "/data/residues_ens.parquet")
    .add_local_dir("cryptic_tabicl", "/app/cryptic_tabicl")
)

app = modal.App("cryptobench-context-ensemble")


@app.function(image=image, gpu=GPU, timeout=10800)
def score_one(payload: dict) -> dict:
    """Fit one context and score a target split. One container per context."""
    import sys
    import time

    sys.path.insert(0, "/app")
    import numpy as np
    import pandas as pd
    from cryptic_tabicl import config as C, model, sampling

    seed = payload["seed"]
    target = payload["target"]          # "test" or a fold name
    n_estimators = payload["n_estimators"]
    budget = payload["budget"]

    table = pd.read_parquet("/data/residues_ens.parquet")
    train = table[table["split"] == "train"]

    if target == "test":
        ctx_src, scored = train, table[table["split"] == "test"]
    else:
        ctx_src, scored = train[train["fold"] != target], train[train["fold"] == target]

    ctx, plan = sampling.build_context(ctx_src, budget=budget, seed=seed)
    print(f"[seed {seed} -> {target}] {plan}", flush=True)

    t0 = time.time()
    clf = model.CrypticSitePredictor(
        n_estimators=n_estimators, random_state=C.RANDOM_SEED  # fixed on purpose
    ).fit(ctx)
    scores = clf.predict_by_protein(scored)
    print(f"[seed {seed} -> {target}] scored {len(scored):,} in "
          f"{time.time()-t0:.0f}s", flush=True)

    return {
        "seed": seed,
        "target": target,
        "scores": [float(s) for s in scores],
        "n_context": len(ctx),
        "n_hard_negative": plan.n_hard_negative,
    }


@app.local_entrypoint()
def main(k: int = 5, n_estimators: int = 8, budget: int = 50_000,
         threshold_fold: str = "train-fold-1"):
    import sys

    sys.path.insert(0, ".")
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    from cryptic_tabicl import config as C, evaluate

    table = pd.read_parquet(TABLE_ENS)
    test = table[table["split"] == "test"]
    val = table[(table["split"] == "train") & (table["fold"] == threshold_fold)]

    jobs = (
        [{"seed": s, "target": threshold_fold, "n_estimators": n_estimators,
          "budget": budget} for s in range(k)]
        + [{"seed": s, "target": "test", "n_estimators": n_estimators,
            "budget": budget} for s in range(k)]
    )
    print(f"K={k} contexts x 2 targets = {len(jobs)} GPU jobs")

    by_target: dict[str, dict[int, np.ndarray]] = {threshold_fold: {}, "test": {}}
    for r in score_one.map(jobs):
        by_target[r["target"]][r["seed"]] = np.asarray(r["scores"])
        print(f"  done seed={r['seed']} target={r['target']}", flush=True)

    def combine(mat: np.ndarray, how: str) -> np.ndarray:
        if how == "prob":
            return mat.mean(axis=0)
        if how == "logit":
            p = np.clip(mat, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p)).mean(axis=0)
        return np.stack([rankdata(row) / len(row) for row in mat]).mean(axis=0)

    val_mat = np.stack([by_target[threshold_fold][s] for s in range(k)])
    test_mat = np.stack([by_target["test"][s] for s in range(k)])

    y_te, g_te = test[C.LABEL_COL].to_numpy(), test[C.GROUP_COL].to_numpy()
    y_va = val[C.LABEL_COL].to_numpy()

    results = {"k": k, "budget": budget, "curves": {}, "final": {}}

    print(f"\n{'rule':<7} {'K':>3} {'AUC':>7} {'AUPRC':>7} {'TPR':>6} {'FPR':>6} {'MCC':>6}")
    print("-" * 48)
    for how in ("prob", "logit", "rank"):
        curve = []
        for kk in range(1, k + 1):
            s_va = combine(val_mat[:kk], how)
            s_te = combine(test_mat[:kk], how)
            thr = evaluate.choose_threshold(y_va, s_va)   # threshold from val only
            r = evaluate.summarise(y_te, s_te, g_te, thr,
                                   label=f"TabICL x{kk} ({how})")
            curve.append(r)
            print(f"{how:<7} {kk:>3} {r['AUC']:>7.3f} {r['AUPRC']:>7.3f} "
                  f"{r['TPR']:>6.3f} {r['FPR']:>6.3f} {r['MCC']:>6.3f}")
        results["curves"][how] = curve
        results["final"][how] = curve[-1]

    best = max(results["final"].values(), key=lambda r: r["AUPRC"])
    print("\n" + evaluate.comparison_table(best))

    single = results["curves"]["prob"][0]
    print(f"\ncontext ensembling, K=1 -> K={k}:")
    for key in ("AUC", "AUPRC", "MCC"):
        print(f"  {key:<6} {single[key]:.3f} -> {best[key]:.3f}   "
              f"{best[key] - single[key]:+.3f}")

    os.makedirs("build", exist_ok=True)
    with open("build/context_ensemble.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote build/context_ensemble.json")
