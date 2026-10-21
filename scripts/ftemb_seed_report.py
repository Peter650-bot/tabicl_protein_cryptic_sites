"""Context-draw variance of the best model, ft117 (RESULTS.md 4.6, redone for step 10; PLAN.md step 12).

WHAT VARIES
    Only `sampling.build_context`'s seed: which non-cryptic residues fill each of the
    four fold-k contexts, and their row order. Fine-tuned models, embeddings, PCA,
    features, the protein sub-fold split and TabICL's own random_state are fixed.

PROTOCOL PER SEED (the headline protocol, unchanged)
    1. per fold model k: out-of-fold scores of fold k's four protein sub-folds give a
       threshold at mean per-protein FPR 0.05; the seed's threshold is the median of four.
    2. test score = mean of the four cross-fitted predictions (k = 0..3).
    Metrics are also reported at seed 0's frozen threshold, for the reader who keeps
    the published threshold and only asks whether the ranking moves.

    Seed 0 is run `fe1`; it must reproduce step10_result.json exactly.

Run: python scripts/ftemb_seed_report.py --run-id fe1seeds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryptic_tabicl import config as C, evaluate  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

ARM, KS, N_SEEDS = "ft117", range(4), 12
METRICS = ("AUC", "AUPRC", "TPR", "FPR", "MCC", "F1", "F1_weighted", "ACC",
           "AUC_pooled", "AUPRC_pooled")
ROWS = ("apo_id", "chain", "seq_id", "y")


def load(path: Path, ref: dict | None = None) -> dict:
    r = json.loads(path.read_text())
    if ref is not None:
        for c in ROWS:
            assert list(r[c]) == list(ref[c]), f"{path.name}: rows differ on {c}"
    return r


def files(seed: int, run_id: str) -> dict:
    if seed == 0:
        d, sfx = C.BUILD_DIR / "ftemb_stack" / "fe1", ""
    else:
        d, sfx = C.BUILD_DIR / "ftemb_stack" / run_id, f"_s{seed}"
    return {(k, q): d / f"{ARM}_k{k}_{q}{sfx}.json"
            for k in KS for q in [f"sub{i}" for i in range(4)] + ["test"]}


def one_seed(seed: int, run_id: str, test_ref: dict | None):
    f = files(seed, run_id)
    per_k = []
    for k in KS:
        rs = [load(f[(k, f"sub{s}")]) for s in range(4)]
        y = np.concatenate([r["y"] for r in rs])
        s_ = np.concatenate([r["score"] for r in rs])
        g = np.concatenate([r["apo_id"] for r in rs])
        per_k.append(float(evaluate.choose_threshold(y, s_, g)))
    tests = [load(f[(k, "test")], test_ref) for k in KS]
    test_ref = test_ref or tests[0]
    for t in tests[1:]:
        for c in ROWS:
            assert list(t[c]) == list(test_ref[c]), "test rows differ across k"
    score = np.mean([np.asarray(t["score"]) for t in tests], axis=0)
    ctx = [t["n_context"] for t in tests]
    return per_k, float(np.median(per_k)), score, test_ref, ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="fe1seeds")
    a = ap.parse_args()

    fixed = json.loads(Path("finetuned_esm/step10_thresholds.json").read_text())[ARM]["frozen_threshold"]
    single = single_chain_pdbs()
    rows, ref = [], None
    for seed in range(N_SEEDS):
        per_k, thr, score, ref, ctx = one_seed(seed, a.run_id, ref)
        df = pd.DataFrame({c: ref[c] for c in ROWS}).assign(score=score)
        for subset, sub in (("CB-full", df),
                            ("CB-P2RANK-apo", df[df["apo_id"].str[:4].str.lower().isin(single)])):
            y, g, s_ = sub["y"].to_numpy(), sub["apo_id"].to_numpy(), sub["score"].to_numpy()
            for rule, t in (("own", thr), ("fixed", fixed)):
                m = evaluate.summarise(y, s_, g, t, f"{ARM}/s{seed}")
                rows.append({"seed": seed, "subset": subset, "threshold_rule": rule,
                             "threshold": t, "k_thresholds": per_k, "n_context": ctx,
                             **{x: m[x] for x in METRICS}})

    tab = pd.DataFrame(rows)
    # Seed 0 must be the published result, to the last digit.
    pub = json.loads(Path("finetuned_esm/step10_result.json").read_text())
    s0 = tab[(tab.seed == 0) & (tab.threshold_rule == "own")].set_index("subset")
    for subset in ("CB-full", "CB-P2RANK-apo"):
        for ours, theirs in (("AUC", "per_protein_AUC"), ("AUPRC", "per_protein_AUPRC"),
                             ("AUC_pooled", "pooled_AUC"), ("AUPRC_pooled", "pooled_AUPRC")):
            want = pub[subset][theirs][ARM]["value"]
            assert abs(s0.loc[subset, ours] - want) < 1e-12, (subset, ours, s0.loc[subset, ours], want)
        assert abs(s0.loc[subset, "MCC"] - pub[subset]["threshold_metrics"][ARM]["MCC"]) < 1e-12
    print("seed 0 reproduces step10_result.json exactly")

    summary = {}
    for (subset, rule), g in tab.groupby(["subset", "threshold_rule"]):
        summary[f"{subset}/{rule}"] = {
            x: {"mean": float(g[x].mean()), "sd": float(g[x].std(ddof=1)),
                "min": float(g[x].min()), "max": float(g[x].max()),
                "seed0": float(g.loc[g.seed == 0, x].iloc[0])} for x in METRICS + ("threshold",)}
    out = {"run_id": a.run_id, "seed0_run": "fe1", "n_seeds": N_SEEDS, "fixed_threshold": fixed,
           "summary": summary, "per_seed": tab.to_dict(orient="records")}
    dest = Path("finetuned_esm/step12_seed_variance.json")
    dest.write_text(json.dumps(out, indent=1))

    for key, v in summary.items():
        print(f"\n== {key}")
        print(f"{'metric':14s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'max':>8s} {'seed 0':>8s}")
        for x, st in v.items():
            print(f"{x:14s} {st['mean']:8.4f} {st['sd']:8.4f} {st['min']:8.4f} {st['max']:8.4f} {st['seed0']:8.4f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
