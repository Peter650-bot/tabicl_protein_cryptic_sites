"""Step 9f context (frozen_step9.json, "reported_not_claimed"): every arm vs the released model.

The released model trained on LIGYSIS labels for 25 test proteins (PLAN.md 9b), so it
is compared only on the other 206. Paired bootstrap over proteins, per-protein and
pooled AUPRC/AUC. Nothing here is a claim.

Run: python scripts/s2p_released_context.py --run-id st2
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
from s2p_evaluate import pooled  # noqa: E402

N_BOOT, N_BOOT_POOLED, SEED = 10_000, 2_000, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    leaked = set(json.loads(Path("finetuned_esm/step9b_ligysis_overlap.json").read_text())
                 ["test"]["near_identical"])

    blob = json.loads((C.BUILD_DIR / "modal_result_esm64.json").read_text())["scores"]
    df = pd.DataFrame({k: blob[k] for k in ("apo_id", "chain", "seq_id", "y")})
    df["tabicl_117"] = blob["score"]
    keyed = pd.read_parquet(C.BUILD_DIR / "s2p_scores_keyed.parquet")
    released = keyed[keyed["split"] == "test"].reset_index(drop=True)
    assert (released["apo_id"].to_numpy() == df["apo_id"].to_numpy()).all()
    df["released"] = released["s2p_score"].to_numpy()
    tab = pd.read_parquet(C.BUILD_DIR / "residues_s2pfull.parquet",
                          columns=["apo_id", "split", "s2p_oof", "s2pfull_oof"])
    tab = tab[tab["split"] == "test"].reset_index(drop=True)
    assert (tab["apo_id"].to_numpy() == df["apo_id"].to_numpy()).all()
    df["plain_alone"], df["full_alone"] = tab["s2p_oof"].to_numpy(), tab["s2pfull_oof"].to_numpy()
    for name, path in (("s2p_117", "build/s2p_stack/st1/s2p_117_final.json"),
                       ("s2pfull_117", f"build/s2p_stack/{a.run_id}/s2pfull_117_final.json")):
        r = json.loads(Path(path).read_text())
        assert list(r["apo_id"]) == df["apo_id"].tolist()
        df[name] = r["score"]

    sub = df[~df["apo_id"].isin(leaked)].reset_index(drop=True)
    arms = ["s2pfull_117", "s2p_117", "full_alone", "plain_alone", "tabicl_117", "released"]
    y, g = sub["y"].to_numpy(), sub["apo_id"].to_numpy()
    pp = {k: evaluate.per_protein(y, sub[k].to_numpy(), g, 0.5).set_index("apo_id")[["AUC", "AUPRC"]]
          for k in arms}
    n = len(pp["released"])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = {k: np.nanmean(v.to_numpy(float)[idx], axis=1) for k, v in pp.items()}
    rows_of = {p: np.flatnonzero(g == p) for p in pp["released"].index}
    plist = np.array(list(rows_of))
    pb = {k: np.empty((N_BOOT_POOLED, 2)) for k in arms}
    for b in range(N_BOOT_POOLED):
        r = np.concatenate([rows_of[p] for p in rng.choice(plist, n)])
        for k in arms:
            pb[k][b] = pooled(y[r], sub[k].to_numpy()[r])
    out = {"n_proteins": n, "excluded_leaked": sorted(leaked)}
    print(f"206 non-leaked test proteins (n={n}); difference vs released, paired 95 % CI")
    for k in arms:
        pt = np.nanmean(pp[k].to_numpy(float), axis=0)
        pl = pooled(y, sub[k].to_numpy())
        row = {"pp_AUC": float(pt[0]), "pp_AUPRC": float(pt[1]),
               "pooled_AUC": float(pl[0]), "pooled_AUPRC": float(pl[1])}
        if k != "released":
            for j, m in enumerate(("pp_AUC", "pp_AUPRC")):
                d = boot[k][:, j] - boot["released"][:, j]
                row[f"{m}_minus_released"] = [float(pt[j] - np.nanmean(pp['released'].to_numpy(float), axis=0)[j]),
                                              *map(float, np.percentile(d, [2.5, 97.5]))]
            d = pb[k][:, 1] - pb["released"][:, 1]
            row["pooled_AUPRC_minus_released"] = [float(pl[1] - pooled(y, sub["released"].to_numpy())[1]),
                                                  *map(float, np.percentile(d, [2.5, 97.5]))]
        out[k] = row
        extra = "" if k == "released" else (
            f"  pp AUPRC diff {row['pp_AUPRC_minus_released'][0]:+.3f} "
            f"[{row['pp_AUPRC_minus_released'][1]:+.3f},{row['pp_AUPRC_minus_released'][2]:+.3f}]"
            f"  pooled AUPRC diff {row['pooled_AUPRC_minus_released'][0]:+.3f} "
            f"[{row['pooled_AUPRC_minus_released'][1]:+.3f},{row['pooled_AUPRC_minus_released'][2]:+.3f}]")
        print(f"  {k:12s} pp AUC {row['pp_AUC']:.3f} AUPRC {row['pp_AUPRC']:.3f} | pooled AUC "
              f"{row['pooled_AUC']:.3f} AUPRC {row['pooled_AUPRC']:.3f}{extra}")
    Path("finetuned_esm/step9_released_context.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
