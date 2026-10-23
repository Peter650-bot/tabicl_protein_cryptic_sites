"""PAM on every test subset for which a paper publishes numbers, under that paper's protocol.

EVALUATION UNIT (verified in the authors' code)
    CryptoBench's pLM-NN numbers come from skrhakv/apolo@cryptobench-v2 (ccc7b60):
    - create-annotations.py writes one row per apo structure: the union of every cryptic
      pocket's residues, under the chain string with the most chains ("A-B");
    - helper.py splits that string on "-" and makes one entry per chain, skipping chains
      whose annotation is empty;
    - stats.py averages over those chains and always drops 7ndl.
    Our test table has 231 records, because 7 multi-chain structures are split into several
    overlapping records with different pocket sets. cryptobench_units() rebuilds their unit:
    the most-chains record's rows and scores, labels OR-ed over all records of the
    structure, one unit per (structure, chain), positive chains only, 7ndl dropped.

SUBSETS
    CB-full          all chain units                      (CryptoBench Table 5, pLM-NN)
    CB-PM            single-chain structures minus the 21 PocketMiner errors logged in
                     H-prediction-evaluation/roc-curve.ipynb, minus 7ndl: 162 (Table 5)
    CB-PM-tool       the same plus 7ndl: 163, the set PocketMiner itself was scored on
                     (pocketminer_stats.py does not drop 7ndl; Figure 4 pooled curves)
    CB-P2RANK-apo    the 184 single-chain structures minus 7ndl: 183 (Table 5, pLM-NN)
    CB-P2RANK-184    all 184 (P2Rank in Table 5; Hidden in protein sequences, pooled)
    CB-records       our 231 records, for reference only

MODELS, each at its own CV-frozen threshold
    PAM          ft117 (scripts/final_comparison.py)
    TabICL-117   zero-shot TabICLv2, 53 features + 64 frozen ESM-2 PCs (RESULTS.md § 1)
    TabICL-53    zero-shot TabICLv2, 53 features, no pLM (RESULTS.md § 1)

Metric estimators are CryptoBench's: AUPRC = metrics.auc(recall, precision), F1 weighted.
Per-protein 95 % CIs resample whole structures (10,000 draws, seed 0), since chains of one
structure are not independent; each draw re-averages the stored per-unit table.

Run: python scripts/published_comparison.py   ->  finetuned_esm/published_comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_comparison as F  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

NB = Path("external/CryptoBench/src/H-prediction-evaluation/roc-curve.ipynb")
DROPPED = "7ndl"   # apolo stats.py: "some bug in the uniprot mapping"
ZERO_SHOT = {"TabICL-117": "modal_result_esm64.json", "TabICL-53": "modal_result_esm0.json"}
MODELS = ("PAM", *ZERO_SHOT)
N_BOOT = 10_000


def pocketminer_skipped() -> set[str]:
    """Records PocketMiner's loader skipped, as printed in CryptoBench's own notebook run."""
    cells = json.loads(NB.read_text())["cells"]
    text = next("".join(o["text"]) for c in cells for o in c.get("outputs", [])
                if o.get("name") == "stdout")
    return {line.strip().lower() for line in text.splitlines() if line.strip()}


def cryptobench_units(df: pd.DataFrame) -> pd.DataFrame:
    """One row per residue of each structure's most-chains record, labels OR-ed over its records."""
    df = df.assign(pdb=df["apo_id"].str[:4].str.lower())
    rank = df.groupby("apo_id").agg(pdb=("pdb", "first"), n=("y", "size"))
    rank["dashes"] = rank.index.str.count("-")
    keep = rank.sort_values(["dashes", "n"]).groupby("pdb").tail(1).index
    out = df[df["apo_id"].isin(keep)].copy()
    multi = rank.groupby("pdb").size().loc[lambda s: s > 1].index
    m = out["pdb"].isin(multi)
    assert not out[m].duplicated(["pdb", "chain", "seq_id"]).any()
    union = df[df["pdb"].isin(multi)].groupby(["pdb", "chain", "seq_id"])["y"].max()
    out.loc[m, "y"] = union.loc[list(zip(out.loc[m, "pdb"], out.loc[m, "chain"],
                                          out.loc[m, "seq_id"]))].to_numpy()
    out["unit"] = out["pdb"] + out["chain"].astype(str)
    return out


def subsets(u: pd.DataFrame) -> dict[str, np.ndarray]:
    single = u["pdb"].isin(single_chain_pdbs()).to_numpy()
    failed = {s[:4] for s in pocketminer_skipped() if "-" not in s}
    pm = single & ~u["pdb"].isin(failed).to_numpy()
    positive = u.groupby("unit")["y"].transform("sum").gt(0).to_numpy()
    kept = positive & (u["pdb"] != DROPPED).to_numpy()
    return {"CB-full": kept, "CB-PM": pm & kept, "CB-PM-tool": pm,
            "CB-P2RANK-apo": single & kept, "CB-P2RANK-184": single}


def score(y, s, p, g, cluster, rng) -> dict:
    pp = F.per_protein_table(y, s, p, g)
    unit_cluster = pd.Series(cluster, index=g).groupby(level=0).first().loc[pp.index]
    sums = pp.groupby(unit_cluster.to_numpy()).sum()
    counts = unit_cluster.value_counts().loc[sums.index].to_numpy()
    draws = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    w = np.apply_along_axis(np.bincount, 1, draws, minlength=len(sums))
    boot = (w @ sums.to_numpy()) / (w @ counts)[:, None]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"per_protein": pp.mean(skipna=True).to_dict(),
            "per_protein_ci": {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(sums.columns)},
            "pooled": F.pooled_all(y, s, p)}


def main() -> None:
    df = F.load(single_chain_only=False)
    pred, desc = F.calls(df)
    df["PAM"], df["PAM_call"] = df["ft117"], pred["ft117"]
    ops = {"PAM": desc["ft117"]}
    for name, fname in ZERO_SHOT.items():
        blob = json.loads((F.C.BUILD_DIR / fname).read_text())
        for c in ("apo_id", "chain", "seq_id"):
            assert (np.asarray(blob["scores"][c]) == df[c].to_numpy()).all(), f"{name}: {c}"
        df[name] = np.asarray(blob["scores"]["score"], dtype=float)
        df[f"{name}_call"] = (df[name] >= blob["threshold"]).astype(int)
        ops[name] = f"score >= {blob['threshold']:.4f} (CV-frozen)"
    u = cryptobench_units(df)
    out = {"operating_points": ops, "n_boot": N_BOOT}
    frames = {name: u[mask] for name, mask in subsets(u).items()}
    frames["CB-records"] = df.assign(unit=df["apo_id"], pdb=df["apo_id"].str[:4].str.lower())
    for name, sub in frames.items():
        y, g = sub["y"].to_numpy(), sub["unit"].to_numpy()
        out[name] = {"n_units": int(sub["unit"].nunique()), "n_structures": int(sub["pdb"].nunique()),
                     "n_residues": int(len(y)), "positives": int(y.sum())}
        print(f"\n== {name}: {out[name]['n_units']} units ({out[name]['n_structures']} structures), "
              f"{len(y):,} residues, {int(y.sum()):,} positives")
        print(f"  {'':22s}" + "".join(f"{k:>13s}" for k in F.METRICS))
        for m in MODELS:
            r = out[name][m] = score(y, sub[m].to_numpy(), sub[f"{m}_call"].to_numpy(), g,
                                     sub["pdb"].to_numpy(), np.random.default_rng(F.SEED))
            for kind in ("per_protein", "pooled"):
                print(f"  {m + ' ' + kind:22s}" + "".join(f"{r[kind][k]:>13.4f}" for k in F.METRICS))
    assert out["CB-P2RANK-184"]["n_units"] == 184 and out["CB-PM"]["n_units"] == 162
    assert out["CB-full"]["n_units"] == 262
    Path("finetuned_esm/published_comparison.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
