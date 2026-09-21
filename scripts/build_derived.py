"""Build the derived feature families: context pooled into each residue's row.

Five families, all computed from columns that already exist plus
`build/residue_coords.parquet`. No GPU, no network, no structure parsing.

    seqwin   sequence-window means, for the SEQUENCE-derived families only
    esmnb    8 A spatial means of the leading ESM components
    nbstat   spread and extreme of the strongest families, and a wider shell
    pctx     protein-level aggregates, broadcast back onto every row
    prel     within-protein rank and z of the strongest columns

WHY THESE FIVE AND NOT OTHERS
    `seqwin` covers everything EXCEPT the `ens_*` pocket family, which is the
    one family windowing clearly damages. Measured sign-free on the train split
    (|AUC-0.5|+0.5, so an anti-correlated feature is not mistaken for a weak
    one -- the trap this very table was first read into):

        cons_prob_self  0.625 -> 0.651 (w5)      ens_mean  0.776 -> 0.742 (w5)
        cons_score      0.633 -> 0.649 (w5)                 0.776 -> 0.664 (w21)
        rsa             0.508 -> 0.550 (w11)
        esm_pc01        0.564 -> 0.616 (w5)      esm_pc00  0.680 -> 0.665 (w5)
        esm_pc07        0.528 -> 0.577 (w11)     esm_pc03  0.583 -> 0.537 (w5)

    So windowing lifts the weakly-oriented, backbone-smooth signals and blurs
    the sharp one. `ens_mean` measures whether THIS residue's pocket opens; a
    five-residue average of that is a different and worse question. The ESM
    components split both ways and are kept whole, because picking the four
    that happen to win univariately is the selection this project has twice
    recorded being misled by (RESULTS.md section 4.3, section 4.3b).

    `esmnb` exists because the 64 ESM components are the one family with no
    spatial pooling, while every neighbour-pooled column in the project
    outranks its own parent (`cons_prob_self_nb8` 0.673 vs 0.627).

    `prel` is here on a LOW prior and is labelled as such: within-protein
    normalisation does not add univariate signal, and for the strongest
    features it subtracts a little (`ens_mean` 0.776 raw, 0.767 rank, 0.750 z).
    It is screened anyway because it is free, and because reading univariate
    AUCs and concluding "does not help here" is the mistake RESULTS.md
    section 4.3 and section 4.3b record this project making twice.

MISSING IS NaN, NOT ZERO
    Inherited from `merge_ensemble.py:8-15` and enforced by `build_table.validate`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C  # noqa: E402
from cryptic_tabicl import derive  # noqa: E402

KEYS = ["apo_id", "chain", "seq_id", "icode"]

WINDOW = 5
ESM_WINDOWED = [f"esm_pc{i:02d}" for i in range(8)]
GEOM_WINDOWED = ["rsa", "protrusion"]
ESM_POOLED = [f"esm_pc{i:02d}" for i in range(16)]
CONS = ["cons_score", "cons_entropy", "cons_prob_self"]
NBSTAT_ENS = ["ens_mean", "ens_max", "ens_std", "ens_opening"]
PREL = ["ens_mean", "ens_max", "ens_q90", "cons_score_nb8", "rsa", "protrusion"]
WIDE_RADIUS = 12.0
EXPOSED_RSA = 0.25


def build(table: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    for d in (table, coords):
        d["icode"] = d["icode"].fillna("").astype(str)
    if coords.duplicated(KEYS).any():
        raise SystemExit("coords table has duplicate keys")

    m = table.merge(coords, on=KEYS, how="left", validate="1:1")
    assert len(m) == len(table), "coord merge changed the row count"
    if m["cx"].isna().any():
        raise SystemExit(f"{int(m['cx'].isna().sum())} rows have no centroid")

    xyz = m[["cx", "cy", "cz"]].to_numpy(float)
    grp = m[C.GROUP_COL].to_numpy()
    out = m[KEYS].copy()

    # -- seqwin: ordered along the backbone, never across a chain boundary ----
    chain_key = (m[C.GROUP_COL].astype(str) + "|" + m["chain"].astype(str)).to_numpy()
    # Backbone order is (seq_id, icode) -- "60" then "60A" then "60B", which is
    # what the empty string sorting before "A" gives. A global lexsort rank is
    # enough because it is only ever compared within one chain.
    _lex = np.lexsort((m["icode"].to_numpy(), m["seq_id"].to_numpy()))
    order = np.empty(len(m), dtype=np.int64)
    order[_lex] = np.arange(len(m))
    win_cols = CONS + GEOM_WINDOWED + ESM_WINDOWED
    win = derive.sequence_window(
        m[win_cols].to_numpy(float), chain_key, order, WINDOW)
    for i, col in enumerate(win_cols):
        out[f"{col}_w{WINDOW}"] = win[:, i]

    # -- esmnb: the family that has never been pooled -------------------------
    pooled = derive.spatial_pool(
        m[ESM_POOLED].to_numpy(float), xyz, grp, 8.0, "mean")
    for i, col in enumerate(ESM_POOLED):
        out[f"{col}_nb8"] = pooled[:, i]

    # -- nbstat: spread, extreme, and a wider shell ---------------------------
    stat_cols = NBSTAT_ENS + ["cons_score"]
    for stat, suffix in (("std", "nb8std"), ("max", "nb8max")):
        vals = derive.spatial_pool(
            m[stat_cols].to_numpy(float), xyz, grp, 8.0, stat)
        for i, col in enumerate(stat_cols):
            out[f"{col}_{suffix}"] = vals[:, i]
    wide = derive.spatial_pool(
        m[stat_cols].to_numpy(float), xyz, grp, WIDE_RADIUS, "mean")
    for i, col in enumerate(stat_cols):
        out[f"{col}_nb{int(WIDE_RADIUS)}"] = wide[:, i]

    # -- pctx: what kind of protein is this ----------------------------------
    ctx = derive.protein_context(
        m, ["ens_mean"], stats=("mean", "max", "std"), group_col=C.GROUP_COL)
    ctx = ctx.join(derive.protein_context(
        m, ["rsa", "cons_score"], stats=("mean",), group_col=C.GROUP_COL))
    g = m.groupby(C.GROUP_COL, sort=False)
    ctx["prot_n_res"] = g[C.GROUP_COL].transform("size").to_numpy(float)
    ctx["prot_n_chains"] = g["chain"].transform("nunique").to_numpy(float)
    ctx["prot_frac_exposed"] = (
        m.assign(__e=(m["rsa"] > EXPOSED_RSA).astype(float))
         .groupby(C.GROUP_COL, sort=False)["__e"].transform("mean").to_numpy(float)
    )
    for col in ctx.columns:
        out[col] = ctx[col].to_numpy(float)

    # -- prel: where does this residue sit within its own protein ------------
    for how in ("rank", "z"):
        vals = derive.within_protein(m[PREL].to_numpy(float), grp, how)
        for i, col in enumerate(PREL):
            out[f"{col}_{how}"] = vals[:, i]

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=C.BUILD_DIR / "residues_esm64.parquet")
    ap.add_argument("--coords", type=Path, default=C.BUILD_DIR / "residue_coords.parquet")
    ap.add_argument("--out", type=Path, default=C.BUILD_DIR / "derived_features.parquet")
    args = ap.parse_args()

    table = pd.read_parquet(args.table)
    coords = pd.read_parquet(args.coords)
    print(f"table  : {len(table):,} rows / {table[C.GROUP_COL].nunique()} proteins")

    out = build(table, coords)
    new = [c for c in out.columns if c not in KEYS]

    # float32 for the derived block. These are new columns, so narrowing them
    # cannot move a published number, and it halves what has to be uploaded to
    # Modal -- the base 117 are deliberately left at whatever dtype they were
    # published with, because the control arm has to reproduce its stored
    # scores elementwise.
    out[new] = out[new].astype("float32")

    print(f"derived: {len(new)} columns")
    nan_rate = out[new].isna().mean()
    worst = nan_rate.sort_values(ascending=False).head(5)
    print("\nhighest NaN rates:")
    print(worst.to_string())
    allnan = [c for c in new if nan_rate[c] == 1.0]
    if allnan:
        raise SystemExit(f"all-NaN columns would be rejected downstream: {allnan}")

    out.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out):,} rows, {len(new)} columns)")


if __name__ == "__main__":
    main()
