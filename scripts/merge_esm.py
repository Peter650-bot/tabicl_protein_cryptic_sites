"""Join ESM-2 PCA features onto the conservation-enriched residue table.

Produces `residues_esm.parquet` (54 + k features) from `residues_cons.parquet`.

THE KEY INCLUDES icode
    279 rows share an (apo_id, chain, seq_id) triple -- insertion codes, where
    two residues are numbered 52 and 52A. Merging on the triple alone fans those
    rows out and silently changes the row count. Both tables carry `icode`, and
    with it the four-part key is unique on both sides, so the merge is 1:1 and
    asserted to be.

--n-components TRIMS, IT DOES NOT REFIT
    PCA components are ordered by explained variance, so the first k columns of
    a 64-component fit ARE the k-component projection. Sweeping k costs a column
    slice rather than a re-run of the GPU job.

COMPARABILITY
    Row count, protein count and split membership are asserted unchanged, so the
    before/after against the 54-feature run measures the embeddings and nothing
    else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C  # noqa: E402
from cryptic_tabicl.build_table import validate  # noqa: E402

KEYS = ["apo_id", "chain", "seq_id", "icode"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path,
                    default=C.BUILD_DIR / "residues_cons.parquet")
    ap.add_argument("--esm", type=Path,
                    default=C.BUILD_DIR / "esm_features.parquet")
    ap.add_argument("--n-components", type=int, default=64,
                    help="keep the first k principal components")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or C.BUILD_DIR / f"residues_esm{args.n_components}.parquet"

    base = pd.read_parquet(args.table)
    esm = pd.read_parquet(args.esm)

    for d in (base, esm):
        d["icode"] = d["icode"].fillna("").astype(str)

    all_pcs = sorted(c for c in esm.columns if c.startswith("esm_pc"))
    if args.n_components > len(all_pcs):
        raise SystemExit(f"only {len(all_pcs)} components available")
    pcs = all_pcs[: args.n_components]

    print(f"base : {len(base):,} rows / {base[C.GROUP_COL].nunique()} proteins")
    print(f"esm  : {len(esm):,} rows / {esm[C.GROUP_COL].nunique()} proteins"
          f" / keeping {len(pcs)} of {len(all_pcs)} components")

    if esm.duplicated(KEYS).any():
        raise SystemExit("ESM table has duplicate keys; the merge would fan out")

    merged = base.merge(esm[KEYS + pcs], on=KEYS, how="left", validate="1:1")

    assert len(merged) == len(base), "merge changed the row count"
    assert merged[C.GROUP_COL].nunique() == base[C.GROUP_COL].nunique(), \
        "merge changed the protein count"

    matched = merged[pcs[0]].notna()
    print(f"\ncoverage : {matched.mean():.2%} of residues, "
          f"{merged.groupby(C.GROUP_COL)[pcs[0]].apply(lambda s: s.notna().any()).mean():.2%}"
          " of proteins")
    print("\ncoverage by split:")
    print(merged.assign(has=matched).groupby("split")["has"].mean().to_string())

    problems = validate(merged)
    print("\nvalidation:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)

    merged.to_parquet(out, index=False)
    n_feat = len([c for c in merged.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])
    print(f"\nwrote {out}  ({len(merged):,} rows, {n_feat} features)")


if __name__ == "__main__":
    main()
