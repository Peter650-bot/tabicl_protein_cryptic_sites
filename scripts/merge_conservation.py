"""Join conservation features onto the ensemble-enriched residue table.

Produces `residues_cons.parquet` (54 features) from `residues_ens.parquet` (48).

MISSING IS NaN, NOT ZERO
    `cons_entropy = 0` means "perfectly conserved" and `cons_score = 0` means
    "not conserved at all" -- both are strong claims. A residue whose
    conservation could not be fetched is neither, so it stays NaN. TabICL imputes
    internally; the sklearn controls impute on training statistics only.

COMPARABILITY
    Row count, protein count and split membership are asserted unchanged, so the
    before/after against the 48-feature run measures the features rather than a
    quietly different test set.
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
    ap.add_argument("--table", type=Path, default=C.BUILD_DIR / "residues_ens.parquet")
    ap.add_argument("--conservation", type=Path,
                    default=C.BUILD_DIR / "conservation_features.parquet")
    ap.add_argument("--out", type=Path, default=C.BUILD_DIR / "residues_cons.parquet")
    args = ap.parse_args()

    base = pd.read_parquet(args.table)
    cons = pd.read_parquet(args.conservation)
    cons_cols = [c for c in cons.columns if c.startswith("cons_")]

    print(f"base         : {len(base):,} rows / {base[C.GROUP_COL].nunique()} proteins")
    print(f"conservation : {len(cons):,} rows / {cons[C.GROUP_COL].nunique()} proteins"
          f" / {len(cons_cols)} features")

    # Normalise icode on both sides: parquet round-trips a blank code as NaN
    # on one side and "" on the other, which would silently fail every join.
    for d in (base, cons):
        d["icode"] = d["icode"].fillna("").astype(str)

    # With icode in the key both sides are unique, so the merge is 1:1 and is
    # asserted to be. Dropping duplicates here instead -- as this did before
    # icode was part of the key -- handed every residue at position 60 the
    # features of whichever one happened to survive.
    if cons.duplicated(KEYS).any():
        raise SystemExit("cons table has duplicate keys; the merge would fan out")
    merged = base.merge(cons[KEYS + cons_cols], on=KEYS, how="left",
                        validate="1:1")

    assert len(merged) == len(base), "merge changed the row count"
    assert merged[C.GROUP_COL].nunique() == base[C.GROUP_COL].nunique(), \
        "merge changed the protein count"

    matched = merged[cons_cols[0]].notna()
    by_protein = merged.groupby(C.GROUP_COL)[cons_cols[0]].apply(
        lambda s: s.notna().any())
    print(f"\ncoverage     : {matched.mean():.2%} of residues, "
          f"{by_protein.mean():.2%} of proteins")

    print("\ncoverage by split:")
    print(merged.assign(has=matched).groupby("split")["has"].mean().to_string())

    problems = validate(merged)
    print("\nvalidation:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)

    merged.to_parquet(args.out, index=False)
    n_feat = len([c for c in merged.columns
                  if c not in C.NON_FEATURE_COLS and c != "pdb_id"])
    print(f"\nwrote {args.out}  ({len(merged):,} rows, {n_feat} features)")


if __name__ == "__main__":
    main()
