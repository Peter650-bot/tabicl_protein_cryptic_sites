"""Join ensemble pocket features onto the Tier A residue table.

WHY A SEPARATE STEP
    The ensemble features cost ~28,000 fpocket runs. Keeping them in their own
    parquet means that expensive computation happens once and can be re-joined,
    re-weighted or ablated freely without recomputing.

MISSING VALUES ARE NOT ZERO
    `ens_static = 0` is meaningful: it says fpocket found no alpha sphere near
    this residue. A protein whose ensemble computation FAILED is a different
    thing entirely, and filling it with 0 would assert "no pocket here" for
    every one of its residues -- a confident false statement, and one the model
    would happily learn from. So unmatched rows are left as NaN (TabICL imputes;
    the sklearn baselines impute explicitly) and coverage is reported loudly.

COMPARABILITY
    Row count and split membership are asserted unchanged. If the enriched table
    silently lost proteins, the before/after comparison against the Tier A run
    would be measuring two different test sets.
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
    ap.add_argument("--table", type=Path, default=C.BUILD_DIR / "residues.parquet")
    ap.add_argument("--ensemble", type=Path,
                    default=C.BUILD_DIR / "ensemble_features.parquet")
    ap.add_argument("--out", type=Path, default=C.BUILD_DIR / "residues_ens.parquet")
    args = ap.parse_args()

    base = pd.read_parquet(args.table)
    ens = pd.read_parquet(args.ensemble)
    ens_cols = [c for c in ens.columns if c.startswith("ens_")]

    print(f"base     : {len(base):,} rows / {base[C.GROUP_COL].nunique()} proteins")
    print(f"ensemble : {len(ens):,} rows / {ens[C.GROUP_COL].nunique()} proteins"
          f" / {len(ens_cols)} features")

    # Normalise icode on both sides: parquet round-trips a blank code as NaN
    # on one side and "" on the other, which would silently fail every join.
    for d in (base, ens):
        d["icode"] = d["icode"].fillna("").astype(str)

    # With icode in the key both sides are unique, so the merge is 1:1 and is
    # asserted to be. Dropping duplicates here instead -- as this did before
    # icode was part of the key -- handed every residue at position 60 the
    # features of whichever one happened to survive.
    if ens.duplicated(KEYS).any():
        raise SystemExit("ens table has duplicate keys; the merge would fan out")
    merged = base.merge(ens[KEYS + ens_cols], on=KEYS, how="left",
                        validate="1:1")

    assert len(merged) == len(base), "merge changed the row count"
    assert merged[C.GROUP_COL].nunique() == base[C.GROUP_COL].nunique(), \
        "merge changed the protein count"

    matched = merged[ens_cols[0]].notna()
    by_protein = merged.groupby(C.GROUP_COL)[ens_cols[0]].apply(lambda s: s.notna().any())
    print(f"\ncoverage : {matched.mean():.2%} of residues, "
          f"{by_protein.mean():.2%} of proteins")
    missing = sorted(by_protein[~by_protein].index)
    if missing:
        print(f"no ensemble data for {len(missing)} proteins: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
        print("  -> left as NaN, not zero (0 would assert 'no pocket here')")

    # split-level coverage matters: if the test set is systematically less
    # covered than train, the comparison is not clean
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
