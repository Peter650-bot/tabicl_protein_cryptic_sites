"""Join the derived feature families onto the 117-feature residue table.

Produces `residues_derived.parquet` from `residues_esm64.parquet` plus any
number of feature parquets keyed on (apo_id, chain, seq_id, icode).

ONE TABLE, MANY ARMS
    The model ladder was produced by building a different parquet per feature
    set, which is why `scripts/model_ladder.py` carries its feature counts as
    hand-typed literals. Here every family lives in one table and an arm is a
    column selection declared in `cryptic_tabicl/feature_sets.py`. The control
    arm is pinned to exactly the 117 columns of the published run, and
    `feature_sets.base_columns` derives that by subtracting the derived
    families rather than by trusting a count.

MISSING IS NaN, NOT ZERO
    Same contract as the other three merge scripts. `build_table.validate`
    rejects an all-NaN column and any float16 column, and enforces both halves
    of the holo-derived guard against every new name.

COMPARABILITY
    Row count, protein count and split membership are asserted unchanged, so a
    before/after against the 117-feature run measures the features and not a
    quietly different test set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C  # noqa: E402
from cryptic_tabicl import feature_sets as FS  # noqa: E402
from cryptic_tabicl.build_table import validate  # noqa: E402

KEYS = ["apo_id", "chain", "seq_id", "icode"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=C.BUILD_DIR / "residues_esm64.parquet")
    ap.add_argument("--extra", type=Path, nargs="+",
                    default=[C.BUILD_DIR / "derived_features.parquet"],
                    help="feature parquets to join, keyed on the residue tuple")
    ap.add_argument("--out", type=Path, default=C.BUILD_DIR / "residues_derived.parquet")
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="supervised nestings to keep, e.g. train-fold-0 final. "
                         "All five blocks are 160 columns; a screen needs one. "
                         "Omit to keep every block.")
    args = ap.parse_args()

    base = pd.read_parquet(args.table)
    base["icode"] = base["icode"].fillna("").astype(str)
    n_rows, n_prot = len(base), base[C.GROUP_COL].nunique()
    print(f"base : {n_rows:,} rows / {n_prot} proteins / "
          f"{len(FS.base_columns(base))} features")

    merged = base
    for path in args.extra:
        new = pd.read_parquet(path)
        new["icode"] = new["icode"].fillna("").astype(str)
        cols = [c for c in new.columns if c not in KEYS]
        if args.blocks is not None and any("__fit_" in c for c in cols):
            keep = {f"__fit_{b}" for b in args.blocks}
            cols = [c for c in cols
                    if "__fit_" not in c or any(c.endswith(k) for k in keep)]
            print(f"  keeping supervised blocks {args.blocks} "
                  f"({len(cols)} columns)")
        clash = sorted(set(cols) & set(merged.columns))
        if clash:
            raise SystemExit(f"{path.name} would overwrite existing columns: {clash}")
        # With icode in the key both sides are unique, so the merge is 1:1 and
        # is asserted to be. 3bv9 chain B has ten residues at position 60;
        # dropping the code hands them all one residue's features.
        if new.duplicated(KEYS).any():
            raise SystemExit(f"{path.name} has duplicate keys; the merge would fan out")
        merged = merged.merge(new[KEYS + cols], on=KEYS, how="left", validate="1:1")
        assert len(merged) == n_rows, f"{path.name} changed the row count"
        assert merged[C.GROUP_COL].nunique() == n_prot, \
            f"{path.name} changed the protein count"
        cov = merged[cols].notna().any(axis=1).mean()
        print(f"  + {path.name:32s} {len(cols):3d} columns, {cov:.2%} of rows covered")

    problems = validate(merged)
    print("\nvalidation:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)
    if problems:
        raise SystemExit("refusing to write a table that fails validation")

    print("\nfeature sets resolvable against this table:")
    # One nesting's block, not all of them: a supervised arm is 53 base columns
    # plus 32, and listing it against every block at once would report a feature
    # count no job will ever use.
    blocks = sorted({c.split("__fit_", 1)[1] for c in merged.columns
                     if c.startswith(FS.SUPERVISED_PREFIX) and "__fit_" in c})
    one = blocks[0] if blocks else None
    sup = sorted(c for c in merged.columns
                 if c.startswith(FS.SUPERVISED_PREFIX) and c.endswith(f"__fit_{one}")) \
        if one else []
    if blocks:
        print(f"  (supervised arms shown against block {one!r}; "
              f"blocks present: {blocks})")
    for name in FS.FEATURE_SETS:
        want = sup if name in FS.USES_SUPERVISED_ESM else ()
        if name in FS.USES_SUPERVISED_ESM and not sup:
            print(f"  {name:24s} pending: no {FS.SUPERVISED_PREFIX}* columns yet")
            continue
        try:
            print(f"  {name:24s} {len(FS.resolve(name, merged, want)):4d} features")
        except (KeyError, ValueError) as exc:
            print(f"  {name:24s} unresolved: {exc}")

    merged.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(merged):,} rows, {merged.shape[1]} columns)")


if __name__ == "__main__":
    main()
