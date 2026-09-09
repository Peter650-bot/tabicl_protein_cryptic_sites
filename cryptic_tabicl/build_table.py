"""Build the per-residue feature table from CryptoBench annotations + apo mmCIFs.

WHY THIS EXISTS
    This is where annotations meet coordinates, and it is the step most likely
    to fail quietly. A residue key that doesn't resolve becomes a NEGATIVE
    rather than an error: the protein still yields rows, the model still trains,
    and the only symptom is a slightly worse score that looks like modelling
    noise. So the join here is instrumented -- every apo structure reports what
    fraction of its annotated cryptic residues actually resolved, and anything
    below `MIN_MATCH_RATE` is quarantined rather than silently accepted.

PARALLELISM
    Downloads are I/O bound (threads); featurisation is CPU bound (processes).
    They are separate stages so a slow RCSB response never idles a core.
"""
from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd

from . import config, dataset as ds_mod
from .features import compute_features
from .structures import fetch_cif, load_structure, residue_records

# An apo whose annotations mostly fail to resolve indicates a numbering
# mismatch, not a hard protein. Quarantine it rather than train on bad labels.
MIN_MATCH_RATE = 0.80


@dataclass
class BuildResult:
    apo_id: str
    rows: List[Dict]
    match_rate: float
    n_annotated: int
    n_matched: int
    error: str | None = None


def _build_one(
    apo_id: str,
    pdb_id: str,
    chains: Sequence[str],
    label_keys: Set[str],
    noncryptic_keys: Set[str],
) -> BuildResult:
    try:
        path = fetch_cif(pdb_id)
        st = load_structure(path)
        recs = residue_records(st, chains)
        if not recs:
            return BuildResult(apo_id, [], 0.0, len(label_keys), 0,
                               "no standard residues in requested chains")

        observed = {r.key for r in recs}
        matched = label_keys & observed
        rate = len(matched) / len(label_keys) if label_keys else 0.0

        rows = compute_features(st, recs, noncryptic_keys)
        for row in rows:
            # Must match ResidueRecord.key exactly, insertion code included.
            key = config.RESIDUE_KEY_FMT.format(
                chain=row["chain"], seq_id=row["seq_id"], icode=row["icode"])
            row[config.LABEL_COL] = int(key in label_keys)
            row[config.GROUP_COL] = apo_id
            row["pdb_id"] = pdb_id
        return BuildResult(apo_id, rows, rate, len(label_keys), len(matched))
    except Exception as exc:  # noqa: BLE001 - one bad entry must not kill the run
        return BuildResult(apo_id, [], 0.0, len(label_keys), 0,
                           f"{type(exc).__name__}: {exc}")


def prefetch(pdb_ids: Sequence[str], workers: int = 8) -> Dict[str, str]:
    """Download every needed mmCIF up front. Returns {pdb_id: error} for failures."""
    errors: Dict[str, str] = {}
    todo = sorted(set(pdb_ids))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_cif, p): p for p in todo}
        for fut in as_completed(futs):
            pdb = futs[fut]
            done += 1
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors[pdb] = f"{type(exc).__name__}: {exc}"
            if done % 100 == 0:
                print(f"  fetched {done}/{len(todo)} ({len(errors)} failed)", flush=True)
    return errors


def build(
    apo_ids: Sequence[str] | None = None,
    workers: int = 4,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature table. Returns (table, manifest)."""
    ds = ds_mod.load_json(config.DATASET_JSON)
    labels = ds_mod.apo_binding_residues(ds)
    chains = ds_mod.apo_chains(ds)
    pdb_of = ds_mod.apo_pdb_ids(ds)

    try:
        noncryptic = ds_mod.noncryptic_residues(ds_mod.load_json(config.NONCRYPTIC_JSON))
    except FileNotFoundError:
        noncryptic = {}

    folds = ds_mod.load_folds()
    fold_of = ds_mod.fold_assignment(folds)

    targets = list(apo_ids) if apo_ids else sorted(labels)
    if limit:
        targets = targets[:limit]

    print(f"prefetching mmCIFs for {len({pdb_of[a] for a in targets})} PDB entries ...")
    fetch_errors = prefetch([pdb_of[a] for a in targets])
    if fetch_errors:
        print(f"  {len(fetch_errors)} downloads failed")

    print(f"featurising {len(targets)} apo structures on {workers} workers ...")
    results: List[BuildResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _build_one, aid, pdb_of[aid], sorted(chains[aid]),
                labels[aid], noncryptic.get(aid, set()),
            ): aid
            for aid in targets
        }
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(targets)}", flush=True)

    manifest = pd.DataFrame([{
        "apo_id": r.apo_id,
        "n_annotated": r.n_annotated,
        "n_matched": r.n_matched,
        "match_rate": r.match_rate,
        "n_rows": len(r.rows),
        "error": r.error,
        "fold": fold_of.get(r.apo_id),
        "quarantined": bool(r.error) or (r.n_annotated > 0 and r.match_rate < MIN_MATCH_RATE),
    } for r in results]).sort_values("apo_id")

    keep = set(manifest.loc[~manifest["quarantined"], "apo_id"])
    frames = [pd.DataFrame(r.rows) for r in results if r.apo_id in keep and r.rows]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not table.empty:
        table["fold"] = table[config.GROUP_COL].map(fold_of)
        table["split"] = table["fold"].map(
            lambda f: "test" if f and "test" in str(f) else "train"
        )
    return table, manifest


def validate(table: pd.DataFrame) -> List[str]:
    """Checks that must pass before the table reaches TabICL."""
    problems: List[str] = []
    if table.empty:
        return ["table is empty"]

    feature_cols = [c for c in table.columns
                    if c not in config.NON_FEATURE_COLS and c != "pdb_id"]

    # TabICL's predict_proba crashes on an all-NaN categorical column.
    for col in feature_cols:
        if table[col].isna().all():
            problems.append(f"column '{col}' is entirely NaN")

    # float16 is unsupported by TabICL.
    for col, dt in table.dtypes.items():
        if dt == "float16":
            problems.append(f"column '{col}' is float16")

    # The leakage invariant: nothing holo-derived may be a feature.
    for col in feature_cols:
        low = col.lower()
        for bad in config.FORBIDDEN_FEATURE_SUBSTRINGS:
            if bad in low:
                problems.append(f"column '{col}' looks holo-derived ('{bad}')")

    # Splits must not share a protein.
    if "split" in table:
        by_split = table.groupby(config.GROUP_COL)["split"].nunique()
        if (by_split > 1).any():
            problems.append("some apo structures appear in more than one split")

    if table[config.LABEL_COL].nunique() < 2:
        problems.append("label column has a single class")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="build only the first N apo structures (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=config.BUILD_DIR / "residues.parquet")
    args = ap.parse_args()

    table, manifest = build(workers=args.workers, limit=args.limit)

    manifest_path = args.out.with_name(args.out.stem + "_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("\n=== build summary ===")
    print(f"apo attempted   : {len(manifest)}")
    print(f"quarantined     : {int(manifest['quarantined'].sum())}")
    ok = manifest.loc[~manifest["quarantined"]]
    if len(ok):
        print(f"median match rate: {ok['match_rate'].median():.3f}")
        print(f"min match rate   : {ok['match_rate'].min():.3f}")
    if not table.empty:
        print(f"rows            : {len(table):,}")
        print(f"positives       : {int(table[config.LABEL_COL].sum()):,} "
              f"({table[config.LABEL_COL].mean():.2%})")
        print(f"features        : {len([c for c in table.columns if c not in config.NON_FEATURE_COLS and c != 'pdb_id'])}")

    problems = validate(table)
    print("\nvalidation:", "PASS" if not problems else "FAIL")
    for p in problems:
        print("  -", p)

    if not table.empty:
        table.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
