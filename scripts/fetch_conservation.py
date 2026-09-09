"""Fetch PDBe conservation for every apo chain and emit per-residue features.

WHY LOCAL AND NOT MODAL
    This is ~2,300 HTTP calls against a public academic API, not compute. It
    runs fine from a laptop with modest concurrency, and hammering EBI from
    dozens of containers to save a few minutes would be rude.

INSTRUMENTED JOIN
    The API keys on `label_seq_id`, CryptoBench on `auth_seq_id`. Where these
    diverge a positional join yields plausible, wrong values, so this reports the
    per-structure match rate and quarantines anything that resolves poorly. A
    silent 60% match would look like "conservation is a weak feature".

CACHING
    Raw API responses are cached per entity, so re-running after a parsing change
    costs no network at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C, conservation, dataset as D  # noqa: E402
from cryptic_tabicl.structures import load_structure, residue_records  # noqa: E402

CACHE = C.DATA_STAGE / "conservation_cache"
MIN_MATCH_RATE = 0.50


def _cached_fetch(pdb: str, chain: str) -> conservation.ChainConservation | None:
    """Fetch with an on-disk cache. ONLY successes are cached.

    An earlier version cached failures as `null` too. That is wrong in a way
    that is easy to miss: `fetch_chain` returns None both for "this entry
    genuinely has no conservation" and for "the network was down". During a
    brief outage that poisoned 1,004 of 1,281 cache entries, and the rerun
    faithfully reproduced the outage instead of recovering from it -- presenting
    as 78% of structures having no conservation, against a probe that had
    measured 97% coverage.

    Since coverage is high, re-requesting the genuine misses each run is cheap
    and strictly safer than persisting an unfalsifiable negative.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{pdb.lower()}_{chain}.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            raw = None
        if raw:
            return conservation.ChainConservation(
                score={int(k): v for k, v in raw["score"].items()},
                entropy={int(k): v for k, v in raw["entropy"].items()},
                probabilities={int(k): v for k, v in raw["probabilities"].items()},
            )
        path.unlink(missing_ok=True)      # poisoned or empty: refetch

    cons = conservation.fetch_chain(pdb, chain)
    if cons is not None:
        path.write_text(json.dumps({
            "score": cons.score, "entropy": cons.entropy,
            "probabilities": cons.probabilities,
        }))
    return cons


def _one(apo_id: str, pdb: str, chains: list[str]) -> dict:
    try:
        st = load_structure(C.STRUCTURE_CACHE / f"{pdb.lower()}.cif.gz")
        recs = residue_records(st, chains)
        if not recs:
            return {"apo_id": apo_id, "rows": [], "match_rate": 0.0,
                    "error": "no residues"}

        # Conservation is per entity; fetch once per distinct chain present.
        per_chain = {ch: _cached_fetch(pdb, ch) for ch in sorted({r.chain for r in recs})}

        rows, matched = [], 0
        for rec in recs:
            cons = per_chain.get(rec.chain)
            feats = conservation.residue_features(
                cons, [rec.label_seq], [rec.resname])[0]
            if not np.isnan(feats["cons_score"]):
                matched += 1
            # icode is part of the residue's identity: without it the merge
            # back onto the residue table collapses 60/60A/60B onto one key.
            rows.append({"apo_id": apo_id, "chain": rec.chain,
                         "seq_id": rec.seq_id, "icode": rec.icode, **feats})

        centroids = np.vstack([r.centroid for r in recs])
        conservation.add_neighbourhood(rows, centroids)
        return {"apo_id": apo_id, "rows": rows,
                "match_rate": matched / len(recs), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"apo_id": apo_id, "rows": [], "match_rate": 0.0,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6,
                    help="kept modest: this is a public academic API")
    ap.add_argument("--out", type=Path,
                    default=C.BUILD_DIR / "conservation_features.parquet")
    args = ap.parse_args()

    ds = D.load_json(C.DATASET_JSON)
    chains = D.apo_chains(ds)
    pdb_of = D.apo_pdb_ids(ds)
    targets = sorted(chains)[: args.limit] if args.limit else sorted(chains)

    print(f"fetching conservation for {len(targets)} apo structures "
          f"({args.workers} workers)")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, a, pdb_of[a], sorted(chains[a])): a for a in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 50 == 0:
                ok = sum(1 for r in results if r["match_rate"] >= MIN_MATCH_RATE)
                print(f"  {i}/{len(targets)}  ({ok} matched) "
                      f"{time.time()-t0:.0f}s", flush=True)

    manifest = pd.DataFrame([{k: r[k] for k in ("apo_id", "match_rate", "error")}
                             for r in results])
    manifest["quarantined"] = (manifest["match_rate"] < MIN_MATCH_RATE)

    keep = set(manifest.loc[~manifest["quarantined"], "apo_id"])
    frames = [pd.DataFrame(r["rows"]) for r in results
              if r["apo_id"] in keep and r["rows"]]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print("\n=== conservation fetch ===")
    print(f"structures attempted : {len(manifest)}")
    print(f"quarantined          : {int(manifest['quarantined'].sum())}")
    good = manifest.loc[~manifest["quarantined"], "match_rate"]
    if len(good):
        print(f"match rate           : median {good.median():.3f}  min {good.min():.3f}")
    errs = manifest["error"].dropna()
    if len(errs):
        print(f"errors               : {len(errs)}  e.g. {errs.iloc[0][:80]}")
    if not table.empty:
        print(f"rows                 : {len(table):,}")
        cols = [c for c in table.columns if c.startswith("cons_")]
        print(table[cols].describe().T[["mean", "std", "min", "max"]].to_string())
        table.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")

    mpath = args.out.with_name(args.out.stem + "_manifest.csv")
    manifest.to_csv(mpath, index=False)
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
