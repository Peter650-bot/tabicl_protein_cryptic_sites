"""Probe ConSurf-DB: is it reachable, what is the URL pattern, what is our coverage?

WHY REMOTE
    consurfdb.tau.ac.il does not resolve from the development sandbox (RCSB and
    OSF do, so it is a host restriction rather than a network failure). Modal
    containers have unrestricted egress.

WHY PROBE BEFORE BUILDING
    ConSurf-DB is precomputed, so if it covers our 1,135 apo chains it is nearly
    free and answers "does conservation help the hard stratum" in an afternoon.
    If coverage is poor we need an MMseqs2 pipeline against UniRef instead --
    tens of GB and hours of compute. That is a large fork to take on assumption,
    and a sample of 40 chains settles it.

WHAT IT REPORTS
    Which URL pattern works, and the hit rate over a random sample of the actual
    apo chains in our dataset (not a convenience sample of famous PDB entries,
    which would flatter coverage).
"""
from __future__ import annotations

import json
import os

import modal

CHAINS_FILE = "build/apo_chains.json"
if modal.is_local() and not os.path.exists(CHAINS_FILE):
    import sys

    sys.path.insert(0, ".")
    from cryptic_tabicl import config as C, dataset as D

    ds = D.load_json(C.DATASET_JSON)
    pdb_of = D.apo_pdb_ids(ds)
    chains = D.apo_chains(ds)
    payload = [
        {"apo_id": a, "pdb": pdb_of[a], "chain": sorted(chains[a])[0]}
        for a in sorted(chains)
    ]
    os.makedirs("build", exist_ok=True)
    with open(CHAINS_FILE, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {CHAINS_FILE} ({len(payload)} chains)")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests")
    .add_local_file(CHAINS_FILE, "/data/apo_chains.json")
)

app = modal.App("consurf-probe")

PATTERNS = [
    "https://consurfdb.tau.ac.il/DB/{pdb_u}{chain}/consurf_summary.txt",
    "https://consurfdb.tau.ac.il/DB/{pdb_l}{chain}/consurf_summary.txt",
    "https://consurfdb.tau.ac.il/DB/{pdb_u}_{chain}/consurf_summary.txt",
    "https://consurfdb.tau.ac.il/DB/{pdb_u}{chain}/consurf.grades",
    "https://consurfdb.tau.ac.il/DB/{pdb_u}{chain}/{pdb_u}{chain}_consurf_summary.txt",
    "https://consurfdb.tau.ac.il/DB/{pdb_u}{chain}/consurf_grades.txt",
]


@app.function(image=image, timeout=1800)
def probe(sample_size: int = 40) -> dict:
    import random

    import requests

    out: dict = {}
    s = requests.Session()
    s.headers["User-Agent"] = "cryptic-tabicl/0.1 (academic coverage probe)"

    # 1. reachable at all?
    try:
        r = s.get("https://consurfdb.tau.ac.il/", timeout=45)
        out["homepage"] = {"status": r.status_code, "bytes": len(r.content)}
        body = r.text.lower()
        out["homepage_mentions_download"] = "download" in body
    except Exception as exc:  # noqa: BLE001
        out["homepage"] = {"error": f"{type(exc).__name__}: {exc}"}
        return out

    # 2. which URL pattern serves a real grades file?
    probes = [("1a4u", "B"), ("4pfs", "A"), ("2f6v", "A")]
    pattern_results = []
    working = None
    for pat in PATTERNS:
        hits = []
        for pdb, ch in probes:
            url = pat.format(pdb_u=pdb.upper(), pdb_l=pdb.lower(), chain=ch)
            try:
                rr = s.get(url, timeout=45)
                ok = rr.status_code == 200 and len(rr.content) > 500
                hits.append({"url": url, "status": rr.status_code,
                             "bytes": len(rr.content), "ok": ok})
                if ok and working is None:
                    working = pat
                    out["sample_head"] = rr.text[:600]
            except Exception as exc:  # noqa: BLE001
                hits.append({"url": url, "error": f"{type(exc).__name__}"})
        pattern_results.append({"pattern": pat, "hits": hits})
    out["patterns"] = pattern_results
    out["working_pattern"] = working

    if working is None:
        return out

    # 3. coverage over a random sample of OUR chains
    chains = json.load(open("/data/apo_chains.json"))
    random.seed(0)
    sample = random.sample(chains, min(sample_size, len(chains)))
    found, missing = 0, []
    for item in sample:
        url = working.format(pdb_u=item["pdb"].upper(), pdb_l=item["pdb"].lower(),
                             chain=item["chain"])
        try:
            rr = s.get(url, timeout=45)
            if rr.status_code == 200 and len(rr.content) > 500:
                found += 1
            else:
                missing.append(item["apo_id"])
        except Exception:  # noqa: BLE001
            missing.append(item["apo_id"])
    out["coverage"] = {
        "sampled": len(sample),
        "found": found,
        "rate": found / max(len(sample), 1),
        "missing_examples": missing[:12],
    }
    return out


@app.local_entrypoint()
def main(sample_size: int = 40):
    res = probe.remote(sample_size=sample_size)
    print(json.dumps(res, indent=2)[:4000])
    cov = res.get("coverage")
    if cov:
        print(f"\nCOVERAGE: {cov['found']}/{cov['sampled']} = {cov['rate']:.1%}")
    else:
        print("\nno working URL pattern found; ConSurf-DB route is not viable as-is")
