"""Download CryptoBench annotations from OSF.

WHY NOT THE PROVIDED ZIP
    The tutorial points at a single archive endpoint that zips the whole project
    server-side. That archive is ~1.1 GB and streams at a few MB/min, and 1.09 GB
    of it is `cif-files.zip` -- structures we fetch from RCSB on demand anyway,
    in parallel, at roughly forty times the rate. Walking the OSF file API and
    taking only the annotations pulls ~51 MB instead.

DESTINATION
    Local disk, not the OneDrive-backed project directory: the structure cache
    grows to a few hundred MB and syncing it is pathological.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config  # noqa: E402

OSF_ROOT = "https://api.osf.io/v2/nodes/pz4a9/files/osfstorage/"
SKIP_LARGER_THAN = 50 * 1024 * 1024


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=90) as resp:
        return json.load(resp)


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.rename(dest)
    return dest.stat().st_size


def walk(url: str, dest_root: Path, skip_large: bool = True) -> None:
    payload = _get(url)
    for item in payload.get("data", []):
        attrs = item["attributes"]
        materialized = attrs["materialized_path"].lstrip("/")
        if attrs["kind"] == "folder":
            walk(item["relationships"]["files"]["links"]["related"]["href"],
                 dest_root, skip_large)
            continue

        size = attrs.get("size") or 0
        if skip_large and size > SKIP_LARGER_THAN:
            print(f"SKIP  {materialized}  ({size / 1048576:.0f} MB)")
            continue

        rel = materialized.split("cryptobench/", 1)[-1]
        out = dest_root / rel
        if out.exists() and out.stat().st_size == size:
            print(f"have  {rel}")
            continue
        try:
            n = _download(item["links"]["download"], out)
            print(f"OK    {rel}  {n / 1048576:.2f} MB")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {rel}: {exc}")

    nxt = payload.get("links", {}).get("next")
    if nxt:
        walk(nxt, dest_root, skip_large)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-cifs", action="store_true",
                    help="also pull the 1.09 GB CIF archive (usually unnecessary: "
                         "build_table fetches structures from RCSB in parallel)")
    ap.add_argument("--dest", type=Path, default=config.CRYPTOBENCH_DIR)
    args = ap.parse_args()

    print(f"downloading CryptoBench annotations -> {args.dest}")
    walk(OSF_ROOT, args.dest, skip_large=not args.include_cifs)

    for required in (config.DATASET_JSON, config.SPLITS_JSON, config.NONCRYPTIC_JSON):
        status = "ok" if required.exists() else "MISSING"
        print(f"  {status:8s} {required.name}")


if __name__ == "__main__":
    main()
