"""Archive PAM's data, fold models and predictions on Zenodo, from Modal.

WHY MODAL
    The four fold models (4 x 5.3 GiB) live on the `cryptobench-s2p` volume. Uploading
    them from the volume to Zenodo avoids routing 21 GB through a home connection.

WHAT GOES UP (build/zenodo_release/README.md describes each file)
    model_cleantest_{0..3}.pt   from ft/s2p_full_cleantest_20261008/ on the volume
    pam_features.zip, pam_predictions.zip   from zenodo_release/ (modal volume put first)
    README.md, bundles_manifest.json        sent from this machine

    Every file is hashed (MD5 and SHA-256) on Modal before upload; the upload is accepted
    only if Zenodo's MD5 equals ours. The token is the Modal secret `ZENODO_TOKEN`
    (key `zenodo`); it never leaves Modal.

USAGE
    modal run deploy/zenodo_release.py::create_draft                 # prints id and reserved DOI
    modal run deploy/zenodo_release.py --dep-id N --upload models     # or zips / all; parallel
    modal run deploy/zenodo_release.py --dep-id N --small --software-doi 10.5281/zenodo.M
    modal run deploy/zenodo_release.py --dep-id N --status-only
    modal run deploy/zenodo_release.py --dep-id N --do-publish
"""
from __future__ import annotations

import modal

API = "https://zenodo.org/api/deposit/depositions"
VOL_ROOT = "/vol"
FOLD_DIR = "ft/s2p_full_cleantest_20261008"
BIG_FILES = [f"{FOLD_DIR}/model_cleantest_{k}.pt" for k in range(4)] + [
    "zenodo_release/pam_features.zip", "zenodo_release/pam_predictions.zip"]
CODE_URL = "https://github.com/Peter650-bot/tabicl_protein_cryptic_sites"

CREATORS = [{"name": "Saba, Peter Riad"}]
DESCRIPTION = (
    "<p>Data, fine-tuned models and predictions accompanying the manuscript "
    "<em>Tabular In-Context Learning for Cryptic Site Prediction</em> (submitted to "
    "<em>Bioinformatics</em>). PAM predicts cryptic binding site residues on CryptoBench "
    "with TabICLv2 over 53 structural features and 64 principal components of a "
    "cross-fitted fine-tuned ESM-2 3B embedding.</p>"
    "<ul>"
    "<li><code>pam_features.zip</code>: per-residue feature tables (381,704 residues, "
    "1,136 apo chains), labels, folds and the LIGYSIS exclusion lists.</li>"
    "<li><code>pam_predictions.zip</code>: per-residue predictions of PAM, of each fold "
    "model and of the released checkpoint of Škrhák et al.; cross-validation thresholds, "
    "bootstrap outputs and comparison tables.</li>"
    "<li><code>model_cleantest_{0..3}.pt</code>: PyTorch state dicts of the four "
    "cross-fitted fine-tuned ESM-2 3B fold models.</li>"
    "</ul>"
    f"<p>Code: <a href=\"{CODE_URL}\">{CODE_URL}</a> (MIT). README.md describes every "
    "file; bundles_manifest.json lists SHA-256 checksums.</p>")
METADATA = {
    "upload_type": "dataset",
    "title": "PAM: features, fine-tuned ESM-2 fold models and predictions for cryptic "
             "binding site prediction",
    "creators": CREATORS,
    "description": DESCRIPTION,
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": "1.0.0",
    "keywords": ["cryptic binding sites", "protein language models", "ESM-2",
                 "tabular in-context learning", "TabICL", "CryptoBench"],
    "related_identifiers": [
        {"identifier": CODE_URL, "relation": "isSupplementTo", "resource_type": "software"},
        {"identifier": "10.17605/OSF.IO/PZ4A9", "relation": "isDerivedFrom",
         "resource_type": "dataset"},
        {"identifier": "10.5281/zenodo.13121413", "relation": "isDerivedFrom",
         "resource_type": "dataset"},
    ],
    "prereserve_doi": True,
}

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.3")
vol = modal.Volume.from_name("cryptobench-s2p")
app = modal.App("pam-zenodo-release")
secret = modal.Secret.from_name("ZENODO_TOKEN")


def _session():
    import os

    import requests
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {os.environ['zenodo']}"
    return s


def _check(r):
    if r.status_code >= 400:
        raise RuntimeError(f"{r.request.method} {r.url.split('?')[0]}: {r.status_code} {r.text[:800]}")
    return r


def _bucket(s, dep_id: int) -> str:
    return _check(s.get(f"{API}/{dep_id}")).json()["links"]["bucket"]


def _hashes(path: str) -> tuple[str, str, int]:
    import hashlib
    md5, sha = hashlib.md5(), hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            md5.update(b)
            sha.update(b)
            n += len(b)
    return md5.hexdigest(), sha.hexdigest(), n


def _put(s, bucket: str, name: str, data, md5: str) -> None:
    r = _check(s.put(f"{bucket}/{name}", data=data, timeout=(60, 6 * 3600)))
    got = r.json()["checksum"].removeprefix("md5:")
    if got != md5:
        raise RuntimeError(f"{name}: Zenodo md5 {got} != local {md5}")


@app.function(image=image, secrets=[secret], timeout=600)
def create_draft() -> dict:
    s = _session()
    dep = _check(s.post(API, json={"metadata": METADATA})).json()
    out = {"id": dep["id"], "doi": dep["metadata"]["prereserve_doi"]["doi"],
           "html": dep["links"]["html"]}
    print(out, flush=True)
    return out


@app.function(image=image, secrets=[secret], volumes={VOL_ROOT: vol}, timeout=6 * 3600,
              cpu=2.0, memory=4096)
def upload_from_volume(dep_id: int, rel: str) -> dict:
    import os
    import time
    s = _session()
    path = f"{VOL_ROOT}/{rel}"
    name = os.path.basename(rel)
    md5, sha, n = _hashes(path)
    t0 = time.time()
    with open(path, "rb") as fh:
        _put(s, _bucket(s, dep_id), name, fh, md5)
    dt = time.time() - t0
    print(f"{name}: {n/1e9:.2f} GB in {dt:.0f}s, md5 verified", flush=True)
    return {"file": name, "bytes": n, "md5": md5, "sha256": sha, "seconds": round(dt)}


@app.function(image=image, secrets=[secret], timeout=600)
def upload_bytes(dep_id: int, name: str, data: bytes) -> dict:
    import hashlib
    s = _session()
    _put(s, _bucket(s, dep_id), name, data, hashlib.md5(data).hexdigest())
    return {"file": name, "bytes": len(data)}


@app.function(image=image, secrets=[secret], timeout=600)
def set_metadata(dep_id: int, metadata: dict) -> None:
    _check(_session().put(f"{API}/{dep_id}", json={"metadata": metadata}))


@app.function(image=image, secrets=[secret], timeout=600)
def status(dep_id: int) -> dict:
    dep = _check(_session().get(f"{API}/{dep_id}")).json()
    return {"state": dep["state"], "submitted": dep["submitted"],
            "doi": dep["metadata"].get("prereserve_doi", {}).get("doi"),
            "files": {f["filename"]: (f["filesize"], f["checksum"]) for f in dep["files"]}}


@app.function(image=image, secrets=[secret], timeout=1800)
def publish(dep_id: int) -> dict:
    r = _check(_session().post(f"{API}/{dep_id}/actions/publish", timeout=1800)).json()
    return {"doi": r["doi"], "record": r["links"].get("record_html") or r["links"]["html"],
            "concept_doi": r.get("conceptdoi")}


@app.local_entrypoint()
def main(dep_id: int = 0, upload: str = "", small: bool = False,
         software_doi: str = "", status_only: bool = False, do_publish: bool = False):
    import json
    import pathlib
    if upload:                                  # "models", "zips" or "all"
        files = [f for f in BIG_FILES if upload == "all" or
                 (f.endswith(".pt") if upload == "models" else f.endswith(".zip"))]
        results = list(upload_from_volume.map([dep_id] * len(files), files))
        out = pathlib.Path(f"build/zenodo_release/uploaded_{upload}.json")
        out.write_text(json.dumps(results, indent=1))
        print(json.dumps(results, indent=1))
    if small:
        root = pathlib.Path("build/zenodo_release")
        for name in ("README.md", "bundles_manifest.json"):
            print(upload_bytes.remote(dep_id, name, (root / name).read_bytes()))
        if software_doi:
            md = dict(METADATA)
            md.pop("prereserve_doi")
            md["related_identifiers"] = METADATA["related_identifiers"] + [
                {"identifier": software_doi, "relation": "isCompiledBy",
                 "resource_type": "software"}]
            set_metadata.remote(dep_id, md)
    if status_only:
        print(json.dumps(status.remote(dep_id), indent=1))
    if do_publish:
        print(publish.remote(dep_id))
