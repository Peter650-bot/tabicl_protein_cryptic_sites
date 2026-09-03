"""One-off: confirm fpocket's output layout before building against it.

CryptoSite defines the residue pocket score as "the maximum druggability score
among the alpha spheres within 5 A of the residue". That needs two things from
fpocket that live in different files -- per-pocket druggability, and the alpha
sphere COORDINATES -- so the parser depends on details of the output tree that
are worth verifying once rather than guessing at.
"""
import modal

image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("fpocket", channels=["conda-forge", "bioconda"])
    .pip_install("gemmi", "numpy")
)

app = modal.App("fpocket-probe")


@app.function(image=image, timeout=900)
def probe() -> str:
    import subprocess
    import urllib.request
    from pathlib import Path

    out = []
    v = subprocess.run(["fpocket", "-v"], capture_output=True, text=True)
    out.append(f"fpocket version: {(v.stdout + v.stderr).strip()[:200]}")

    work = Path("/tmp/w")
    work.mkdir(exist_ok=True)
    pdb = work / "1a4u.pdb"
    urllib.request.urlretrieve("https://files.rcsb.org/download/1A4U.pdb", pdb)
    out.append(f"downloaded {pdb.stat().st_size} bytes")

    r = subprocess.run(["fpocket", "-f", str(pdb)], capture_output=True, text=True)
    out.append(f"fpocket rc={r.returncode}")
    if r.returncode != 0:
        out.append(f"stderr: {r.stderr[:500]}")

    tree = sorted(p.relative_to(work) for p in work.rglob("*"))
    out.append(f"\n--- output tree ({len(tree)} entries) ---")
    out.extend(f"  {t}" for t in tree[:25])

    info = work / "1a4u_out" / "1a4u_info.txt"
    if info.exists():
        out.append("\n--- info.txt head ---")
        out.append("\n".join(info.read_text().splitlines()[:22]))

    pockets_dir = work / "1a4u_out" / "pockets"
    if pockets_dir.exists():
        files = sorted(pockets_dir.iterdir())
        out.append(f"\n--- pockets/ ({len(files)} files) ---")
        out.extend(f"  {f.name}" for f in files[:6])
        vert = next((f for f in files if f.name.endswith("_vert.pqr")), None)
        if vert:
            out.append(f"\n--- {vert.name} head ---")
            out.append("\n".join(vert.read_text().splitlines()[:8]))

    allp = work / "1a4u_out" / "1a4u_pockets.pqr"
    if allp.exists():
        out.append(f"\n--- {allp.name} head ---")
        out.append("\n".join(allp.read_text().splitlines()[:6]))

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
