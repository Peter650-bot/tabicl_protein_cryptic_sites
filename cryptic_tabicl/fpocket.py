"""fpocket subprocess handling and CryptoSite's per-residue pocket score.

THE DEFINITION WE IMPLEMENT
    CryptoSite, verbatim: "the Fpocket residue pocket score [is] the maximum
    druggability score among the alpha spheres within 5 A of the residue, or 0
    if there are no alpha spheres (and hence pockets) in its neighbourhood."

    So an alpha sphere inherits the druggability of the pocket it belongs to,
    and a residue takes the max over spheres near it. Note this is a property of
    the residue's SURROUNDINGS, not of any pocket it is formally assigned to --
    a residue can score highly without being in fpocket's own residue list.

OUTPUT LAYOUT (verified against fpocket in a bioconda container)
    <stem>_out/<stem>_info.txt        "Pocket N :" blocks, "Druggability Score : x"
    <stem>_out/<stem>_pockets.pqr     ALL alpha spheres; the residue-sequence
                                      field carries the pocket index
    <stem>_out/pockets/pocketN_vert.pqr   per-pocket spheres (not needed)

    Reading the single combined .pqr rather than N per-pocket files matters:
    this runs tens of thousands of times, and the per-pocket route costs ~60
    extra file opens per invocation.

CALIBRATION
    fpocket's druggability score is calibrated at its default parameters. The
    -m/-M/-i flags are deliberately left alone; tuning them would silently
    invalidate the score's meaning and any comparison to published numbers.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

# CryptoSite's neighbourhood radius for attributing spheres to residues.
SPHERE_RADIUS = 5.0

_DRUG_RE = re.compile(r"Druggability Score\s*:\s*([-\d.eE+]+)")
_POCKET_RE = re.compile(r"^Pocket\s+(\d+)\s*:")


@dataclass
class PocketSpheres:
    """Alpha sphere centres and the druggability of the pocket each belongs to."""
    xyz: np.ndarray               # (n_spheres, 3)
    druggability: np.ndarray      # (n_spheres,)

    @classmethod
    def empty(cls) -> "PocketSpheres":
        return cls(np.zeros((0, 3)), np.zeros(0))

    def __len__(self) -> int:
        return len(self.xyz)


def fpocket_available() -> bool:
    return shutil.which("fpocket") is not None


def run_fpocket(pdb_text: str, workdir: Path | None = None,
                timeout: int = 120) -> PocketSpheres:
    """Run fpocket on PDB text and return its alpha spheres.

    Returns an empty result rather than raising when fpocket finds nothing or
    fails: across ~23k invocations a handful of degenerate conformations is
    expected, and one of them must not abort a whole protein's ensemble.
    """
    tmp = tempfile.mkdtemp(dir=workdir)
    try:
        base = Path(tmp)
        pdb = base / "s.pdb"
        pdb.write_text(pdb_text)
        proc = subprocess.run(
            ["fpocket", "-f", str(pdb)],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return PocketSpheres.empty()

        out_dir = base / "s_out"
        drug = _parse_druggability(out_dir / "s_info.txt")
        if not drug:
            return PocketSpheres.empty()
        return _parse_spheres(out_dir / "s_pockets.pqr", drug)
    except (subprocess.TimeoutExpired, OSError):
        return PocketSpheres.empty()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _parse_druggability(info_path: Path) -> Dict[int, float]:
    """pocket index -> druggability score, from <stem>_info.txt."""
    if not info_path.exists():
        return {}
    out: Dict[int, float] = {}
    current: int | None = None
    for line in info_path.read_text().splitlines():
        m = _POCKET_RE.match(line.strip())
        if m:
            current = int(m.group(1))
            continue
        if current is not None:
            d = _DRUG_RE.search(line)
            if d:
                try:
                    out[current] = float(d.group(1))
                except ValueError:
                    pass
                current = None
    return out


def _parse_spheres(pqr_path: Path, drug: Dict[int, float]) -> PocketSpheres:
    """Alpha sphere centres from the combined .pqr.

    Columns follow the PDB convention; the residue-sequence field (23-26) holds
    the pocket index, which is how a sphere is tied back to its druggability.
    """
    if not pqr_path.exists():
        return PocketSpheres.empty()

    xyz: List[tuple] = []
    scores: List[float] = []
    for line in pqr_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            pocket_id = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except (ValueError, IndexError):
            continue
        if pocket_id not in drug:
            continue
        xyz.append((x, y, z))
        scores.append(drug[pocket_id])

    if not xyz:
        return PocketSpheres.empty()
    return PocketSpheres(np.asarray(xyz, dtype=float), np.asarray(scores, dtype=float))


def residue_pocket_scores(
    spheres: PocketSpheres,
    residue_atom_xyz: Sequence[np.ndarray],
    radius: float = SPHERE_RADIUS,
) -> np.ndarray:
    """CryptoSite's residue pocket score, per residue.

    Max druggability over alpha spheres within `radius` of ANY atom of the
    residue; 0 when there are none.
    """
    n = len(residue_atom_xyz)
    if len(spheres) == 0 or n == 0:
        return np.zeros(n)

    tree = cKDTree(spheres.xyz)
    scores = np.zeros(n)
    for i, atoms in enumerate(residue_atom_xyz):
        near = tree.query_ball_point(atoms, radius)
        hit = {j for sub in near for j in sub}
        if hit:
            scores[i] = float(spheres.druggability[list(hit)].max())
    return scores


def coords_to_pdb(
    atom_xyz: np.ndarray, atom_names: Sequence[str], resnames: Sequence[str],
    chains: Sequence[str], seq_ids: Sequence[int], elements: Sequence[str],
) -> str:
    """Minimal PDB text for fpocket.

    Written directly rather than by round-tripping through gemmi: this is on the
    hot path (once per conformation per protein) and fpocket needs only ATOM
    records with well-formed columns.
    """
    lines = []
    for i, (xyz, name, resn, ch, seq, el) in enumerate(
        zip(atom_xyz, atom_names, resnames, chains, seq_ids, elements), start=1
    ):
        # PDB is a FIXED-COLUMN format and fpocket reads it positionally, so
        # every field below is placed by column, not by whitespace. Omitting the
        # altLoc column (17) shifts resName, chainID, resSeq and all three
        # coordinates one place left, which mistypes atoms and corrupts the
        # alpha-sphere radii fpocket derives from them.
        #   0:6 record  6:11 serial  12:16 name  16 altLoc  17:20 resName
        #   21 chain  22:26 resSeq  30:38 x  38:46 y  46:54 z  76:78 element
        atom_name = f" {name:<3s}" if len(name) < 4 else name[:4]
        lines.append(
            f"ATOM  {i % 100000:5d} "
            f"{atom_name:<4s}"
            f" "
            f"{resn[:3]:>3s}"
            f" "
            f"{(ch[:1] or 'A')}"
            f"{seq % 10000:4d}"
            f" "
            f"   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"  1.00  0.00          {el[:2]:>2s}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"
