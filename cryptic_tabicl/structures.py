"""mmCIF -> per-residue geometry, on apo structures only.

WHY THIS EXISTS
    The join between CryptoBench annotations and atomic coordinates is the most
    fragile step in the pipeline, and it fails silently. CryptoBench keys
    residues by auth numbering ("B_12"); their own example script warns that the
    naive index can be off by two because unobserved residues are skipped and
    author numbering need not start at 1. So we never index positionally --
    every residue carries its auth (chain, seq_id, icode) and the join is on
    that tuple, with the match rate asserted downstream.

    Verified empirically against 1a4uB: gemmi's `setup_entities()` names chains
    by auth_asym_id and `residue.seqid.num` is auth_seq_id -- all 30 annotated
    pocket residues resolve.

SOLVENT ACCESSIBILITY
    freesasa needs a C toolchain that isn't present, so Shrake-Rupley is
    implemented here directly, vectorised over sphere points with a KD-tree for
    neighbour lookup. Ligands, ions and waters are excluded: the apo surface we
    describe must be the protein's own.
"""
from __future__ import annotations

import functools
import gzip
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from . import config


# ------------------------------------------------------------------ fetching
def fetch_cif(pdb_id: str, cache_dir: Path | None = None) -> Path:
    """Download an mmCIF (gzipped) once and cache it.

    mmCIF rather than PDB, per CryptoBench's explicit recommendation: some
    entries have no legacy PDB file at all.
    """
    cache = Path(cache_dir or config.STRUCTURE_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{pdb_id.lower()}.cif.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = config.RCSB_CIF_URL.format(pdb_id=pdb_id.lower())
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.rename(dest)
    return dest


def load_structure(path: Path) -> gemmi.Structure:
    """Parse a (possibly gzipped) mmCIF into auth-named chains."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        text = fh.read()
    st = gemmi.read_structure_string(text, format=gemmi.CoorFormat.Mmcif)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    return st


# ------------------------------------------------------------------ records
@dataclass
class ResidueRecord:
    chain: str
    seq_id: int          # auth_seq_id -- what CryptoBench annotations use
    label_seq: int | None  # label_seq_id -- what the PDBe conservation API uses
    icode: str
    resname: str
    atom_xyz: np.ndarray          # (n_atoms, 3)
    atom_elements: List[str]
    atom_names: List[str]
    bfactors: np.ndarray          # (n_atoms,)
    ca_xyz: np.ndarray | None
    cb_xyz: np.ndarray | None
    # The name as the PDB spells it. Equal to `resname` for the standard 20; for
    # a modified residue `resname` carries the PARENT and this carries the
    # original, so the substitution is auditable and countable. Never a feature.
    resname_observed: str | None = None

    @property
    def key(self) -> str:
        return config.RESIDUE_KEY_FMT.format(
            chain=self.chain, seq_id=self.seq_id, icode=self.icode)

    @property
    def centroid(self) -> np.ndarray:
        return self.atom_xyz.mean(axis=0)


# Selenomethionine is the one modified residue common enough to matter (it was
# 191 of 263 dropped residues in a 200-structure sample, and 8% of structures
# carry it). It is ISOSTERIC with methionine by design -- crystallographers use
# it precisely because swapping the sulfur for selenium does not change the
# shape -- so its side-chain heteroatom is presented to fpocket as a methionine
# sulfur. That keeps fpocket's druggability calibration, which is defined at
# default flags over ordinary protein atoms, on ground it was calibrated for.
#
# This aliasing does NOT generalise: LLP is a lysine carrying a whole pyridoxal
# phosphate, so there is no atom-level correspondence to invent. Every other
# modified residue keeps its real atoms and real elements -- the extra bulk is
# genuinely there in the apo structure, and the SASA probe should see it.
ATOM_ALIAS = {
    "MSE": {"SE": ("SD", "S")},        # observed name -> (parent name, element)
}


@functools.lru_cache(maxsize=None)
def parent_residue(name: str) -> str | None:
    """The standard amino acid a residue stands in for, or None to exclude it.

    WHY THIS EXISTS
        `residue_records` used to keep only the literal 20 standard names, which
        removed modified residues from the structure ENTIRELY -- not merely from
        the prediction set. A dropped MSE leaves a methionine-shaped void, so
        every neighbour's RSA is overestimated, its packing and neighbour counts
        are understated, and fpocket can report a cavity where a side chain
        should be. Measured cost before this fix: 47 of 20,387 annotated cryptic
        residues (0.23%) never matched a record and were silently discarded --
        `4yt8`'s match rate of 0.857 was exactly three MSE residues, A_243,
        A_283 and B_214, all three annotated cryptic by CryptoBench.

    HOW
        gemmi's chemical-component table already knows the parent: a modified
        amino acid carries its parent's one-letter code in LOWER case
        (MSE -> 'm', LLP -> 'k', PCA -> 'e'), while the standard 20 are upper
        case. So the parent falls out of the table and no hand-maintained list
        of modifications has to be kept correct.

    WHAT IS STILL EXCLUDED, DELIBERATELY
        - DNA and RNA chains. Their atoms are left out of the surface entirely,
          which is a modelling choice, not an oversight: the task is defined on
          the apo protein. For a protein crystallised with its nucleic acid this
          means the computed surface is not the one in the complex.
        - `UNK` and anything whose parent is unknown ('X'). There is no honest
          MAX_ASA, hydropathy or charge to give a residue of unknown identity,
          and inventing one would be a confident false statement.
        - Waters, ions, ligands, and non-amino-acid components that gemmi does
          not class as amino acids at all (HS8, IL0, ASB and friends).
    """
    info = gemmi.find_tabulated_residue(name)
    if info is None or not info.is_amino_acid():
        return None
    olc = (info.one_letter_code or " ").strip().upper()
    return config.ONE_TO_THREE.get(olc)


def residue_records(
    st: gemmi.Structure, chains: Sequence[str] | None = None
) -> List[ResidueRecord]:
    """Amino-acid polymer residues of the requested chains.

    Modified residues (MSE, SEP, PTR, ...) are KEPT, under their parent standard
    name -- see `parent_residue` for why, and for what is still excluded. Waters,
    ions, ligands and nucleic acids are excluded: they are not prediction targets
    and, for the apo task, not part of the surface being described.
    """
    wanted = set(chains) if chains else None
    model = st[0]
    out: List[ResidueRecord] = []
    for chain in model:
        if wanted is not None and chain.name not in wanted:
            continue
        for res in chain.get_polymer():
            parent = parent_residue(res.name)
            if parent is None:
                continue
            alias = ATOM_ALIAS.get(res.name, {})
            xyz, elems, names, bs = [], [], [], []
            ca = cb = None
            for atom in res:
                p = atom.pos
                atom_name, element = alias.get(
                    atom.name, (atom.name, atom.element.name.upper()))
                xyz.append((p.x, p.y, p.z))
                elems.append(element)
                names.append(atom_name)
                bs.append(atom.b_iso)
                if atom_name == "CA":
                    ca = np.array((p.x, p.y, p.z))
                elif atom_name == "CB":
                    cb = np.array((p.x, p.y, p.z))
            if not xyz:
                continue
            out.append(ResidueRecord(
                chain=chain.name,
                seq_id=res.seqid.num,
                # Both numberings are kept because different sources key on
                # different ones: CryptoBench annotates by auth_seq_id, the PDBe
                # conservation API indexes by label_seq_id, and they diverge
                # whenever a structure's author numbering does not start at 1 or
                # skips. Carrying both makes the join explicit instead of
                # positional.
                label_seq=res.label_seq,
                icode=(res.seqid.icode or " ").strip(),
                # The PARENT standard name: every downstream lookup keys on it
                # (MAX_ASA for RSA, hydropathy, formal charge, the one-letter
                # sequence handed to ESM-2, and the conservation join), and none
                # of those tables has an entry for a modified residue.
                resname=parent,
                resname_observed=res.name,
                atom_xyz=np.asarray(xyz, dtype=np.float64),
                atom_elements=elems,
                atom_names=names,
                bfactors=np.asarray(bs, dtype=np.float64),
                ca_xyz=ca,
                cb_xyz=cb,
            ))
    return out


# ------------------------------------------------------------------ SASA
def _sphere_points(n: int = 92) -> np.ndarray:
    """Near-uniform points on the unit sphere (golden-spiral)."""
    idx = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def shrake_rupley(
    xyz: np.ndarray, elements: Sequence[str], n_points: int = 92,
    probe: float = config.PROBE_RADIUS,
) -> np.ndarray:
    """Per-atom solvent accessible surface area (A^2).

    Vectorised over sphere points; a KD-tree restricts the occlusion test to
    atoms that could possibly overlap.
    """
    radii = np.array([config.VDW_RADII.get(e, config.DEFAULT_VDW) for e in elements])
    expanded = radii + probe
    sphere = _sphere_points(n_points)
    tree = cKDTree(xyz)
    max_reach = float(expanded.max() * 2)
    areas = np.empty(len(xyz))
    for i in range(len(xyz)):
        test = xyz[i] + expanded[i] * sphere                     # (n_points, 3)
        neigh = [j for j in tree.query_ball_point(xyz[i], max_reach) if j != i]
        if neigh:
            d2 = ((test[:, None, :] - xyz[neigh][None, :, :]) ** 2).sum(-1)
            buried = (d2 < (expanded[neigh] ** 2)[None, :]).any(axis=1)
            frac = 1.0 - buried.mean()
        else:
            frac = 1.0
        areas[i] = 4.0 * np.pi * expanded[i] ** 2 * frac
    return areas


def residue_sasa(records: Sequence[ResidueRecord]) -> np.ndarray:
    """Per-residue SASA, computed in the context of the whole selection."""
    if not records:
        return np.zeros(0)
    xyz = np.vstack([r.atom_xyz for r in records])
    elems: List[str] = []
    for r in records:
        elems.extend(r.atom_elements)
    atom_areas = shrake_rupley(xyz, elems)
    out = np.empty(len(records))
    pos = 0
    for i, r in enumerate(records):
        n = len(r.atom_xyz)
        out[i] = atom_areas[pos:pos + n].sum()
        pos += n
    return out
