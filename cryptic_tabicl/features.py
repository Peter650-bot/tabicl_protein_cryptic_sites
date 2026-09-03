"""Tier A per-residue features. Apo-only, no GPU, no external binaries.

WHY THESE FEATURES
    CryptoSite (Cimermancic et al.) ranked its 58 residue descriptors by
    KS-test significance. Excluding the MD-ensemble features (which need a
    conformational ensemble we don't build in Tier A), its strongest signals are
    sequence conservation, protrusion, fragment-docking, then a long tail of
    shape and neighbourhood terms. Crucially, the "...of neighbours" variants
    (SQCn, PRTn, CVXn, SASn) rank near the top -- a residue's environment
    carries as much signal as the residue. So every numeric feature here also
    gets an 8 A spatial-neighbourhood mean.

    Where CryptoSite gives an exact definition we use it verbatim, and say so:
      - packing            "number of atoms of other residues within 4 A from
                            any atom in the residue, divided by the number of
                            atoms in the residue"
      - n_neighbors        "number of different residues within the same distance"
      - depth_to_surface   "smallest distance between any atom of a given residue
                            and the closest atom with surface area > 2 A^2"
    Where it cites prior work without a formula (protrusion, convexity), we use a
    named, documented proxy rather than pretending to reproduce the original.

APO-ONLY
    Nothing here touches the holo structure. That is the single invariant that
    makes the resulting model honest; `tests/test_no_leakage.py` enforces it.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from . import config
from .structures import ResidueRecord, residue_sasa

# Numeric features that additionally get an 8 A neighbourhood mean.
NEIGHBOURHOOD_RADIUS = 8.0
AGGREGATED = (
    "sasa", "rsa", "depth_to_surface", "packing", "n_neighbors_8A",
    "protrusion", "convexity", "hse_up", "hse_down", "bfactor_z",
    "kd_hydropathy", "formal_charge",
)


# ------------------------------------------------------------------ helpers
def _virtual_cb(rec: ResidueRecord) -> np.ndarray | None:
    """CB position, reconstructed for glycine from N/CA/C.

    Half-sphere exposure needs a CA->CB direction; glycine has no CB, so we
    place a virtual one using the standard tetrahedral construction.
    """
    if rec.cb_xyz is not None:
        return rec.cb_xyz
    if rec.ca_xyz is None:
        return None
    idx = {n: i for i, n in enumerate(_atom_names(rec))}
    if "N" not in idx or "C" not in idx:
        return None
    n = rec.atom_xyz[idx["N"]] - rec.ca_xyz
    c = rec.atom_xyz[idx["C"]] - rec.ca_xyz
    n /= (np.linalg.norm(n) or 1.0)
    c /= (np.linalg.norm(c) or 1.0)
    bisector = -(n + c)
    perp = np.cross(n, c)
    nb, np_ = np.linalg.norm(bisector) or 1.0, np.linalg.norm(perp) or 1.0
    direction = (bisector / nb) * math.cos(math.radians(37.6)) + \
                (perp / np_) * math.sin(math.radians(37.6))
    return rec.ca_xyz + 1.53 * direction / (np.linalg.norm(direction) or 1.0)


def _atom_names(rec: ResidueRecord) -> List[str]:
    return rec.atom_names


def _secondary_structure(records: Sequence[ResidueRecord]) -> List[str]:
    """Coarse H/E/C assignment from backbone dihedrals.

    A real DSSP needs hydrogen-bond energetics and a compiled dependency; the
    Ramachandran region is a reasonable stand-in and, unlike DSSP, degrades
    gracefully across chain breaks. Reported as a categorical -- TabICL encodes
    categoricals natively, so no manual one-hot is needed.
    """
    out: List[str] = []
    for i, rec in enumerate(records):
        phi = rec.phi if hasattr(rec, "phi") else None
        psi = rec.psi if hasattr(rec, "psi") else None
        if phi is None or psi is None or np.isnan(phi) or np.isnan(psi):
            out.append("C")
            continue
        p, s = math.degrees(phi), math.degrees(psi)
        if -160.0 <= p <= -20.0 and -120.0 <= s <= 50.0:
            out.append("H")
        elif -180.0 <= p <= -40.0 and (s >= 90.0 or s <= -150.0):
            out.append("E")
        else:
            out.append("C")
    return out


def _dihedrals(st: gemmi.Structure, records: Sequence[ResidueRecord]) -> None:
    """Attach phi/psi to each record, in place, keyed by (chain, seq_id)."""
    index = {(r.chain, r.seq_id): r for r in records}
    for chain in st[0]:
        polymer = chain.get_polymer()
        n = len(polymer)
        for i in range(n):
            res = polymer[i]
            rec = index.get((chain.name, res.seqid.num))
            if rec is None:
                continue
            phi = psi = float("nan")
            try:
                prev_res = polymer[i - 1] if i > 0 else None
                next_res = polymer[i + 1] if i < n - 1 else None
                if prev_res is not None and next_res is not None:
                    phi, psi = gemmi.calculate_phi_psi(prev_res, res, next_res)
            except Exception:
                pass
            rec.phi, rec.psi = phi, psi


# ------------------------------------------------------------------ main
def compute_features(
    st: gemmi.Structure,
    records: Sequence[ResidueRecord],
    noncryptic_keys: Set[str] | None = None,
) -> List[Dict]:
    """One feature dict per residue. Pure function of the apo structure."""
    if not records:
        return []
    noncryptic_keys = noncryptic_keys or set()

    # --- atom-level scaffolding -------------------------------------------
    atom_xyz = np.vstack([r.atom_xyz for r in records])
    atom_owner = np.concatenate([
        np.full(len(r.atom_xyz), i, dtype=np.int32) for i, r in enumerate(records)
    ])
    atom_elements: List[str] = []
    for r in records:
        atom_elements.extend(r.atom_elements)

    from .structures import shrake_rupley
    atom_sasa = shrake_rupley(atom_xyz, atom_elements)
    atom_tree = cKDTree(atom_xyz)

    # per-residue SASA / RSA
    res_sasa = np.zeros(len(records))
    np.add.at(res_sasa, atom_owner, atom_sasa)
    rsa = np.array([
        res_sasa[i] / config.MAX_ASA.get(r.resname, 200.0)
        for i, r in enumerate(records)
    ])

    # CryptoSite: "closest atom with surface area > 2 A^2"
    surface_atoms = atom_xyz[atom_sasa > 2.0]
    surface_tree = cKDTree(surface_atoms) if len(surface_atoms) else None

    centroids = np.vstack([r.centroid for r in records])
    centroid_tree = cKDTree(centroids)
    protein_centroid = atom_xyz.mean(axis=0)

    _dihedrals(st, records)
    ss = _secondary_structure(records)

    # B-factor z-score within chain: absolute B is not comparable across
    # crystals (resolution, refinement protocol), the relative profile is.
    b_mean = np.array([r.bfactors.mean() for r in records])
    by_chain: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_chain.setdefault(r.chain, []).append(i)
    b_z = np.zeros(len(records))
    for idxs in by_chain.values():
        vals = b_mean[idxs]
        sd = vals.std()
        b_z[idxs] = (vals - vals.mean()) / sd if sd > 1e-9 else 0.0

    chain_len = {c: len(idxs) for c, idxs in by_chain.items()}
    chain_order = {c: {i: k for k, i in enumerate(idxs)} for c, idxs in by_chain.items()}

    rows: List[Dict] = []
    for i, rec in enumerate(records):
        own = rec.atom_xyz

        # --- CryptoSite exact definitions ---------------------------------
        near4 = set()
        for a in own:
            near4.update(atom_tree.query_ball_point(a, 4.0))
        other4 = [j for j in near4 if atom_owner[j] != i]
        packing = len(other4) / max(len(own), 1)
        n_res_4 = len({int(atom_owner[j]) for j in other4})

        n_res_6 = len({
            int(atom_owner[j])
            for a in own for j in atom_tree.query_ball_point(a, 6.0)
            if atom_owner[j] != i
        })
        n_res_8 = len(centroid_tree.query_ball_point(rec.centroid, 8.0)) - 1

        if surface_tree is not None:
            depth = float(min(surface_tree.query(a)[0] for a in own))
        else:
            depth = 0.0

        # --- shape proxies -------------------------------------------------
        # protrusion: empty fraction of a 10 A sphere (PSAIA-style). A residue
        # sticking out of the body has more empty space around it.
        in_sphere = atom_tree.query_ball_point(rec.centroid, 10.0)
        v_sphere = (4.0 / 3.0) * math.pi * 10.0 ** 3
        protrusion = max(0.0, 1.0 - len(in_sphere) * 20.0 / v_sphere)

        # convexity: how far the residue sits from the centre of its own 12 A
        # neighbourhood, signed by whether that displacement points away from
        # the protein centre. Positive = bulging outward.
        neigh12 = centroid_tree.query_ball_point(rec.centroid, 12.0)
        if len(neigh12) > 1:
            local = centroids[[j for j in neigh12 if j != i]].mean(axis=0)
            disp = rec.centroid - local
            outward = rec.centroid - protein_centroid
            no = np.linalg.norm(outward) or 1.0
            convexity = float(np.dot(disp, outward / no))
        else:
            convexity = 0.0

        # half-sphere exposure about the CA->CB axis
        hse_up = hse_down = 0
        cb = _virtual_cb(rec)
        if rec.ca_xyz is not None and cb is not None:
            axis = cb - rec.ca_xyz
            na = np.linalg.norm(axis)
            if na > 1e-6:
                axis = axis / na
                for j in centroid_tree.query_ball_point(rec.ca_xyz, 13.0):
                    if j == i or records[j].ca_xyz is None:
                        continue
                    if float(np.dot(records[j].ca_xyz - rec.ca_xyz, axis)) > 0:
                        hse_up += 1
                    else:
                        hse_down += 1

        rows.append({
            "chain": rec.chain,
            "seq_id": rec.seq_id,
            "icode": rec.icode,
            "resname": rec.resname,
            "sasa": float(res_sasa[i]),
            "rsa": float(rsa[i]),
            "depth_to_surface": depth,
            "packing": float(packing),
            "n_neighbors_4A": n_res_4,
            "n_neighbors_6A": n_res_6,
            "n_neighbors_8A": n_res_8,
            "protrusion": float(protrusion),
            "convexity": convexity,
            "hse_up": hse_up,
            "hse_down": hse_down,
            "bfactor_mean": float(b_mean[i]),
            "bfactor_z": float(b_z[i]),
            "ss": ss[i],
            "phi": float(getattr(rec, "phi", float("nan"))),
            "psi": float(getattr(rec, "psi", float("nan"))),
            "kd_hydropathy": config.KD_HYDROPATHY.get(rec.resname, 0.0),
            "formal_charge": config.FORMAL_CHARGE.get(rec.resname, 0.0),
            "rel_seq_pos": chain_order[rec.chain][i] / max(chain_len[rec.chain] - 1, 1),
            "chain_len": chain_len[rec.chain],
            "in_noncryptic_pocket": int(rec.key in noncryptic_keys),
        })

    _add_neighbourhood_means(rows, centroid_tree, centroids)
    return rows


def _add_neighbourhood_means(
    rows: List[Dict], tree: cKDTree, centroids: np.ndarray
) -> None:
    """Append the 8 A neighbourhood mean of each aggregated feature.

    CryptoSite's neighbour-averaged descriptors rank among its most
    discriminative; a cryptic site is a property of a surface patch, not of an
    isolated residue.
    """
    neighbours = tree.query_ball_point(centroids, NEIGHBOURHOOD_RADIUS)
    base = {f: np.array([r[f] for r in rows], dtype=float) for f in AGGREGATED}
    for i, row in enumerate(rows):
        idx = [j for j in neighbours[i] if j != i] or [i]
        for f, vals in base.items():
            row[f"{f}_nb8"] = float(vals[idx].mean())


def feature_columns(rows: Sequence[Dict]) -> List[str]:
    """Model-visible columns: everything except identifiers and the label."""
    if not rows:
        return []
    return [c for c in rows[0] if c not in config.NON_FEATURE_COLS]
