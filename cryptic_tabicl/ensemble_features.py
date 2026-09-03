"""Ensemble pocket-score features: the strongest signal CryptoSite reports.

WHAT THESE CORRESPOND TO
    CryptoSite's five most discriminative descriptors are all ensemble pocket
    statistics (-log10 P in brackets):

        CNC_mean  mean pocket score over snapshots        [138.6]
        CNS       fraction of snapshots scoring > 0.4     [134.4]
        CNC_std   std of pocket score over snapshots      [121.0]
        CN5_std   std of the top-5% pocket scores         [ 94.4]
        CN5_mean  mean of the top-5% pocket scores        [ 91.9]

    For comparison the best static descriptor, sequence conservation, is 67.5
    and the static pocket score is 28.9. The ensemble is where the signal is.

WHERE WE DEVIATE, AND WHY
    CryptoSite averaged over ~100k MD snapshots, so its top-5% statistic covered
    thousands of samples. We use ~24 ANM conformations, where "top 5%" would be
    a single draw -- a statistic of one, dominated by noise. So instead of
    pretending to reproduce CN5 we report `q90` and `max` explicitly, which are
    honest at this sample size and carry the same intent (what does the pocket
    look like in its most open observed state).

THE DERIVED FEATURE THAT MATTERS MOST
    `ens_opening` = max over the ensemble minus the static crystal score. A
    cryptic site is by definition one that scores poorly in the unbound crystal
    and well once it opens, so this difference states the hypothesis directly.
    It is computed entirely from the apo structure and its own motion -- no holo
    information is involved.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

from . import anm, fpocket
from .structures import ResidueRecord

N_CONFORMATIONS = 24
DRUGGABLE_CUTOFF = 0.4          # CryptoSite's threshold for the CNS analogue
NEIGHBOURHOOD_RADIUS = 8.0

FEATURE_NAMES = (
    "ens_mean", "ens_std", "ens_frac_druggable", "ens_max", "ens_q90",
    "ens_static", "ens_opening",
)


def compute_ensemble_features(
    records: Sequence[ResidueRecord],
    n_conformations: int = N_CONFORMATIONS,
    target_rmsd: float = anm.TARGET_RMSD,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Per-residue ensemble pocket statistics. One dict per record.

    Returns zero-filled features (never None) when the ensemble cannot be built,
    so the resulting table has no all-NaN column -- TabICL's predict_proba
    crashes on those.
    """
    n = len(records)
    if n == 0:
        return []

    atom_xyz, atom_names, resnames, chains, seq_ids, elements, owner = _flatten(records)
    residue_atoms = [r.atom_xyz for r in records]

    # --- static score on the crystal conformation --------------------------
    static_pdb = fpocket.coords_to_pdb(
        atom_xyz, atom_names, resnames, chains, seq_ids, elements)
    static = fpocket.residue_pocket_scores(
        fpocket.run_fpocket(static_pdb), residue_atoms)

    # --- ensemble ----------------------------------------------------------
    ca_index = [i for i, r in enumerate(records) if r.ca_xyz is not None]
    modes = None
    if len(ca_index) >= 20:
        ca_xyz = np.vstack([records[i].ca_xyz for i in ca_index])
        modes = anm.compute_modes(ca_xyz)

    if modes is None:
        # No usable network (too small, or fragmented). Fall back to the static
        # score with zero spread rather than dropping the protein: the features
        # stay defined and simply carry no ensemble information.
        return _rows(static, np.tile(static[:, None], (1, 1)), records)

    # Map each atom to its residue's position within the Ca subset, so an atom
    # of a residue lacking a Ca still moves with the nearest modelled residue.
    ca_rank = {ri: k for k, ri in enumerate(ca_index)}
    atom_ca_slot = np.array([
        ca_rank.get(int(o), _nearest_slot(int(o), ca_rank)) for o in owner
    ])

    conformations = anm.sample_conformations(
        modes, n_conformations, target_rmsd=target_rmsd, seed=seed)

    per_conf = np.zeros((n, len(conformations)))
    for c, ca_new in enumerate(conformations):
        moved = anm.apply_to_all_atoms(atom_xyz, atom_ca_slot, modes.ca_xyz, ca_new)
        pdb = fpocket.coords_to_pdb(
            moved, atom_names, resnames, chains, seq_ids, elements)
        spheres = fpocket.run_fpocket(pdb)
        split = _split_by_owner(moved, owner, n)
        per_conf[:, c] = fpocket.residue_pocket_scores(spheres, split)

    rows = _rows(static, per_conf, records)
    _add_neighbourhood(rows, records)
    return rows


def _rows(static: np.ndarray, per_conf: np.ndarray,
          records: Sequence[ResidueRecord]) -> List[Dict[str, float]]:
    mean = per_conf.mean(axis=1)
    std = per_conf.std(axis=1)
    frac = (per_conf > DRUGGABLE_CUTOFF).mean(axis=1)
    mx = per_conf.max(axis=1)
    q90 = np.quantile(per_conf, 0.9, axis=1)
    return [{
        "ens_mean": float(mean[i]),
        "ens_std": float(std[i]),
        "ens_frac_druggable": float(frac[i]),
        "ens_max": float(mx[i]),
        "ens_q90": float(q90[i]),
        "ens_static": float(static[i]),
        # The cryptic hypothesis, stated as a number: how much more druggable
        # this residue's neighbourhood becomes once the protein moves.
        "ens_opening": float(mx[i] - static[i]),
    } for i in range(len(records))]


def _add_neighbourhood(rows: List[Dict[str, float]],
                       records: Sequence[ResidueRecord]) -> None:
    """8 A neighbourhood means, matching the Tier A convention.

    CryptoSite lists `CNCn` (pocket score of neighbours) separately from `CNC`
    and ranks it higher, so the patch-level view is worth carrying here too.
    """
    centroids = np.vstack([r.centroid for r in records])
    neighbours = cKDTree(centroids).query_ball_point(centroids, NEIGHBOURHOOD_RADIUS)
    base = {f: np.array([r[f] for r in rows]) for f in FEATURE_NAMES}
    for i, row in enumerate(rows):
        idx = [j for j in neighbours[i] if j != i] or [i]
        for f, vals in base.items():
            row[f"{f}_nb8"] = float(vals[idx].mean())


def _flatten(records: Sequence[ResidueRecord]):
    xyz, names, resn, chains, seqs, elems, owner = [], [], [], [], [], [], []
    for i, r in enumerate(records):
        for k in range(len(r.atom_xyz)):
            xyz.append(r.atom_xyz[k])
            names.append(r.atom_names[k])
            elems.append(r.atom_elements[k])
            resn.append(r.resname)
            chains.append(r.chain)
            seqs.append(r.seq_id)
            owner.append(i)
    return (np.asarray(xyz), names, resn, chains, seqs, elems,
            np.asarray(owner, dtype=int))


def _split_by_owner(xyz: np.ndarray, owner: np.ndarray, n: int) -> List[np.ndarray]:
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(n + 1))
    return [xyz[order[bounds[i]:bounds[i + 1]]] for i in range(n)]


def _nearest_slot(residue_index: int, ca_rank: Dict[int, int]) -> int:
    """Slot of the nearest residue that has a Ca, by sequence position."""
    if not ca_rank:
        return 0
    best = min(ca_rank, key=lambda ri: abs(ri - residue_index))
    return ca_rank[best]
