"""Anisotropic Network Model ensembles, as a cheap stand-in for MD.

WHY THIS EXISTS
    CryptoSite's single strongest feature is the mean fpocket pocket score across
    an MD ensemble (-log10 P = 138.6, roughly double the next feature). The
    intuition is direct: a cryptic site is one that is closed in the crystal but
    opens under thermal motion, so scoring a single static structure cannot see
    it -- you have to look at the structure moving.

    CryptoSite got that ensemble from AllosMod: 50 simulations, 6 ns each, five
    temperatures, 100k snapshots per protein. That is far out of reach here. ANM
    is the standard cheap surrogate: it gives the low-frequency collective modes
    that dominate large-scale domain motion, from the Ca contact topology alone,
    in one eigendecomposition per protein rather than nanoseconds of dynamics.

    The trade is real and worth stating: ANM captures harmonic, collective
    motion around the crystal conformation. It will not find a cryptic site that
    needs a side-chain rotamer flip or a loop to cross a barrier. So these
    features should recover part of CryptoSite's ensemble signal, not all of it.

UNBIASED BY CONSTRUCTION
    Amplitudes are drawn from the ANM's own thermal distribution. Nothing in
    this module reads fpocket output or optimises towards open pockets -- if it
    did, the resulting features would encode "we searched until a pocket
    appeared", which is a different (and circular) quantity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree

# Standard ANM parameters (Atilgan et al. 2001): a uniform spring between Ca
# pairs within a cutoff. 13 A is the usual choice for Ca-only networks.
CUTOFF = 13.0
N_MODES = 10
N_RIGID_BODY = 6          # 3 translations + 3 rotations, eigenvalue ~0

# Target Ca RMSD of a generated conformation from the crystal. Large enough to
# open a closed pocket, small enough to stay a plausible near-native state.
TARGET_RMSD = 1.5


@dataclass
class ANMModes:
    ca_xyz: np.ndarray            # (n_ca, 3) reference coordinates
    eigenvalues: np.ndarray       # (n_modes,) ascending, rigid-body removed
    eigenvectors: np.ndarray      # (3*n_ca, n_modes)

    @property
    def n_ca(self) -> int:
        return len(self.ca_xyz)


def compute_modes(ca_xyz: np.ndarray, n_modes: int = N_MODES,
                  cutoff: float = CUTOFF) -> ANMModes | None:
    """Lowest-frequency non-trivial ANM modes. None if the network is degenerate."""
    n = len(ca_xyz)
    if n < N_RIGID_BODY + n_modes + 1:
        return None

    hessian = _hessian(ca_xyz, cutoff)
    if hessian is None:
        return None

    # Only the lowest few modes matter, so solve the partial problem rather than
    # the full 3N x 3N spectrum.
    hi = min(N_RIGID_BODY + n_modes - 1, 3 * n - 1)
    try:
        vals, vecs = eigh(hessian, subset_by_index=[0, hi])
    except Exception:
        return None

    # Drop the six rigid-body modes. They are numerically near-zero rather than
    # exactly zero, so select by index (their count is known) rather than by a
    # magnitude threshold, which would be fragile.
    vals, vecs = vals[N_RIGID_BODY:], vecs[:, N_RIGID_BODY:]
    keep = vals > 1e-8
    if not keep.any():
        return None
    return ANMModes(ca_xyz=ca_xyz, eigenvalues=vals[keep], eigenvectors=vecs[:, keep])


def _hessian(ca_xyz: np.ndarray, cutoff: float) -> np.ndarray | None:
    """3N x 3N ANM Hessian for a uniform-spring Ca network."""
    n = len(ca_xyz)
    h = np.zeros((3 * n, 3 * n))
    pairs = cKDTree(ca_xyz).query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return None

    for i, j in pairs:
        d = ca_xyz[j] - ca_xyz[i]
        d2 = float(d @ d)
        if d2 < 1e-8:
            continue
        block = np.outer(d, d) / d2          # superelement for the pair
        bi, bj = 3 * i, 3 * j
        h[bi:bi + 3, bj:bj + 3] -= block
        h[bj:bj + 3, bi:bi + 3] -= block
        h[bi:bi + 3, bi:bi + 3] += block
        h[bj:bj + 3, bj:bj + 3] += block
    return h


def sample_conformations(
    modes: ANMModes, n_conformations: int, target_rmsd: float = TARGET_RMSD,
    seed: int = 0,
) -> np.ndarray:
    """(n_conformations, n_ca, 3) Ca coordinates drawn from the ANM distribution.

    Mode amplitudes scale as 1/sqrt(eigenvalue), the equipartition result: soft
    modes move further. Each conformation is then rescaled to `target_rmsd` so
    the ensemble's magnitude is comparable across proteins of different size and
    stiffness -- otherwise large floppy proteins would get systematically bigger
    pocket-score fluctuations than small rigid ones, and the feature would encode
    protein size as much as crypticity.
    """
    rng = np.random.default_rng(seed)
    n_ca = modes.n_ca
    scale = 1.0 / np.sqrt(modes.eigenvalues)

    coeffs = rng.normal(size=(n_conformations, len(modes.eigenvalues))) * scale
    disp = coeffs @ modes.eigenvectors.T          # (n_conf, 3*n_ca)
    disp = disp.reshape(n_conformations, n_ca, 3)

    rmsd = np.sqrt((disp ** 2).sum(axis=2).mean(axis=1))
    rmsd[rmsd < 1e-9] = 1.0
    disp *= (target_rmsd / rmsd)[:, None, None]
    return modes.ca_xyz[None, :, :] + disp


def apply_to_all_atoms(
    atom_xyz: np.ndarray, atom_residue_index: np.ndarray,
    ca_reference: np.ndarray, ca_perturbed: np.ndarray,
) -> np.ndarray:
    """Move every atom by its residue's Ca displacement.

    A rigid-residue approximation: ANM only describes Ca motion, so side chains
    ride along. This preserves internal residue geometry exactly, which keeps
    fpocket's alpha-sphere construction well behaved -- interpolating atoms
    independently would distort bond lengths and produce spurious cavities.
    """
    shift = ca_perturbed - ca_reference           # (n_ca, 3)
    return atom_xyz + shift[atom_residue_index]
