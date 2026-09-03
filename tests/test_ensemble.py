"""ANM ensembles and fpocket parsing.

The fpocket parser runs ~28,000 times inside containers where nothing is
watching, so its failure mode is a silently empty result rather than a crash --
which would show up only as weak features. These tests pin the format against
real fpocket output captured from a bioconda container.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import anm, fpocket


# ---------------------------------------------------------------- ANM
def _helix(n=60, rise=1.5, radius=2.3, turn=100.0):
    """A synthetic alpha helix: a well-conditioned Ca network."""
    t = np.arange(n) * np.radians(turn)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.arange(n) * rise], 1)


def test_modes_drop_rigid_body():
    """Six near-zero modes are translations/rotations and carry no dynamics."""
    modes = anm.compute_modes(_helix(), n_modes=6)
    assert modes is not None
    # every retained mode must be a genuine vibration
    assert (modes.eigenvalues > 1e-8).all()
    assert modes.eigenvectors.shape[0] == 3 * modes.n_ca


def test_modes_are_ascending():
    """Softest modes first — sampling weights by 1/sqrt(eigenvalue)."""
    modes = anm.compute_modes(_helix(), n_modes=6)
    assert np.all(np.diff(modes.eigenvalues) >= -1e-9)


def test_too_small_returns_none():
    """A handful of atoms cannot support a network; caller must handle None."""
    assert anm.compute_modes(_helix(n=8)) is None


def test_conformations_hit_target_rmsd():
    """Rescaling per conformation is what makes proteins comparable."""
    modes = anm.compute_modes(_helix(), n_modes=6)
    conf = anm.sample_conformations(modes, 8, target_rmsd=3.0, seed=0)
    rmsd = np.sqrt(((conf - modes.ca_xyz) ** 2).sum(2).mean(1))
    assert np.allclose(rmsd, 3.0, atol=1e-6)


def test_sampling_is_deterministic():
    modes = anm.compute_modes(_helix(), n_modes=6)
    a = anm.sample_conformations(modes, 4, seed=3)
    b = anm.sample_conformations(modes, 4, seed=3)
    assert np.array_equal(a, b)


def test_all_atom_move_is_rigid_per_residue():
    """Internal residue geometry must survive, or fpocket sees fake cavities."""
    ca_ref = _helix(n=10)
    ca_new = ca_ref + np.random.default_rng(0).normal(scale=0.5, size=ca_ref.shape)
    atoms = np.repeat(ca_ref, 4, axis=0) + np.tile(np.arange(4)[:, None], (10, 3)) * 0.1
    owner = np.repeat(np.arange(10), 4)
    moved = anm.apply_to_all_atoms(atoms, owner, ca_ref, ca_new)
    for r in range(10):
        before = atoms[owner == r]
        after = moved[owner == r]
        # pairwise distances within the residue are unchanged
        assert np.allclose(
            np.linalg.norm(before[:, None] - before[None], axis=-1),
            np.linalg.norm(after[:, None] - after[None], axis=-1),
        )


# ---------------------------------------------------------------- fpocket
INFO_TXT = """Pocket 1 :
\tScore : \t0.248
\tDruggability Score : \t0.577
\tNumber of Alpha Spheres : \t98
\tVolume : \t962.616
Pocket 2 :
\tScore : \t0.101
\tDruggability Score : \t0.0026
\tNumber of Alpha Spheres : \t42
"""

# Real fpocket .pqr columns: the residue-sequence field holds the pocket index.
PQR = """HEADER
HEADER This is a pqr format file writen by the programm fpocket.
ATOM      1    O STP     1       5.097  -4.299  -2.284    0.00     3.44
ATOM      2    C STP     1      -1.705  -5.374  -1.515    0.00     3.94
ATOM      3    C STP     2      50.000  50.000  50.000    0.00     3.40
"""


def test_druggability_parsed_per_pocket(tmp_path):
    p = tmp_path / "s_info.txt"
    p.write_text(INFO_TXT)
    drug = fpocket._parse_druggability(p)
    assert drug == {1: pytest.approx(0.577), 2: pytest.approx(0.0026)}


def test_spheres_inherit_their_pocket_druggability(tmp_path):
    p = tmp_path / "s_pockets.pqr"
    p.write_text(PQR)
    spheres = fpocket._parse_spheres(p, {1: 0.577, 2: 0.0026})
    assert len(spheres) == 3
    assert spheres.druggability[0] == pytest.approx(0.577)
    assert spheres.druggability[2] == pytest.approx(0.0026)
    assert spheres.xyz[0] == pytest.approx([5.097, -4.299, -2.284])


def test_spheres_of_unscored_pockets_are_dropped(tmp_path):
    """A sphere with no druggability entry cannot contribute a score."""
    p = tmp_path / "s_pockets.pqr"
    p.write_text(PQR)
    assert len(fpocket._parse_spheres(p, {1: 0.577})) == 2


def test_residue_score_is_max_within_5A():
    """CryptoSite: max druggability among alpha spheres within 5 A."""
    spheres = fpocket.PocketSpheres(
        xyz=np.array([[0.0, 0, 0], [1.0, 0, 0], [100.0, 0, 0]]),
        druggability=np.array([0.2, 0.9, 1.0]),
    )
    residues = [np.array([[0.0, 0, 0]]), np.array([[100.0, 0, 0]]),
                np.array([[50.0, 0, 0]])]
    scores = fpocket.residue_pocket_scores(spheres, residues)
    assert scores[0] == pytest.approx(0.9)   # sees both near spheres, takes max
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == 0.0                  # nothing within 5 A -> 0, not NaN


def test_no_spheres_gives_zeros_not_nan():
    """All-NaN columns crash TabICL's predict_proba."""
    scores = fpocket.residue_pocket_scores(
        fpocket.PocketSpheres.empty(), [np.zeros((1, 3))] * 4)
    assert scores.shape == (4,)
    assert not np.isnan(scores).any()


def test_pdb_columns_are_well_formed():
    """fpocket parses by column position; a shifted field silently breaks it."""
    txt = fpocket.coords_to_pdb(
        np.array([[1.234, 5.678, 9.012]]), ["CA"], ["ALA"], ["A"], [42], ["C"])
    line = txt.splitlines()[0]
    assert line.startswith("ATOM")
    assert line[17:20] == "ALA"
    assert int(line[22:26]) == 42
    assert float(line[30:38]) == pytest.approx(1.234)
    assert float(line[38:46]) == pytest.approx(5.678)
    assert float(line[46:54]) == pytest.approx(9.012)
