"""Modified residues must stay in the structure, under their parent name.

Keeping only the literal 20 standard names removed modified residues ENTIRELY --
not merely from the prediction set. A dropped MSE leaves a methionine-shaped
void, so every neighbour's RSA is overestimated, its packing understated, and
fpocket can find a cavity where a side chain should be. It also threw away
ground truth: 47 of 20,387 annotated cryptic residues never matched a record,
and `4yt8`'s three missing annotations were all selenomethionine.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gemmi

from cryptic_tabicl import config, structures


# ------------------------------------------------- the parent resolver
@pytest.mark.parametrize("observed,parent", [
    ("MSE", "MET"),      # selenomethionine -- by far the most common
    ("SEP", "SER"),      # phosphoserine
    ("TPO", "THR"),      # phosphothreonine
    ("PTR", "TYR"),      # phosphotyrosine
    ("CSO", "CYS"),      # oxidised cysteine
    ("YCM", "CYS"),
    ("KCX", "LYS"),      # carboxylated lysine
    ("LLP", "LYS"),      # lysine-pyridoxal phosphate
    ("M3L", "LYS"),      # trimethyllysine
    ("PCA", "GLU"),      # pyroglutamate
    ("FME", "MET"),      # N-formylmethionine
    ("ALA", "ALA"),      # the standard 20 map to themselves
    ("MET", "MET"),
])
def test_modified_residues_resolve_to_their_parent(observed, parent):
    assert structures.parent_residue(observed) == parent


@pytest.mark.parametrize("name", [
    "UNK",               # identity unknown: no honest MAX_ASA or hydropathy
    "DT", "DA", "DC", "DG",   # DNA -- excluded from the apo surface by choice
    "A", "U",            # RNA
    "HOH",               # water
    "ZN",                # ion
    "HS8", "IL0", "ASB", "AYA",   # components gemmi does not class as amino acids
])
def test_non_amino_acids_stay_excluded(name):
    assert structures.parent_residue(name) is None


def test_every_parent_has_a_complete_feature_row():
    """A parent name is only useful if every lookup keyed on it has an entry."""
    for observed in ("MSE", "SEP", "TPO", "PTR", "LLP", "PCA", "KCX"):
        parent = structures.parent_residue(observed)
        assert parent in config.MAX_ASA
        assert parent in config.KD_HYDROPATHY
        assert parent in config.STANDARD_AA


def test_resolver_never_invents_a_nonstandard_parent():
    """Whatever gemmi reports, the parent must be one of the standard 20."""
    for name in ("MSE", "SEP", "LLP", "UNK", "DT", "HOH", "XXX", ""):
        parent = structures.parent_residue(name)
        assert parent is None or parent in config.MAX_ASA


# ------------------------------------------------- the isosteric MSE alias
def _residue(name, num, atoms):
    res = gemmi.Residue()
    res.name = name
    res.seqid = gemmi.SeqId(num, " ")
    for atom_name, element, xyz in atoms:
        atom = gemmi.Atom()
        atom.name, atom.element = atom_name, gemmi.Element(element)
        atom.pos = gemmi.Position(*xyz)
        atom.b_iso = 20.0
        res.add_atom(atom)
    return res


def _backbone(offset):
    return [("N", "N", (offset, 0.0, 0.0)), ("CA", "C", (offset + 1.5, 0.0, 0.0)),
            ("C", "C", (offset + 2.0, 1.4, 0.0)), ("O", "O", (offset + 3.2, 1.6, 0.0)),
            ("CB", "C", (offset + 2.0, -1.0, 1.0))]


def _mse_structure(middle="MSE"):
    """ALA-<middle>-ALA. Three residues, because gemmi will not call a single
    residue a polymer and `residue_records` iterates `get_polymer()`."""
    side = [("CG", "C", (7.4, -1.3, 0.8)), ("SE", "SE", (8.0, -2.5, 2.0)),
            ("CE", "C", (9.8, -2.3, 1.9))]
    if middle != "MSE":
        side = [("CG", "C", (7.4, -1.3, 0.8)), ("SD", "S", (8.0, -2.5, 2.0)),
                ("CE", "C", (9.8, -2.3, 1.9))]
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    chain.add_residue(_residue("ALA", 1, _backbone(0.0)))
    chain.add_residue(_residue(middle, 2, _backbone(4.0) + side))
    chain.add_residue(_residue("ALA", 3, _backbone(8.0)))
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st


def test_selenomethionine_is_presented_as_methionine():
    """Se is isosteric with S and fpocket is calibrated on protein atoms."""
    recs = structures.residue_records(_mse_structure())
    assert [r.resname for r in recs] == ["ALA", "MET", "ALA"]
    r = recs[1]
    assert r.resname == "MET"
    assert r.resname_observed == "MSE"
    assert "SD" in r.atom_names and "SE" not in r.atom_names
    assert "S" in r.atom_elements and "SE" not in r.atom_elements
    # No atom is lost -- the void is what the old filter created.
    assert len(r.atom_xyz) == 8
    assert r.ca_xyz is not None and r.cb_xyz is not None


def test_alias_does_not_apply_to_other_modified_residues():
    """LLP carries a whole pyridoxal phosphate; there is nothing to alias."""
    assert set(structures.ATOM_ALIAS) == {"MSE"}


def test_standard_residues_keep_their_observed_name():
    r = structures.residue_records(_mse_structure(middle="MET"))[1]
    assert r.resname == r.resname_observed == "MET"
    assert "SD" in r.atom_names


# ------------------------------------------------- against a real structure
_CACHED = config.STRUCTURE_CACHE / "4yt8.cif.gz"


@pytest.mark.skipif(not _CACHED.exists(),
                    reason="needs the staged mmCIF cache")
def test_4yt8_recovers_its_three_annotated_selenomethionines():
    """The concrete cost of the old filter, on the structure that showed it.

    CryptoBench annotates A_243, A_283 and B_214 as cryptic. All three are MSE,
    so all three used to vanish -- that is exactly why 4yt8's label match rate
    was 0.857 rather than 1.000.
    """
    st = gemmi.read_structure(str(_CACHED))
    st.setup_entities()
    by_key = {r.key: r for r in structures.residue_records(st)}
    for key in ("A_243", "A_283", "B_214"):
        assert key in by_key, f"{key} is still missing"
        assert by_key[key].resname == "MET"
        assert by_key[key].resname_observed == "MSE"
