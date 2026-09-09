"""Insertion codes are part of a residue's identity.

WHY THIS FILE EXISTS
    Author numbering is not dense. Proteases and immunoglobulins insert residues
    without renumbering the rest of the chain, so 3bv9 chain B holds ten
    residues at position 60 -- 60, 60A, 60B ... 60I -- and six at 149.
    CryptoBench keys them "B_60", "B_60A", ...; an earlier version of this code
    keyed on (chain, seq_id) alone and collapsed all ten onto "B_60".

    That failed in BOTH directions simultaneously, which is what made it hard to
    see:
      - an annotated "B_60A" matched nothing, so a true positive became negative
      - an annotated "B_149" matched all six residues at 149, so five residues
        were handed a positive label they never earned

    Measured on the built table before the fix: 19 residues wrongly positive and
    1 wrongly negative for the cryptic label, 9 wrongly set and 3 wrongly unset
    for `in_noncryptic_pocket`, across 6 structures -- plus 3bv9B quarantined
    outright at match rate 0.793 (23 of 29 annotated keys resolving).

    All of it sat in the train split, so no published test metric was wrong.
    That is luck, not design, which is exactly why it is pinned here.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config
from cryptic_tabicl.structures import ResidueRecord

import numpy as np


def _rec(chain="B", seq_id=60, icode=""):
    return ResidueRecord(
        chain=chain, seq_id=seq_id, label_seq=None, icode=icode, resname="ALA",
        atom_xyz=np.zeros((1, 3)), atom_elements=["C"], atom_names=["CA"],
        bfactors=np.zeros(1), ca_xyz=np.zeros(3), cb_xyz=None)


def test_key_includes_the_insertion_code():
    """"B_60A" is CryptoBench's own spelling; we must produce it verbatim."""
    assert _rec(icode="A").key == "B_60A"
    assert _rec(icode="").key == "B_60"


def test_residues_at_one_position_get_distinct_keys():
    """The 3bv9B 60-loop: ten residues, ten keys, no collisions."""
    keys = [_rec(icode=c).key for c in ["", *"ABCDEFGHI"]]
    assert len(set(keys)) == 10, "insertion codes collapsed onto one key"
    assert keys[0] == "B_60" and keys[1] == "B_60A"


def test_annotated_icode_key_resolves():
    """The false-negative half: "B_60A" must match the residue that bears it."""
    observed = {_rec(icode=c).key for c in ["", *"ABCDEFGHI"]}
    for annotated in ("B_60A", "B_60D", "B_60F"):
        assert annotated in observed


def test_bare_key_does_not_label_the_whole_position():
    """The false-positive half, which was the more common of the two.

    Annotating only "B_149" must label residue 149 and NOT 149A-149E.
    """
    label_keys = {"B_149"}
    recs = [_rec(seq_id=149, icode=c) for c in ["", *"ABCDE"]]
    labels = [int(r.key in label_keys) for r in recs]
    assert labels == [1, 0, 0, 0, 0, 0]


def test_build_table_key_matches_residue_record_key():
    """build_table.py builds the key from a DataFrame row, not from the record.

    Two spellings of the same rule is how they drift apart; this pins them
    together.
    """
    rec = _rec(seq_id=149, icode="B")
    row = {"chain": rec.chain, "seq_id": rec.seq_id, "icode": rec.icode}
    from_row = config.RESIDUE_KEY_FMT.format(
        chain=row["chain"], seq_id=row["seq_id"], icode=row["icode"])
    assert from_row == rec.key == "B_149B"


def test_chain_parse_survives_an_insertion_code():
    """dataset.py recovers the chain with rsplit('_', 1)[0]; "B_60A" must work."""
    from cryptic_tabicl import dataset
    ds = {"3bv9": [{"apo_chain": "B",
                    "apo_pocket_selection": ["B_60A", "C_149B"]}]}
    assert dataset.apo_chains(ds)["3bv9B"] == {"B", "C"}
    assert dataset.apo_binding_residues(ds)["3bv9B"] == {"B_60A", "C_149B"}


# ------------------------------------------------------------------ merges
@pytest.mark.parametrize("module", ["merge_ensemble", "merge_conservation",
                                    "merge_esm"])
def test_every_merge_joins_on_icode(module):
    """A side table joined without icode gives 60A the features of 60.

    All three feature merges must agree on the four-part key, or the residue
    table and its enrichments refer to different residues.
    """
    import importlib
    m = importlib.import_module(f"scripts.{module}")
    assert "icode" in m.KEYS, f"{module}.KEYS is missing icode"


def test_merge_on_icode_key_is_one_to_one():
    """The shape the merges assert: four-part key, no fan-out, no collapse."""
    base = pd.DataFrame({
        "apo_id": ["3bv9B"] * 3, "chain": ["B"] * 3,
        "seq_id": [60, 60, 60], "icode": ["", "A", "B"]})
    side = base.assign(feat=[1.0, 2.0, 3.0])
    keys = ["apo_id", "chain", "seq_id", "icode"]
    merged = base.merge(side, on=keys, how="left", validate="1:1")
    assert len(merged) == 3
    assert merged["feat"].tolist() == [1.0, 2.0, 3.0]

    # ...and what the old three-part key did instead.
    keys3 = ["apo_id", "chain", "seq_id"]
    collapsed = base.merge(side.drop_duplicates(keys3)[keys3 + ["feat"]],
                           on=keys3, how="left")
    assert collapsed["feat"].tolist() == [1.0, 1.0, 1.0], \
        "the three-part key should hand all three residues the same features"


def test_anm_seed_is_stable_across_processes():
    """`hash()` on a str is salted by PYTHONHASHSEED, so it cannot seed an ensemble.

    The previous `abs(hash(apo_id))` drew different ANM conformations on every
    run, which meant the ensemble features could not be reproduced at all.
    """
    import subprocess
    code = ("import sys; sys.path.insert(0, '.');"
            "from deploy.ensemble_modal import _seed_for; print(_seed_for('3bv9B'))")
    seen = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env={"PYTHONHASHSEED": s, "PATH": "/usr/bin:/bin"},
                           cwd=str(Path(__file__).resolve().parent.parent)).stdout.strip()
            for s in ("0", "1", "12345")}
    assert len(seen) == 1 and seen != {""}, f"seed varies with PYTHONHASHSEED: {seen}"
