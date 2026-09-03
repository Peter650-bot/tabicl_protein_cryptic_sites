"""Conservation parsing and the auth/label join.

The join is the whole risk here. The PDBe API indexes conservation by
`label_seq_id` while CryptoBench annotates by `auth_seq_id`. For many structures
the two coincide, so a positional join passes casual inspection and then assigns
one protein's conservation profile shifted by a few residues for the structures
where they diverge. That produces plausible numbers and a weak feature, which
reads as "conservation does not help" rather than as a bug.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import conservation


def _cons(score=None, entropy=None, probs=None):
    return conservation.ChainConservation(
        score=score or {}, entropy=entropy or {}, probabilities=probs or {})


# ---------------------------------------------------------------- entropy
def test_certain_position_has_zero_entropy():
    """One residue with probability 1 is maximally conserved."""
    probs = {a: 0.0 for a in conservation.AMINO_ACIDS}
    probs["A"] = 1.0
    assert conservation._normalised_entropy(probs) == pytest.approx(0.0)


def test_uniform_position_has_entropy_one():
    """Normalising by log(20) puts the uniform distribution at exactly 1."""
    probs = {a: 1 / 20 for a in conservation.AMINO_ACIDS}
    assert conservation._normalised_entropy(probs) == pytest.approx(1.0)


def test_entropy_is_normalised_within_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(20):
        raw = rng.random(20)
        probs = dict(zip(conservation.AMINO_ACIDS, raw / raw.sum()))
        h = conservation._normalised_entropy(probs)
        assert 0.0 <= h <= 1.0


def test_entropy_handles_unnormalised_input():
    """The API's probabilities are rounded and need not sum to exactly 1."""
    probs = {a: 0.0 for a in conservation.AMINO_ACIDS}
    probs["A"] = probs["C"] = 0.4          # sums to 0.8
    h = conservation._normalised_entropy(probs)
    assert h == pytest.approx(math.log(2) / math.log(20))


def test_zero_total_probability_is_nan_not_zero():
    """Zero would claim perfect conservation; absence must stay absent."""
    probs = {a: 0.0 for a in conservation.AMINO_ACIDS}
    assert math.isnan(conservation._normalised_entropy(probs))


# ---------------------------------------------------------------- join
def test_features_align_on_label_seq_not_position():
    """The join must follow label_seq, even when auth numbering differs.

    Here residue 0 sits at label_seq 5 and residue 1 at label_seq 7; a positional
    join would hand them the values for label_seq 1 and 2.
    """
    cons = _cons(score={5: 9.0, 6: 1.0, 7: 3.0},
                 entropy={5: 0.1, 6: 0.9, 7: 0.5})
    rows = conservation.residue_features(cons, [5, 7], ["ALA", "GLY"])
    assert rows[0]["cons_score"] == 9.0
    assert rows[1]["cons_score"] == 3.0


def test_unmatched_residue_is_nan():
    cons = _cons(score={1: 5.0}, entropy={1: 0.2})
    rows = conservation.residue_features(cons, [99], ["ALA"])
    assert math.isnan(rows[0]["cons_score"])
    assert math.isnan(rows[0]["cons_entropy"])


def test_missing_label_seq_is_nan():
    """Residues absent from the mmCIF sequence have no label_seq at all."""
    cons = _cons(score={1: 5.0})
    rows = conservation.residue_features(cons, [None], ["ALA"])
    assert math.isnan(rows[0]["cons_score"])


def test_no_conservation_available_gives_nan_rows():
    rows = conservation.residue_features(None, [1, 2, 3], ["ALA"] * 3)
    assert len(rows) == 3
    assert all(math.isnan(r["cons_score"]) for r in rows)


def test_prob_self_uses_the_observed_residue():
    probs = {a: 0.01 for a in conservation.AMINO_ACIDS}
    probs["W"] = 0.8
    cons = _cons(score={1: 1.0}, entropy={1: 0.3}, probs={1: probs})
    assert conservation.residue_features(cons, [1], ["TRP"])[0]["cons_prob_self"] \
        == pytest.approx(0.8)
    assert conservation.residue_features(cons, [1], ["ALA"])[0]["cons_prob_self"] \
        == pytest.approx(0.01)


# ---------------------------------------------------------------- envelope
def test_unwrap_handles_all_observed_shapes():
    """The API returned three different envelopes across calls."""
    record = {"data": {"index": [1], "conservation_score": [1.0]}}
    assert conservation._unwrap({"1abc": record}, "1abc") == record
    assert conservation._unwrap({"1abc": [record]}, "1abc") == record
    assert conservation._unwrap(record, "1abc") == record
    assert conservation._unwrap(None, "1abc") is None


# ---------------------------------------------------------------- neighbourhood
def test_neighbourhood_mean_ignores_nan_neighbours():
    """A NaN neighbour must not poison the patch average."""
    rows = [
        {"cons_score": 1.0, "cons_entropy": 0.1, "cons_prob_self": 0.5},
        {"cons_score": 3.0, "cons_entropy": 0.3, "cons_prob_self": 0.5},
        {"cons_score": float("nan"), "cons_entropy": float("nan"),
         "cons_prob_self": float("nan")},
    ]
    centroids = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    conservation.add_neighbourhood(rows, centroids)
    # residue 0's neighbours are 1 (3.0) and 2 (NaN) -> mean of the finite one
    assert rows[0]["cons_score_nb8"] == pytest.approx(3.0)


def test_neighbourhood_all_nan_stays_nan():
    rows = [{"cons_score": float("nan"), "cons_entropy": float("nan"),
             "cons_prob_self": float("nan")} for _ in range(2)]
    centroids = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    conservation.add_neighbourhood(rows, centroids)
    assert math.isnan(rows[0]["cons_score_nb8"])
