"""Cross-row pooling: the conventions it has to match, and the edges it must survive.

The published `_nb8` columns were produced by three separate private copies of
the same loop. `derive.spatial_pool` replaces all three, so the thing worth
testing is not that it computes a mean -- it is that it computes THE SAME mean,
including in the two cases where the copies disagree with each other:
NaN neighbours, and a residue with no neighbour at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import derive  # noqa: E402


def _line(n=5, spacing=5.0):
    """n residues on a line. At radius 8 each has only its immediate neighbours."""
    return np.column_stack([np.arange(n) * spacing, np.zeros(n), np.zeros(n)])


def test_spatial_pool_matches_a_hand_computed_neighbourhood():
    coords = _line(5)
    coords[4, 0] = 100.0                      # residue 4 is isolated
    vals = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    g = np.array(["p"] * 5)
    got = derive.spatial_pool(vals, coords, g, 8.0, "mean")
    # 0 -> {1}; 1 -> {0,2}; 2 -> {1,3}; 3 -> {2}; 4 -> isolated, keeps its own
    assert got == pytest.approx([2.0, 2.0, 3.0, 3.0, 9.0])


def test_the_residue_is_excluded_from_its_own_neighbourhood():
    coords = _line(3)
    vals = np.array([0.0, 100.0, 0.0])
    g = np.array(["p"] * 3)
    got = derive.spatial_pool(vals, coords, g, 8.0, "mean")
    assert got[1] == pytest.approx(0.0), "residue 1 pooled its own 100 into its mean"


def test_isolated_residue_falls_back_to_its_own_value():
    """`features.py:287`'s `or [i]`: no neighbour means the residue itself."""
    coords = np.array([[0.0, 0, 0], [500.0, 0, 0]])
    vals = np.array([7.0, 11.0])
    got = derive.spatial_pool(vals, coords, np.array(["p", "p"]), 8.0, "mean")
    assert got == pytest.approx([7.0, 11.0])


def test_nan_neighbours_are_dropped_not_propagated():
    coords = _line(3)
    vals = np.array([1.0, np.nan, 3.0])
    got = derive.spatial_pool(vals, coords, np.array(["p"] * 3), 8.0, "mean")
    assert got[1] == pytest.approx(2.0), "a NaN residue still pools its neighbours"
    assert np.isnan(got[0]), "residue 0's only neighbour is NaN, so its pool is empty"


def test_an_all_nan_neighbourhood_is_nan_not_the_residues_own_value():
    """The distinction the published columns make, worth three rows of cons_score_nb8.

    The fallback in `conservation.py:229-231` is applied to the INDEX SET, before
    the finite filter. So "I have neighbours, they are all NaN" is NaN, while
    "I have no neighbours" is the residue's own value. Collapsing the two would
    quietly invent data.
    """
    coords = _line(2)
    vals = np.array([5.0, np.nan])
    got = derive.spatial_pool(vals, coords, np.array(["p", "p"]), 8.0, "mean")
    assert np.isnan(got[0]), "residue 0 has a neighbour; that it is NaN is not licence"


def test_pooling_never_crosses_proteins():
    coords = np.vstack([_line(2), _line(2)])   # identical coordinates
    vals = np.array([1.0, 1.0, 99.0, 99.0])
    g = np.array(["a", "a", "b", "b"])
    got = derive.spatial_pool(vals, coords, g, 8.0, "mean")
    assert got == pytest.approx([1.0, 1.0, 99.0, 99.0])


@pytest.mark.parametrize("stat,expected", [
    ("mean", [2.0, 2.0, 3.0, 3.0]),
    ("max", [2.0, 3.0, 4.0, 3.0]),
    ("std", [0.0, 1.0, 1.0, 0.0]),
])
def test_each_statistic(stat, expected):
    coords = _line(4)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    got = derive.spatial_pool(vals, coords, np.array(["p"] * 4), 8.0, stat)
    assert got == pytest.approx(expected)


def test_multi_column_pooling_equals_column_by_column():
    rng = np.random.default_rng(0)
    n = 40
    coords = rng.normal(scale=6.0, size=(n, 3))
    vals = rng.normal(size=(n, 4))
    vals[rng.random((n, 4)) < 0.1] = np.nan
    g = np.where(np.arange(n) < 25, "a", "b")
    both = derive.spatial_pool(vals, coords, g, 8.0, "mean")
    for c in range(4):
        one = derive.spatial_pool(vals[:, c], coords, g, 8.0, "mean")
        assert np.allclose(both[:, c], one, equal_nan=True)


def test_unknown_statistic_is_rejected():
    with pytest.raises(ValueError, match="unknown stat"):
        derive.spatial_pool(np.zeros(2), _line(2), np.array(["p"] * 2), 8.0, "median")


def test_non_finite_coordinates_are_rejected():
    coords = _line(2)
    coords[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        derive.spatial_pool(np.zeros(2), coords, np.array(["p"] * 2), 8.0, "mean")


# ------------------------------------------------------------ sequence window
def test_sequence_window_never_crosses_a_chain_boundary():
    vals = np.array([1.0, 2.0, 3.0, 40.0, 50.0])
    key = np.array(["A", "A", "A", "B", "B"])
    got = derive.sequence_window(vals, key, np.arange(5), 3)
    assert got == pytest.approx([1.5, 2.0, 2.5, 45.0, 45.0])


def test_sequence_window_respects_the_order_it_is_given():
    """Rows arrive in table order, which need not be backbone order."""
    vals = np.array([3.0, 1.0, 2.0])
    key = np.array(["A"] * 3)
    got = derive.sequence_window(vals, key, np.array([2, 0, 1]), 3)
    # backbone order is 1.0, 2.0, 3.0 -> windows 1.5, 2.0, 2.5
    assert got == pytest.approx([2.5, 1.5, 2.0])


def test_sequence_window_skips_nan_instead_of_erasing_the_row():
    vals = np.array([1.0, np.nan, 3.0])
    got = derive.sequence_window(vals, np.array(["A"] * 3), np.arange(3), 3)
    assert got == pytest.approx([1.0, 2.0, 3.0])


def test_even_window_width_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        derive.sequence_window(np.zeros(3), np.array(["A"] * 3), np.arange(3), 4)


# ------------------------------------------------------------ within protein
def test_within_protein_rank_is_a_fraction_of_the_protein():
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    got = derive.within_protein(vals, np.array(["p"] * 4), "rank")
    assert got == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_within_protein_z_of_a_constant_protein_is_zero_not_nan():
    got = derive.within_protein(np.array([2.0, 2.0, 2.0]), np.array(["p"] * 3), "z")
    assert got == pytest.approx([0.0, 0.0, 0.0])


def test_within_protein_of_a_single_residue_protein_is_defined():
    vals = np.array([5.0, 1.0, 2.0])
    g = np.array(["solo", "p", "p"])
    assert derive.within_protein(vals, g, "z")[0] == pytest.approx(0.0)
    assert derive.within_protein(vals, g, "rank")[0] == pytest.approx(1.0)


def test_within_protein_keeps_missing_missing():
    vals = np.array([1.0, np.nan, 3.0])
    for how in ("rank", "z"):
        assert np.isnan(derive.within_protein(vals, np.array(["p"] * 3), how)[1])


def test_within_protein_is_invariant_to_a_shift_of_the_whole_protein():
    rng = np.random.default_rng(1)
    vals = rng.normal(size=20)
    g = np.where(np.arange(20) < 11, "a", "b")
    shifted = vals + np.where(g == "a", 100.0, 0.0)
    for how in ("rank", "z"):
        assert np.allclose(derive.within_protein(vals, g, how),
                           derive.within_protein(shifted, g, how))


# ---------------------------------------------------------- protein context
def test_protein_context_broadcasts_one_value_per_protein():
    t = pd.DataFrame({"apo_id": ["a", "a", "b"], "x": [1.0, 3.0, 7.0]})
    got = derive.protein_context(t, ["x"], stats=("mean", "max"))
    assert got["prot_x_mean"].tolist() == pytest.approx([2.0, 2.0, 7.0])
    assert got["prot_x_max"].tolist() == pytest.approx([3.0, 3.0, 7.0])
