"""The per-protein decision rules, and the invariance the whole idea rests on."""
from __future__ import annotations

import numpy as np
import pytest

from cryptic_tabicl import evaluate, operating_point as op


def _toy(seed=0, sizes=(40, 17, 300, 5)):
    """Distinct scores per residue, so no tie-handling ambiguity."""
    rng = np.random.default_rng(seed)
    y, s, g = [], [], []
    for i, n in enumerate(sizes):
        gid = f"prot{i}"
        scores = rng.permutation(np.linspace(0.01, 0.99, n))
        labels = (rng.random(n) < 0.2).astype(int)
        labels[0], labels[1] = 1, 0  # guarantee both classes present
        y.extend(labels.tolist())
        s.extend(scores.tolist())
        g.extend([gid] * n)
    return np.array(y), np.array(s), np.array(g)


def test_rank_flags_floor_q_n_per_protein():
    y, s, g = _toy()
    t = op.rank(s, g)
    for q in (0.05, 0.08, 0.10, 0.25):
        threshold = 1.0 - q
        for gid in np.unique(g):
            m = g == gid
            n = int(m.sum())
            flagged = int((t[m] >= threshold).sum())
            assert flagged == int(np.floor(n * q + 1e-9)), (gid, n, q, flagged)


def test_rank_is_scale_free_across_proteins():
    """Shifting one protein's scores cannot change which of its residues are flagged."""
    y, s, g = _toy()
    shifted = s.copy()
    shifted[g == "prot2"] = shifted[g == "prot2"] * 0.1 + 0.005
    before = op.rank(s, g) >= 0.92
    after = op.rank(shifted, g) >= 0.92
    assert np.array_equal(before, after)
    # the incumbent rule is NOT scale-free -- this is the whole point
    assert not np.array_equal(s >= 0.5, shifted >= 0.5)


def test_rank_ties_are_admitted_together():
    y = np.array([1, 0, 0, 1])
    s = np.array([0.7, 0.7, 0.2, 0.1])
    g = np.array(["a"] * 4)
    t = op.rank(s, g)
    assert t[0] == t[1]  # tied scores share a rank


def test_per_protein_auc_is_invariant_under_every_transform():
    """The load-bearing claim: the headline cannot move."""
    y, s, g = _toy(seed=3)
    base = evaluate.per_protein(y, s, g, threshold=0.5).set_index("apo_id")
    for rule in ("rank", "zscore"):
        t = op.transform(rule, s, g)
        got = evaluate.per_protein(y, t, g, threshold=0.5).set_index("apo_id")
        np.testing.assert_allclose(got["AUC"].to_numpy(), base["AUC"].to_numpy())
        np.testing.assert_allclose(got["AUPRC"].to_numpy(), base["AUPRC"].to_numpy())


def test_choose_threshold_on_a_transform_hits_the_target():
    y, s, g = _toy(seed=5)
    for rule in ("global_threshold", "rank", "zscore"):
        t = op.transform(rule, s, g)
        thr = evaluate.choose_threshold(y, t, g, target_fpr=0.05)
        assert evaluate.mean_per_protein_fpr(y, t, g, thr) <= 0.05 + 1e-12


def test_zscore_of_a_constant_protein_is_zero_not_nan():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.4, 0.4, 0.9, 0.1])
    g = np.array(["flat", "flat", "b", "b"])
    z = op.zscore(s, g)
    assert np.isfinite(z).all()
    assert (z[g == "flat"] == 0.0).all()


def test_single_residue_protein_is_finite():
    y = np.array([1, 0, 1])
    s = np.array([0.8, 0.3, 0.6])
    g = np.array(["solo", "pair", "pair"])
    for rule in ("rank", "zscore"):
        out = op.transform(rule, s, g)
        assert np.isfinite(out).all()


def test_transform_rejects_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        op.transform("nope", [0.1], ["a"])


def test_top_q_round_trip():
    assert op.top_q_from_threshold(0.92) == pytest.approx(0.08)
    assert "top 8.00%" in op.describe("rank", 0.92)
