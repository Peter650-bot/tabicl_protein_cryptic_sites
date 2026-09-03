"""Context construction and annotation handling.

The sampler decides what TabICL ever sees, so a bug here is invisible: the model
simply learns from a worse table. These tests pin the properties that make the
context defensible -- every positive kept, no protein allowed to dominate, and
hard negatives actually present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config, dataset, sampling


# ---------------------------------------------------------------- annotations
def test_labels_union_over_all_holo_partners():
    """One apo maps to ~5 holo on average; taking one would mislabel positives."""
    ds = {"1a4u": [
        {"apo_chain": "B", "apo_pocket_selection": ["B_12", "B_14"]},
        {"apo_chain": "B", "apo_pocket_selection": ["B_14", "B_99"]},
    ]}
    assert dataset.apo_binding_residues(ds)["1a4uB"] == {"B_12", "B_14", "B_99"}


def test_binding_site_spanning_two_chains_pulls_both():
    """161 CryptoBench apo structures have multi-chain sites."""
    ds = {"1xyz": [{"apo_chain": "A", "apo_pocket_selection": ["A_5", "C_7"]}]}
    assert dataset.apo_chains(ds)["1xyzA"] == {"A", "C"}


def test_apo_identity_is_pdb_plus_chain():
    """Two chains of one entry are different structures with different labels."""
    ds = {"1a4u": [
        {"apo_chain": "A", "apo_pocket_selection": ["A_1"]},
        {"apo_chain": "B", "apo_pocket_selection": ["B_2"]},
    ]}
    lab = dataset.apo_binding_residues(ds)
    assert lab["1a4uA"] == {"A_1"} and lab["1a4uB"] == {"B_2"}


def test_fold_assignment_rejects_overlap():
    """A protein in two folds silently invalidates cross-validation."""
    folds = {
        "train-fold-0": {"1abc": [{"apo_chain": "A", "apo_pocket_selection": ["A_1"]}]},
        "train-fold-1": {"1abc": [{"apo_chain": "A", "apo_pocket_selection": ["A_1"]}]},
    }
    with pytest.raises(ValueError, match="not disjoint"):
        dataset.fold_assignment(folds)


# ---------------------------------------------------------------- sampling
def _table(n_prot=10, per_prot=300, pos_rate=0.05, hard_rate=0.15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prot):
        for i in range(per_prot):
            rows.append({
                config.GROUP_COL: f"prot{p}",
                config.LABEL_COL: int(rng.random() < pos_rate),
                "in_noncryptic_pocket": int(rng.random() < hard_rate),
                "rsa": rng.random(),
            })
    return pd.DataFrame(rows)


def test_every_positive_is_kept():
    """Positives are ~5% of rows and are the entire signal."""
    t = _table()
    ctx, plan = sampling.build_context(t, budget=50_000)
    assert plan.n_positive == int(t[config.LABEL_COL].sum())
    assert ctx[config.LABEL_COL].sum() == t[config.LABEL_COL].sum()


def test_negative_ratio_is_respected():
    t = _table()
    ctx, plan = sampling.build_context(t, budget=50_000, negative_ratio=8)
    assert plan.n_negative <= plan.n_positive * 8


def test_per_protein_cap_limits_domination():
    """Without a cap, context composition tracks protein size, not biology."""
    t = pd.concat([_table(n_prot=2, per_prot=2000, seed=1),
                   _table(n_prot=8, per_prot=100, seed=2)])
    ctx, _ = sampling.build_context(t, budget=50_000, max_negatives_per_protein=50)
    neg = ctx[ctx[config.LABEL_COL] == 0]
    assert neg.groupby(config.GROUP_COL).size().max() <= 50


def test_pocket_lining_negatives_are_not_over_represented():
    """Negatives must occur at their natural rate, not a forced quota.

    Forcing 40% of the negative budget to come from residues lining an
    already-open pocket cost 0.087 AUPRC: only ~17k residues are in that
    stratum, so the quota over-samples it by an order of magnitude and distorts
    the distribution the model ranks against. This pins the default off.
    """
    t = _table(hard_rate=0.15)
    ctx, plan = sampling.build_context(t, budget=50_000)
    neg = ctx[ctx[config.LABEL_COL] == 0]
    observed = plan.n_hard_negative / max(len(neg), 1)
    natural = t.loc[t[config.LABEL_COL] == 0, "in_noncryptic_pocket"].mean()
    assert abs(observed - natural) < 0.05, (
        f"pocket-lining negatives at {observed:.2f} vs natural {natural:.2f}"
    )


def test_forced_share_is_off_by_default():
    """A regression guard on the config value itself."""
    assert config.HARD_NEGATIVE_SHARE == 0.0


def test_context_fits_budget():
    t = _table(n_prot=60, per_prot=1000)
    ctx, _ = sampling.build_context(t, budget=5_000, negative_ratio=8)
    n_pos = int(t[config.LABEL_COL].sum())
    # positives are never dropped, so the bound is budget or all-positives
    assert len(ctx) <= max(5_000, n_pos) + 1


def test_sampling_is_deterministic():
    t = _table()
    a, _ = sampling.build_context(t, seed=7)
    b, _ = sampling.build_context(t, seed=7)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- evaluation
def test_threshold_selection_respects_target_fpr():
    from cryptic_tabicl import evaluate

    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    s = np.clip(y * 0.5 + rng.normal(0.3, 0.2, 5000), 0, 1)
    thr = evaluate.choose_threshold(y, s, target_fpr=0.05)
    achieved = ((s >= thr) & (y == 0)).sum() / (y == 0).sum()
    assert achieved <= 0.05 + 1e-9


def test_perfect_scores_give_auc_one():
    from cryptic_tabicl import evaluate

    y = np.array([0, 0, 1, 1] * 10)
    s = y.astype(float)
    g = np.array(["p"] * 40)
    res = evaluate.summarise(y, s, g, threshold=0.5)
    assert res["AUC"] == pytest.approx(1.0)
    assert res["TPR"] == pytest.approx(1.0)
    assert res["FPR"] == pytest.approx(0.0)
