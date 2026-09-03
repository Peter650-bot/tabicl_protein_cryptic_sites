"""The invariant that makes this project's numbers meaningful.

A cryptic site is DEFINED by comparing apo to holo. So any quantity computed
from the holo structure reconstructs the label almost perfectly, and a model
using one would post excellent metrics while being useless on a real apo
structure -- where no holo exists. These tests are the tripwire.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config
from cryptic_tabicl.model import feature_matrix


def _fake_table(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "apo_id": ["1a4uB"] * n,
        "pdb_id": ["1a4u"] * n,
        "chain": ["B"] * n,
        "seq_id": np.arange(n),
        "icode": [""] * n,
        "resname": rng.choice(list(config.MAX_ASA), n),
        "ss": rng.choice(["H", "E", "C"], n),
        "rsa": rng.random(n),
        "protrusion": rng.random(n),
        "in_noncryptic_pocket": rng.integers(0, 2, n),
        "fold": ["train-fold-0"] * n,
        "split": ["train"] * n,
        config.LABEL_COL: rng.integers(0, 2, n),
    })


def test_identifiers_never_reach_the_model():
    """apo_id would let the model memorise proteins instead of learning biology."""
    X = feature_matrix(_fake_table())
    for forbidden in ("apo_id", "pdb_id", "chain", "seq_id", "icode",
                      "fold", "split", config.LABEL_COL):
        assert forbidden not in X.columns, f"{forbidden} leaked into X"


def test_no_holo_derived_column_survives():
    """Any column named after a holo-derived quantity must be caught."""
    from cryptic_tabicl.build_table import validate

    t = _fake_table()
    t["holo_druggability"] = 1.0        # the exact mistake the first plan made
    t["pRMSD"] = 2.0
    problems = validate(t)
    joined = " ".join(problems).lower()
    assert "holo_druggability" in joined
    assert "prmsd" in joined


def test_clean_table_passes_validation():
    from cryptic_tabicl.build_table import validate
    assert validate(_fake_table()) == []


def test_all_nan_column_is_rejected():
    """TabICL's predict_proba crashes on an entirely-NaN column."""
    from cryptic_tabicl.build_table import validate

    t = _fake_table()
    t["dead_feature"] = np.nan
    assert any("entirely NaN" in p for p in validate(t))


def test_split_overlap_is_rejected():
    """One protein appearing in both train and test invalidates every metric."""
    from cryptic_tabicl.build_table import validate

    t = _fake_table()
    t.loc[t.index[:10], "split"] = "test"      # same apo_id, two splits
    assert any("more than one split" in p for p in validate(t))


def test_categoricals_typed_for_tabicl():
    X = feature_matrix(_fake_table())
    assert str(X["resname"].dtype) == "category"
    assert str(X["ss"].dtype) == "category"


def test_no_float16():
    """TabICL does not support float16 arrays."""
    t = _fake_table()
    t["rsa"] = t["rsa"].astype("float16")
    X = feature_matrix(t)
    assert X["rsa"].dtype != np.float16
