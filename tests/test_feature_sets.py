"""The named feature sets: that they are what they say, and that the control is pinned.

The control arm is the load-bearing one. If `control` resolves to anything other
than the exact 117 columns of the published run, every delta in the screen is
measuring two things at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config, feature_sets as FS  # noqa: E402

BUILT = config.BUILD_DIR / "residues_derived.parquet"

# The published feature set, as RESULTS.md section 3 counts it.
PUBLISHED_N_FEATURES = 117


def _fake_table():
    """A table shaped like the merged one, without needing build/ to exist."""
    base = ["resname", "sasa", "rsa", "ss"] + [f"esm_pc{i:02d}" for i in range(64)]
    cols = {c: np.zeros(4) for c in base}
    cols.update({c: np.zeros(4) for c in FS.ALL_DERIVED})
    cols.update({
        "apo_id": ["a", "a", "b", "b"], "chain": list("AABB"),
        "seq_id": [1, 2, 1, 2], "icode": [""] * 4, "pdb_id": ["p"] * 4,
        "is_cryptic": [0, 1, 0, 1], "fold": ["f"] * 4, "split": ["train"] * 4,
        "in_noncryptic_pocket": [0, 0, 0, 0],
    })
    return pd.DataFrame(cols)


def test_pdb_id_is_a_non_feature_column():
    """It used to be excluded by hand at a dozen call sites; one would eventually miss."""
    assert "pdb_id" in config.NON_FEATURE_COLS


def test_base_columns_exclude_every_identifier_and_the_label():
    t = _fake_table()
    base = FS.base_columns(t)
    for col in config.NON_FEATURE_COLS:
        assert col not in base
    assert "in_noncryptic_pocket" not in base, "the holo-derived column is not a feature"


def test_base_columns_exclude_every_derived_family():
    """Adding a family to the parquet must not silently enlarge the control arm."""
    base = FS.base_columns(_fake_table())
    for col in FS.ALL_DERIVED:
        assert col not in base


def test_families_do_not_overlap():
    seen = {}
    for name, cols in FS.FAMILIES.items():
        for c in cols:
            assert c not in seen, f"{c} is in both {seen[c]} and {name}"
            seen[c] = name


def test_every_arm_resolves_and_has_no_duplicates():
    t = _fake_table()
    sup = tuple(f"esm_sup{i:02d}" for i in range(32))
    for c in sup:
        t[c] = 0.0
    for name in FS.FEATURE_SETS:
        want = sup if name in FS.USES_SUPERVISED_ESM else ()
        cols = FS.resolve(name, t, want)
        assert len(cols) == len(set(cols)), f"{name} names a column twice"


def test_supervised_arms_replace_the_unsupervised_components():
    t = _fake_table()
    sup = tuple(f"esm_sup{i:02d}" for i in range(32))
    for c in sup:
        t[c] = 0.0
    cols = FS.resolve("esmsup32", t, sup)
    assert not any(c.startswith("esm_pc") for c in cols), \
        "the supervised block is a replacement, not an addition"
    assert set(sup) <= set(cols)


def test_supervised_columns_must_match_the_arm():
    """Passing a supervised block to an arm that does not use one is a grid bug."""
    t = _fake_table()
    with pytest.raises(ValueError, match="does not use"):
        FS.resolve("control", t, ("esm_sup00",))
    with pytest.raises(ValueError, match="requires"):
        FS.resolve("esmsup32", t, ())


def test_unknown_arm_raises():
    with pytest.raises(KeyError, match="unknown feature set"):
        FS.resolve("no_such_arm", _fake_table())


def test_an_arm_naming_a_missing_column_raises_rather_than_narrowing():
    t = _fake_table().drop(columns=[FS.SEQWIN[0]])
    with pytest.raises(KeyError, match="does not have"):
        FS.resolve("seqwin", t)


def test_trim_names_real_components_and_removes_them():
    t = _fake_table()
    assert set(FS.TRIM_WEAK_PCS) <= set(FS.ALL_PCS)
    cols = FS.resolve("trim", t)
    assert not (set(FS.TRIM_WEAK_PCS) & set(cols))
    assert len(cols) == len(FS.base_columns(t)) - len(FS.TRIM_WEAK_PCS)


@pytest.mark.skipif(not BUILT.exists(), reason="merged table not built")
def test_control_is_exactly_the_published_feature_set():
    t = pd.read_parquet(BUILT)
    assert len(FS.resolve("control", t)) == PUBLISHED_N_FEATURES


@pytest.mark.skipif(not BUILT.exists(), reason="merged table not built")
def test_every_arm_resolves_against_the_real_table():
    t = pd.read_parquet(BUILT)
    sup = sorted(c for c in t.columns if c.startswith(FS.SUPERVISED_PREFIX))
    for name in FS.FEATURE_SETS:
        if name in FS.USES_SUPERVISED_ESM and not sup:
            continue
        want = sup if name in FS.USES_SUPERVISED_ESM else ()
        assert FS.resolve(name, t, want)


def test_the_context_draw_does_not_depend_on_which_features_are_present():
    """The assumption the entire paired screen rests on.

    Every arm is scored at the same seed so the comparison is paired and the
    0.0015 AUPRC context-draw noise cancels. That is only true if adding or
    removing feature COLUMNS cannot change which ROWS `build_context` draws.
    It samples on label and protein, so it does not -- but if it ever started
    to, every delta in the screen would silently become a mixture of a feature
    effect and a different context, and nothing in the output would say so.
    """
    import numpy as np

    from cryptic_tabicl import sampling

    rng = np.random.default_rng(0)
    n = 600
    base = pd.DataFrame({
        "apo_id": [f"p{i % 20}" for i in range(n)],
        "chain": ["A"] * n,
        "seq_id": np.arange(n),
        "icode": [""] * n,
        "fold": [f"train-fold-{i % 4}" for i in range(n)],
        "split": ["train"] * n,
        "is_cryptic": (rng.random(n) < 0.1).astype(int),
        "in_noncryptic_pocket": (rng.random(n) < 0.05).astype(int),
        "keeper": rng.normal(size=n),
    })
    wide = base.assign(**{f"extra{i}": rng.normal(size=n) for i in range(40)})

    keys = ["apo_id", "chain", "seq_id", "icode"]
    for held in (None, "train-fold-0"):
        a = base if held is None else base[base["fold"] != held]
        b = wide if held is None else wide[wide["fold"] != held]
        ca, _ = sampling.build_context(a, budget=200, seed=0)
        cb, _ = sampling.build_context(b, budget=200, seed=0)
        pd.testing.assert_frame_equal(
            ca[keys].reset_index(drop=True), cb[keys].reset_index(drop=True))
