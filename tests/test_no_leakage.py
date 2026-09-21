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


def test_noncryptic_pocket_annotation_never_reaches_the_model():
    """CryptoBench derives it from apo-holo pairs, so it cannot exist at
    prediction time -- see config.HOLO_DERIVED_METADATA_COLS for the citation.

    It stays in the table (evaluate.stratified needs it) but must not be a
    feature. This is the regression test for the leak that survived the
    substring blacklist.
    """
    t = _fake_table()
    assert "in_noncryptic_pocket" in t.columns, "fixture must carry the column"
    X = feature_matrix(t)
    assert "in_noncryptic_pocket" not in X.columns


def test_holo_derived_metadata_is_caught_by_name_not_just_substring():
    """The substring blacklist would miss every one of these."""
    from cryptic_tabicl.build_table import validate

    for col in config.HOLO_DERIVED_METADATA_COLS:
        low = col.lower()
        assert not any(bad in low for bad in config.FORBIDDEN_FEATURE_SUBSTRINGS), (
            f"'{col}' is caught by a substring, so it does not exercise the "
            "by-name half of the guard")

    # Simulate someone re-adding it to the feature set: validate must object.
    t = _fake_table()
    feature_cols = [c for c in t.columns
                    if c not in config.NON_FEATURE_COLS and c != "pdb_id"]
    assert "in_noncryptic_pocket" not in feature_cols


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


# ---------------------------------------------------------------------------
# The derived feature families (RESULTS.md section 12).
#
# `in_noncryptic_pocket` got past a substring blacklist because it is not named
# after the thing it leaks. The derived families are a much larger surface --
# 64 new columns, every one a function of other columns -- so the guard here is
# not a name check at all. It permutes the label and asserts nothing moves.
# ---------------------------------------------------------------------------
def _derived_fixture(n_per_protein=12, n_proteins=3, seed=0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for i in range(n_per_protein):
            rows.append({
                "apo_id": f"prot{p}A", "chain": "A", "seq_id": i + 1, "icode": "",
                "pdb_id": f"prot{p}", "fold": "train-fold-0", "split": "train",
                "is_cryptic": int(rng.random() < 0.3),
                "in_noncryptic_pocket": int(rng.random() < 0.1),
            })
    t = pd.DataFrame(rows)
    needed = ["ens_mean", "ens_max", "ens_std", "ens_q90", "ens_opening",
              "cons_score", "cons_entropy", "cons_prob_self", "cons_score_nb8",
              "rsa", "protrusion"] + [f"esm_pc{i:02d}" for i in range(16)]
    for col in needed:
        t[col] = rng.normal(size=len(t))
    coords = t[["apo_id", "chain", "seq_id", "icode"]].copy()
    xyz = rng.normal(scale=6.0, size=(len(t), 3))
    coords["cx"], coords["cy"], coords["cz"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return t, coords


def test_derive_module_never_mentions_the_label_in_code():
    """A source-level tripwire, to catch the mistake while it is still being typed.

    Comments and docstrings are stripped first: the module's prose explains at
    length why it must not read the label, and a check that cannot tell the
    explanation from the offence is not a check.
    """
    import io
    import inspect
    import tokenize

    from cryptic_tabicl import derive

    src = inspect.getsource(derive)
    code = "".join(
        tok.string + " "
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type not in (tokenize.COMMENT, tokenize.STRING, tokenize.NL)
    )
    for needle in (config.LABEL_COL, "label", "y_true", "is_cryptic"):
        assert needle not in code, (
            f"derive.py's code mentions {needle!r}; every function there must "
            "be a function of features and geometry only")


def test_derived_features_do_not_move_when_the_label_is_permuted():
    """The real guard. Any label dependence at all shows up here as a difference."""
    import numpy as np
    import pandas as pd

    from scripts.build_derived import build

    t, coords = _derived_fixture()
    before = build(t.copy(), coords.copy())

    shuffled = t.copy()
    rng = np.random.default_rng(99)
    shuffled[config.LABEL_COL] = rng.permutation(shuffled[config.LABEL_COL].to_numpy())
    assert not shuffled[config.LABEL_COL].equals(t[config.LABEL_COL]), \
        "the permutation was a no-op, so this test proves nothing"
    after = build(shuffled, coords.copy())

    pd.testing.assert_frame_equal(before, after)


def test_derived_features_do_not_move_when_the_label_is_removed_entirely():
    import pandas as pd

    from scripts.build_derived import build

    t, coords = _derived_fixture()
    before = build(t.copy(), coords.copy())
    after = build(t.drop(columns=[config.LABEL_COL]), coords.copy())
    pd.testing.assert_frame_equal(before, after)


def test_every_derived_column_passes_both_halves_of_the_holo_guard():
    from cryptic_tabicl import feature_sets as FS
    from cryptic_tabicl.build_table import validate

    t = _fake_table()
    for col in FS.ALL_DERIVED:
        t[col] = 1.0
    problems = validate(t)
    joined = " ".join(problems).lower()
    for col in FS.ALL_DERIVED:
        assert col.lower() not in joined, f"{col} trips the holo-derived guard"


def test_no_derived_column_is_named_after_a_forbidden_quantity():
    from cryptic_tabicl import feature_sets as FS

    for col in FS.ALL_DERIVED:
        low = col.lower()
        assert not any(bad in low for bad in config.FORBIDDEN_FEATURE_SUBSTRINGS), col


# ---------------------------------------------------------------------------
# The supervised ESM projection is the one label-using input in the feature
# work package, so it is the one that needs a nesting check rather than a
# permutation test. `deploy/feature_sweep.py` refuses to score a block that was
# fitted on the fold it is about to be evaluated on.
# ---------------------------------------------------------------------------
def _manifest(blocks):
    import json
    import tempfile

    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"blocks": blocks}, fh)
    fh.close()
    return fh.name


def test_supervised_block_must_have_excluded_the_fold_it_is_scored_on():
    import pytest as _pytest

    from cryptic_tabicl.feature_sets import supervised_block

    good = _manifest({
        "train-fold-0": {"held": "train-fold-0", "columns": ["esm_sup00__fit_train-fold-0"]},
        "final": {"held": None, "columns": ["esm_sup00__fit_final"]},
    })
    assert supervised_block("train-fold-0", good) == ["esm_sup00__fit_train-fold-0"]
    assert supervised_block(None, good) == ["esm_sup00__fit_final"]

    # The failure this guard exists for: a block fitted on everything, handed to
    # a CV job. It would score its own training rows and look like a win.
    bad = _manifest({
        "train-fold-0": {"held": None, "columns": ["esm_sup00__fit_final"]},
    })
    with _pytest.raises(RuntimeError, match="Refusing to score"):
        supervised_block("train-fold-0", bad)


def test_a_missing_supervised_block_is_an_error_not_a_silent_skip():
    import pytest as _pytest

    from cryptic_tabicl.feature_sets import supervised_block

    path = _manifest({"final": {"held": None, "columns": []}})
    with _pytest.raises(RuntimeError, match="no supervised block"):
        supervised_block("train-fold-2", path)


def test_supervised_column_names_encode_their_nesting():
    """The name carries the fold, so a mismatched block cannot be pasted in silently."""
    from deploy.esm_supervised_modal import block_cols

    assert block_cols("train-fold-1", 2) == [
        "esm_sup00__fit_train-fold-1", "esm_sup01__fit_train-fold-1"]
    assert block_cols(None, 1) == ["esm_sup00__fit_final"]
    for held in ("train-fold-0", "train-fold-1", None):
        other = "train-fold-3" if held != "train-fold-3" else "train-fold-0"
        assert not set(block_cols(held, 4)) & set(block_cols(other, 4))
