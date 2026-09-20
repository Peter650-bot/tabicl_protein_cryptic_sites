"""The TabICL constructor knobs surfaced on `CrypticSitePredictor`.

The point of these tests is that surfacing a knob must not change what the model
does. If a tabicl upgrade moves one of its defaults, the first test fails loudly
rather than silently re-measuring the headline.
"""
from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from cryptic_tabicl import config
from cryptic_tabicl.model import CrypticSitePredictor

# The only two places this project deliberately differs from the library.
DELIBERATE = {"batch_size": 2, "random_state": 0}


def _library_defaults():
    from tabicl import TabICLClassifier

    return {
        name: p.default
        for name, p in inspect.signature(TabICLClassifier.__init__).parameters.items()
        if name != "self"
    }


def _context():
    return pd.DataFrame({config.LABEL_COL: [0, 1, 0, 1], "sasa": [1.0, 2.0, 3.0, 4.0]})


class _Recorder:
    """Stands in for TabICLClassifier and remembers how it was constructed."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def fit(self, X, y):
        return self


def test_surfaced_defaults_match_library_defaults():
    lib = _library_defaults()
    forwarded = CrypticSitePredictor().tabicl_kwargs()
    for name, value in forwarded.items():
        assert name in lib, f"{name} is not a TabICLClassifier argument"
        if name in DELIBERATE:
            assert value == DELIBERATE[name]
            continue
        assert value == lib[name], (
            f"{name}: wrapper default {value!r} != library default {lib[name]!r}. "
            "Either the library changed or the wrapper drifted; both silently "
            "re-measure every published number."
        )


def test_fit_forwards_every_knob(monkeypatch):
    monkeypatch.setattr("tabicl.TabICLClassifier", _Recorder)
    clf = CrypticSitePredictor(
        n_estimators=32,
        norm_methods=["none", "power", "quantile", "robust"],
        outlier_threshold=8.0,
        average_logits=False,
        kv_cache="repr",
    )
    clf.fit(_context())
    got = _Recorder.last_kwargs
    assert got["n_estimators"] == 32
    assert got["norm_methods"] == ["none", "power", "quantile", "robust"]
    assert got["outlier_threshold"] == 8.0
    assert got["average_logits"] is False
    assert got["kv_cache"] == "repr"
    # untouched knobs still ride along at their library defaults
    assert got["feat_shuffle_method"] == "latin"
    assert got["offload_mode"] == "auto"
    assert got == clf.tabicl_kwargs()


def test_default_predictor_forwards_the_published_configuration(monkeypatch):
    monkeypatch.setattr("tabicl.TabICLClassifier", _Recorder)
    CrypticSitePredictor().fit(_context())
    assert _Recorder.last_kwargs == {
        "n_estimators": 8,
        "batch_size": 2,
        "device": None,
        "random_state": 0,
        "verbose": False,
        "model_path": None,
        "norm_methods": None,
        "feat_shuffle_method": "latin",
        "class_shuffle_method": "shift",
        "outlier_threshold": 4.0,
        "softmax_temperature": 0.9,
        "average_logits": True,
        "kv_cache": False,
        "offload_mode": "auto",
    }


@pytest.mark.parametrize("bad", ["quantil", "yeo-johnson", ""])
def test_invalid_norm_method_is_rejected_early(bad):
    with pytest.raises(ValueError, match="unknown norm_methods"):
        CrypticSitePredictor(norm_methods=[bad]).fit(_context())


def test_invalid_shuffle_method_is_rejected_early():
    with pytest.raises(ValueError, match="unknown feat_shuffle_method"):
        CrypticSitePredictor(feat_shuffle_method="latin-square").fit(_context())
    with pytest.raises(ValueError, match="unknown class_shuffle_method"):
        CrypticSitePredictor(class_shuffle_method="rotate").fit(_context())


def test_a_single_norm_method_string_is_accepted(monkeypatch):
    monkeypatch.setattr("tabicl.TabICLClassifier", _Recorder)
    CrypticSitePredictor(norm_methods="power").fit(_context())
    assert _Recorder.last_kwargs["norm_methods"] == "power"


def test_config_is_json_serialisable():
    cfg = CrypticSitePredictor(norm_methods=("none", "power")).config()
    assert json.loads(json.dumps(cfg))["norm_methods"] == ["none", "power"]
