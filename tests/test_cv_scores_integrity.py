"""Provenance of the staged CV scores, and a guard on the published headline.

These skip when `build/` has not been populated, so a clean checkout still runs
green; they are the checks that matter on a machine that HAS the artefacts.

Note `build/` is gitignored, so "has the headline moved?" cannot be answered
with `git diff`. It is answered here instead, by asserting the numbers.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from cryptic_tabicl import config

CV_DIR = config.BUILD_DIR / "cv_scores"
MANIFEST = config.PROJECT_ROOT / "operating_point" / "cv_scores_manifest.json"
TEST_RESULT = config.BUILD_DIR / "modal_result_esm64.json"
FOLDS = ["train-fold-0", "train-fold-1", "train-fold-2", "train-fold-3"]

# The published zero-shot headline (RESULTS.md section 1). Nothing in the
# operating-point or knob work is allowed to move these.
HEADLINE = {"AUC": 0.8618362991984249, "AUPRC": 0.44130501771474306,
            "TPR": 0.5593051200704021, "FPR": 0.07068750309158402,
            "MCC": 0.37396276891577773, "F1_weighted": 0.912141905367498}


def _load(path):
    d = json.loads(path.read_text())
    return d["scores"] if "scores" in d else d


needs_build = pytest.mark.skipif(not TEST_RESULT.exists(),
                                 reason="build/modal_result_esm64.json not present")
needs_cv = pytest.mark.skipif(not (CV_DIR / "k64_final.json").exists(),
                              reason="CV scores not staged; see operating_point/README.md")


@needs_build
def test_published_headline_is_unchanged():
    got = json.loads(TEST_RESULT.read_text())["result"]
    for metric, expected in HEADLINE.items():
        assert got[metric] == pytest.approx(expected, abs=1e-12), (
            f"{metric} moved: {got[metric]} != {expected}. The zero-shot headline "
            "must not change; something re-wrote build/modal_result_esm64.json."
        )


@needs_cv
@needs_build
def test_staged_final_scores_match_the_published_run():
    """The gate: the staged CV bundle is from the run that produced the headline."""
    staged = np.asarray(_load(CV_DIR / "k64_final.json")["score"], dtype=float)
    published = np.asarray(_load(TEST_RESULT)["score"], dtype=float)
    assert staged.shape == published.shape
    assert np.array_equal(staged, published)


@needs_cv
def test_manifest_checksums_match_the_staged_files():
    manifest = json.loads(MANIFEST.read_text())
    for name, meta in manifest["files"].items():
        path = CV_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not staged")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meta["sha256"], f"{name} differs from the manifest"


@needs_cv
def test_cv_folds_are_disjoint_in_proteins():
    seen: dict[str, str] = {}
    for fold in FOLDS:
        for apo in set(_load(CV_DIR / f"k64_{fold}.json")["apo_id"]):
            assert apo not in seen, f"{apo} appears in {seen[apo]} and {fold}"
            seen[apo] = fold
    assert len(seen) > 800


@needs_cv
@needs_build
def test_test_proteins_never_appear_in_a_cv_fold():
    """The leakage check that matters for the operating-point work."""
    test_proteins = set(_load(TEST_RESULT)["apo_id"])
    for fold in FOLDS:
        fold_proteins = set(_load(CV_DIR / f"k64_{fold}.json")["apo_id"])
        assert not (test_proteins & fold_proteins), f"{fold} overlaps the test split"
