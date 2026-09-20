"""The knob sweep grid, in version control rather than a CLI string.

WHY A MODULE AND NOT A FLAG
    Pre-registration. The set of configurations, the screening fold and the
    seed are all fixed here before anything runs, so "we tried nine things and
    kept the best" is auditable rather than asserted. Adding a config after
    seeing results means a commit that says so.

WHAT IS BEING TESTED
    `cryptic_tabicl/model.py` pinned 16 of TabICLClassifier's 22 arguments at
    library defaults that were never checked against this feature set. These
    are the ones that can plausibly move a 117-feature, 5.3%-prevalence,
    50k-row problem.

SCREENING FOLD, NOT TEST
    Every config is scored on CV fold 0 only. The test split is touched once, by
    the single finalist. Ranking the nine against test and keeping the winner
    would be nine looks at the test set.

SEED
    Held at 0 throughout. The context draw is then identical across configs, so
    each comparison is PAIRED and the 0.0015 AUPRC context-draw noise cancels
    rather than swamping the effect.
"""
from __future__ import annotations

# The fold every config is screened on. Fold 0's control scores already exist in
# the fanout volume, so the control job here doubles as a determinism gate.
SCREEN_FOLD = "train-fold-0"
CONFIRM_FOLDS = ["train-fold-1", "train-fold-2", "train-fold-3"]
SEED = 0

#: name -> kwargs for CrypticSitePredictor. `control` must stay empty: it is the
#: paired baseline AND the check that surfacing the knobs changed nothing.
CONFIGS: dict[str, dict] = {
    "control": {},

    # -- ensemble diversity. TabICL pairs feature-shuffles with preprocessors to
    # build its members, then truncates the list to n_estimators. Widening the
    # preprocessor pool therefore costs NOTHING: still 8 members, just more
    # varied ones. Highest expected value per dollar in the grid.
    "norm4": {"norm_methods": ["none", "power", "quantile", "robust"]},
    "norm5": {"norm_methods": ["none", "power", "quantile", "quantile_rtdl",
                               "robust"]},

    # -- more members. The paper's own TALENT benchmark used 32; its default of
    # 8 is described as unlocking "most" of the benefit, which leaves some.
    "est32": {"n_estimators": 32},
    "est32_norm5": {"n_estimators": 32,
                    "norm_methods": ["none", "power", "quantile",
                                     "quantile_rtdl", "robust"]},

    # -- 'latin' is a systematic Latin-square permutation; with 117 features it
    # may cover the feature space less well than independent random draws.
    "shuffle_random": {"feat_shuffle_method": "random"},

    # -- the suspicious one. TabICL clips every feature at |z| > 4. Sixty-four of
    # the 117 features are ESM-2 principal components, which are heavy-tailed,
    # so the default may be clipping real signal off the tails.
    "outlier8": {"outlier_threshold": 8.0},

    # -- the only averaging setting that can change the RANKING at all. With
    # average_logits=True the temperature is rank-preserving, so AUC/AUPRC are
    # invariant to it; with False, members are combined in probability space and
    # the ordering can move.
    "avg_probs": {"average_logits": False},

    # -- speed only. Expected |dAUPRC| ~ 0; any movement is a numerical bug.
    "batch8": {"batch_size": 8},
    "kv_repr": {"kv_cache": "repr"},
}

#: Configs judged on wall-clock, not accuracy. Accept iff AUPRC does not move.
SPEED_ONLY = {"batch8", "kv_repr"}

#: Promote the best accuracy config only if it clears the single-draw noise band
#: (RESULTS.md 4.6: AUPRC sd = 0.0015 over 12 context draws).
MIN_DELTA_AUPRC = 0.003
