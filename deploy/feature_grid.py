"""The pre-registered feature screen. Pure data -- no modal import, no I/O.

This file is committed BEFORE the screen runs, exactly as `deploy/knob_grid.py`
was for the knob sweep. Its job is to make the protocol unfalsifiable after the
fact: which arms, on which fold, at which bar, with which prior stated in
advance.

PROTOCOL
    Every arm is scored on CV fold 0 and on nothing else. Ranking ten arms
    against the test split and keeping the winner is ten looks at the test set.
    The finalist is confirmed on folds 1-3 and scored on test exactly once.

    The context seed is the same for every arm, so the context draw is
    identical and each comparison is PAIRED -- the 0.0015 AUPRC context-draw
    noise (RESULTS.md section 4.6) cancels instead of swamping the effect.

THE BAR
    +0.003 AUPRC, the single-draw noise band from RESULTS.md section 4.6, and
    the same bar the knob sweep used. Set here, before the run. A finalist is
    promoted only if it clears the bar on fold 0; a WIN is declared only if the
    four-fold mean also clears it AND all four fold deltas share a sign.

PRIORS, STATED IN ADVANCE
    Recorded so that a null is read as a null rather than reinterpreted. Two
    arms are entered with an explicitly LOW prior and are expected to fail:
    `prel`, because within-protein normalisation does not add univariate signal
    here, and `trim`, because arXiv 2609.13202 finds feature selection
    consistently harms TabICLv2 and this project's own ladder has 117 features
    beating 85. They are screened because they are cheap, and because reading a
    univariate table and concluding "does not help" is the mistake RESULTS.md
    section 4.3 and section 4.3b record this project making twice, in both
    directions.
"""
from __future__ import annotations

SCREEN_FOLD = "train-fold-0"
CONFIRM_FOLDS = ["train-fold-1", "train-fold-2", "train-fold-3"]
SEED = 0
N_COMPONENTS = 64          # the unsupervised block the control arm keeps

# The promotion bar, in AUPRC, against the control on the same fold.
MIN_DELTA_AUPRC = 0.003

# Arm -> (prior, one-line rationale). The key is the name in
# `cryptic_tabicl.feature_sets.FEATURE_SETS`; `control` must be first.
ARMS: dict[str, tuple[str, str]] = {
    "control": (
        "baseline",
        "the published 117 features; also the determinism gate against "
        "build/cv_scores/k64_train-fold-0.json",
    ),
    "seqwin": (
        "high",
        "5-residue backbone windows on the sequence-derived families; measured "
        "sign-free on train, cons_prob_self 0.625->0.651 and esm_pc01 "
        "0.564->0.616, while ens_mean falls 0.776->0.742 and is excluded",
    ),
    "esmnb": (
        "high",
        "8 A spatial mean of the 16 leading ESM components; the pLM block is "
        "the one family with no spatial pooling, and every pooled column in "
        "the project outranks its parent",
    ),
    "nbstat": (
        "medium",
        "spread and extreme at 8 A plus a 12 A shell; a pocket is a patch, and "
        "the mean is the only shape statistic currently taken of it",
    ),
    "pctx": (
        "medium",
        "protein-level aggregates broadcast to each row; every metric in the "
        "project is per protein, yet a row carries no protein context beyond "
        "chain_len",
    ),
    "prel": (
        "low",
        "within-protein rank and z; expected to fail -- rank costs ens_mean "
        "0.776->0.767 and z 0.776->0.750 univariately",
    ),
    "esmsup32": (
        "high",
        "32 fold-nested PLS components replace the 64 unsupervised PCs; "
        "SMALLER than the control at 85 features",
    ),
    "esmsup32_seqwin_esmnb": (
        "high",
        "the supervised block plus the two pooling families, pre-registered as "
        "a combination so it is not a post-hoc union of whatever won",
    ),
    "free_combo": (
        "medium",
        "seqwin + pctx + prel, the union that costs nothing to build",
    ),
    "trim": (
        "low",
        "drop the 24 ESM PCs at |AUC-0.5| <= 0.02 on train; the "
        "'removing features' arm, expected null or negative",
    ),
}

# Arms whose feature set replaces the unsupervised PCs with a supervised block.
# Kept here as well as in `feature_sets` so the grid is readable on its own.
NEEDS_SUPERVISED = ("esmsup32", "esmsup32_seqwin_esmnb")

# Dropped first if wall-clock or budget runs over. Named in advance so the
# decision is not made by looking at results.
RELEASE_VALVE = ("free_combo", "trim")
