"""Choosing the in-context training set under TabICL's context budget.

WHY THIS EXISTS
    TabICL is in-context: the "training" table is fed through the network in a
    forward pass, so its size is a memory cost at *inference* time, every time.
    ~300k residues will not fit in 5 GB, and TabICL v2 is pretrained for
    contexts up to roughly 60k rows. Subsampling is therefore part of the
    method, not a shortcut.

WHAT WE KEEP
    Every positive. Positives are ~6% of rows and are the entire signal; there
    is no reason to discard any.

WHICH NEGATIVES -- MEASURED, NOT ASSUMED
    An earlier version of this module reserved 40% of the negative budget for
    residues lining already-open pockets, on the reasoning that those are the
    hard negatives and the decision boundary runs through them. That reasoning
    was wrong, and expensively so. Isolating each choice with logistic
    regression at a fixed ~49k context:

        capped, no forced share                  AUPRC 0.420
        uncapped, no forced share                AUPRC 0.417
        uncapped, 40% forced pocket share        AUPRC 0.317
        capped + 40% forced share (old default)  AUPRC 0.333
        all 288k negatives (the target)          AUPRC 0.417

    Only ~17k residues in the whole dataset line a non-cryptic pocket, so
    forcing 40% of ~33k negative slots to come from that stratum over-represents
    it by an order of magnitude and distorts the negative distribution the model
    learns to rank against. The per-protein cap is harmless; the forced share
    cost 0.087 AUPRC.

    The headline consequence: a ~49k context ALREADY matches training on all
    288k negatives. There is no context-size ceiling here. Sampling negatives in
    proportion to how they actually occur is what matters.

    Hard-negative mining was also tried -- rank negatives by a cheap logistic
    regression and take the most cryptic-looking. It does not beat plain
    sampling (0.398-0.410 across mixtures) and pure mining collapses entirely
    (AUPRC 0.057, having removed every example of an easy negative), so it is
    not used.

    A per-protein cap stops a few large structures from crowding out the rest;
    without it, context composition tracks protein size rather than biology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from . import config


@dataclass
class SamplePlan:
    n_positive: int
    n_negative: int
    n_hard_negative: int
    n_proteins: int
    budget: int

    def __str__(self) -> str:
        total = self.n_positive + self.n_negative
        return (
            f"context: {total:,} rows from {self.n_proteins} proteins "
            f"(budget {self.budget:,})\n"
            f"  positives      : {self.n_positive:,}\n"
            f"  negatives      : {self.n_negative:,} "
            f"({self.n_hard_negative:,} lining non-cryptic pockets)\n"
            f"  ratio          : 1:{self.n_negative / max(self.n_positive,1):.1f}"
        )


def build_context(
    train: pd.DataFrame,
    budget: int = config.CONTEXT_BUDGET_ROWS,
    negative_ratio: int = config.NEGATIVE_RATIO,
    max_negatives_per_protein: int = config.MAX_NEGATIVES_PER_PROTEIN,
    hard_negative_share: float = config.HARD_NEGATIVE_SHARE,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, SamplePlan]:
    """Subsample `train` into a TabICL context. Returns (context, plan).

    `hard_negative_share` defaults to 0: negatives are drawn in proportion to
    how they occur. Raising it re-introduces the distortion documented above and
    is kept only so the effect stays reproducible.
    """
    rng = np.random.default_rng(seed)
    label = config.LABEL_COL

    pos = train[train[label] == 1]
    neg = train[train[label] == 0]

    target_neg = min(len(neg), int(len(pos) * negative_ratio), max(budget - len(pos), 0))

    if hard_negative_share <= 0:
        # No stratification at all. Note this is NOT the same as a zero quota on
        # the "hard" stratum: splitting the pool and then taking zero from one
        # side would EXCLUDE pocket-lining negatives entirely, which is the same
        # distortion as the old 40% quota, mirrored. Sample the pool as it is.
        neg_sample = _capped_sample(neg, target_neg, max_negatives_per_protein, rng)
        context = pd.concat([pos, neg_sample]).sample(frac=1.0, random_state=seed)
        plan = SamplePlan(
            n_positive=len(pos),
            n_negative=len(neg_sample),
            n_hard_negative=int(neg_sample.get(
                "in_noncryptic_pocket", pd.Series(dtype=int)).sum()),
            n_proteins=context[config.GROUP_COL].nunique(),
            budget=budget,
        )
        return context, plan

    hard_mask = neg["in_noncryptic_pocket"] == 1 if "in_noncryptic_pocket" in neg else pd.Series(False, index=neg.index)
    hard, easy = neg[hard_mask], neg[~hard_mask]

    n_hard = min(len(hard), int(target_neg * hard_negative_share))
    n_easy = max(target_neg - n_hard, 0)

    # The cap is split between the pools rather than applied to each in full,
    # so a protein contributing both hard and easy negatives still cannot
    # exceed `max_negatives_per_protein` overall.
    cap_hard = max(1, int(round(max_negatives_per_protein * hard_negative_share)))
    cap_easy = max(1, max_negatives_per_protein - cap_hard)

    picked = []
    for pool, n_take, cap in ((hard, n_hard, cap_hard), (easy, n_easy, cap_easy)):
        if n_take <= 0 or pool.empty:
            continue
        picked.append(_capped_sample(pool, n_take, cap, rng))

    neg_sample = pd.concat(picked) if picked else neg.iloc[:0]
    context = pd.concat([pos, neg_sample]).sample(frac=1.0, random_state=seed)

    plan = SamplePlan(
        n_positive=len(pos),
        n_negative=len(neg_sample),
        n_hard_negative=int(neg_sample.get("in_noncryptic_pocket", pd.Series(dtype=int)).sum()),
        n_proteins=context[config.GROUP_COL].nunique(),
        budget=budget,
    )
    return context, plan


def _capped_sample(
    pool: pd.DataFrame, n_take: int, per_protein_cap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample `n_take` rows, taking at most `per_protein_cap` from any protein.

    The cap is applied BEFORE any size check. Returning the pool early when it
    is smaller than the request would bypass the cap in precisely the situation
    it exists for -- a few large proteins supplying most of the negatives.
    """
    chunks = []
    for _, sub in pool.groupby(config.GROUP_COL, sort=False):
        k = min(len(sub), per_protein_cap)
        pos = rng.choice(len(sub), size=k, replace=False)
        chunks.append(sub.iloc[pos])
    capped = pd.concat(chunks) if chunks else pool.iloc[:0]

    if len(capped) <= n_take:
        return capped
    pos = rng.choice(len(capped), size=n_take, replace=False)
    return capped.iloc[pos]
