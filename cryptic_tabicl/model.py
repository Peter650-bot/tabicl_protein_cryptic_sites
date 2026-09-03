"""Zero-shot TabICLv2 for per-residue cryptic site prediction.

WHY ZERO-SHOT
    TabICL's claim is that no training happens: the labelled context is fed
    through the network in one forward pass. Fine-tuning would forfeit exactly
    that property, so it is deliberately not used here. (`tabicl[finetune]`
    exists and could be added later as a separate ceiling measurement -- it
    should be reported alongside, never in place of, the zero-shot number.)

MEMORY
    On CPU with 5 GB, the context size dominates. `batch_size` bounds the
    feature-ensemble memory and `offload_mode` lets TabICL spill to disk, so the
    knobs are surfaced here rather than buried.

CATEGORICALS
    `resname` and `ss` are left as pandas `category`/object dtype on purpose:
    TabICL detects and ordinal-encodes them itself. Hand-rolled one-hot would
    inflate the feature count towards the memory ceiling for no benefit.

PREDICTION UNIT
    Scores are produced for EVERY residue of each test protein, never a
    subsample. Subsampling applies only to the context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import config

CATEGORICAL_COLS = ("resname", "ss")


def feature_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Model-visible columns only, with categoricals typed for TabICL.

    Dropping the identifier columns here (rather than at build time) keeps the
    parquet self-describing while guaranteeing the model never sees `apo_id`,
    which would otherwise let it memorise proteins.
    """
    drop = set(config.NON_FEATURE_COLS) | {"pdb_id"}
    X = table[[c for c in table.columns if c not in drop]].copy()
    for col in CATEGORICAL_COLS:
        if col in X:
            X[col] = X[col].astype("category")
    # TabICL rejects float16; make the contract explicit rather than hoping.
    for col in X.columns:
        if X[col].dtype == "float16":
            X[col] = X[col].astype("float32")
    return X


@dataclass
class CrypticSitePredictor:
    """Thin wrapper: hold a context, score residues of unseen proteins."""

    n_estimators: int = 8
    batch_size: int = 2
    device: str | None = None
    random_state: int = config.RANDOM_SEED
    verbose: bool = False
    _clf: object | None = field(default=None, init=False, repr=False)
    _columns: List[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, context: pd.DataFrame) -> "CrypticSitePredictor":
        from tabicl import TabICLClassifier

        X = feature_matrix(context)
        y = context[config.LABEL_COL].to_numpy()
        self._columns = list(X.columns)
        self._clf = TabICLClassifier(
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._clf.fit(X, y)
        return self

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        """P(cryptic) for every row, column-aligned to the fitted context."""
        if self._clf is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        X = feature_matrix(table)
        missing = [c for c in self._columns if c not in X.columns]
        if missing:
            raise ValueError(f"test table missing feature columns: {missing}")
        X = X[self._columns]
        proba = self._clf.predict_proba(X)
        return proba[:, 1]

    def predict_by_protein(
        self, table: pd.DataFrame, chunk_proteins: int = 40
    ) -> np.ndarray:
        """Score in protein-sized chunks so peak memory stays bounded.

        Chunking on protein boundaries (not row count) keeps every residue of a
        structure in one call, which matters if per-call preprocessing ever
        becomes context-dependent.
        """
        groups = list(dict.fromkeys(table[config.GROUP_COL]))
        out = np.empty(len(table), dtype=float)
        for i in range(0, len(groups), chunk_proteins):
            batch = set(groups[i:i + chunk_proteins])
            mask = table[config.GROUP_COL].isin(batch).to_numpy()
            out[mask] = self.predict_proba(table[mask])
        return out
