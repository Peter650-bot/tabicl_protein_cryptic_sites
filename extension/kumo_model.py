"""NVIDIA Kumo Tabular as a drop-in for `CrypticSitePredictor` (extension/PLAN.md).

WHY A SECOND WRAPPER, NOT A FLAG ON THE FIRST
    `cryptic_tabicl/model.py` is pinned to `tabicl==2.1.1` and every published number
    went through it. Kumo Tabular ships in a different library (`structured-data-models`,
    import name `sdm`, Python >= 3.11, torch >= 2.7), so it gets its own class with the
    same four methods -- `fit(context)`, `predict_proba`, `predict_by_protein`, `config`
    -- and `extension/kumo_stack.py` swaps one for the other and changes nothing else.

WHAT IS HELD FIXED
    Column selection is `cryptic_tabicl.model.feature_matrix`, the same function TabICL
    gets its columns from, so the two models see the same 117 columns in the same order.
    `resname` and `ss` are passed as `categorical`; Kumo's recipe aligns their
    categories to the context by value and hands them to the model as ordinal codes
    with a categorical mask, the counterpart of TabICL encoding them itself. NaNs
    (phi/psi, conservation) pass through: Kumo has a learned missingness projection.

HOW IT IS CALLED
    As in NVIDIA's own benchmark adapter (`benchmark/tabular/model.py`, sdm v0.2.1):
    `fit` caches the context once under fp16 autocast with a seeded CUDA generator,
    then `predict` scores query chunks against that cache. `estimator_batch_size=1`
    because the adapter uses 1 whenever a context exceeds 3,000 rows; ours are ~23-31k.

`backbone="tabiclv2"` runs `sdm.models.TabICLv2` (the same tabicl-classifier-v2-20260212
checkpoint PAM used) through this exact code path. It exists only for the harness
check in PLAN.md step 2: if it reproduces PAM, a Kumo-PAM gap is the model, not the
plumbing.

`sdm` and `torch` are imported inside methods, so this module imports on the local
Python 3.10 box where the tests run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from cryptic_tabicl import config
from cryptic_tabicl.model import CATEGORICAL_COLS, feature_matrix

BACKBONES = ("kumo", "tabiclv2")
SIZES = ("small", "medium", "large")
POSITIVE_CLASS = "1"


def to_stypes(X: pd.DataFrame) -> Dict[str, str]:
    """Semantic type per model-visible column, in `X`'s column order.

    Only `resname` and `ss` are categorical; everything else in the table is numeric.
    A non-numeric column outside that pair would be silently ignored by
    `TableTensor.from_pandas` if left out of the mapping, so it is an error here.
    """
    out = {}
    for col in X.columns:
        if col in CATEGORICAL_COLS:
            out[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(X[col]):
            out[col] = "numerical"
        else:
            raise TypeError(f"column {col!r} has dtype {X[col].dtype} and is not a known categorical")
    return out


@dataclass
class KumoTabularPredictor:
    """Thin wrapper: hold a context, score residues of unseen proteins."""

    backbone: str = "kumo"
    size: str = "large"
    num_estimators: int = 16
    estimator_batch_size: int = 1
    fp16: bool = True
    seed: int = config.RANDOM_SEED
    device: str = "cuda"
    _model: object | None = field(default=None, init=False, repr=False)
    _columns: List[str] = field(default_factory=list, init=False, repr=False)
    _stypes: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    n_context_columns: int = field(default=0, init=False)

    def config(self) -> Dict[str, object]:
        """JSON-safe record of what produced a result file."""
        import sdm
        import torch

        return {"backbone": self.backbone, "size": self.size if self.backbone == "kumo" else None,
                "num_estimators": self.num_estimators,
                "estimator_batch_size": self.estimator_batch_size, "fp16": self.fp16,
                "seed": self.seed, "device": self.device, "sdm": sdm.__version__,
                "torch": torch.__version__, "n_context_columns": self.n_context_columns}

    def _build(self):
        import sdm

        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if self.backbone == "kumo":
            if self.size not in SIZES:
                raise ValueError(f"size must be one of {SIZES}, got {self.size!r}")
            return sdm.models.KumoTabular(task="classification", size=self.size, device=self.device)
        return sdm.models.TabICLv2(task="classification", device=self.device)

    def _autocast(self):
        import torch

        return torch.amp.autocast("cuda", torch.float16,
                                  enabled=self.fp16 and self.device.startswith("cuda"))

    def _table(self, X: pd.DataFrame):
        import sdm

        return sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=self.device)

    def fit(self, context: pd.DataFrame) -> "KumoTabularPredictor":
        import sdm
        import torch

        X = feature_matrix(context)
        self._columns = list(X.columns)
        self._stypes = to_stypes(X)
        constant = [c for c in X.columns if X[c].nunique(dropna=False) < 2]
        assert not constant, f"constant context columns would be dropped by the recipe: {constant}"
        x = self._table(X)
        assert x.size(-1) == len(self._columns), \
            f"{x.size(-1)} columns reached the tensor, wanted {len(self._columns)}"
        self.n_context_columns = int(x.size(-1))
        y = sdm.TableTensor.from_pandas(
            df=context[[config.LABEL_COL]].astype(int).reset_index(drop=True),
            stypes={config.LABEL_COL: "categorical"}, device=self.device)
        self._model = self._build()
        self._model.eval()
        generator = torch.Generator(self.device).manual_seed(self.seed)
        with self._autocast():
            self._model.fit(x=x, y=y, num_estimators=self.num_estimators,
                            estimator_batch_size=self.estimator_batch_size, generator=generator)
        return self

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        """P(cryptic) for every row, column-aligned to the fitted context."""
        import sdm

        if self._model is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        X = feature_matrix(table)
        missing = [c for c in self._columns if c not in X.columns]
        if missing:
            raise ValueError(f"query table missing feature columns: {missing}")
        x = self._table(X[self._columns])
        with self._autocast():
            out = self._model.predict(x)
        names = list(out.columns[sdm.Stype.numerical])
        assert sorted(names) == ["0", "1"], f"unexpected output columns {names}"
        proba = out.numerical[..., names.index(POSITIVE_CLASS)].float().cpu().numpy()
        assert proba.shape == (len(table),), f"output shape {proba.shape}, wanted ({len(table)},)"
        return proba

    def predict_by_protein(self, table: pd.DataFrame, chunk_proteins: int = 40) -> np.ndarray:
        """Score in protein-sized chunks, exactly as `CrypticSitePredictor` does."""
        groups = list(dict.fromkeys(table[config.GROUP_COL]))
        out = np.empty(len(table), dtype=float)
        for i in range(0, len(groups), chunk_proteins):
            batch = set(groups[i:i + chunk_proteins])
            mask = table[config.GROUP_COL].isin(batch).to_numpy()
            out[mask] = self.predict_proba(table[mask])
        return out
