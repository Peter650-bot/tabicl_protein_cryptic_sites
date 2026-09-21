"""Cross-row feature construction: pooling a residue's context into its row.

WHY THIS EXISTS
    TabICL sees one residue per row and cannot look at the other residues of the
    same protein. So anything about a residue's surroundings has to be written
    into its row or it is invisible to the model. The repo already does this in
    three near-identical private copies -- `features._add_neighbourhood_means`,
    `ensemble_features._add_neighbourhood`, `conservation.add_neighbourhood` --
    each of which pools its own family at 8 A inside its own structure loop.

    That arrangement has a blind spot with a name: the 64 ESM principal
    components arrive from a GPU job that never touches a coordinate, so they
    are the one family with no spatial pooling at all.

    These functions take the pooling out of the structure loop. Given
    `build/residue_coords.parquet` they work on any column of the built table,
    at any radius, with any of several statistics.

THIS IS NOT RE-REPRESENTATION
    "Do Tabular Foundation Models Still Need Feature Engineering?"
    (arXiv 2609.13202) measures interactions and group aggregation on TabICLv2
    at a median of +0.00 macro-F1, and finds feature selection actively harmful.
    Its conclusion is that the frontier has moved "from representation
    engineering toward information augmentation". Pooling is on the right side
    of that line: a neighbour's conservation is not a rearrangement of this
    row's columns, it is a fact this row did not contain.

NO FUNCTION HERE MAY READ THE LABEL
    Everything in this module is a function of feature columns and geometry
    only. `tests/test_no_leakage.py` enforces that by permuting the label column
    and asserting the outputs do not move. The one label-using step in the
    feature work package lives in `scripts/esm_supervised.py`, where it is
    fold-nested and carries its own manifest.

CONVENTIONS INHERITED FROM THE EXISTING `_nb8` COLUMNS
    - Neighbours are found on residue CENTROIDS, not alpha carbons
      (`features.py:271`).
    - The residue is EXCLUDED from its own neighbourhood, falling back to its
      own value when it has no neighbour (`features.py:287`).
    - Pooling is NaN-aware: non-finite neighbours are dropped before the
      statistic, and the result is NaN only when nothing finite remains
      (`conservation.py:229-231`). The other two copies propagate NaN; this one
      does not.
    - Missing stays NaN, never 0 (`merge_ensemble.py:8-15`).
    - Pooling never crosses proteins.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STATS = ("mean", "std", "max")


def _as_2d(values: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    if arr.ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D, got {arr.ndim}-D")
    return arr, False


def _pool_one_protein(
    vals: np.ndarray, coords: np.ndarray, radius: float, stat: str
) -> np.ndarray:
    """Pool every column over each row's neighbours. vals is (n, k)."""
    n, k = vals.shape
    out = np.full((n, k), np.nan, dtype=float)
    if n == 0:
        return out

    # query_pairs excludes self by construction; both directions are needed
    # because "j is a neighbour of i" must also write into j's pool.
    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        nbr = np.concatenate([pairs[:, 1], pairs[:, 0]])
    else:
        src = np.empty(0, dtype=int)
        nbr = np.empty(0, dtype=int)

    # Two different emptinesses, and the published columns treat them
    # differently. A residue with no neighbour AT ALL falls back to its own
    # value (`features.py:287`'s `or [i]`). A residue that has neighbours whose
    # values all happen to be NaN gets NaN, because the fallback in the
    # published code is applied to the INDEX SET, before the finite filter
    # (`conservation.py:229-231`). Collapsing the two cases changes three rows
    # of `cons_score_nb8`.
    isolated = np.bincount(src, minlength=n) == 0

    for c in range(k):
        col = vals[:, c]
        v = col[nbr]
        fin = np.isfinite(v)
        i, x = src[fin], v[fin]
        count = np.bincount(i, minlength=n)
        have = count > 0

        if stat == "mean":
            total = np.bincount(i, weights=x, minlength=n)
            out[have, c] = total[have] / count[have]
        elif stat == "std":
            total = np.bincount(i, weights=x, minlength=n)
            sqs = np.bincount(i, weights=x * x, minlength=n)
            mean = np.divide(total, count, out=np.zeros(n), where=have)
            var = np.divide(sqs, count, out=np.zeros(n), where=have) - mean ** 2
            out[have, c] = np.sqrt(np.clip(var[have], 0.0, None))
        elif stat == "max":
            acc = np.full(n, -np.inf)
            np.maximum.at(acc, i, x)
            out[have, c] = acc[have]
        else:
            raise ValueError(f"unknown stat {stat!r}; expected one of {STATS}")

        out[~have & isolated, c] = col[~have & isolated]
        # `out` is pre-filled with NaN, so ~have & ~isolated needs no write.
    return out


def spatial_pool(
    values: np.ndarray,
    coords: np.ndarray,
    groups: np.ndarray,
    radius: float,
    stat: str = "mean",
) -> np.ndarray:
    """Pool `values` over each row's neighbours within `radius`, per protein.

    `values` is (n,) or (n, k); the return has the same shape. Passing every
    column at once is much faster than one call per column, because the KD-tree
    is built once per protein either way.
    """
    if stat not in STATS:
        raise ValueError(f"unknown stat {stat!r}; expected one of {STATS}")
    vals, was_1d = _as_2d(values)
    xyz = np.asarray(coords, dtype=float)
    grp = np.asarray(groups)
    if not (len(vals) == len(xyz) == len(grp)):
        raise ValueError("values, coords and groups must have the same length")
    if xyz.shape[1] != 3:
        raise ValueError(f"coords must be (n, 3), got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("coords contain non-finite values; pooling would be undefined")

    out = np.full(vals.shape, np.nan, dtype=float)
    order = np.argsort(grp, kind="stable")
    bounds = np.flatnonzero(np.r_[True, grp[order][1:] != grp[order][:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = order[lo:hi]
        out[sel] = _pool_one_protein(vals[sel], xyz[sel], radius, stat)
    return out.ravel() if was_1d else out


def sequence_window(
    values: np.ndarray, chain_key: np.ndarray, order: np.ndarray, width: int
) -> np.ndarray:
    """Centred rolling mean along the chain, never crossing a chain boundary.

    A cryptic site is a patch in space, but the sequence-derived families --
    conservation and the pLM components -- also vary smoothly along the
    backbone, and their windowed forms measure better than their raw forms
    (RESULTS.md section 12). Structure-derived families do not; do not window
    them.
    """
    if width < 1 or width % 2 == 0:
        raise ValueError(f"width must be a positive odd number, got {width}")
    vals, was_1d = _as_2d(values)
    frame = pd.DataFrame(vals)
    frame["__key"] = np.asarray(chain_key)
    frame["__ord"] = np.asarray(order)
    frame = frame.sort_values(["__key", "__ord"], kind="stable")
    cols = [c for c in frame.columns if c not in ("__key", "__ord")]
    # min_periods=1 keeps chain termini finite, and pandas' rolling mean skips
    # NaN, so a missing neighbour shrinks the window instead of erasing the row.
    rolled = frame.groupby("__key", sort=False)[cols].transform(
        lambda s: s.rolling(width, center=True, min_periods=1).mean()
    )
    out = rolled.reindex(range(len(vals))).to_numpy(dtype=float)
    return out.ravel() if was_1d else out


def within_protein(values: np.ndarray, groups: np.ndarray, how: str) -> np.ndarray:
    """Rank (as a fraction) or z-score of each value among its own protein.

    The same two transforms the decision rule uses (`operating_point.RULES`),
    applied to features rather than to scores. A constant protein has no spread,
    so its z is 0 -- not a division by zero, and not NaN.
    """
    vals, was_1d = _as_2d(values)
    grp = pd.Series(np.asarray(groups))
    frame = pd.DataFrame(vals)
    if how == "rank":
        out = frame.groupby(grp.to_numpy(), sort=False).rank(pct=True)
    elif how == "z":
        g = frame.groupby(grp.to_numpy(), sort=False)
        centred = frame - g.transform("mean")
        sd = g.transform("std").replace(0.0, np.nan)
        out = (centred / sd).fillna(0.0).where(frame.notna())
    else:
        raise ValueError(f"unknown how {how!r}; expected 'rank' or 'z'")
    arr = out.to_numpy(dtype=float)
    return arr.ravel() if was_1d else arr


def protein_context(
    table: pd.DataFrame,
    columns: Sequence[str],
    stats: Iterable[str] = ("mean",),
    group_col: str = "apo_id",
) -> pd.DataFrame:
    """Protein-level aggregates of `columns`, broadcast back onto every row.

    Each row currently carries almost nothing about the protein it belongs to --
    `chain_len` and nothing else. Yet every metric in this project is computed
    per protein, so "high for this protein" is the quantity that matters and the
    model has no way to form it.
    """
    grouped = table.groupby(group_col, sort=False)[list(columns)]
    pieces = {}
    for stat in stats:
        agg = grouped.transform(stat)
        for col in columns:
            pieces[f"prot_{col}_{stat}"] = agg[col].to_numpy(dtype=float)
    return pd.DataFrame(pieces, index=table.index)
