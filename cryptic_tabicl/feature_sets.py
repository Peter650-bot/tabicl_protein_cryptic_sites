"""Named feature sets, so an ablation is a name rather than a different parquet.

WHY THIS EXISTS
    Until now the feature set was defined NEGATIVELY -- "every column that is
    not an identifier, the label, the group, the fold, the split, or the one
    holo-derived column" -- and that idiom is repeated verbatim in about a dozen
    places. Two consequences:

      - `pdb_id` is not in `config.NON_FEATURE_COLS`, so every call site
        re-excludes it by hand (`model.feature_matrix`, all three merge
        scripts, ...). One forgotten call site feeds the model a structure
        identifier. That hole is closed in `config.py` rather than papered over
        here.
      - Ablating a family means building a different parquet. That is how the
        model ladder was produced, and it is why `scripts/model_ladder.py` has
        the feature counts 33/47/53/69/85/117 hand-typed as literals.

    With every family living in one table, an arm is a column selection, and
    the selection is declared here, in version control, before it is run.

PRE-REGISTRATION
    This file IS the pre-registration for the feature screen, in the same way
    `deploy/knob_grid.py` was for the knob sweep. It is committed before the
    screen runs. The `trim` list in particular was computed on the TRAIN split
    only, by a criterion fixed in advance, and is frozen here as literals so it
    cannot be quietly re-derived after seeing a result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd

from . import config

# ---------------------------------------------------------------- families
# Derived families, built by `scripts/build_derived.py`. Every column here is a
# function of feature columns and geometry only -- never of the label.

SEQWIN: Tuple[str, ...] = tuple(
    [f"{c}_w5" for c in ("cons_score", "cons_entropy", "cons_prob_self",
                         "rsa", "protrusion")]
    + [f"esm_pc{i:02d}_w5" for i in range(8)]
)

ESMNB: Tuple[str, ...] = tuple(f"esm_pc{i:02d}_nb8" for i in range(16))

NBSTAT: Tuple[str, ...] = tuple(
    f"{c}_{suffix}"
    for suffix in ("nb8std", "nb8max", "nb12")
    for c in ("ens_mean", "ens_max", "ens_std", "ens_opening", "cons_score")
)

PCTX: Tuple[str, ...] = (
    "prot_ens_mean_mean", "prot_ens_mean_max", "prot_ens_mean_std",
    "prot_rsa_mean", "prot_cons_score_mean",
    "prot_n_res", "prot_n_chains", "prot_frac_exposed",
)

PREL: Tuple[str, ...] = tuple(
    f"{c}_{how}"
    for how in ("rank", "z")
    for c in ("ens_mean", "ens_max", "ens_q90", "cons_score_nb8", "rsa", "protrusion")
)

FAMILIES: Dict[str, Tuple[str, ...]] = {
    "seqwin": SEQWIN,
    "esmnb": ESMNB,
    "nbstat": NBSTAT,
    "pctx": PCTX,
    "prel": PREL,
}

# Every column produced by `scripts/build_derived.py`. The base 117 is defined
# as "the table, minus these, minus the supervised block, minus non-features",
# so adding a family to the parquet cannot silently change the control arm.
ALL_DERIVED: Tuple[str, ...] = tuple(c for fam in FAMILIES.values() for c in fam)

SUPERVISED_PREFIX = "esm_sup"

# ------------------------------------------------------------------- trim
# The 24 ESM principal components whose univariate |AUC - 0.5| on the TRAIN
# split is at most 0.02. Frozen as literals: the criterion and the threshold
# were fixed before the screen, and re-deriving the list later would make it a
# free parameter. This is the "removing features" arm, and it is expected to be
# null or negative -- arXiv 2609.13202 finds feature selection consistently
# harms TabICLv2, and this project's own ladder has 117 features beating 85.
TRIM_WEAK_PCS: Tuple[str, ...] = (
    "esm_pc04", "esm_pc05", "esm_pc08", "esm_pc15", "esm_pc18", "esm_pc19",
    "esm_pc30", "esm_pc32", "esm_pc35", "esm_pc36", "esm_pc39", "esm_pc40",
    "esm_pc41", "esm_pc43", "esm_pc45", "esm_pc47", "esm_pc48", "esm_pc49",
    "esm_pc50", "esm_pc51", "esm_pc52", "esm_pc54", "esm_pc55", "esm_pc59",
)

ALL_PCS: Tuple[str, ...] = tuple(f"esm_pc{i:02d}" for i in range(64))

# The out-of-fold score of Škrhák et al.'s fine-tuned ESM-2 (finetuned_esm/PLAN.md
# step 8; `scripts/s2p_build_stack_table.py`). Excluded from the base explicitly:
# it is not derived from features, and a table that carries it must still give the
# control arm exactly the published 117.
S2P_OOF: Tuple[str, ...] = ("s2p_oof",)
# The same, from their full multitask + LIGYSIS recipe with test/fold-homologous LIGYSIS
# proteins removed (PLAN.md step 9, `clean` variant).
S2PFULL_OOF: Tuple[str, ...] = ("s2pfull_oof",)
# Step 10: the fine-tuned model's own embedding, 64 PCs of its last hidden state, and
# its score. The table carries one block per fold model (`ftk{k}_pc00..63`, `ftk{k}_score`);
# the harness renames block k to these generic names for jobs whose context is fold k,
# because each fold model has its own embedding space.
FT_PCS: Tuple[str, ...] = tuple(f"ft_pc{i:02d}" for i in range(64))
FT_SCORE: Tuple[str, ...] = ("ft_score",)
FT_BLOCK_PREFIX = "ftk"
# Step 11: the same hidden state, projected by PLS on the label instead of PCA, fit on
# fold k only (`deploy/ftpls_stack.py`). Components are nested, so the d-dim arm is the
# first d columns of one 128-component fit.
FT_PLS: Tuple[str, ...] = tuple(f"ft_pls{i:03d}" for i in range(128))
PLS_DIMS: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)

# ---------------------------------------------------------------- the arms
# add  : derived families joined onto the base
# drop : base columns removed
# Anything not named here keeps the published 117.
FEATURE_SETS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "control": {"add": (), "drop": ()},
    "seqwin": {"add": SEQWIN, "drop": ()},
    "esmnb": {"add": ESMNB, "drop": ()},
    "nbstat": {"add": NBSTAT, "drop": ()},
    "pctx": {"add": PCTX, "drop": ()},
    "prel": {"add": PREL, "drop": ()},
    "free_combo": {"add": SEQWIN + PCTX + PREL, "drop": ()},
    "trim": {"add": (), "drop": TRIM_WEAK_PCS},
    # The supervised block replaces the unsupervised PCs outright, which is why
    # these two arms DROP all 64. Their `add` is filled in at resolve time from
    # whatever supervised columns the table carries, because the block name
    # depends on which fold is held out.
    "esmsup32": {"add": (), "drop": ALL_PCS},
    "esmsup32_seqwin_esmnb": {"add": SEQWIN + ESMNB, "drop": ALL_PCS},
    # Stacking arms, pre-registered in finetuned_esm/frozen_step8.json.
    "s2p_117": {"add": S2P_OOF, "drop": ()},
    "s2p_53": {"add": S2P_OOF, "drop": ALL_PCS},
    # Pre-registered in finetuned_esm/frozen_step9.json.
    "s2pfull_117": {"add": S2PFULL_OOF, "drop": ()},
    "s2pfull_53": {"add": S2PFULL_OOF, "drop": ALL_PCS},
    # Step 10, pre-registered in finetuned_esm/frozen_step10.json. `base: False` starts
    # from no columns at all: the fine-tuned embedding alone.
    "ft117": {"add": FT_PCS, "drop": ALL_PCS},
    "ft64only": {"add": FT_PCS, "drop": (), "base": False},
    "score118": {"add": FT_SCORE, "drop": ()},
}
# Step 11, pre-registered in finetuned_esm/frozen_step11.json.
for _d in PLS_DIMS:
    FEATURE_SETS[f"pls{_d}_53"] = {"add": FT_PLS[:_d], "drop": ALL_PCS}
    FEATURE_SETS[f"pls{_d}_only"] = {"add": FT_PLS[:_d], "drop": (), "base": False}

USES_SUPERVISED_ESM = ("esmsup32", "esmsup32_seqwin_esmnb")
USES_S2P = ("s2p_117", "s2p_53", "s2pfull_117", "s2pfull_53")
USES_FT = ("ft117", "ft64only", "score118") + tuple(
    f"pls{d}_{b}" for d in PLS_DIMS for b in ("53", "only"))


def base_columns(table: pd.DataFrame) -> list[str]:
    """The published 117: model-visible columns that are not derived."""
    excluded = (set(config.NON_FEATURE_COLS) | set(ALL_DERIVED) | set(S2P_OOF)
                | set(S2PFULL_OOF) | set(FT_PCS) | set(FT_SCORE) | set(FT_PLS))
    return [c for c in table.columns
            if c not in excluded and not c.startswith((SUPERVISED_PREFIX, FT_BLOCK_PREFIX))]


def resolve(name: str, table: pd.DataFrame,
            supervised: Sequence[str] = ()) -> list[str]:
    """Model-visible columns for one arm, in a deterministic order.

    Raises rather than silently narrowing: an arm that names a column the table
    does not carry is a mistake in the grid, not a smaller experiment.
    """
    if name not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {name!r}; "
                       f"known: {sorted(FEATURE_SETS)}")
    spec = FEATURE_SETS[name]
    if bool(supervised) != (name in USES_SUPERVISED_ESM):
        raise ValueError(
            f"feature set {name!r} "
            f"{'requires' if name in USES_SUPERVISED_ESM else 'does not use'} "
            f"supervised ESM columns, but {len(supervised)} were supplied")

    cols = base_columns(table)
    drop = set(spec["drop"])
    unknown_drop = drop - set(cols)
    if unknown_drop:
        raise KeyError(f"{name!r} drops columns not in the table: "
                       f"{sorted(unknown_drop)}")
    kept = [c for c in cols if c not in drop] if spec.get("base", True) else []

    added = list(spec["add"]) + list(supervised)
    missing = [c for c in added if c not in table.columns]
    if missing:
        raise KeyError(f"{name!r} needs columns the table does not have: {missing}")
    return kept + added


def supervised_block(held: str | None, manifest: str | Path | dict) -> list[str]:
    """Columns of the PLS block fitted WITHOUT `held`, or raise saying why not.

    `deploy/esm_supervised_modal.py` fits one projection per nesting, each
    excluding the fold it will be evaluated on, and records which fold that was.
    This is the check that the block a job picks up is the block that excluded
    its own fold. It lives here rather than in the sweep script so it can be
    tested without Modal, and so there is exactly one implementation of it.

    A mismatch is a hard error. A projection fitted on the rows it is then
    scored on would look like a large, clean win.
    """
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    blocks = manifest["blocks"]
    name = held or "final"
    if name not in blocks:
        raise RuntimeError(
            f"no supervised block for {name!r}; manifest has {sorted(blocks)}")
    block = blocks[name]
    if block["held"] != held:
        raise RuntimeError(
            f"supervised block {name!r} was fitted holding out {block['held']!r}, "
            f"but this job holds out {held!r}. Refusing to score a projection "
            "that has seen its own evaluation fold.")
    return list(block["columns"])
