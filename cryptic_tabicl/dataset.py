"""CryptoBench annotation layer: labels, chains, splits, hard negatives.

WHY THIS EXISTS
    Three correctness traps live in this file, and all three silently produce a
    *better-looking* model if you get them wrong:

    1. ONE APO -> MANY HOLO. `dataset.json` maps an apo id to a LIST of holo
       structures, each contributing its own pocket. The label for an apo is the
       UNION of `apo_pocket_selection` over all of them. Taking only the first
       (or only the `is_main_holo_structure`) marks genuine cryptic residues as
       negatives, which quietly inflates precision.

    2. IDENTITY IS (pdb_id, chain), NOT pdb_id. Pocket keys are
       "{auth_asym_id}_{auth_seq_id}" and a binding site may span chains. The
       benchmark's own example uses '7w19A'. We key apo structures the same way.

    3. HOLO IS FOR LABELS ONLY. Nothing in this module returns a holo-derived
       quantity for use as a feature -- not the ligand, not `pRMSD`. At
       prediction time only the apo structure exists, so any holo-derived
       feature is the answer smuggled into X.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from . import config

ResidueKey = str          # "B_12"
ApoId = str               # "1a4uB"


# ------------------------------------------------------------------ io
def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _apo_id(pdb_id: str, chain: str) -> ApoId:
    return f"{pdb_id}{chain}"


# ------------------------------------------------------------------ labels
def apo_binding_residues(dataset: dict) -> Dict[ApoId, Set[ResidueKey]]:
    """Union of cryptic binding residues per apo structure.

    Mirrors the reference `get_apo_binding_residues` in CryptoBench's tutorial,
    but keys by (pdb_id, apo_chain) instead of pdb_id alone so that two apo
    chains of the same entry stay distinct.
    """
    out: Dict[ApoId, Set[ResidueKey]] = defaultdict(set)
    for pdb_id, holo_structures in dataset.items():
        for holo in holo_structures:
            aid = _apo_id(pdb_id, holo["apo_chain"])
            out[aid].update(holo["apo_pocket_selection"])
    return dict(out)


def apo_chains(dataset: dict) -> Dict[ApoId, Set[str]]:
    """Chains that must be parsed for each apo structure.

    This is the apo_chain plus any chain appearing in a pocket key, because a
    binding site may stretch over more than one chain.
    """
    out: Dict[ApoId, Set[str]] = defaultdict(set)
    for pdb_id, holo_structures in dataset.items():
        for holo in holo_structures:
            aid = _apo_id(pdb_id, holo["apo_chain"])
            out[aid].add(holo["apo_chain"])
            for key in holo["apo_pocket_selection"]:
                out[aid].add(key.rsplit("_", 1)[0])
    return dict(out)


def apo_pdb_ids(dataset: dict) -> Dict[ApoId, str]:
    """apo id -> 4-character PDB id, for structure download."""
    out = {}
    for pdb_id, holo_structures in dataset.items():
        for holo in holo_structures:
            out[_apo_id(pdb_id, holo["apo_chain"])] = pdb_id
    return out


def noncryptic_residues(noncryptic: dict) -> Dict[ApoId, Set[ResidueKey]]:
    """Residues lining pockets that are already open in the apo form.

    These are HARD NEGATIVES: pocket-like geometry that is nevertheless not
    cryptic. Without them the task collapses towards "find any pocket", which is
    not the question CryptoBench asks. Used both as a feature and as an
    evaluation stratum.
    """
    out: Dict[ApoId, Set[ResidueKey]] = defaultdict(set)
    for pdb_id, entries in noncryptic.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            chain = entry.get("apo_chain")
            selection = entry.get("apo_pocket_selection") or entry.get("pocket_selection") or []
            if chain is None:
                # Fall back to the chain embedded in the keys themselves.
                for key in selection:
                    out[_apo_id(pdb_id, key.rsplit("_", 1)[0])].update(selection)
                continue
            out[_apo_id(pdb_id, chain)].update(selection)
    return dict(out)


# ------------------------------------------------------------------ splits
def load_splits(splits_path: Path | None = None) -> Dict[str, List[str]]:
    """Subset name -> list of apo structure ids, as shipped by CryptoBench.

    We never construct our own split. CryptoBench's folds are built with
    sequence-identity clustering; a random split over residues (or even over
    proteins, ignoring homology) leaks and inflates every metric.
    """
    path = splits_path or config.SPLITS_JSON
    raw = load_json(path)
    return {k: list(v) for k, v in raw.items()}


def load_folds(folds_dir: Path | None = None) -> Dict[str, dict]:
    """Fold name -> raw fold dataset (same schema as dataset.json)."""
    d = folds_dir or config.FOLDS_DIR
    return {p.stem: load_json(p) for p in sorted(Path(d).glob("*.json"))}


def fold_assignment(folds: Dict[str, dict]) -> Dict[ApoId, str]:
    """apo id -> fold name, asserting an apo never appears in two folds."""
    out: Dict[ApoId, str] = {}
    for fold_name, fold_data in folds.items():
        for aid in apo_binding_residues(fold_data):
            if aid in out and out[aid] != fold_name:
                raise ValueError(
                    f"{aid} appears in both {out[aid]} and {fold_name}; "
                    "splits are not disjoint"
                )
            out[aid] = fold_name
    return out


# ------------------------------------------------------------------ summary
def describe(dataset: dict) -> str:
    """Human-readable shape of the dataset; used to validate assumptions."""
    labels = apo_binding_residues(dataset)
    chains = apo_chains(dataset)
    n_holo = sum(len(v) for v in dataset.values())
    multi_chain = sum(1 for v in chains.values() if len(v) > 1)
    sizes = sorted(len(v) for v in labels.values())
    med = sizes[len(sizes) // 2] if sizes else 0
    return (
        f"apo pdb entries      : {len(dataset)}\n"
        f"apo structures (id+ch): {len(labels)}\n"
        f"holo pairings        : {n_holo}\n"
        f"apo w/ multi-chain BS: {multi_chain}\n"
        f"cryptic residues/apo : min {sizes[0] if sizes else 0}, "
        f"median {med}, max {sizes[-1] if sizes else 0}\n"
        f"total cryptic residues: {sum(sizes)}"
    )
