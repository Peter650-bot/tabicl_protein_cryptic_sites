"""Sequence conservation from the PDBe Graph API.

WHAT CONSERVATION IS
    How strongly evolution has resisted substitution at a sequence position.
    Align a protein against its homologues across species and look down each
    column of the alignment: if only one amino acid ever appears, the position
    is doing something the organism cannot afford to change. If many appear
    freely, it is not. Functional sites -- catalytic residues, binding sites --
    are under selection and show up conserved.

WHY IT IS THE RIGHT FEATURE TO ADD HERE
    Not because CryptoSite ranks it second (-log10 P 67.5): that ranking is
    univariate and over-promised for the ensemble features already.

    The reason is that it is orthogonal to everything we have. All 48 current
    features are geometry, and geometry has repeatedly failed on one specific
    stratum: telling a cryptic site from a pocket that is ALREADY OPEN. Its AUC
    sat at 0.677, 0.680, 0.681 across three very different interventions. Both
    classes are concave and pocket-shaped, so shape descriptors cannot separate
    them. But a cryptic binding site is a functional site under selection, while
    an incidental open surface pocket that binds nothing is not -- so
    conservation should discriminate exactly where geometry is blind.

WHY THIS SOURCE
    The CryptoBench authors used this same API to analyse conservation of their
    own dataset (`src/F-statistics/conservation/conservation-scores.ipynb`), so
    the scores are consistent with the benchmark rather than a different
    pipeline's numbers. It is precomputed, which avoids an MMseqs2 search against
    UniRef -- tens of gigabytes and hours of compute -- for a question we can
    answer in an afternoon.

THE JOIN, WHICH IS THE DANGEROUS PART
    The API indexes by `label_seq_id`; CryptoBench annotates by `auth_seq_id`.
    These coincide for many structures and diverge for many others, so a
    positional join silently produces plausible, wrong conservation values for a
    subset of proteins. Every lookup here goes through the residue's own
    `label_seq` carried from the mmCIF, and the caller reports match rate.

    Conservation is served per ENTITY, not per chain, so every chain of the same
    polymer entity shares one score vector.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

MOLECULES_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb}"
CONSERVATION_URL = (
    "https://www.ebi.ac.uk/pdbe/graph-api/pdb/sequence_conservation/{pdb}/{entity}"
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_ENTROPY = math.log(20)          # uniform over 20 amino acids

FEATURE_NAMES = ("cons_score", "cons_entropy", "cons_prob_self")
NEIGHBOURHOOD_RADIUS = 8.0


@dataclass
class ChainConservation:
    """Per-label_seq_id conservation for one polymer entity."""
    score: Dict[int, float]           # PDBe conservation grade
    entropy: Dict[int, float]         # Shannon entropy, normalised to [0, 1]
    probabilities: Dict[int, Dict[str, float]]

    def __len__(self) -> int:
        return len(self.score)


def _get(url: str, retries: int = 3, backoff: float = 1.5) -> dict | None:
    """GET with retries. Transient failures are common over ~2,300 calls."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None          # genuinely absent, not worth retrying
            if attempt == retries - 1:
                return None
        except Exception:  # noqa: BLE001 - URLError, timeouts, malformed JSON
            if attempt == retries - 1:
                return None
        time.sleep(backoff ** attempt)
    return None


def _unwrap(payload: dict | list | None, pdb: str) -> dict | None:
    """The response envelope varies between calls; normalise it.

    Observed shapes: {pdb: {...}}, {pdb: [{...}]}, and the record itself.
    Guessing one shape works until it doesn't, so handle all three.
    """
    if payload is None:
        return None
    node = payload.get(pdb, payload) if isinstance(payload, dict) else payload
    if isinstance(node, list):
        node = node[0] if node else None
    return node if isinstance(node, dict) else None


def polymer_entity_id(pdb: str, chain: str) -> int | None:
    """Entity id of the polypeptide containing `chain`.

    Selected by `molecule_type`, not by position: entry 1a4u lists a water
    entity alongside the protein and both report the same `in_chains`.
    """
    node = _get(MOLECULES_URL.format(pdb=pdb.lower()))
    if not node:
        return None
    entities = node.get(pdb.lower()) or node.get(pdb.upper()) or []
    for entity in entities:
        if not str(entity.get("molecule_type", "")).startswith("polypeptide"):
            continue
        if chain in (entity.get("in_chains") or []):
            return entity.get("entity_id")
    return None


def fetch_chain(pdb: str, chain: str) -> ChainConservation | None:
    """Conservation for one chain, or None when the entry has none."""
    entity = polymer_entity_id(pdb, chain)
    if entity is None:
        return None

    data = _unwrap(_get(CONSERVATION_URL.format(pdb=pdb.lower(), entity=entity)),
                   pdb.lower())
    data = (data or {}).get("data")
    if not data or "conservation_score" not in data or "index" not in data:
        return None

    index = data["index"]
    scores = data["conservation_score"]
    prob_keys = [f"probability_{a}" for a in AMINO_ACIDS]
    have_probs = all(k in data for k in prob_keys)

    score_map: Dict[int, float] = {}
    entropy_map: Dict[int, float] = {}
    prob_map: Dict[int, Dict[str, float]] = {}

    for i, (pos, sc) in enumerate(zip(index, scores)):
        score_map[int(pos)] = float(sc)
        if not have_probs:
            continue
        probs = {a: float(data[f"probability_{a}"][i]) for a in AMINO_ACIDS}
        prob_map[int(pos)] = probs
        entropy_map[int(pos)] = _normalised_entropy(probs)

    return ChainConservation(score_map, entropy_map, prob_map)


def _normalised_entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy over the 20 amino acids, scaled to [0, 1].

    This is CryptoSite's actual definition of conservation, and it is finer than
    the integer grade the API also returns -- the grade is binned, the entropy is
    continuous. Normalising by log(20) keeps it comparable across positions and
    proteins. 0 means one residue is certain (fully conserved); 1 means uniform.
    """
    total = sum(probs.values())
    if total <= 0:
        return float("nan")
    h = 0.0
    for p in probs.values():
        q = p / total
        if q > 0:
            h -= q * math.log(q)
    return h / MAX_ENTROPY


def residue_features(
    cons: ChainConservation | None,
    label_seqs: Sequence[int | None],
    resnames: Sequence[str],
) -> List[Dict[str, float]]:
    """Per-residue conservation features, aligned by label_seq_id.

    Missing positions become NaN rather than 0: a zero would assert "maximally
    conserved", a confident false statement the model would learn from.
    """
    from .config import MAX_ASA  # three-letter -> one-letter via a shared table

    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    assert set(three_to_one) == set(MAX_ASA), "residue tables disagree"

    nan = float("nan")
    out: List[Dict[str, float]] = []
    for label_seq, resname in zip(label_seqs, resnames):
        if cons is None or label_seq is None or label_seq not in cons.score:
            out.append({"cons_score": nan, "cons_entropy": nan,
                        "cons_prob_self": nan})
            continue
        probs = cons.probabilities.get(int(label_seq))
        one = three_to_one.get(resname)
        out.append({
            "cons_score": cons.score[int(label_seq)],
            "cons_entropy": cons.entropy.get(int(label_seq), nan),
            "cons_prob_self": probs.get(one, nan) if (probs and one) else nan,
        })
    return out


def add_neighbourhood(rows: List[Dict[str, float]], centroids: np.ndarray) -> None:
    """8 A neighbourhood means, matching the convention of the other tiers.

    CryptoSite lists conservation-of-neighbours (SQCn) separately from
    conservation (SQC) and ranks it comparably, because a binding site is a
    conserved PATCH rather than a lone conserved residue.
    """
    from scipy.spatial import cKDTree

    neighbours = cKDTree(centroids).query_ball_point(centroids, NEIGHBOURHOOD_RADIUS)
    base = {f: np.array([r[f] for r in rows], dtype=float) for f in FEATURE_NAMES}
    for i, row in enumerate(rows):
        idx = [j for j in neighbours[i] if j != i] or [i]
        for f, vals in base.items():
            window = vals[idx]
            finite = window[np.isfinite(window)]
            row[f"{f}_nb8"] = float(finite.mean()) if len(finite) else float("nan")
