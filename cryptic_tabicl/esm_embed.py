"""Per-residue ESM-2 embeddings: sequence extraction and long-chain chunking.

WHY THIS EXISTS AS A SEPARATE, PURE MODULE
    The GPU part of ESM is trivial (`model(tokens)`). The part that goes wrong
    silently is the bookkeeping around it: which residue does row `i` of the
    embedding matrix correspond to. Get that off by one and you still produce a
    plausible feature matrix with slightly degraded performance, which reads as
    "embeddings did not help" rather than as a bug. So the index-carrying logic
    lives here, in CPU-only functions with tests, and the Modal script only
    calls them.

WHAT SEQUENCE WE EMBED
    The OBSERVED residues of one chain, in order, as a single string.

    CryptoBench embedded UniProt sequences and mapped back, keeping "only those
    embeddings corresponding to residues observed in the structure". We embed the
    observed residues directly. The two differ wherever a chain has an unmodelled
    gap: they show the language model the missing residues as context, we do not,
    so at a gap boundary our two neighbours look adjacent to the model when they
    are not. That affects a handful of positions per chain and avoids depending on
    a SIFTS mapping, but it IS a difference and it is not in our favour.

WHY WE CANNOT JOIN ON SEQUENCE POSITION LATER
    159 of 380,930 residues share an (apo_id, chain, seq_id) key with another
    residue -- insertion codes, where crystallographers number two residues 52A
    and 52B. A join on that key alone would fan those rows out. Every function
    here therefore carries an explicit `res_ix` ordinal alongside the key, and
    the merge uses it.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# ESM-2 was trained with a 1024-token context, two of which are BOS/EOS.
MAX_WINDOW = 1022
# Overlap discarded from each side of an interior chunk. Residues within ~100
# positions of a chunk edge see a truncated context; keeping only the middle
# means every embedding we retain had at least OVERLAP//2 residues of context on
# both sides (except at the true chain termini, where none exists).
CHUNK_OVERLAP = 256

THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def chunk_spans(
    n: int, window: int = MAX_WINDOW, overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[int, int, int, int]]:
    """Cover [0, n) with overlapping windows, each keeping a disjoint middle.

    Returns (start, end, keep_start, keep_end) with `start <= keep_start <
    keep_end <= end`. The keep intervals tile [0, n) exactly once, so
    concatenating `emb[keep_start-start : keep_end-start]` across spans
    reconstructs one embedding per residue in order.

    Chunking exists for 5 chains out of 1,308. It is still worth doing properly:
    the alternative -- truncating at 1022 -- would silently drop up to 464
    residues of a chain and quietly shrink the evaluation set.
    """
    if n <= 0:
        return []
    if n <= window:
        return [(0, n, 0, n)]

    step = window - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than window")

    spans: List[Tuple[int, int, int, int]] = []
    start = 0
    while True:
        end = min(start + window, n)
        # Trim the half-overlap from each interior edge; keep the true termini.
        keep_start = start if start == 0 else start + overlap // 2
        keep_end = end if end == n else end - overlap // 2
        spans.append((start, end, keep_start, keep_end))
        if end == n:
            break
        start += step

    # The final window is clamped to end at n, so it can start before the
    # previous keep_end and leave an overlap. Snap it forward to stay disjoint.
    fixed: List[Tuple[int, int, int, int]] = []
    cursor = 0
    for s, e, ks, ke in spans:
        ks = max(ks, cursor)
        if ke <= ks:
            continue
        fixed.append((s, e, ks, ke))
        cursor = ke
    assert fixed and fixed[0][2] == 0 and fixed[-1][3] == n, "spans do not tile"
    return fixed


def chain_sequences(records: Sequence) -> List[dict]:
    """Group ResidueRecords into per-chain sequences with index bookkeeping.

    Each entry is {chain, seq, keys} where `keys` is the list of
    (seq_id, icode, res_ix) parallel to `seq`, and `res_ix` is the residue's
    position in the full ordered record list for this apo structure.
    """
    order: List[str] = []
    grouped: Dict[str, dict] = {}
    for res_ix, rec in enumerate(records):
        if rec.chain not in grouped:
            grouped[rec.chain] = {"chain": rec.chain, "seq": [], "keys": []}
            order.append(rec.chain)
        g = grouped[rec.chain]
        # Anything non-standard was already filtered upstream; X is a safety net
        # rather than an expected path.
        g["seq"].append(THREE_TO_ONE.get(rec.resname, "X"))
        g["keys"].append((rec.seq_id, getattr(rec, "icode", "") or "", res_ix))

    out = []
    for chain in order:
        g = grouped[chain]
        out.append({"chain": chain, "seq": "".join(g["seq"]), "keys": g["keys"]})
    return out
