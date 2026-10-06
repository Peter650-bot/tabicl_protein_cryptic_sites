"""The exact input the published fine-tuned ESM-2 model was trained and tested on.

WHY REBUILD THEIR INPUT RATHER THAN FEED IT OURS
    Škrhák et al.'s model reads one string per chain: every residue that has
    peptide-backbone atoms, in file order, from biotite (`get_structure`, model
    1, author fields). Our `esm_embed.chain_sequences` builds a near-identical
    string from gemmi, but "near" is the problem: a residue present in one and
    not the other shifts every position after it, and the model then scores a
    slightly different protein than the one the papers evaluated. So their
    extraction is replicated line for line (their `generate-sequences.ipynb`,
    their `deeplife_utils.three_to_one`) under their pinned biotite 0.41.2.

WHY THE JOIN IS ON AUTHOR KEYS, NOT ON SEQUENCE POSITION
    biotite hands back each residue's author number and insertion code next to
    its position, so a score at position i joins to our table on
    (chain, seq_id, icode) directly. No alignment, nothing to guess.

THEIR LABELS ARE REPLICATED, QUIRKS INCLUDED, FOR ONE PURPOSE ONLY
    `label_positions` reproduces how they marked binding residues, including an
    insertion-code shortcut that approximates codes by counting repeats. It
    exists so the reproduction gate scores their model against their own labels.
    Every result we report uses OUR labels through the key join instead.

The residue-name table and `three_to_one` are copied from
skrhakv/deeplife-project, src/deeplife_utils.py (MIT).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# skrhakv/deeplife-project src/deeplife_utils.py, verbatim.
S2P_MAPPING: Dict[str, str] = {
    'Aba': 'A', 'Ace': 'X', 'Acr': 'X', 'Ala': 'A', 'Aly': 'K', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cas': 'C',
    'Ccs': 'C', 'Cme': 'C', 'Csd': 'C', 'Cso': 'C', 'Csx': 'C', 'Cys': 'C', 'Dal': 'A', 'Dbb': 'T', 'Dbu': 'T',
    'Dha': 'S', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'Glz': 'G', 'His': 'H', 'Hse': 'S', 'Ile': 'I', 'Leu': 'L',
    'Llp': 'K', 'Lys': 'K', 'Men': 'N', 'Met': 'M', 'Mly': 'K', 'Mse': 'M', 'Nh2': 'X', 'Nle': 'L', 'Ocs': 'C',
    'Pca': 'E', 'Phe': 'F', 'Pro': 'P', 'Ptr': 'Y', 'Sep': 'S', 'Ser': 'S', 'Thr': 'T', 'Tih': 'A', 'Tpo': 'T',
    'Trp': 'W', 'Tyr': 'Y', 'Unk': 'X', 'Val': 'V', 'Ycm': 'C', 'Sec': 'U', 'Pyl': 'O', 'Mhs': 'H', 'Snm': 'S',
    'Mis': 'S', 'Seb': 'S', 'Hic': 'H', 'Fme': 'M', 'Asb': 'D', 'Sah': 'C', 'Smc': 'C', 'Tpq': 'Y', 'Onl': 'X',
    'Tox': 'W', '5x8': 'X', 'Ddz': 'A'}

# Their loader drops any sequence longer than this (finetuning_utils.MAX_LENGTH).
THEIR_MAX_LENGTH = 1024

Residue = Tuple[int, str, str]          # (auth seq_id, insertion code, residue name)


def three_to_one(three_letter_code: str) -> str:
    """deeplife_utils.three_to_one, verbatim behaviour: unknown -> 'X'."""
    if three_letter_code[0].upper() + three_letter_code[1:].lower() not in S2P_MAPPING:
        return 'X'
    return S2P_MAPPING[three_letter_code[0].upper() + three_letter_code[1:].lower()]


def sequence_of(residues: Sequence[Residue]) -> str:
    return "".join(three_to_one(name) for _, _, name in residues)


def label_positions(seq_ids: Sequence[int], binding: Set[str]) -> List[int]:
    """0-based positions their extraction marks as binding -- quirks included.

    `binding` holds author numbers as strings ("12", "60A"). Their loop does not
    read insertion codes; it counts consecutive repeats of a number and tries the
    letter of that count, so a repeat run 60, 60, 60 is tried as 60/60A, 60/60B,
    60/60C. Replicated exactly (generate-sequences.ipynb, extract_annotations).
    """
    out: List[int] = []
    previous_seq_id = float('-inf')
    letter_counter = 0
    for idx, auth_seq_id in enumerate(seq_ids):
        if previous_seq_id == auth_seq_id:
            letter_counter += 1
        elif letter_counter > 0:
            letter_counter = 0
        if (str(auth_seq_id) in binding
                or (str(auth_seq_id) + chr(ord('A') + letter_counter)) in binding):
            out.append(idx)
        previous_seq_id = auth_seq_id
    return out


def their_test_entries(entries: dict) -> Dict[str, Tuple[str, Set[str]]]:
    """pdb_id -> (apo chain, binding author numbers), as their extraction builds it.

    Multi-chain entries (an apo_chain like "A-B") are skipped. The chain is taken
    from the FIRST holo record only, and binding numbers are pooled over ALL holo
    records with the chain prefix stripped -- both exactly as they do it.
    """
    out: Dict[str, Tuple[str, Set[str]]] = {}
    for pdb_id, holo_structures in entries.items():
        chain_id = holo_structures[0]['apo_chain']
        if '-' in chain_id:
            continue
        binding: Set[str] = set()
        for holo in holo_structures:
            binding.update(r.split('_')[1] for r in holo['apo_pocket_selection'])
        out[pdb_id] = (chain_id, binding)
    return out


def their_line(pdb_id: str, chain: str, seq: str, positions: Iterable[int]) -> str:
    """One row of their annotation file: `pdb;chain;UNKNOWN;A12 G13 ...;SEQUENCE`."""
    labels = " ".join(f"{seq[i]}{i}" for i in positions)
    return f"{pdb_id};{chain};UNKNOWN;{labels};{seq}"


def chain_residues(cif_path, chain: str) -> List[Residue]:
    """Their residue list for one chain, with the true insertion code alongside.

    Requires biotite 0.41.2 (their pin): `get_structure(model=1)`, keep atoms of
    `chain` passing `filter_peptide_backbone`, one entry per residue start. The
    order and membership are asserted equal to `get_residues`, which is what
    their code calls, so carrying the insertion code changes nothing they did.
    """
    import gzip

    import biotite
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import numpy as np

    if biotite.__version__ != "0.41.2":
        raise RuntimeError(f"biotite {biotite.__version__}; their pipeline pins 0.41.2")

    path = str(cif_path)
    handle = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with handle as fh:
        cif_file = pdbx.CIFFile.read(fh)
    auth = pdbx.get_structure(cif_file, model=1)
    auth = auth[(auth.chain_id == chain) & struc.filter_peptide_backbone(auth)]
    if len(auth) == 0:
        return []

    starts = struc.get_residue_starts(auth)
    ids, names = struc.get_residues(auth)
    assert np.array_equal(ids, auth.res_id[starts]), "residue starts disagree"
    assert np.array_equal(names, auth.res_name[starts]), "residue names disagree"
    return [(int(r), str(c), str(n)) for r, c, n in
            zip(auth.res_id[starts], auth.ins_code[starts], auth.res_name[starts])]


def key_of(chain: str, residue: Residue) -> Tuple[str, int, str]:
    """(chain, seq_id, icode) in our table's convention: '' for no insertion code."""
    seq_id, icode, _ = residue
    return (chain, seq_id, icode.strip() if icode else "")


def find_duplicate_keys(chain: str, residues: Sequence[Residue]) -> List[Tuple[str, int, str]]:
    seen: Set[Tuple[str, int, str]] = set()
    dup: List[Tuple[str, int, str]] = []
    for r in residues:
        k = key_of(chain, r)
        if k in seen:
            dup.append(k)
        seen.add(k)
    return dup


def first_holo_chain(entries: dict, pdb_id: str) -> Optional[str]:
    hs = entries.get(pdb_id)
    return hs[0]['apo_chain'] if hs else None
