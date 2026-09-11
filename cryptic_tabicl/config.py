"""Paths and constants for the CryptoBench -> TabICLv2 per-residue pipeline.

WHY THIS EXISTS
    Every stage needs to agree on where the data lives, what a residue key looks
    like, and which columns are allowed to reach the model. Centralising that
    here makes the leakage guard in `evaluate.py` checkable: the set of feature
    columns is declared in one place, so a test can assert that nothing
    holo-derived ever enters X.

SEPARATION
    This package intentionally shares no code with the PPO/HOP project. Ideas
    borrowed from it (ANM ensembles, fpocket scoring) are reimplemented here.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data is staged on the local disk rather than under OneDrive: the CIF archive
# is large enough that letting OneDrive sync it would be pathological.
DATA_STAGE = Path(os.environ.get("CRYPTOBENCH_STAGE", Path.home() / "cryptobench_stage"))
CRYPTOBENCH_DIR = DATA_STAGE / "cryptobench"

# Populated after extraction; resolved lazily so import never depends on the
# download having finished.
DATASET_JSON = CRYPTOBENCH_DIR / "cryptobench-dataset" / "dataset.json"
SPLITS_JSON = CRYPTOBENCH_DIR / "cryptobench-dataset" / "folds.json"
FOLDS_DIR = CRYPTOBENCH_DIR / "cryptobench-dataset" / "folds"
NONCRYPTIC_JSON = (
    CRYPTOBENCH_DIR / "cryptobench-dataset" / "auxiliary-data"
    / "non-cryptic-pockets" / "noncryptic-pockets.json"
)

STRUCTURE_CACHE = DATA_STAGE / "structures"      # downloaded mmCIF files
BUILD_DIR = PROJECT_ROOT / "build"               # parquet shards, tables
BUILD_DIR.mkdir(exist_ok=True)

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

# ---------------------------------------------------------------- task
# A residue is keyed exactly as CryptoBench keys it:
# "{auth_asym_id}_{auth_seq_id}{pdbx_PDB_ins_code}".
#
# THE INSERTION CODE IS PART OF THE KEY, NOT DECORATION. Author numbering is not
# dense: proteases and immunoglobulins insert residues without renumbering the
# rest, so a chain can hold 60, 60A, 60B ... 60I as ten DISTINCT residues.
# CryptoBench writes those as "B_60A"; dropping the code collapses all ten onto
# "B_60", which fails in both directions at once -- an annotated "B_60A" matches
# nothing, and an annotated "B_60" is handed to every residue at position 60.
# Measured before the fix: 20 mislabelled residues across 6 structures, plus
# 3bv9B quarantined outright at match rate 0.793. See tests/test_icode_keys.py.
RESIDUE_KEY_FMT = "{chain}_{seq_id}{icode}"

LABEL_COL = "is_cryptic"
GROUP_COL = "apo_id"          # grouping unit for every split; never split within a protein
RESIDUE_ID_COLS = ("apo_id", "chain", "seq_id", "icode")

# Columns computed FROM THE HOLO STRUCTURE. They are kept in the table because
# they are legitimate for stratifying an evaluation, and must never be handed to
# the model, because at prediction time there is no holo structure to compute
# them from.
#
# `in_noncryptic_pocket` is the one that matters, and it is not obviously
# holo-derived from its name -- which is exactly why it survived a substring
# blacklist for so long. CryptoBench builds that annotation in
# `src/E-add-noncryptic-pockets/main.py`:
#
#     additional_noncryptic[additional_noncryptic['apo_pocket_rms']
#                           < filter_utils.MAX_POCKET_RMSD]
#
# `apo_pocket_rms` is how far the pocket moved between apo and holo. That is the
# SAME measurement as the label, with the inequality flipped: the label says
# "this pocket changes shape when a ligand binds", the annotation says "this one
# does not". Both need the holo partner. Measured on the built table it carries a
# 44.0% positive rate against 3.5% elsewhere, and 36% of every cryptic residue in
# the test set sits inside the 4.4% of rows it flags. Feeding it to the model was
# worth ~0.066 AUPRC of unearned performance (sklearn controls, both directions).
HOLO_DERIVED_METADATA_COLS = frozenset({"in_noncryptic_pocket"})

# Columns that exist for bookkeeping and must never be handed to the model.
NON_FEATURE_COLS = (set(RESIDUE_ID_COLS) | {LABEL_COL, GROUP_COL, "fold", "split"}
                   | set(HOLO_DERIVED_METADATA_COLS))

# Anything derived from the holo structure is the label in disguise. Named here
# so a test can assert none of these ever appear as a feature column.
#
# A substring blacklist alone is NOT sufficient -- it only catches leaks that
# happen to be named after the leak. `HOLO_DERIVED_METADATA_COLS` above is the
# by-name half of the guard, and `build_table.validate` enforces both.
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "holo", "prmsd", "delta_drug", "ddrug", "dvolume", "dvol", "rmsd_to_holo",
)

# ---------------------------------------------------------------- model
# TabICL v2 is pretrained for contexts up to ~60k rows and is in-context, so the
# whole training table sits in memory during a forward pass. With ~300k residues
# and 5 GB of RAM, subsampling is mandatory, not an optimisation.
CONTEXT_BUDGET_ROWS = 50_000
NEGATIVE_RATIO = 8            # negatives kept per positive
MAX_NEGATIVES_PER_PROTEIN = 120   # stops large proteins dominating the context
RANDOM_SEED = 0

# Fraction of the negative budget forced to come from residues lining an
# already-open pocket. MUST be 0. At 0.4 it over-represents a ~17k-residue
# stratum into ~13k of ~33k negative slots and costs 0.087 AUPRC; see the
# measurement table in sampling.py.
HARD_NEGATIVE_SHARE = 0.0

# The benchmark reports TPR at a low FPR; we sweep the threshold on train folds
# to hit this, rather than assuming 0.5.
#
# The sweep targets the MEAN PER-PROTEIN FPR, which is the statistic CryptoBench
# publishes and the one this project reports. It used to target the pooled FPR
# instead -- a different quantity, dominated by the largest proteins. See
# `evaluate.choose_threshold`.
TARGET_FPR = 0.05

# ---------------------------------------------------------------- chemistry
# Kyte-Doolittle hydropathy.
KD_HYDROPATHY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

FORMAL_CHARGE = {
    "ARG": 1.0, "LYS": 1.0, "HIS": 0.5, "ASP": -1.0, "GLU": -1.0,
}

# Tien et al. (2013) theoretical maxima, for relative solvent accessibility.
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

STANDARD_AA = tuple(sorted(MAX_ASA))

# One-letter -> three-letter, for resolving a modified residue to its parent.
# gemmi's component table reports the parent as a LOWER-case one-letter code
# (MSE -> 'm'); see `structures.parent_residue`.
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
assert set(ONE_TO_THREE.values()) == set(MAX_ASA), "residue tables disagree"

# Bondi van der Waals radii, for Shrake-Rupley.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
             "SE": 1.90, "H": 1.20}
DEFAULT_VDW = 1.70
PROBE_RADIUS = 1.4
