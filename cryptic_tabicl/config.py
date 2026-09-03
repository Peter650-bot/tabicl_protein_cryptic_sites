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
# A residue is keyed exactly as CryptoBench keys it: "{auth_asym_id}_{auth_seq_id}".
RESIDUE_KEY_FMT = "{chain}_{seq_id}"

LABEL_COL = "is_cryptic"
GROUP_COL = "apo_id"          # grouping unit for every split; never split within a protein
RESIDUE_ID_COLS = ("apo_id", "chain", "seq_id", "icode")

# Columns that exist for bookkeeping and must never be handed to the model.
NON_FEATURE_COLS = set(RESIDUE_ID_COLS) | {LABEL_COL, GROUP_COL, "fold", "split"}

# Anything derived from the holo structure is the label in disguise. Named here
# so a test can assert none of these ever appear as a feature column.
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

# Bondi van der Waals radii, for Shrake-Rupley.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
             "SE": 1.90, "H": 1.20}
DEFAULT_VDW = 1.70
PROBE_RADIUS = 1.4
