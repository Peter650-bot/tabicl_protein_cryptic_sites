"""Step 9b of finetuned_esm/PLAN.md: what the LIGYSIS pre-training stage shares with CryptoBench.

Two different questions, with different consequences:
  test        A CryptoBench TEST protein in LIGYSIS training would leak into every model
              built with this recipe, including the released one. Step 2 could only
              infer this was filtered; their actual training list now settles it.
  train folds Their filter was aimed at the test set only. A LIGYSIS protein matching a
              CryptoBench TRAINING protein is harmless for their model, but in our
              out-of-fold scheme it means fold model h saw (general) binding sites of
              fold h's proteins during the shared LIGYSIS stage -- context scores that
              are less "unseen" than test scores.

Method: their loader's filters first (the 46 excluded UniProt IDs, length > 1,024), then
exact UniProt matches, then a 5-mer screen of every LIGYSIS x CryptoBench chain pair and a
local alignment (BLOSUM62) of every candidate. Identity = identical aligned positions /
aligned length; coverage = aligned length / shorter sequence.

Needs Biopython on PYTHONPATH. Writes finetuned_esm/step9b_ligysis_overlap.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C, dataset as D  # noqa: E402

LIGYSIS_EXCLUDED = {'P28907', 'P40925', 'Q8TDX7', 'P62068', 'P13929', 'P18669', 'P22303', 'P23141', 'P46597', 'P68106', 'Q86WA6', 'P13686', 'P00480', 'O75530', 'P50053', 'Q13303', 'O75317', 'P15328', 'P08236', 'Q9H477', 'P52895', 'P09936', 'P14207', 'P30613', 'P14618', 'P42330', 'P17900', 'Q04828', 'P12004', 'Q8IVL8', 'Q7L266', 'P51857', 'Q9BUT1', 'P30838', 'O43924', 'P09104', 'P07864', 'Q9BYC5', 'P06733', 'P15121', 'P17516', 'P40926', 'O60218', 'P00338', 'P07738', 'P07195'}
K, MIN_SHARED = 5, 15
NEAR_IDENTICAL, HOMOLOG, MIN_COVERAGE = 0.90, 0.40, 0.50


def kmers(s: str) -> set:
    return {s[i:i + K] for i in range(len(s) - K + 1)}


def main() -> None:
    from Bio import Align
    from Bio.Align import substitution_matrices

    lig = []
    for line in (C.BUILD_DIR / "s2p_full" / "ligysis_train.txt").read_text().splitlines():
        row = line.split(";")
        if row[0] in LIGYSIS_EXCLUDED or len(row[4]) > 1024:   # their loader's filters
            continue
        lig.append((row[0], row[4]))

    ds = D.load_json(C.DATASET_JSON)
    uniprot = {f"{pdb}{h['apo_chain']}": h["uniprot_id"] for pdb, hs in ds.items() for h in hs}
    table = pd.read_parquet(C.BUILD_DIR / "residues_esm64.parquet", columns=["apo_id", "fold"])
    fold_of = dict(table.drop_duplicates("apo_id").itertuples(index=False))
    inputs = pd.read_parquet(C.BUILD_DIR / "s2p_inputs.parquet").sort_values(["apo_id", "chain", "pos"])
    chains = [(a, c, "".join(g["aa"])) for (a, c), g in inputs.groupby(["apo_id", "chain"])]

    index = defaultdict(set)
    for i, (_, _, s) in enumerate(chains):
        for k in kmers(s):
            index[k].add(i)

    aligner = Align.PairwiseAligner(mode="local", open_gap_score=-10, extend_gap_score=-0.5)
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    best = {}   # apo_id -> (identity, coverage, ligysis uniprot)
    hits = defaultdict(set)   # apo_id -> LIGYSIS UniProt IDs at >= HOMOLOG identity
    n_aligned = 0
    for uid, seq in lig:
        shared = defaultdict(int)
        for k in kmers(seq):
            for i in index.get(k, ()):
                shared[i] += 1
        for i, n in shared.items():
            if n < MIN_SHARED:
                continue
            apo_id, _, cs = chains[i]
            aln = aligner.align(seq, cs)[0]
            n_aligned += 1
            ident = aln.counts().identities
            length = aln.length
            idt, cov = ident / length, length / min(len(seq), len(cs))
            if cov >= MIN_COVERAGE and idt >= HOMOLOG:
                hits[apo_id].add(uid)
            if cov >= MIN_COVERAGE and idt > best.get(apo_id, (0, 0, ""))[0]:
                best[apo_id] = (idt, cov, uid)

    lig_ids = {u for u, _ in lig}
    out = {"ligysis_trained_on": len(lig), "pairs_aligned": n_aligned,
           "thresholds": {"near_identical": NEAR_IDENTICAL, "homolog": HOMOLOG,
                          "min_coverage": MIN_COVERAGE, "kmer": K, "min_shared_kmers": MIN_SHARED}}
    for fold in sorted(set(fold_of.values())):
        apos = [a for a, f in fold_of.items() if f == fold]
        near = sorted(a for a in apos if best.get(a, (0,))[0] >= NEAR_IDENTICAL)
        hom = sorted(a for a in apos if best.get(a, (0,))[0] >= HOMOLOG)
        out[fold] = {"proteins": len(apos),
                     "exact_uniprot_in_ligysis": sorted(a for a in apos if uniprot.get(a) in lig_ids),
                     "near_identical_n": len(near), "homolog_n": len(hom),
                     "near_identical": {a: [round(best[a][0], 3), round(best[a][1], 3), best[a][2]] for a in near}}
    Path("finetuned_esm/step9b_ligysis_overlap.json").write_text(json.dumps(out, indent=1))

    # The clean recipe's exclusion lists: for each set a model will score held-out,
    # every LIGYSIS protein homologous to any protein in it (identity >= HOMOLOG,
    # coverage >= MIN_COVERAGE). Exact UniProt matches are included by construction
    # (they align at ~100 %), and are added explicitly in case an alignment missed one.
    excl = {}
    for fold in sorted(set(fold_of.values())):
        apos = {a for a, f in fold_of.items() if f == fold}
        ids = set().union(*(hits[a] for a in apos)) | {uniprot[a] for a in apos if uniprot.get(a) in lig_ids}
        excl[fold] = sorted(ids)
    Path("finetuned_esm/ligysis_exclusions.json").write_text(json.dumps(
        {"rule": f"LIGYSIS proteins >= {HOMOLOG:.0%} identical over >= {MIN_COVERAGE:.0%} coverage "
                 "to any protein in the set, plus exact UniProt matches",
         "counts": {k: len(v) for k, v in excl.items()}, "exclude": excl}, indent=1))
    print("exclusion list sizes:", {k: len(v) for k, v in excl.items()})
    for fold in sorted(set(fold_of.values())):
        o = out[fold]
        print(f"{fold:13s} {o['proteins']:4d} proteins | exact UniProt in LIGYSIS {len(o['exact_uniprot_in_ligysis']):3d} "
              f"| >= {NEAR_IDENTICAL:.0%} id {o['near_identical_n']:3d} | >= {HOMOLOG:.0%} id {o['homolog_n']:3d}")
    print(f"LIGYSIS proteins trained on: {len(lig)}; pairs aligned: {n_aligned}")


if __name__ == "__main__":
    main()
