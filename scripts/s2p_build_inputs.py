"""Step 3 of finetuned_esm/PLAN.md: rebuild their model's inputs and map them to ours.

Writes
    build/s2p_inputs.parquet     one row per (apo_id, chain, position) of THEIR
                                 sequence, with our (seq_id, icode) key beside it
    build/s2p_their_test.txt     their annotation file for the test set, rebuilt
                                 with their own extraction logic (reproduction gate)
    finetuned_esm/step3_mapping.json   every count the gate is read from

Needs biotite 0.41.2 (their pin) and numpy<2 on the path, e.g.
    PYTHONPATH=<dir with biotite 0.41.2 + numpy 1.26.4>:. python3 scripts/s2p_build_inputs.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryptic_tabicl import config as C, dataset as D, s2p_inputs as S  # noqa: E402
from cryptic_tabicl.esm_embed import THREE_TO_ONE  # noqa: E402
from f1_by_subset import single_chain_pdbs  # noqa: E402

KEY = ["apo_id", "chain", "seq_id", "icode"]


def main() -> None:
    ds = D.load_json(C.DATASET_JSON)
    pdb_of = D.apo_pdb_ids(ds)
    table = pd.read_parquet(C.BUILD_DIR / "residues_esm64.parquet",
                            columns=KEY + ["resname", "split", C.LABEL_COL])

    # ---- 1. their sequence for every chain of every apo structure we evaluate
    rows, dup_keys = [], []
    for apo_id, chains in table.groupby("apo_id")["chain"].unique().items():
        cif = C.STRUCTURE_CACHE / f"{pdb_of[apo_id].lower()}.cif.gz"
        for chain in sorted(chains):
            res = S.chain_residues(cif, chain)
            dup_keys += [(apo_id, *k) for k in S.find_duplicate_keys(chain, res)]
            for pos, r in enumerate(res):
                _, seq_id, icode = S.key_of(chain, r)
                rows.append((apo_id, chain, pos, S.three_to_one(r[2]), r[2], seq_id, icode))
    theirs = pd.DataFrame(rows, columns=["apo_id", "chain", "pos", "aa", "their_resname",
                                         "seq_id", "icode"])
    theirs["chain_len"] = theirs.groupby(["apo_id", "chain"])["pos"].transform("size")
    print(f"their sequences: {theirs.groupby(['apo_id','chain']).ngroups} chains, "
          f"{len(theirs):,} residues; duplicate keys: {len(dup_keys)}")

    # ---- 2. join on author keys, both directions
    m = table.merge(theirs, on=KEY, how="outer", indicator=True)
    ours_only = m[m["_merge"] == "left_only"]
    theirs_only = m[m["_merge"] == "right_only"]
    both = m[m["_merge"] == "both"].copy()
    both["our_aa"] = both["resname"].map(THREE_TO_ONE).fillna("X")
    aa_mismatch = both[both["our_aa"] != both["aa"]]

    def by_split(df):
        return {s: int(n) for s, n in df.groupby("split").size().items()}

    summary = {
        "n_table_residues": len(table),
        "n_their_residues": len(theirs),
        "duplicate_keys_in_their_sequences": len(dup_keys),
        "matched": len(both),
        "ours_without_their_position": {
            "n": len(ours_only), "by_split": by_split(ours_only),
            "positives": int(ours_only[C.LABEL_COL].sum()),
            "by_resname": dict(Counter(ours_only["resname"]).most_common(15)),
            "n_structures": int(ours_only["apo_id"].nunique()),
        },
        "theirs_without_our_row": {
            "n": len(theirs_only),
            "by_resname": dict(Counter(theirs_only["their_resname"]).most_common(15)),
            "n_structures": int(theirs_only["apo_id"].nunique()),
        },
        "aa_letter_mismatch_on_matched": {
            "n": len(aa_mismatch),
            "pairs": dict(Counter(zip(aa_mismatch["resname"], aa_mismatch["their_resname"]))
                          .most_common(15)),
        },
        "chains_over_their_max_length": sorted(
            {f"{a}/{c}" for a, c, n in theirs[["apo_id", "chain", "chain_len"]]
             .drop_duplicates().itertuples(index=False) if n > S.THEIR_MAX_LENGTH}),
    }
    summary["ours_without_their_position"]["by_resname"] = {
        str(k): v for k, v in summary["ours_without_their_position"]["by_resname"].items()}
    summary["aa_letter_mismatch_on_matched"]["pairs"] = {
        f"{a}->{b}": v for (a, b), v in summary["aa_letter_mismatch_on_matched"]["pairs"].items()}

    # ---- 3. their test file, rebuilt with their own extraction logic
    test_entries = D.load_json(C.FOLDS_DIR / "test.json")
    their = S.their_test_entries(test_entries)
    lines, label_rows = [], []
    for pdb_id, (chain, binding) in sorted(their.items()):
        res = S.chain_residues(C.STRUCTURE_CACHE / f"{pdb_id.lower()}.cif.gz", chain)
        seq = S.sequence_of(res)
        pos = S.label_positions([r[0] for r in res], binding)
        lines.append(S.their_line(pdb_id, chain, seq, pos))
        apo_id = f"{pdb_id}{chain}"
        pos_set = set(pos)
        for i, r in enumerate(res):
            _, seq_id, icode = S.key_of(chain, r)
            label_rows.append((apo_id, chain, seq_id, icode, int(i in pos_set)))
    (C.BUILD_DIR / "s2p_their_test.txt").write_text("\n".join(lines))
    lens = [len(l.rsplit(";", 1)[1]) for l in lines]

    their_labels = pd.DataFrame(label_rows, columns=KEY + ["their_label"])
    lab = their_labels.merge(table[KEY + [C.LABEL_COL]], on=KEY, how="left")
    single = single_chain_pdbs()
    their_pdbs = {p.lower() for p in their}
    summary["their_test_file"] = {
        "n_entries": len(lines),
        "n_over_max_length (they skip)": sum(n > S.THEIR_MAX_LENGTH for n in lens),
        "n_residues": sum(lens),
        "n_positive_by_their_labels": int(their_labels["their_label"].sum()),
        "paper_reports": {"structures": 184, "residues": 55572, "binding": 3158},
        "apo_ids_absent_from_our_table": sorted(set(their_labels["apo_id"]) - set(table["apo_id"])),
        "label_disagreement_vs_ours": {
            "their1_our0": int(((lab["their_label"] == 1) & (lab[C.LABEL_COL] == 0)).sum()),
            "their0_our1": int(((lab["their_label"] == 0) & (lab[C.LABEL_COL] == 1)).sum()),
            "no_our_row": int(lab[C.LABEL_COL].isna().sum()),
        },
        "vs_our_CB_P2RANK_apo": {
            "ours": len(single), "theirs": len(their_pdbs),
            "only_ours": sorted(single - their_pdbs), "only_theirs": sorted(their_pdbs - single),
        },
    }

    theirs.to_parquet(C.BUILD_DIR / "s2p_inputs.parquet", index=False)
    out = Path("finetuned_esm/step3_mapping.json")
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
