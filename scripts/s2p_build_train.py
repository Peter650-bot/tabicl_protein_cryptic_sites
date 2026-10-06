"""Step 8a of finetuned_esm/PLAN.md: their training files, one per CryptoBench train fold.

Built with exactly the extraction that rebuilt their test file residue-for-residue
(`s2p_inputs.their_annotation_rows`): single-chain entries only, chain from the first
holo record, their label loop. Their loader's length filter (> 1,024 skipped) is
applied by the trainer, not here, so the file still records what it drops.

Writes build/s2p_train/train-fold-{0..3}.txt and finetuned_esm/step8a_train_files.json.
Needs biotite 0.41.2 + numpy<2 on PYTHONPATH (as scripts/s2p_build_inputs.py).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C, dataset as D, s2p_inputs as S  # noqa: E402

FOLDS = [f"train-fold-{i}" for i in range(4)]


def main() -> None:
    out_dir = C.BUILD_DIR / "s2p_train"
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for fold in FOLDS:
        entries = D.load_json(C.FOLDS_DIR / f"{fold}.json")
        rows = S.their_annotation_rows(entries, C.STRUCTURE_CACHE)
        lines = [S.their_line(p, c, S.sequence_of(res), pos) for p, c, res, pos in rows]
        text = "\n".join(lines)
        (out_dir / f"{fold}.txt").write_text(text)
        lens = [len(res) for _, _, res, _ in rows]
        kept = [i for i, n in enumerate(lens) if n <= S.THEIR_MAX_LENGTH]
        summary[fold] = {
            "entries_in_fold": len(entries),
            "single_chain_entries": len(rows),
            "skipped_multichain": len(entries) - len(rows),
            "empty_chains": sum(n == 0 for n in lens),
            "over_max_length_skipped_by_loader": len(rows) - len(kept),
            "proteins_trained_on": len(kept),
            "residues_trained_on": sum(lens[i] for i in kept),
            "positives_trained_on": sum(len(rows[i][3]) for i in kept),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    tot = {k: sum(v[k] for v in summary.values())
           for k in ("proteins_trained_on", "residues_trained_on", "positives_trained_on")}
    tot["positive_rate"] = tot["positives_trained_on"] / tot["residues_trained_on"]
    # Their pos_weight 8.7481 is 1/(2p) for the balanced-weight formula, so it pins p.
    tot["their_pos_weight_implies_rate"] = 1 / (2 * 8.7481)
    summary["all_folds"] = tot
    Path("finetuned_esm/step8a_train_files.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
