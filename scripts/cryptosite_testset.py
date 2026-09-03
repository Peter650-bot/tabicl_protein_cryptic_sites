"""Build the CryptoSite 14-pair held-out test set as a labelled residue table.

SOURCE
    Table S5 of the CryptoSite supplementary (Cimermancic et al.), transcribed
    verbatim below: apo+chain, holo+chain, the ligand that binds the cryptic
    site, the protein length, the cryptic-site size, and CryptoSite's own
    FPR/TPR/MCC at thresholds 0.05 / 0.10 / 0.15 plus the AUC.

    The published site sizes are the reason this script can be trusted: we
    re-derive the sites from the structures and check the counts against the
    paper. A mismatch means our definition is not theirs.

GROUND TRUTH
    Cryptic-site residues are the holo residues contacting the ligand, carried
    back to the apo numbering by sequence alignment. The apo structure is what
    the model sees; the holo structure only supplies the label.

WHY IT IS A CLEAN EVALUATION SET
    None of the 14 apo structures appears anywhere in CryptoBench (train or
    test), so a CryptoBench-context TabICL model can be scored on them with no
    leakage. The 79-pair CryptoSite sweep is NOT clean -- six of its structures
    sit in CryptoBench's train split.

Run:  python scripts/cryptosite_testset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import gemmi
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cryptic_tabicl import config as C, structures as S  # noqa: E402

# Table S5, verbatim. apo, apo_chain, holo, holo_chain, ligand, size, site_size,
# then (FPR, TPR, MCC) at 0.05 / 0.10 / 0.15 and the AUC.
MANIFEST = [
    ("1E2X", "A", "1H9G", "A", "MYR", 243, 28, 0.96, (.32, .96, .42), (.08, .89, .68), (.03, .61, .63)),
    ("1MY0", "B", "1N0T", "D", "AT1", 263, 18, 0.94, (.31, 1.00, .36), (.10, .78, .49), (.02, .22, .28)),
    ("1ZAH", "B", "2OT1", "D", "N3P", 363,  9, 0.86, (.32, 1.00, .22), (.12, .56, .20), (.03, .11, .08)),
    ("4HB2", "C", "4HAT", "C", "LMB", 1023, 22, 0.85, (.19, .68, .18), (.04, .32, .19), (.01, .09, .12)),
    ("1BSQ", "A", "1GX8", "A", "RTL", 162, 16, 0.83, (.44, .88, .26), (.21, .75, .37), (.12, .63, .41)),
    ("2GPO", "A", "1S9Q", "B", "CHD", 230, 28, 0.83, (.46, 1.00, .36), (.26, .89, .44), (.14, .32, .16)),
    ("3FDL", "A", "2YXJ", "A", "N3C", 158, 24, 0.82, (.43, .83, .29), (.22, .63, .33), (.10, .50, .40)),
    ("1B6B", "A", "1KUV", "A", "CA5", 174, 32, 0.82, (.39, .91, .40), (.23, .72, .40), (.11, .47, .36)),
    ("1KZ7", "D", "1GRN", "A", "AF3", 188, 13, 0.75, (.54, .92, .20), (.29, .69, .22), (.13, .31, .13)),
    ("1JWP", "A", "1PZO", "A", "CBT", 263, 22, 0.72, (.33, .68, .20), (.10, .27, .15), (.02, .00, -.04)),
    ("3GXD", "B", "2WCG", "A", "MT5", 497, 18, 0.71, (.40, .67, .10), (.12, .39, .15), (.02, .28, .27)),
    ("1BNC", "B", "2V5A", "A", "LZL", 449, 18, 0.69, (.17, .61, .22), (.03, .17, .14), (.01, .00, -.02)),
    ("1Z92", "A", "1PY2", "A", "FRH", 133, 15, 0.65, (.55, .73, .12), (.18, .40, .17), (.10, .20, .10)),
    ("1JBU", "H", "1WUN", "H", "P5B", 254, 23, 0.62, (.46, .61, .09), (.13, .22, .07), (.04, .17, .16)),
]

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def one_letter(records) -> str:
    """Works for gemmi residues (.name) and our ResidueRecord (.resname)."""
    return "".join(AA3TO1.get(getattr(r, "resname", None) or r.name, "X")
                   for r in records)


def ligand_contacts(holo: gemmi.Structure, holo_chain: str, ligand: str,
                    cutoff: float, union: bool = False):
    """Holo polymer residues with any heavy atom within `cutoff` of the ligand.

    A ligand code can appear several times in an entry; we take the copy that
    contacts the named chain most, which is the site the paper annotates.
    """
    model = holo[0]
    chain = next((c for c in model if c.name == holo_chain), None)
    if chain is None:
        return None, "holo chain absent"
    poly = [r for r in chain if r.name in AA3TO1]

    copies = [r for c in model for r in c if r.name == ligand]
    if not copies:
        return None, f"ligand {ligand} absent"

    best, best_hits, all_hits = None, -1, set()
    for lig in copies:
        lig_atoms = [a.pos for a in lig if a.element != gemmi.Element("H")]
        hits = set()
        for i, res in enumerate(poly):
            for a in res:
                if a.element == gemmi.Element("H"):
                    continue
                if any(a.pos.dist(lp) <= cutoff for lp in lig_atoms):
                    hits.add(i)
                    break
        all_hits |= hits
        if len(hits) > best_hits:
            best, best_hits = hits, len(hits)
    return (poly, sorted(all_hits if union else best)), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", type=float, default=5.0,
                    help="heavy-atom distance defining a ligand contact (A)")
    # 5.0 A + union over ligand copies reproduces the published site sizes best
    # (8/14 exact, mean error 3.1); a sweep of 4.0-6.5 A in both modes is worse
    # on both counts, which is the evidence that this is CryptoSite's definition.
    ap.add_argument("--best-copy", dest="union", action="store_false",
                    help="use only the best-contacting ligand copy (default: union of all)")
    ap.set_defaults(union=True)
    ap.add_argument("--cache", type=Path, default=C.STRUCTURE_CACHE)
    ap.add_argument("--out", type=Path, default=Path("data/cryptosite_testset.parquet"))
    args = ap.parse_args()

    rows, report = [], []
    for (apo, ach, holo, hch, lig, size, site_size, auc, t05, t10, t15) in MANIFEST:
        try:
            apo_st = S.load_structure(S.fetch_cif(apo, args.cache))
            holo_st = S.load_structure(S.fetch_cif(holo, args.cache))
        except Exception as e:                        # noqa: BLE001
            report.append(dict(pair=f"{apo}_{holo}", status=f"fetch failed: {e}"))
            continue

        apo_recs = S.residue_records(apo_st, [ach])
        if not apo_recs:
            report.append(dict(pair=f"{apo}_{holo}", status=f"apo chain {ach} absent"))
            continue

        got, err = ligand_contacts(holo_st, hch, lig, args.cutoff, args.union)
        if err:
            report.append(dict(pair=f"{apo}_{holo}", status=err))
            continue
        holo_poly, site_idx = got

        # Carry holo site indices onto the apo chain by sequence alignment.
        a_seq, h_seq = one_letter(apo_recs), one_letter(holo_poly)
        h2a = {}
        for blk in SequenceMatcher(None, h_seq, a_seq, autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                h2a[blk.a + k] = blk.b + k
        mapped = {h2a[i] for i in site_idx if i in h2a}

        for i, r in enumerate(apo_recs):
            rows.append({
                "apo_id": f"{apo}{ach}", "pair": f"{apo}_{holo}",
                "chain": ach, "seq_id": r.seq_id, "icode": r.icode or "",
                "res_ix": i, "res_name": r.resname,
                C.LABEL_COL: int(i in mapped),
            })
        report.append(dict(
            pair=f"{apo}_{holo}", status="ok", ligand=lig,
            apo_len=len(apo_recs), published_len=size,
            site=len(mapped), published_site=site_size,
            holo_contacts=len(site_idx), unmapped=len(site_idx) - len(mapped),
            published_auc=auc))

    rep = pd.DataFrame(report)
    print(rep.to_string(index=False))
    ok = rep[rep.status == "ok"]
    if len(ok):
        print(f"\ncutoff {args.cutoff} A   "
              f"site-size error: mean {(ok.site - ok.published_site).abs().mean():.1f}  "
              f"exact {(ok.site == ok.published_site).sum()}/{len(ok)}   "
              f"length error: exact {(ok.apo_len == ok.published_len).sum()}/{len(ok)}")

    if rows:
        df = pd.DataFrame(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(df):,} residues, "
              f"{df[C.LABEL_COL].sum():,} positive, "
              f"{df[C.LABEL_COL].mean():.2%} prevalence)")
        Path("data/cryptosite_testset_manifest.json").write_text(
            json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
