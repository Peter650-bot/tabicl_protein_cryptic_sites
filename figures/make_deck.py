"""Build the pipeline slide deck (16:9, white background).

Every figure comes from figures/slides/, produced by make_slide_figures.py.
Every number is quoted from RESULTS.md or build/*.json.
"""
from __future__ import annotations

import os
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
SLIDES = HERE / "slides"
OUT = Path(os.environ.get("DECK_OUT", HERE.parent / "Cryptic_site_prediction_pipeline.pptx"))

W, H = Inches(13.333), Inches(7.5)
INK = RGBColor(0x1A, 0x20, 0x2C)
MUTED = RGBColor(0x4A, 0x55, 0x68)
BLUE = RGBColor(0x2B, 0x6C, 0xB0)
ORANGE = RGBColor(0xDD, 0x6B, 0x20)
GREEN = RGBColor(0x2F, 0x85, 0x5A)
RED = RGBColor(0xC5, 0x30, 0x30)
PURPLE = RGBColor(0x6B, 0x46, 0xC1)
RULE = RGBColor(0xCB, 0xD5, 0xE0)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, size=16, color=INK, bold=False, space_before=0, space_after=6,
         first=False, align=PP_ALIGN.LEFT, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = FONT
    return p


def new(title, kicker=None):
    """Slide with a title, a coloured rule, and an optional kicker line."""
    s = prs.slides.add_slide(BLANK)
    tf = textbox(s, 0.62, 0.38, 12.1, 0.65)
    para(tf, title, size=30, bold=True, first=True, space_after=0)
    line = s.shapes.add_shape(1, Inches(0.62), Inches(1.10), Inches(1.5), Pt(3))
    line.fill.solid(); line.fill.fore_color.rgb = BLUE
    line.line.fill.background(); line.shadow.inherit = False
    if kicker:
        tf = textbox(s, 0.62, 1.26, 12.1, 0.4)
        para(tf, kicker, size=15, color=MUTED, first=True, space_after=0)
    return s


def picture(slide, name, top, height=None, width=None):
    p = SLIDES / f"{name}.png"
    if width:
        pic = slide.shapes.add_picture(str(p), 0, Inches(top), width=Inches(width))
    else:
        pic = slide.shapes.add_picture(str(p), 0, Inches(top), height=Inches(height))
    pic.left = int((W - pic.width) / 2)
    return pic


def note(slide, text, top=6.75, color=MUTED, size=13, bold=False):
    tf = textbox(slide, 0.62, top, 12.1, 0.5)
    para(tf, text, size=size, color=color, bold=bold, first=True, space_after=0)


def table(slide, rows, x, y, w, col_w=None, header=True, size=13, row_h=0.36):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * nr)).table
    if col_w:
        for i, cw in enumerate(col_w):
            shp.columns[i].width = Inches(cw)
    for ri, row in enumerate(rows):
        shp.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = shp.cell(ri, ci)
            cell.text = str(val)
            cell.margin_left = Inches(0.10); cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.02); cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(size)
                r.font.name = FONT
                r.font.bold = header and ri == 0
                r.font.color.rgb = INK if (header and ri == 0) else MUTED
    return shp


def bullets(slide, items, x=0.62, y=1.85, w=12.1, h=4.6, size=17, gap=13):
    """items: (text, colour, bold) or plain str."""
    tf = textbox(slide, x, y, w, h)
    for i, it in enumerate(items):
        text, color, bold = it if isinstance(it, tuple) else (it, INK, False)
        para(tf, text, size=size, color=color, bold=bold,
             first=(i == 0), space_after=gap)
    return tf


# ============================================================ 1. title
s = prs.slides.add_slide(BLANK)
tf = textbox(s, 1.1, 2.25, 11.1, 1.6)
para(tf, "Predicting cryptic binding sites", size=44, bold=True, first=True, space_after=4)
para(tf, "one residue at a time, from the unbound structure alone", size=24, color=MUTED)
line = s.shapes.add_shape(1, Inches(1.1), Inches(4.15), Inches(2.2), Pt(4))
line.fill.solid(); line.fill.fore_color.rgb = ORANGE
line.line.fill.background(); line.shadow.inherit = False
tf = textbox(s, 1.1, 4.55, 11.1, 1.4)
para(tf, "Zero-shot TabICLv2  ·  118 apo-only features  ·  CryptoBench",
     size=20, color=BLUE, bold=True, first=True, space_after=10)
para(tf, "No training. No fine-tuning. No holo structure at prediction time.",
     size=16, color=MUTED)

# ============================================================ 2. the problem
s = new("The problem", "Most druggable pockets do not exist in the structure you can crystallise.")
picture(s, "01_cryptic_site", top=1.72, height=4.75)

# ============================================================ 3. dataset
s = new("The dataset:  CryptoBench",
        "The largest curated set of cryptic sites — built from PDB apo/holo pairs, labelled per residue.")
picture(s, "02_dataset", top=1.80, width=10.3)
note(s, "A residue is POSITIVE if it lines the ligand pocket in ANY holo partner of that "
        "apo structure — the union over all 5,493 pairings, not just the first.", top=6.62)

# ============================================================ 4. rules
s = new("CryptoBench's rules for testing",
        "Four constraints. Break any one and the number stops meaning anything.")
bullets(s, [
    ("1.  Split by sequence identity, never at random.", INK, True),
    ("     Homologues would otherwise sit on both sides. We use CryptoBench's own shipped folds — "
     "we never make our own split.", MUTED, False),
    ("2.  A protein is the unit. All of its residues stay together.", INK, True),
    ("     Residues within one structure are correlated; splitting inside a protein leaks.", MUTED, False),
    ("3.  Score per structure, then average the scores.", INK, True),
    ("     Not pooled over all 76,636 residues — that would let big proteins dominate. "
     "This is CryptoBench's own Protein.py convention.", MUTED, False),
    ("4.  Nothing derived from the holo structure may become a feature.", INK, True),
    ("     A cryptic site is DEFINED by the apo→holo change, so any holo-derived column "
     "reconstructs the label. A test fails the build on one.", MUTED, False),
], y=1.80, size=16, gap=7)
note(s, "The decision threshold is chosen on training folds and frozen before the test set is "
        "touched. Each published method sets its own (pLM-NN 0.95, PocketMiner 0.7, P2Rank none).",
     top=6.55)

# ============================================================ 5. pipeline
s = new("The pipeline", "One apo PDB file in, one probability per residue out.")
picture(s, "03_pipeline", top=1.90, width=10.8)
note(s, "Steps run once, offline, and are cached: the ensemble and ESM stages are the expensive ones "
        "(Modal GPU/CPU); Tier A and conservation are cheap.", top=6.55)

# ============================================================ 6. features
s = new("118 features", "Four independent sources of evidence about one residue.")
picture(s, "04_features", top=1.72, width=8.6)
table(s, [
    ["source", "n", "what it measures", "how it is extracted"],
    ["Tier A — geometry", "34", "is this residue in a pocket-like place?",
     "computed locally from the apo PDB (gemmi + scipy)"],
    ["Ensemble — pockets", "14", "does a pocket OPEN when the protein moves?",
     "ANM normal modes → 24 conformations → fpocket on each"],
    ["Conservation", "6", "is evolution protecting this position?",
     "PDBe Graph API — the same source CryptoBench used"],
    ["ESM-2", "64", "what does a language model make of the sequence?",
     "esm2_t36_3B → 2560-d → PCA to 64, fit on train only"],
], x=0.62, y=4.45, w=12.1, col_w=[2.5, 0.6, 4.2, 4.8], size=12.5, row_h=0.38)
note(s, "Every one of the 118 is computable from the apo structure alone. That invariant is what "
        "makes the result honest, and a test enforces it.", top=6.88, color=RED, bold=True, size=12.5)

# ============================================================ 7. Tier A
s = new("Tier A  ·  34 geometric features", "Cheap, local, and the backbone of the model.")
bullets(s, [
    ("Shape and exposure", BLUE, True),
    ("sasa · rsa · depth_to_surface · protrusion · convexity · half-sphere exposure (up/down)",
     MUTED, False),
    ("Packing and neighbourhood", BLUE, True),
    ("packing · n_neighbors at 4 Å / 6 Å / 8 Å", MUTED, False),
    ("Chemistry, flexibility, position", BLUE, True),
    ("residue type · Kyte–Doolittle hydropathy · formal charge · B-factor and its z-score · "
     "secondary structure · φ/ψ · position along the chain", MUTED, False),
    ("Plus an 8 Å neighbourhood mean of 12 of them", GREEN, True),
    ("CryptoSite found the “…of neighbours” variants rank near the top: a residue's "
     "environment carries as much signal as the residue itself.", MUTED, False),
], y=1.80, size=16, gap=8)
note(s, "Definitions are taken verbatim from CryptoSite where it gives a formula (packing, "
        "n_neighbors, depth_to_surface); where it only cites prior work, we use a named documented "
        "proxy and say so.", top=6.45)

# ============================================================ 8. ensemble
s = new("Ensemble features  ·  14", "The one idea that is specific to crypticity: make the protein move.")
picture(s, "05_ensemble", top=1.85, width=12.2)
note(s, "A cryptic site is closed in the crystal and opens under thermal motion, so scoring a single "
        "static structure cannot see it. ANM gives the low-frequency collective modes from the Cα "
        "contact map alone — no MD. The trade is real: it will not find a site that needs a side-chain "
        "flip or a loop to cross a barrier.", top=6.10)

# ============================================================ 9. conservation
s = new("Conservation  ·  6", "Evolution has already labelled the functional sites for us.")
bullets(s, [
    ("Align a protein against its homologues. A column that never varies is doing something the "
     "organism cannot afford to change — and functional sites are under exactly that selection.",
     INK, False),
    ("cons_score  ·  cons_entropy  ·  cons_prob_self     + the 8 Å neighbourhood mean of each",
     PURPLE, True),
    ("Source: the PDBe Graph API, which serves precomputed per-residue conservation. The "
     "CryptoBench authors used the same API to analyse their own dataset, so the numbers are "
     "consistent with the benchmark rather than a different pipeline's.", MUTED, False),
    ("It returns full per-amino-acid probability distributions, so we compute proper Shannon "
     "entropy — CryptoSite's actual definition — rather than a binned integer grade.", MUTED, False),
    ("Coverage 1,136 / 1,136 structures. Median join match rate 1.000. ~2,300 API calls.",
     GREEN, True),
], y=1.80, size=16, gap=14)
note(s, "The strongest individual features in the project: cons_prob_self_nb8 reaches 0.673 AUC "
        "on its own.", top=6.35, color=GREEN, bold=True)

# ============================================================ 10. ESM
s = new("ESM-2  ·  64", "The same protein language model the published benchmark uses.")
bullets(s, [
    ("facebook/esm2_t36_3B_UR50D  →  2,560 dimensions per residue", ORANGE, True),
    ("We embed the observed residues of one chain, in order, as a single string. Chains longer "
     "than 1,022 tokens are chunked with a 256-residue overlap, and only the middle of each chunk "
     "is kept, so every embedding had real context on both sides.", MUTED, False),
    ("2,560 raw dimensions would swamp 54 hand-built features and blow past TabICL's context "
     "budget, so they are reduced by PCA to 64 components — fit on TRAIN ROWS ONLY, then applied "
     "to test.", INK, False),
    ("Bookkeeping is the part that goes wrong silently.", RED, True),
    ("159 residues share an (apo_id, chain, seq_id) key with another — insertion codes, where two "
     "residues are numbered 52A and 52B. The residue key now carries the insertion code everywhere, "
     "and all three feature merges join on that four-part key under a 1:1 assertion.", MUTED, False),
], y=1.80, size=16, gap=13)
note(s, "Gains shrink but do not stop: 16 → 32 components adds +0.008 AUPRC, 32 → 64 adds +0.004 — the last step sits at the context-draw noise floor.",
     top=6.45)

# ============================================================ 11. TabICL
s = new("What TabICLv2 actually does", "It is a tabular foundation model. There is no training step.")
picture(s, "06_tabicl", top=1.80, width=12.0)

# ============================================================ 12. results
s = new("Results", "Against the published CryptoBench benchmark, on the same test split.")
picture(s, "07_results", top=1.78, width=10.6)
note(s, "The 54-feature row is the stronger claim — it beats pLM-NN using no language model at all. "
        "The 118-feature row is the better predictor. Neither supersedes the other.",
     top=6.60, color=INK)

# ============================================================ 13. other models
s = new("The other methods on CryptoBench", "Only CB-full is a like-for-like comparison with us.")
table(s, [
    ["Method", "Subset", "AUC", "AUPRC", "TPR", "FPR", "MCC", "F1 (wtd)"],
    ["TabICLv2 + ESM  (this work)", "CB-full", "0.882", "0.476", "0.591", "0.062", "0.410", "0.921"],
    ["TabICLv2, no pLM  (this work)", "CB-full", "0.867", "0.450", "0.565", "0.064", "0.389", "0.918"],
    ["pLM-NN  (benchmark)", "CB-full", "0.86", "0.36", "0.48", "0.05", "0.39", "0.92"],
    ["TabICLv2 + ESM  (this work)", "CB-P2RANK-apo", "0.885", "0.501", "0.602", "0.062", "0.425", "0.922"],
    ["pLM-NN", "CB-P2RANK-apo", "0.88", "0.42", "0.51", "0.04", "0.43", "0.93"],
    ["P2Rank", "CB-P2RANK-apo", "0.81", "0.21", "0.62", "0.14", "0.27", "0.81"],
    ["pLM-NN", "CB-PM", "0.88", "0.43", "0.52", "0.04", "0.44", "0.93"],
    ["PocketMiner", "CB-PM", "0.76", "0.19", "0.51", "0.16", "0.22", "0.78"],
], x=0.62, y=1.80, w=12.1, col_w=[3.5, 2.1, 1.1, 1.1, 1.0, 1.0, 1.1, 1.2], size=13, row_h=0.38)
bullets(s, [
    ("The subsets are not interchangeable.", INK, True),
    ("CB-P2RANK-apo is the 184 single-chain test structures P2Rank could run on — we reconstruct it "
     "from the shipped dataset. CB-PM is defined by which 60 structures PocketMiner failed on, which "
     "is a property of that tool's run, not of the data, so it cannot be reconstructed.", MUTED, False),
    ("Quote AUPRC, not F1. At 5 % prevalence a weighted F1 is governed by the 95 % of residues that "
     "are trivially negative — every pLM-based row sits at 0.92–0.93.", RED, True),
], y=5.50, h=1.85, size=14, gap=8)

# ============================================================ 14. what moved it
s = new("What actually moved the number", "Ranked by ΔAUPRC, the honest summary at 5 % prevalence.")
picture(s, "08_gains", top=1.82, width=10.8)
note(s, "A separate bug — a missing altLoc column, which shifted every field in the PDB writer — was "
        "worth +0.144 AUPRC on the ensemble features alone. fpocket parses positionally, so it still "
        "ran and still reported pockets; it simply typed the atoms wrongly.", top=6.35)

# ============================================================ 15. weakness
s = new("Where the model is weak", "One stratum resists everything we have tried.")
picture(s, "09_strata", top=1.80, width=10.2)
note(s, "Every addition lifts BOTH strata by about the same amount, so the gap never closes: "
        "28,000 fpocket runs over conformational ensembles are the weakest of the three (+0.005 here), "
        "and ESM buys +0.024 on this stratum against +0.027 on the bulk. What separates these two "
        "classes appears to be a property of MOTION, not of sequence or of static shape.",
     top=6.30, color=INK)

# ============================================================ 16. status
s = new("Where it stands", None)
bullets(s, [
    ("Settled", GREEN, True),
    ("0.882 AUC / 0.476 AUPRC on CB-full, zero-shot. AUPRC and TPR beat pLM-NN by margins that "
     "survive resampling the 231 test proteins. MCC does not, and the AUC margin holds only for the "
     "118-feature run on CB-full — not on CB-P2RANK-apo, and not without ESM.", MUTED, False),
    ("Measured, not assumed", GREEN, True),
    ("Two separate uncertainties are quantified: test-set sampling (95 % CIs, 10,000 bootstrap "
     "resamples) and the 50k context draw (12 seeds on Modal). Together they set a noise floor — "
     "differences below ~0.008 AUPRC are not real.", MUTED, False),
    ("Pending", ORANGE, True),
    ("External validation on CryptoSite's 14 held-out apo/holo pairs. The evaluation set is built "
     "(4,278 residues, 5.9 % positive, zero overlap with CryptoBench); scoring is not yet run.",
     MUTED, False),
    ("Open", RED, True),
    ("The hard stratum at 0.714 AUC. Telling a cryptic site from a pocket that is already open is "
     "the discrimination the task is actually about.", MUTED, False),
], y=1.70, size=16, gap=10)

prs.save(OUT)
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
