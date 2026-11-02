"""PAM's feature-family ablation (finetuned_esm/PLAN.md step 13) as pooled ROC and
precision-recall curves, one line per nested feature set.

WHY POOLED CURVES
    A per-protein AUPRC is a mean of 263 curves and has no single curve to draw.
    The pooled curve (every residue of the subset ranked together) is the one a
    reader can see; the legend quotes its area (pooled AUC / AUPRC, the estimator of
    final_comparison.pooled_all). The three pLM arms overlap almost exactly, so each
    line has its own colour, line style and marker.

SAME SET AS THE MANUSCRIPT TABLES
    CB-full here is every positive chain of the 222 structures with 7ndl KEPT
    (263 chains, 74,048 residues), the set of the pooled table; step13_ablation.json
    used 262 (7ndl dropped), so the legend values are computed here, not read from it.
    CB-P2RANK-184 has no 7ndl difference and is checked against step13_ablation.json.

JOURNAL FORMAT (OUP)
    Pure white background, 178 mm (double column) wide, no title in the image (it goes
    in \\caption), panel labels (a)/(b), all text >= 7 pt at final size, Okabe-Ito
    colours (CVD-checked), TIFF at 600 dpi plus PDF and PNG.

Data:  build/ftemb_stack/{ab1,fe1}/  (scripts/ftemb_ablation_report.py --run-id ab1)
Run:   python figures/make_ablation_figure.py
Out:   figures/ablation_curves.{pdf,png,tif} (CB-full), figures/ablation_curves_184.{pdf,png,tif}
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from sklearn import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import ftemb_ablation_report as R  # noqa: E402
import published_comparison as P  # noqa: E402

RESULT = ROOT / "finetuned_esm" / "step13_ablation.json"
THRESHOLDS = ROOT / "finetuned_esm" / "step13_thresholds.json"

# Okabe-Ito, adjacent pairs CVD ΔE >= 9.6 (dataviz validate_palette.js, white surface);
# PAM in black and drawn first and thickest, so the two overlapping arms stay visible on top
ARMS = [("fam33", "33  geometry + physico-chemistry", "#56B4E9", "-",           "o"),
        ("fam47", "47  + ensemble pocket",            "#E69F00", (0, (6, 2)),   "s"),
        ("fam53", "53  + conservation",               "#0072B2", (0, (6, 2, 1.5, 2)), "^"),
        ("ft69",  "69  + FT ESM-2 PCs 1–16",          "#CC79A7", (0, (3, 1.5)), "D"),
        ("ft85",  "85  + FT ESM-2 PCs 1–32",          "#D55E00", (0, (1, 1.2)), "v"),
        ("ft117", "117 + FT ESM-2 PCs 1–64 (PAM)",    "#000000", "-",           None)]
DRAW_ORDER = ["ft117", "fam33", "fam47", "fam53", "ft69", "ft85"]
INK, INK2, GRID = "#000000", "#333333", "#d9d9d9"
MM = 1 / 25.4
WIDTH, HEIGHT = 178 * MM, 110 * MM
SUBSETS = {"CB-full": "ablation_curves", "CB-P2RANK-184": "ablation_curves_184"}
# Table B (pooled), PAM: the figure must quote the same numbers
TABLE_B_PAM = {"CB-full": (0.903, 0.432), "CB-P2RANK-184": (0.906, 0.488)}

# Arial if this machine has it (narrower than DejaVu: the two-column legend fits 178 mm)
ARIAL = [f for f in (Path("/mnt/c/Windows/Fonts") / n for n in ("arial.ttf", "arialbd.ttf")) if f.exists()]
for f in ARIAL:
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Arial" if ARIAL else "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "pdf.fonttype": 42})

if not RESULT.exists():
    sys.exit("finetuned_esm/step13_ablation.json missing: run scripts/ftemb_ablation_report.py --run-id ab1")
res = json.loads(RESULT.read_text())
units = P.cryptobench_units(R.load_test("ab1", json.loads(THRESHOLDS.read_text())))
positive = units.groupby("unit")["y"].transform("sum").gt(0).to_numpy()
masks = {"CB-full": positive, "CB-P2RANK-184": P.subsets(units)["CB-P2RANK-184"]}

for name, stem in SUBSETS.items():
    sub = units[masks[name]]
    y = sub["y"].to_numpy()
    print(f"{name}: {sub['pdb'].nunique()} structures, {sub['unit'].nunique()} chains, "
          f"{len(y):,} residues, {int(y.sum()):,} positive")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))
    handles = {}
    for i, (arm, label, color, ls, marker) in sorted(enumerate(ARMS), key=lambda t: DRAW_ORDER.index(t[1][0])):
        s = sub[arm].to_numpy()
        fpr, tpr, _ = metrics.roc_curve(y, s)
        prec, rec, _ = metrics.precision_recall_curve(y, s)
        auc, auprc = metrics.roc_auc_score(y, s), metrics.auc(rec, prec)
        if name == "CB-P2RANK-184":
            for m, v in (("AUC", auc), ("AUPRC", auprc)):
                assert abs(res[name]["pooled"][arm][m]["value"] - v) < 1e-9, (name, arm, m)
        if arm == "ft117":
            assert (round(auc, 3), round(auprc, 3)) == TABLE_B_PAM[name], (name, auc, auprc)
        print(f"  {arm:6s} pooled AUC {auc:.4f}  AUPRC {auprc:.4f}")
        style = dict(color=color, ls=ls, lw=1.8 if arm == "ft117" else 1.0, marker=marker,
                     ms=3.6, mew=0, markevery=(0.12 + 0.03 * i, 0.24))
        handles[arm] = ax_roc.plot(fpr, tpr, label=f"{label}   AUC {auc:.3f} · AUPRC {auprc:.3f}", **style)[0]
        ax_pr.plot(rec, prec, **style)
    prevalence = y.mean()
    ax_roc.plot([0, 1], [0, 1], color=INK2, lw=0.7, ls=":")
    ax_pr.axhline(prevalence, color=INK2, lw=0.7, ls=":")
    ax_pr.text(0.02, prevalence + 0.015, f"prevalence {prevalence:.3f}", color=INK2,
               fontsize=7.5, ha="left", va="bottom")
    for ax, xl, yl, panel in ((ax_roc, "False positive rate", "True positive rate", "(a) ROC"),
                              (ax_pr, "Recall", "Precision", "(b) Precision–recall")):
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel=xl, ylabel=yl)
        ax.set_title(panel, fontsize=8.5, color=INK, loc="left", fontweight="bold")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
    order = [a for a, *_ in ARMS]
    fig.legend([handles[a] for a in order], [handles[a].get_label() for a in order],
               loc="lower center", ncol=2, fontsize=7.5, frameon=False, handlelength=2.8,
               labelcolor=INK, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=2.5)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=350)
    fig.savefig(OUT / f"{stem}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    Image.open(OUT / f"{stem}.tif").convert("RGB").save(OUT / f"{stem}.tif", dpi=(600, 600),
                                                       compression="tiff_lzw")   # no alpha channel
    plt.close(fig)
    print(f"wrote figures/{stem}.pdf/.png/.tif")
