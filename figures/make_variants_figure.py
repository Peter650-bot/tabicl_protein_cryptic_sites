"""The seven ESM-2 3B variants (tab:variants) on CB-full: per-protein AUC and AUPRC with
95 % bootstrap intervals, one row per variant, PAM highlighted.

CB-full here is the manuscript's unit: 263 chains of 222 structures, 7ndl kept, metrics
per chain then averaged; intervals resample whole structures (10,000 draws, seed 0).
A vertical dashed line marks PAM's value; a filled marker means the paired interval of
variant - PAM excludes zero, an open marker that it does not. The key below the panels
says so, so the figure reads without its caption.

JOURNAL FORMAT (Bioinformatics, academic.oup.com/bioinformatics/pages/author-guidelines)
    Fitted to a single column (86 mm) at final size; colour/line art, so TIFF at 1200 dpi
    (the guideline's line-drawing resolution, above its 350 dpi for colour) plus vector PDF.
    White background, no title in the image (it goes in \caption), panel labels (a)/(b),
    Arial, all text >= 7 pt at final size, Okabe-Ito vermilion as the only hue, and
    significance shown by fill as well as colour.

Data:  finetuned_esm/variants_cbfull.json
Run:   python figures/make_variants_figure.py
Out:   figures/variants_auc_auprc.{pdf,png,tif}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
DATA = ROOT / "finetuned_esm" / "variants_cbfull.json"
STEM = "variants_auc_auprc"

VARIANTS = [("V1", "1  FT ESM-2 3B (released)"),
            ("V2", "2  Fixed rank blend"),
            ("V3", "3  Stack (54)"),
            ("V4", "4  PAM (117)"),
            ("V5", "5  PCA only (64)"),
            ("V6", "6  Struct. + PLS (117)"),
            ("V7", "7  PLS only (64)")]
PANELS = (("AUC", "(a) AUC", (0.78, 0.82, 0.86, 0.90)), ("AUPRC", "(b) AUPRC", (0.36, 0.42, 0.48, 0.54)))
PAM, INK, INK2, GRID, ACCENT = "V4", "#000000", "#333333", "#d9d9d9", "#D55E00"
MM = 1 / 25.4
WIDTH, HEIGHT = 86 * MM, 66 * MM

ARIAL = [f for f in (Path("/mnt/c/Windows/Fonts") / n for n in ("arial.ttf", "arialbd.ttf")) if f.exists()]
for f in ARIAL:
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Arial" if ARIAL else "DejaVu Sans", "font.size": 7,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "pdf.fonttype": 42})

r = json.loads(DATA.read_text())["CB-full_263"]
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True)
ys = range(len(VARIANTS))[::-1]
for ax, (metric, title, ticks) in zip(axes, PANELS):
    pam = r["per_protein"][PAM][metric]
    ax.axvline(pam, color=ACCENT, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for y, (v, _) in zip(ys, VARIANTS):
        val = r["per_protein"][v][metric]
        lo, hi = r["per_protein_ci"][v][metric]
        color = ACCENT if v == PAM else INK2
        if v == PAM:
            filled = True
        else:
            dlo, dhi = r["per_protein_diff_vs_PAM"][v][metric]["ci"]
            filled = dlo > 0 or dhi < 0
        ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot(val, y, "o", ms=4.2, mew=0.9, mec=color, mfc=color if filled else "white", zorder=3)
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_title(title, fontsize=7.5, color=INK, loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Per-protein {metric}")
axes[0].set_yticks(list(ys), [lab for _, lab in VARIANTS])
axes[0].set_ylim(-0.6, len(VARIANTS) - 0.4)
for tick, (v, _) in zip(axes[0].get_yticklabels(), VARIANTS):
    if v == PAM:
        tick.set_fontweight("bold")
key = [Line2D([], [], color=ACCENT, lw=1.0, marker="o", ms=4.2, mew=0.9, mfc=ACCENT, mec=ACCENT),
       Line2D([], [], color=ACCENT, lw=0.7, ls=(0, (3, 2))),
       Line2D([], [], color=INK2, lw=1.0, marker="o", ms=4.2, mew=0.9, mfc=INK2, mec=INK2),
       Line2D([], [], color=INK2, lw=1.0, marker="o", ms=4.2, mew=0.9, mfc="white", mec=INK2)]
fig.legend(key, ["PAM, with 95% CI", "PAM's value",
                 "Differs from PAM (paired CI excludes 0)", "Does not differ from PAM"],
           loc="lower center", ncol=2, fontsize=7, frameon=False, handlelength=2.2,
           columnspacing=1.2, labelcolor=INK)
fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.0)
fig.savefig(OUT / f"{STEM}.pdf")
fig.savefig(OUT / f"{STEM}.png", dpi=350)
fig.savefig(OUT / f"{STEM}.tif", dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
Image.open(OUT / f"{STEM}.tif").convert("RGB").save(OUT / f"{STEM}.tif", dpi=(1200, 1200),
                                                   compression="tiff_lzw")   # no alpha channel
plt.close(fig)
print(f"wrote figures/{STEM}.pdf/.png/.tif")
