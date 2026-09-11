"""Static (report/slide) version of figures/benchmark_comparison.html.

Same numbers, same palette, same slot order. Regenerate with:
    python figures/make_benchmark_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── palette (validated dataviz categorical slots, light steps) ───────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED = "#101312", "#4f534f", "#8b8f8a"
GRID, BASELINE, BAND = "#e6e8e2", "#c2c5bd", "#f2efe6"
SURFACE = "#fcfcfb"

# fixed slot order — colour follows the entity, never its rank.
# dp = published precision: our runs are measured to 3 dp, the benchmark
# paper reports 2 dp, and rendering 0.86 as "0.860" would invent a digit.
METHODS = [
    ("tabicl_esm",  "TabICLv2 + ESM (this work)",   3),
    ("tabicl",      "TabICLv2, no pLM (this work)", 3),
    ("plmnn",       "pLM-NN (benchmark)",           2),
    ("p2rank",      "P2Rank (benchmark)",           2),
    ("pocketminer", "PocketMiner (benchmark)",      2),
]
SUBSETS = [
    ("full",  "CB-full",        "231 apo structures",           False),
    ("p2apo", "CB-P2RANK-apo",  "184 single-chain apo",         False),
    ("pm",    "CB-PM",          "PocketMiner-runnable",         False),
    ("holo",  "CB-P2RANK-holo", "holo input — diff. task", True),
]
DATA = {
    "full":  {"tabicl_esm":  dict(auprc=.441, auc=.862, mcc=.374, f1=.912, tpr=.559, fpr=.071),
              "tabicl":      dict(auprc=.388, auc=.848, mcc=.336, f1=.906, tpr=.515, fpr=.073),
              "plmnn":       dict(auprc=.36,  auc=.86,  mcc=.39,  f1=.92,  tpr=.48,  fpr=.05)},
    "p2apo": {"tabicl_esm":  dict(auprc=.469, auc=.861, mcc=.392, f1=.914, tpr=.569, fpr=.068),
              "tabicl":      dict(auprc=.411, auc=.845, mcc=.350, f1=.910, tpr=.512, fpr=.068),
              "plmnn":       dict(auprc=.42,  auc=.88,  mcc=.43,  f1=.93,  tpr=.51,  fpr=.04),
              "p2rank":      dict(auprc=.21,  auc=.81,  mcc=.27,  f1=.81,  tpr=.62,  fpr=.14)},
    "pm":    {"plmnn":       dict(auprc=.43,  auc=.88,  mcc=.44,  f1=.93,  tpr=.52,  fpr=.04),
              "pocketminer": dict(auprc=.19,  auc=.76,  mcc=.22,  f1=.78,  tpr=.51,  fpr=.16)},
    "holo":  {"p2rank":      dict(auprc=.34,  auc=.89,  mcc=.38,  f1=.81,  tpr=.84,  fpr=.15)},
}
METRICS = [
    ("auprc", "AUPRC", 0.60, [0, .2, .4, .6],       False),
    ("auc",   "AUC",   1.00, [0, .25, .5, .75, 1.], False),
    ("mcc",   "MCC",   0.50, [0, .25, .5],          False),
    ("f1",    "F1 (weighted)", 1.00, [0, .25, .5, .75, 1.], False),
    ("tpr",   "TPR",   1.00, [0, .25, .5, .75, 1.], False),
    ("fpr",   "FPR",   0.20, [0, .05, .1, .15, .2], True),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    "font.monospace": ["IBM Plex Mono", "DejaVu Sans Mono"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

fig, axes = plt.subplots(4, 2, figsize=(14.6, 13.6))
fig.subplots_adjust(left=.055, right=.985, top=.918, bottom=.035, hspace=.62, wspace=.13)

SLOT, BARW = 1.0, 0.62  # one slot per method; absent methods keep their slot

for ax, (key, label, ymax, ticks, lower) in zip(axes.ravel(), METRICS):
    for si, (sid, sname, ssub, flag) in enumerate(SUBSETS):
        x0 = si * len(METHODS)
        if flag:
            ax.axvspan(x0 - .5, x0 + len(METHODS) - .5, color=BAND, zorder=0, lw=0)
        for mi, (mid, _, dp) in enumerate(METHODS):
            x = x0 + mi
            row = DATA[sid].get(mid)
            v = row[key] if row else None
            if v is None:
                ax.text(x, ymax * .012, "–", ha="center", va="bottom",
                        color=MUTED, fontsize=9, zorder=3)
                continue
            ax.bar(x, v, width=BARW, color=SERIES[mi], zorder=3)
            ax.text(x, v + ymax * .028, f"{v:.{dp}f}", ha="center", va="bottom",
                    color=INK2, fontsize=6.4, zorder=4, fontfamily="monospace")
        ax.text(x0 + (len(METHODS) - 1) / 2, -ymax * .105, sname, ha="center", va="top",
                fontsize=8.4, color=INK, fontweight="medium")
        ax.text(x0 + (len(METHODS) - 1) / 2, -ymax * .20, ssub, ha="center", va="top",
                fontsize=7, color="#7a6a3d" if flag else MUTED)

    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlim(-.7, len(SUBSETS) * len(METHODS) - .3)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=7.4, color=MUTED,
                       fontfamily="monospace")
    ax.set_xticks([])
    ax.yaxis.grid(True, color=GRID, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(.9)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_title(label + ("   ↓ lower is better" if lower else ""),
                 loc="left", fontsize=10.5, color=INK, fontweight="semibold", pad=9,
                 fontfamily="monospace")

# legend + caveats occupy the sixth cell
lax = axes.ravel()[6]
lax.axis("off")
lax.legend(handles=[Patch(facecolor=SERIES[i], label=n) for i, (_, n, _dp) in enumerate(METHODS)],
           loc="upper left", bbox_to_anchor=(-.02, 1.02), frameon=False,
           fontsize=9, labelspacing=.85, handlelength=1.1, handleheight=1.1)
lax.text(-.02, .40,
         "–  method not evaluated on that subset.\n"
         "CB-PM is defined by which structures PocketMiner failed on, so it\n"
         "cannot be reconstructed from the shipped files.\n\n"
         "Shaded group: P2Rank on holo (ligand-bound) input — a different\n"
         "task, shown for context, not a fifth column of apo results.\n\n"
         "TPR / FPR / MCC read each method at its own threshold\n"
         "(pLM-NN 0.95, PocketMiner 0.7, P2Rank binary, ours frozen on\n"
         "train folds at FPR 0.05). AUC and AUPRC are threshold-free.",
         transform=lax.transAxes, fontsize=7.6, color=INK2, va="top", linespacing=1.6)

nax = axes.ravel()[7]
nax.axis("off")
nax.text(-.02, 1.02,
         "On F1\n"
         "The published F1 is a weighted average over BOTH classes, so at\n"
         "5.28 % positives it is governed by the 95 % of residues that are\n"
         "trivially negative. Every pLM-based row lands at 0.92–0.93; the\n"
         "column separates the structure-only tools and nothing else.\n\n"
         "Our positive-class F1 — the number that carries information — is\n"
         "0.379 (117 feat) and 0.345 (53 feat) on CB-full. It is NOT\n"
         "comparable to the published column and is deliberately not plotted\n"
         "beside it.\n\n"
         "Reproduce every value above with:\n"
         "    python scripts/f1_by_subset.py",
         transform=nax.transAxes, fontsize=7.6, color=INK2, va="top", linespacing=1.6)

fig.suptitle("Per-residue cryptic site prediction on CryptoBench",
             x=.055, y=.975, ha="left", fontsize=15.5, color=INK, fontweight="semibold")
fig.text(.055, .946,
         "All metrics per structure then averaged (CryptoBench Protein.py convention). "
         "Test split: 231 apo structures, 76,636 residues, 5.28 % positive.",
         ha="left", fontsize=8.8, color=INK2)

# Anchored to this file, not the CWD: the README documents running these as
# `python figures/make_benchmark_figure.py` from the repo root, and a bare
# relative path dropped the artifacts there instead of in figures/ -- silently,
# because the script still printed "wrote".
_HERE = Path(__file__).resolve().parent
fig.savefig(_HERE / "benchmark_comparison.png", dpi=200,
            bbox_inches="tight", pad_inches=.28)
fig.savefig(_HERE / "benchmark_comparison.pdf",
            bbox_inches="tight", pad_inches=.28)
print("wrote figures/benchmark_comparison.png and .pdf")
