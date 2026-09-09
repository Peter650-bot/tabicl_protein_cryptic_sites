"""Figures for the pipeline slide deck. White background, large type, one idea each.

Every number here is copied from RESULTS.md / build/*.json. Nothing is invented;
if a value is not measured it does not appear.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Wedge, Rectangle
from pathlib import Path

OUT = Path(__file__).resolve().parent / "slides"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GRAY, GREEN, RED, PURPLE = (
    "#2b6cb0", "#dd6b20", "#a0aec0", "#2f855a", "#c53030", "#6b46c1")
INK = "#1a202c"
MUTED = "#4a5568"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.edgecolor": "#cbd5e0", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, label, sub=None, fc="white", ec=BLUE, fs=13, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.16 * h if sub else 0), label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.20 * h, sub, ha="center", va="center",
                fontsize=fs - 3, color=MUTED, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 color=color, lw=lw, zorder=1))


# ------------------------------------------------------------------ 1. cryptic site
def fig_cryptic():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    th = np.linspace(0, 2 * np.pi, 400)

    def blob_r(a):
        return 1.0 + 0.05 * np.sin(3 * a) + 0.035 * np.cos(5 * a)

    for ax, holo in zip(axes, (False, True)):
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.95, 1.75)
        ax.set_aspect("equal"); ax.axis("off")
        body = ax.fill(blob_r(th) * np.cos(th), blob_r(th) * np.sin(th),
                       color="#dbe6f5", ec=BLUE, lw=2.4, zorder=2)[0]

        # A wedge cut in from the right. Both rays start at the apex and run well
        # past the surface; everything is clipped to the body so the notch mouth
        # follows the real boundary instead of a straight chord.
        half = np.radians(21)
        apex_t, apex_b = np.array([0.24, 0.085]), np.array([0.24, -0.085])
        out_t = apex_t + 1.7 * np.array([np.cos(half), np.sin(half)])
        out_b = apex_b + 1.7 * np.array([np.cos(half), -np.sin(half)])
        xs = [out_t[0], apex_t[0], apex_b[0], out_b[0]]
        ys = [out_t[1], apex_t[1], apex_b[1], out_b[1]]

        if holo:
            ax.fill(xs, ys, color="white", zorder=3)
            edge, = ax.plot(xs, ys, color=BLUE, lw=2.4, solid_joinstyle="round", zorder=4)
            edge.set_clip_path(body)
            ax.add_patch(Circle((0.60, 0.0), 0.19, fc=ORANGE, ec="white", lw=2.2, zorder=5))
            ax.text(0.60, 0.0, "L", ha="center", va="center", color="white",
                    fontweight="bold", fontsize=14, zorder=6)
        else:
            # the same residues, still buried \u2014 no surface feature betrays them
            edge, = ax.plot(xs, ys, color="#7f9dc4", lw=1.5, ls=(0, (3, 3)), zorder=3)
            edge.set_clip_path(body)

        ax.text(0, 1.58, "APO  \u2014  unbound" if not holo else "HOLO  \u2014  ligand bound",
                ha="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(0, -1.62,
                "pocket is CLOSED\nnothing on the surface marks it" if not holo
                else "pocket has OPENED\nthe site was there all along",
                ha="center", fontsize=11.5, color=MUTED, linespacing=1.6)

    fig.text(0.5, 0.52, "\u2192", ha="center", va="center", fontsize=32, color=GRAY)
    fig.suptitle("A cryptic site is invisible in the structure we are given",
                 fontsize=16, fontweight="bold", y=1.0)
    fig.text(0.5, 0.01, "We only ever see the apo structure. The task: label every residue "
             "that WILL line the pocket.", ha="center", fontsize=12.5, color=MUTED)
    save(fig, "01_cryptic_site")


# ------------------------------------------------------------------ 2. dataset
def fig_dataset():
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.42)

    # (a) totals
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "CryptoBench", ha="center", fontsize=15, fontweight="bold")
    rows = [("381,189", "residues"), ("1,136", "apo structures"),
            ("5,493", "apo→holo pairings"), ("5.3 %", "are cryptic")]
    for i, (big, small) in enumerate(rows):
        y = 0.76 - i * 0.20
        ax.text(0.30, y, big, ha="right", va="center", fontsize=19,
                fontweight="bold", color=BLUE)
        ax.text(0.36, y, small, ha="left", va="center", fontsize=12.5, color=MUTED)

    # (b) split
    ax = fig.add_subplot(gs[1])
    labels = ["train\nfold 0", "fold 1", "fold 2", "fold 3", "TEST"]
    vals = [223, 228, 226, 228, 231]      # measured from build/residues.parquet
    cols = [BLUE] * 4 + [ORANGE]
    b = ax.bar(range(5), vals, color=cols, width=0.68)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("apo structures")
    ax.set_title("Split by sequence identity\n905 train  /  231 test",
                 fontsize=13, fontweight="bold", pad=12)
    ax.text(1.5, 250, "used only to pick\nthe threshold", ha="center", fontsize=10.5,
            color=MUTED, linespacing=1.4)
    ax.set_ylim(0, 300)
    ax.bar_label(b, fontsize=10, color=MUTED, padding=2)

    # (c) prevalence
    ax = fig.add_subplot(gs[2]); ax.axis("equal"); ax.axis("off")
    ax.add_patch(Wedge((0, 0), 1, 0, 360, fc="#e2e8f0", ec="white", lw=2))
    ax.add_patch(Wedge((0, 0), 1, 90, 90 + 360 * 0.0528, fc=ORANGE, ec="white", lw=2))
    ax.add_patch(Circle((0, 0), 0.58, fc="white", ec="none", zorder=3))
    ax.text(0, 0.10, "5.3 %", ha="center", va="center", fontsize=21,
            fontweight="bold", color=ORANGE, zorder=4)
    ax.text(0, -0.16, "positive", ha="center", va="center", fontsize=12,
            color=MUTED, zorder=4)
    ax.set_title("Heavily imbalanced\n→ report AUPRC, not accuracy",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    save(fig, "02_dataset")


# ------------------------------------------------------------------ 3. pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.1))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.1); ax.axis("off")

    box(ax, 0.15, 2.55, 1.9, 1.15, "apo PDB", "structure only", ec=GRAY)

    feats = [("Tier A", "34  geometry", BLUE, 3.55),
             ("Ensemble", "14  ANM+fpocket", GREEN, 2.45),
             ("Conservation", "6  PDBe API", PURPLE, 1.35),
             ("ESM-2", "64  PCs", ORANGE, 0.25)]
    for name, sub, col, y in feats:
        box(ax, 2.75, y, 2.6, 0.9, name, sub, ec=col, fs=12)
        arrow(ax, 2.05, 3.12, 2.72, y + 0.45)

    box(ax, 6.15, 1.9, 2.3, 1.4, "118 features", "per residue", ec=INK, fc="#f7fafc")
    for _, _, _, y in feats:
        arrow(ax, 5.35, y + 0.45, 6.12, 2.6)

    box(ax, 9.0, 1.9, 2.2, 1.4, "TabICLv2", "zero-shot", ec=RED, fc="#fff5f5")
    arrow(ax, 8.45, 2.6, 8.97, 2.6)
    box(ax, 11.55, 2.05, 1.35, 1.1, "P(cryptic)", "per residue", ec=INK, fs=11)
    arrow(ax, 11.2, 2.6, 11.52, 2.6)

    # context feed
    box(ax, 8.5, 0.25, 3.2, 0.85, "50,000 labelled residues", "the in-context 'training' set",
        ec=RED, fc="white", fs=11.5, lw=1.6)
    arrow(ax, 10.1, 1.12, 10.1, 1.86, color=RED, lw=2.0)

    ax.text(6.5, 4.82, "One pass. No gradient step anywhere in this diagram.",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)
    save(fig, "03_pipeline")


# ------------------------------------------------------------------ 4. feature stack
def fig_features():
    fig, ax = plt.subplots(figsize=(11, 3.3))
    groups = [("Tier A — apo geometry", 34, BLUE),
              ("Ensemble — ANM + fpocket", 14, GREEN),
              ("Conservation — PDBe", 6, PURPLE),
              ("ESM-2 — 64 PCs of a 2560-d embedding", 64, ORANGE)]
    left = 0
    for name, n, col in groups:
        ax.barh(0, n, left=left, color=col, height=0.5, ec="white", lw=2)
        ax.text(left + n / 2, 0, str(n), ha="center", va="center", color="white",
                fontweight="bold", fontsize=15)
        left += n
    ax.set_xlim(0, 118); ax.set_ylim(-1.15, 0.9); ax.axis("off")
    ax.text(0, 0.55, "118 features, every one computed from the apo structure alone",
            fontsize=14, fontweight="bold")
    # Legend rather than labels centred under each segment: the 14- and 6-wide
    # segments are far too narrow to caption in place, and the captions collide.
    # Reading order (left-to-right, then down) matches the bar order.
    sw_w, sw_h = 1.9, 0.11          # swatch sized to look square at this aspect
    for i, (name, n, col) in enumerate(groups):
        r, c = divmod(i, 2)
        x = 1.0 + c * 58.0
        y = -0.44 - r * 0.42
        ax.add_patch(Rectangle((x, y - sw_h / 2), sw_w, sw_h, fc=col, ec="none"))
        ax.text(x + sw_w + 1.4, y, name, ha="left", va="center",
                fontsize=11, color=INK)
    save(fig, "04_features")


# ------------------------------------------------------------------ 5. ANM ensemble
def fig_ensemble():
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.9); ax.axis("off")
    box(ax, 0.15, 1.35, 1.9, 1.15, "apo PDB", None, ec=GRAY)
    box(ax, 2.6, 1.35, 2.2, 1.15, "ANM", "Cα springs, 13 Å", ec=BLUE)
    arrow(ax, 2.05, 1.92, 2.57, 1.92)
    # 24 conformations
    ax.text(6.15, 3.30, "24 conformations", ha="center", fontsize=12.5,
            fontweight="bold", color=GREEN)
    rng = np.random.default_rng(0)
    for i in range(24):
        r, c = divmod(i, 8)
        x = 5.15 + c * 0.26
        y = 2.55 - r * 0.36
        ax.add_patch(Circle((x, y), 0.10, fc="#c6f6d5", ec=GREEN, lw=1.3))
    arrow(ax, 4.8, 1.92, 5.08, 1.92)
    ax.text(6.15, 1.10, "1.5 Å target Cα RMSD\nswept out-of-sample", ha="center",
            fontsize=10.5, color=MUTED, linespacing=1.4)
    box(ax, 7.5, 1.35, 2.0, 1.15, "fpocket", "on each one", ec=GREEN)
    arrow(ax, 7.2, 1.92, 7.47, 1.92)
    box(ax, 10.0, 1.35, 1.85, 1.15, "14 features", "mean, max, q90…", ec=INK, fc="#f7fafc", fs=12)
    arrow(ax, 9.5, 1.92, 9.97, 1.92)
    ax.text(6.0, 0.55, "ens_opening  =  best pocket score across the ensemble  −  the static crystal score",
            ha="center", fontsize=12.5, fontweight="bold", color=GREEN)
    ax.text(6.0, 0.18, "≈ 28,000 fpocket runs, on Modal.  A cheap stand-in for CryptoSite's 100k MD snapshots.",
            ha="center", fontsize=10.5, color=MUTED)
    save(fig, "05_ensemble")


# ------------------------------------------------------------------ 6. TabICL
def fig_tabicl():
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.0); ax.axis("off")

    ax.text(2.6, 4.62, "INPUT", ha="center", fontsize=12, fontweight="bold", color=MUTED)
    ax.add_patch(FancyBboxPatch((0.3, 2.75), 4.6, 1.5,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc="#fff5f5", ec=RED, lw=2.0, zorder=2))
    ax.text(2.6, 3.88, "Context:  50,000 labelled residues", ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK, zorder=3)
    ax.text(2.6, 3.20, "16,282 positive  +  33,718 negative\n"
            "every positive kept; negatives sampled as they occur",
            ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.6, zorder=3)
    box(ax, 0.3, 0.95, 4.6, 1.25, "Query:  every residue of a test protein",
        "same 118 columns, labels hidden", ec=BLUE, fs=13)

    box(ax, 5.6, 1.95, 2.3, 2.0, "TabICLv2", "frozen weights", ec=INK, fc="#f7fafc", fs=16)
    arrow(ax, 4.9, 3.50, 5.57, 3.25, color=RED)
    arrow(ax, 4.9, 1.57, 5.57, 2.55, color=BLUE)

    ax.text(10.05, 4.62, "OUTPUT", ha="center", fontsize=12, fontweight="bold", color=MUTED)
    box(ax, 8.6, 2.90, 2.9, 1.25, "P(cryptic) \u2208 [0,1]", "one score per residue", ec=GREEN, fs=13)
    arrow(ax, 7.9, 2.95, 8.57, 3.35)

    arrow(ax, 10.05, 2.85, 10.05, 2.35, color=GRAY)
    ax.text(10.05, 2.05, "cut at 0.568", ha="center", fontsize=12, fontweight="bold")
    ax.text(10.05, 1.62, "chosen on train folds,\nfrozen before test is touched", ha="center",
            fontsize=10, color=MUTED, linespacing=1.5)
    ax.text(10.05, 1.05, "\u2192  binary call", ha="center", fontsize=12.5,
            fontweight="bold", color=GREEN)

    ax.text(6.0, 0.28, "The 50,000 rows ARE the model. Change them and you change the "
            "predictor \u2014 no weights are updated, ever.",
            ha="center", fontsize=12.5, fontweight="bold", color=RED)
    save(fig, "06_tabicl")


# ------------------------------------------------------------------ 7. results
def fig_results():
    metrics = ["AUC", "AUPRC", "TPR", "MCC"]
    ours = [0.882, 0.476, 0.591, 0.410]
    ours_lo = [0.863, 0.439, 0.546, 0.377]
    ours_hi = [0.901, 0.514, 0.635, 0.443]
    noplm = [0.867, 0.450, 0.565, 0.389]
    plmnn = [0.86, 0.36, 0.48, 0.39]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(4); w = 0.26
    err = [np.array(ours) - np.array(ours_lo), np.array(ours_hi) - np.array(ours)]
    ax.bar(x - w, ours, w, color=ORANGE, label="TabICLv2 + ESM  (118 feat)",
           yerr=err, capsize=4, ecolor=INK, error_kw={"lw": 1.4})
    ax.bar(x, noplm, w, color=BLUE, label="TabICLv2, no pLM  (54 feat)")
    ax.bar(x + w, plmnn, w, color=GRAY, label="pLM-NN  (CryptoBench benchmark)")

    for xi, v in zip(x - w, ours):
        ax.text(xi, v + 0.055, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold", color=ORANGE)
    for xi, v in zip(x, noplm):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=10.5, color=BLUE)
    for xi, v in zip(x + w, plmnn):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=10.5, color=MUTED)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.02); ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=1)
    ax.set_title("CB-full  —  all 231 test structures, per structure then averaged",
                 fontsize=14, fontweight="bold", pad=14)
    ax.text(0.0, -0.20, "Bars for our best model carry a 95 % CI from resampling the 231 test proteins. "
            "AUPRC and TPR clear pLM-NN's value; MCC does not, and the AUC margin clears only here \u2014 not on\nCB-P2RANK-apo, and not without ESM \u2014 so it is not a lead to report.",
            transform=ax.transAxes, fontsize=11, color=MUTED)
    save(fig, "07_results")


# ------------------------------------------------------------------ 8. what moved it
def fig_gains():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    items = [("Averaging over 5 contexts", 0.004, GRAY, "5× GPU inference"),
             ("+14 ensemble pocket features", 0.014, GREEN, "~28,000 fpocket runs"),
             ("+6 conservation features", 0.017, PURPLE, "~2,300 API calls"),
             ("+64 ESM-2 components", 0.026, ORANGE, "0.06 A10G-hours"),
             ("Fixing the negative sampler", 0.113, RED, "one line"),
             ]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    cols = [i[2] for i in items]
    costs = [i[3] for i in items]
    y = np.arange(len(items))
    ax.barh(y, vals, color=cols, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=12)
    for yi, v, c in zip(y, vals, costs):
        ax.text(v + 0.003, yi, f"+{v:.3f}", va="center", fontsize=11.5, fontweight="bold")
        ax.text(v + 0.026, yi, c, va="center", fontsize=10.5, color=MUTED)
    ax.set_xlim(0, 0.16); ax.set_xlabel("Δ AUPRC")
    ax.set_title("What actually moved the number", fontsize=15, fontweight="bold", pad=12)
    ax.text(0.0, -0.30, "The biggest single gain was a bug fix. The three deliberate capability "
            "additions were the three smallest.\nA fourth bug fix, the insertion-code key (\u00a7 7), is not plotted: it "
            "corrected 32 mislabelled training residues for \u22120.004 AUPRC, inside the context-draw noise.",
            transform=ax.transAxes, fontsize=11.5, color=MUTED)
    save(fig, "08_gains")


# ------------------------------------------------------------------ 9. hard stratum
def fig_strata():
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    steps = ["Tier A", "+ ensemble", "+ conservation", "+ ESM-16", "+ ESM-32", "+ ESM-64"]
    hard = [0.667, 0.672, 0.690, 0.705, 0.710, 0.714]
    bulk = [0.824, 0.838, 0.852, 0.867, 0.876, 0.879]
    x = np.arange(len(steps))
    ax.plot(x, bulk, "o-", color=BLUE, lw=2.6, ms=8, label="bulk surface  (73,296 residues)")
    ax.plot(x, hard, "o-", color=RED, lw=2.6, ms=8, label="lines an already-open pocket  (3,340)")
    ax.fill_between(x, hard, bulk, color="#fed7d7", alpha=0.55, zorder=0)
    ax.annotate("", xy=(5, 0.879), xytext=(5, 0.714),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
    ax.text(4.82, 0.796, "gap\n0.165", ha="right", fontsize=11.5, color=RED,
            fontweight="bold", linespacing=1.4)
    ax.set_xticks(x); ax.set_xticklabels(steps, fontsize=11)
    ax.set_ylabel("AUC"); ax.set_ylim(0.63, 0.93)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_title("Where the model is weak", fontsize=15, fontweight="bold", pad=12)
    ax.text(0.0, -0.22, "Nothing closes the gap: it sits at 0.157\u20130.166 across every feature set, and each "
            "addition lifts both strata together.\nTelling a cryptic site from a pocket that is already open is the task itself.",
            transform=ax.transAxes, fontsize=11.5, color=MUTED)
    save(fig, "09_strata")


for f in (fig_cryptic, fig_dataset, fig_pipeline, fig_features, fig_ensemble,
          fig_tabicl, fig_results, fig_gains, fig_strata):
    f()
