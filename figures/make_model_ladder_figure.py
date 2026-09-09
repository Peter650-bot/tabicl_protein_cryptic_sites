"""Every model configuration on CB-full, as a dot-strip plot.

WHY THIS FORM AND NOT MORE BARS
    Adding LogisticRegression and RandomForest to the grouped-bar figure would
    need two more colour channels and would leave them empty on three of the
    four subsets (our sklearn controls exist on CB-full only). A dot strip puts
    one ROW per experimental condition -- same features, same negatives -- and
    one DOT per model family, so the comparison the reader needs (model choice,
    holding everything else fixed) is a single short horizontal span, and the
    comparison that actually moved the project (features, sampler) is the
    vertical distance between rows.

Data:  python scripts/model_ladder.py  ->  build/model_ladder.json
Run:   python figures/make_model_ladder_figure.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "build" / "model_ladder.json"

# validated dataviz slots 1-3 (all-pairs clean in both modes)
FAMILY = {
    "TabICLv2 zero-shot": ("TabICLv2 zero-shot", "#2a78d6"),
    "LogisticRegression": ("LogisticRegression", "#eb6834"),
    "RandomForest":       ("RandomForest",       "#1baf7a"),
}
ORDER = ["TabICLv2 zero-shot", "LogisticRegression", "RandomForest"]
INK, INK2, MUTED = "#101312", "#4f534f", "#8b8f8a"
GRID, BASELINE, SURFACE = "#e6e8e2", "#c2c5bd", "#fcfcfb"
REF = "#7a6a3d"          # published-benchmark reference rule
STRIP = "#d8dad3"        # connector between the dots of one row

# pLM-NN, CB-full, from external/CryptoBench/README.md
PLMNN = {"AUPRC": 0.36, "AUC": 0.86, "MCC": 0.39}

# Axes are cropped to the model range. A dot's POSITION encodes the value --
# there is no bar length to mislead -- and the full 0-1 range would collapse
# every dot in a row onto the same pixel, which is the opposite of the point.
COLUMNS = [("AUPRC", (0.385, 0.487), [.40, .42, .44, .46]),
           ("AUC",   (0.826, 0.890), [.84, .85, .86, .87, .88]),
           ("MCC",   (0.215, 0.425), [.25, .30, .35, .40])]

# "old sampler" no longer appears in build/model_ladder.json — every row was
# re-measured on the fixed sampler. Kept so an archived ladder still renders.
REGIME_ORDER = {"50k context · fixed sampler": 0,
                "50k context · old sampler": 1,
                "full 304k train": 2}
REGIME_LABEL = {"50k context · fixed sampler": "50k context, fixed sampler",
                "50k context · old sampler": "50k context, old sampler",
                "full 304k train": "full 304k train"}

if not LADDER.exists():
    sys.exit("build/model_ladder.json missing — run: python scripts/model_ladder.py")
rows = json.loads(LADDER.read_text())

# ── group into one row per (n_features, negatives) condition ─────────────────
conds, floor = {}, None
for r in rows:
    if r["family"] == "RSA alone (floor)":
        floor = r
        continue
    key = (r["n_features"], r["negatives"])
    conds.setdefault(key, {})[r["family"]] = r

# The RSA floor is a sanity anchor, not a model configuration: at AUPRC 0.089 it
# would stretch the axis until every real dot collapsed. It lives in the footnote.
ordered = sorted(conds, key=lambda k: (-k[0], REGIME_ORDER[k[1]]))
labels = [f"{n} feat · {REGIME_LABEL[neg]}" for n, neg in ordered]

n = len(ordered)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
                     "font.monospace": ["IBM Plex Mono", "DejaVu Sans Mono"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})

fig, axes = plt.subplots(1, 4, figsize=(14.6, 0.46 * n + 3.0),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.62], wspace=.16),
                         sharey=True)
fig.subplots_adjust(left=.215, right=.985, top=.815, bottom=.085)

for ax, (metric, xlim, ticks) in zip(axes[:3], COLUMNS):
    ax.axvline(PLMNN[metric], color=REF, lw=1.1, zorder=1)
    ax.text(PLMNN[metric], n - 0.28, "  pLM-NN", color=REF, fontsize=7,
            ha="left", va="center")

    for i, key in enumerate(ordered):
        y = n - 1 - i
        got = conds[key]
        vals = [got[f][metric] for f in ORDER if f in got and got[f][metric] is not None]
        if len(vals) > 1:                     # strip connecting the families
            ax.plot([min(vals), max(vals)], [y, y], color=STRIP, lw=2,
                    solid_capstyle="round", zorder=2)
        for fam in ORDER:
            if fam not in got or got[fam][metric] is None:
                continue
            ax.plot(got[fam][metric], y, "o", ms=8, color=FAMILY[fam][1],
                    mec=SURFACE, mew=2, zorder=4)

    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7.6, color=MUTED,
                       fontfamily="monospace")
    ax.xaxis.grid(True, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(.9)
    ax.tick_params(length=0, pad=4)
    ax.set_title(metric, loc="left", fontsize=10.5, color=INK,
                 fontweight="semibold", pad=10, fontfamily="monospace")

axes[0].set_ylim(-0.7, n - 0.3)
axes[0].set_yticks(range(n))
axes[0].set_yticklabels(list(reversed(labels)), fontsize=8.4, color=INK)

# ── numeric gutter: AUPRC per family, so no value is colour-only ─────────────
gut = axes[3]
gut.axis("off")
gut.set_xlim(0, 1)
gut.set_title("AUPRC, read off", loc="left", fontsize=9, color=INK2, pad=10)
XS = [0.10, 0.44, 0.78]
for fam, x in zip(ORDER, XS):
    gut.plot([x - .055], [n - 0.30], "o", ms=6, color=FAMILY[fam][1],
             mec=SURFACE, mew=1.5, clip_on=False)
    short = {"TabICLv2 zero-shot": "TabICL", "LogisticRegression": "LogReg",
             "RandomForest": "RF"}[fam]
    gut.text(x - .015, n - 0.30, short, fontsize=7, color=MUTED, va="center")
for i, key in enumerate(ordered):
    y = n - 1 - i
    for fam, x in zip(ORDER, XS):
        v = conds[key].get(fam, {}).get("AUPRC")
        gut.text(x, y, f"{v:.3f}" if v is not None else "·", fontsize=7.4,
                 color=INK2 if v is not None else MUTED, va="center",
                 fontfamily="monospace")

fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, mec=SURFACE, mew=2,
                           color=c, label=l) for l, c in FAMILY.values()],
           loc="upper left", bbox_to_anchor=(.215, .885), ncol=3, frameon=False,
           fontsize=9, handletextpad=.4, columnspacing=2.2)

fig.suptitle("Model choice moves the number far less than the features do",
             x=.028, y=.975, ha="left", fontsize=15.5, color=INK,
             fontweight="semibold")
fig.text(.028, .945,
         "Every configuration we ran, on CB-full (231 apo structures). One row per condition — same features, same negatives — so the "
         "horizontal spread inside a row\nis the model effect alone (0.003–0.030 AUPRC). The vertical distance between rows is the feature "
         "effect (0.398 → 0.476). Axes are cropped to the\nmodel range: a dot's position encodes the value, and the full 0–1 range "
         "would collapse every dot in a row onto one pixel.",
         ha="left", fontsize=8.8, color=INK2, linespacing=1.55, va="top")
fig.text(.028, .022,
         "Our sklearn controls exist on CB-full only — the stored artifacts keep summary metrics, not per-residue scores, so they cannot be "
         "re-scored on CB-P2RANK-apo.   ·   TabICL has no full-304k row by construction: 304k rows do not fit in context, so the\nfull-304k "
         "rows are the sklearn models’ best case — and TabICL in 50k of context still leads them.   ·   RSA alone, the single-feature floor, sits far below the cropped axes at AUPRC 0.089 / AUC 0.499 / MCC 0.034.   ·   "
         "Rebuild: python scripts/model_ladder.py",
         ha="left", fontsize=7.2, color=MUTED, linespacing=1.6)

fig.savefig(Path(__file__).parent / "model_ladder.png", dpi=200,
            bbox_inches="tight", pad_inches=.28)
fig.savefig(Path(__file__).parent / "model_ladder.pdf",
            bbox_inches="tight", pad_inches=.28)
print("wrote figures/model_ladder.png and .pdf")
