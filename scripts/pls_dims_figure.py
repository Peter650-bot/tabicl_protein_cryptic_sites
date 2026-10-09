"""AUPRC against the number of PLS components, from finetuned_esm/step11_result.json.

Two series: TabICLv2 on 53 features + d PLS components, and on the d components alone,
d = 2..128, on Škrhák et al.'s test set (CB-P2RANK-apo, 184 proteins) so the numbers
line up with the final comparison figure. Hollow diamonds are step 10's PCA at d = 64
for the same two feature sets, annotated with the paired PLS - PCA difference; the
dashed line is the released fine-tuned model (finetuned_esm/final_comparison.json).
Marginal CIs are not drawn: they are ~0.08 wide (between-protein spread), overlap
for every pair, and hide that the paired differences are tight. d = 64 was fixed in
advance (frozen_step11.json); the other dimensions are descriptive.

Run: python scripts/pls_dims_figure.py [--out ../Stacked_Figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = json.loads(Path("finetuned_esm/step11_result.json").read_text())
FINAL = json.loads(Path("finetuned_esm/final_comparison.json").read_text())
SUBSET = "CB-P2RANK-apo"
DIMS = [2, 4, 8, 16, 32, 64, 128]
SERIES = {"53": ("53 features + d PLS components", "#2A78D6", "ft117", "53 + 64 PCA"),
          "only": ("d PLS components only", "#008300", "ft64only", "64 PCA only")}
RELEASED = "#4A3AAA"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def panel(ax, key: str, final_kind: str, title: str, rel_above: bool) -> None:
    r = RES[SUBSET]
    for b, (label, color, pca, _) in SERIES.items():
        v = [r["sweep"][b][str(d)][key]["value"] for d in DIMS]
        ax.plot(DIMS, v, color=color, lw=2, zorder=3)
        ax.plot(DIMS, v, "o", ms=7, mfc=color, mec="white", mew=1.5, zorder=4)
        ax.text(DIMS[-1] * 1.1, v[-1], f"{v[-1]:.3f}", va="center", fontsize=8.5, color=INK)
        # step 10's PCA at d = 64, nudged right of the PLS point so neither hides the other
        p = r[key][pca]["value"]
        x = 64 * 1.18
        ax.plot([64, x], [v[DIMS.index(64)], p], color=color, lw=0.8, ls=":", zorder=2)
        ax.plot(x, p, "D", ms=7, mfc="white", mec=color, mew=1.8, zorder=4)
        d = r[key][f"pls64_{b} - {pca}"] if f"pls64_{b} - {pca}" in r[key] else None
        if d is not None:
            ax.text(x * 1.06, p, f"PCA {p:.3f}\nPLS − PCA {d['diff']:+.3f}\n"
                    f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]", va="center", fontsize=7.6,
                    color=MUTED, linespacing=1.25, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=1.5))
    rel = FINAL[final_kind]["released"]["AUPRC"]["value"]
    ax.axhline(rel, color=RELEASED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(2 if rel_above else 8, rel, f"released fine-tuned ESM-2 3B  {rel:.3f}", fontsize=8.3,
            color=MUTED, va="bottom" if rel_above else "top")
    ax.axvline(64, color=GRID, lw=6, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlim(1.6, 300)
    ax.set_xlabel("number of PLS components, d", fontsize=9.5, color=INK)
    ax.set_ylabel("AUPRC", fontsize=9.5, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbdbd")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../Stacked_Figures")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.21, wspace=0.18)
    panel(axes[0], "per_protein_AUPRC", "per_protein",
          "Per protein, then averaged — CryptoBench's convention", rel_above=True)
    panel(axes[1], "pooled_AUPRC", "pooled", "Pooled over all residues — Škrhák et al.'s convention",
          rel_above=False)
    n = RES[SUBSET]["n_proteins"]
    fig.text(0.06, 0.945, "TabICLv2 on PLS components of the fine-tuned ESM-2 3B embedding: "
             "AUPRC vs number of components", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.06, 0.865, f"Škrhák et al.'s test set (CB-P2RANK-apo, {n} proteins).\nEach point is "
             "the mean of four cross-fitted TabICLv2 predictions (context = train fold k, PLS fit on "
             "fold k only). PLS − PCA: paired difference at d = 64 with its 95 % bootstrap CI.", fontsize=9.3, color=MUTED)
    handles = []
    for b, (label, color, _, pca_label) in SERIES.items():
        handles.append(Line2D([], [], color=color, lw=2, marker="o", ms=7, mec="white", label=f"PLS: {label}"))
        handles.append(Line2D([], [], color=color, lw=0, marker="D", ms=7, mfc="white", mew=1.8,
                              label=f"PCA, step 10: {pca_label}"))
    handles.append(Line2D([], [], color=RELEASED, lw=1, ls=(0, (4, 3)),
                          label="Released fine-tuned ESM-2 3B (multitask + LIGYSIS)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.01), ncol=3,
               frameon=False, fontsize=8.8, labelcolor=INK)
    fig.text(0.97, 0.03, "Shaded column: d = 64, fixed in advance and used in the final comparison; "
             "other d are descriptive.", fontsize=8, color=MUTED, ha="right")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pls_dimension_sweep.{ext}", dpi=200 if ext == "png" else None)
    print(f"wrote {out / 'pls_dimension_sweep.pdf'} and .png")


if __name__ == "__main__":
    main()
