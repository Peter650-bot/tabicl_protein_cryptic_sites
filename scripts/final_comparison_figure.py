"""Draw the final comparison from finetuned_esm/final_comparison.json (no scoring here).

Dot + 95 % CI per model and metric: the axes are zoomed to the data, so the marks are
dots, not bars (a bar on a zoomed axis implies a zero it does not have). A filled dot
differs from Škrhák et al.'s released model with a paired 95 % CI that excludes zero.

Run: python scripts/final_comparison_figure.py [--out ../Stacked_Figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = json.loads(Path("finetuned_esm/final_comparison.json").read_text())
ORDER = ["released", "blend", "stack9", "ft117"]               # top to bottom
LABEL = {
    "released": "Fine-tuned ESM-2 3B\n(Škrhák et al., released;\nmultitask + LIGYSIS)",
    "blend": "Fixed rank blend\nTabICLv2 117 + full-recipe FT score",
    "stack9": "Stack (step 9, Fig. 6's best)\nTabICLv2 117 + full-recipe FT score",
    "ft117": "TabICLv2, 53 features +\n64 fine-tuned ESM-2 components",
}
COLOR = {"released": "#4A3AAA", "blend": "#E9A100", "stack9": "#E24A4A", "ft117": "#13958F"}
METRICS = [("AUC", "AUC"), ("AUPRC", "AUPRC"), ("MCC", "MCC"),
           ("F1_weighted", "F1 (weighted)"), ("ACC", "Accuracy"), ("TPR", "TPR (recall)"),
           ("FPR", "FPR")]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def panel(ax, kind: str, metric: str, title: str, show_labels: bool) -> None:
    block, diffs = RES[kind], RES[kind]["diff_vs_released"]
    lo = min(block[m][metric]["ci"][0] for m in ORDER)
    hi = max(block[m][metric]["ci"][1] for m in ORDER)
    pad = (hi - lo) * 0.12
    ax.set_xlim(lo - pad, hi + pad * 3.2)
    for i, m in enumerate(ORDER):
        v, (a, b) = block[m][metric]["value"], block[m][metric]["ci"]
        sig = m != "released" and (diffs[m][metric]["ci"][0] > 0 or diffs[m][metric]["ci"][1] < 0)
        ax.plot([a, b], [i, i], color=COLOR[m], lw=2, solid_capstyle="round", zorder=2)
        filled = m == "released" or sig
        ax.plot(v, i, "o", ms=8, zorder=3, mec="white", mew=2 if filled else 0,
                mfc=COLOR[m] if filled else "white")
        if not filled:
            ax.plot(v, i, "o", ms=7, zorder=4, mfc="white", mec=COLOR[m], mew=2)
        ax.text(b + pad * 0.35, i, f"{v:.4f}", va="center", ha="left", fontsize=8.5, color=INK)
    ref = block["released"][metric]["value"]
    ax.axvline(ref, color=COLOR["released"], lw=0.8, ls=(0, (3, 3)), alpha=0.6, zorder=1)
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABEL[m] for m in ORDER] if show_labels else [], fontsize=8.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=2)
    ax.locator_params(axis="x", nbins=4)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bdbdbd")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK, pad=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../Stacked_Figures")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(19, 11.6))
    gs = fig.add_gridspec(3, len(METRICS), height_ratios=[1, 1, 0.95], hspace=0.52, wspace=0.16,
                          left=0.135, right=0.985, top=0.83, bottom=0.02)
    for row, (kind, rowtitle) in enumerate((
            ("pooled", "Pooled over all residues — the convention of Škrhák et al.'s papers"),
            ("per_protein", "Per protein, then averaged — CryptoBench's and this project's convention"))):
        for j, (metric, title) in enumerate(METRICS):
            ax = fig.add_subplot(gs[row, j])
            panel(ax, kind, metric, title, show_labels=j == 0)
            if j == 0:
                ax.text(-0.02, 1.25, rowtitle, transform=ax.transAxes, fontsize=11.5,
                        fontweight="bold", color=INK, ha="left")
    fig.text(0.135, 0.955, "Final comparison — on Škrhák et al.'s own test set "
             "(CB-P2RANK-apo: 184 single-chain proteins, 55,567 residues)",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.135, 0.93, "Dot = point estimate; line = 95 % bootstrap CI (2,000 protein resamples). "
             "Filled = differs from the released model (dashed line) with a paired 95 % CI excluding zero; "
             "hollow = not distinguishable from it. Axes are zoomed to the data.",
             fontsize=9.5, color=MUTED)

    # ---- bottom row: verification, operating points, reading
    rel = RES["gate"]["these_184_vs_their_file"]
    paper = RES["gate"]["paper_vs_their_file"]
    nb = RES["gate"]["notebook_vs_their_file"]
    ax = fig.add_subplot(gs[2, 0:3])
    ax.axis("off")
    rows = [("", "paper", "notebook", "our run,\ntheir file", "our run,\nthese 184")]
    for k, name in (("AUC", "AUC"), ("AUPRC", "AUPRC"), ("MCC", "MCC @ 0.7"),
                    ("F1_weighted", "F1 weighted"), ("ACC", "Accuracy")):
        rows.append((name, f"{paper[k]['published']:.3f}" if k in paper else "—",
                     f"{nb[k]['published']:.4f}", f"{nb[k]['their_file']:.4f}",
                     f"{RES['pooled']['released'][k]['value']:.4f}"))
    ax.text(0, 1.02, "Check: their model reproduces their published numbers",
            fontsize=10.5, fontweight="bold", color=INK, transform=ax.transAxes)
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper left", cellLoc="center",
                   colWidths=[0.2, 0.17, 0.17, 0.2, 0.2], bbox=[0, 0.12, 0.92, 0.82])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=MUTED, fontsize=8)
        if c == 4 and r > 0:
            cell.set_text_props(fontweight="bold")
    ax.text(0, 0.0, "Their test file has 5 residues our table lacks (3,158 vs 3,157 positives); "
            "every number agrees within 0.0005.\nPaper = Hidden in protein sequences (ACM BCB 2025); "
            "notebook = their seq2pocket evaluation output.",
            fontsize=7.5, color=MUTED, transform=ax.transAxes, va="top")

    ax = fig.add_subplot(gs[2, 3:5])
    ax.axis("off")
    ops = RES["operating_points"]
    ax.text(0, 1.02, "Operating point behind MCC, F1, accuracy, TPR, FPR",
            fontsize=10.5, fontweight="bold", color=INK, transform=ax.transAxes)
    y = 0.86
    for m in ORDER:
        ax.plot(0.012, y + 0.005, "o", ms=7, color=COLOR[m], transform=ax.transAxes, clip_on=False)
        ax.text(0.04, y, LABEL[m].replace("\n", " "), fontsize=8.3, color=INK,
                transform=ax.transAxes, va="center", fontweight="bold")
        ax.text(0.04, y - 0.085, ops[m], fontsize=8, color=MUTED, transform=ax.transAxes, va="center")
        y -= 0.215

    ax = fig.add_subplot(gs[2, 5:7])
    ax.axis("off")
    ax.text(0, 1.02, "Reading the figure", fontsize=10.5, fontweight="bold", color=INK,
            transform=ax.transAxes)
    notes = [
        "Ranking metrics (AUC, AUPRC) need no threshold: they are the fair comparison. "
        "Threshold metrics also reflect where each model chose to operate; theirs was "
        "chosen on this very test set, and the step-9 stack's CV threshold overshoots "
        "on test (FPR 0.070).",
        "Per protein, all three TabICLv2 models beat the released model on AUC and AUPRC. "
        "Pooled, the two stacks match or beat it; the blend's within-protein ranks lose the "
        "cross-protein calibration that pooled AUPRC rewards.",
        "The released model saw LIGYSIS labels for 18 of these 184 proteins; every "
        "TabICLv2 row uses fine-tuned models trained without them. No row is "
        "zero-shot: all four use a fine-tuned ESM-2 3B.",
    ]
    yy = 0.9
    for n in notes:
        import textwrap
        t = "\n".join(textwrap.wrap(n, 62))
        ax.text(0, yy, "• " + t.replace("\n", "\n   "), fontsize=8.2, color=INK,
                transform=ax.transAxes, va="top")
        yy -= 0.08 + 0.085 * t.count("\n")

    for ext in ("pdf", "png"):
        fig.savefig(out / f"final_comparison.{ext}", dpi=200 if ext == "png" else None)
    print(f"wrote {out / 'final_comparison.pdf'} and .png")


if __name__ == "__main__":
    main()
