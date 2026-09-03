"""Render every run into one comparison table.

WHY
    By the end there are several numbers in flight -- TabICL on Tier A, TabICL
    with ensemble features, sklearn controls at two context sizes, and three
    published baselines. Kept in separate JSON files they are easy to quote
    selectively. This prints them together, which makes it hard to report the
    flattering subset by accident.

    Every row uses the same test proteins, the same per-protein averaging and a
    threshold chosen off the test set, so the columns are comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C, evaluate  # noqa: E402


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, default=C.BUILD_DIR)
    args = ap.parse_args()

    rows = []

    for variant, label in (("tierA", "TabICLv2 zero-shot (Tier A)"),
                           ("ensemble", "TabICLv2 zero-shot (+ ensemble)")):
        d = _load(args.build / f"modal_result_{variant}.json")
        if d:
            rows.append((label, d["result"], "ours"))

    legacy = _load(args.build / "modal_result.json")
    if legacy and not any(r[2] == "ours" for r in rows):
        rows.append(("TabICLv2 zero-shot (Tier A)", legacy["result"], "ours"))

    for tag, suffix in (("", " (Tier A)"), ("ens", " (+ ensemble)")):
        d = _load(args.build / f"baselines{('_' + tag) if tag else ''}.json")
        for r in d or []:
            if r["method"].startswith("RSA"):
                continue
            rows.append((r["method"] + suffix, r, "control"))

    floor = None
    for tag in ("", "ens"):
        d = _load(args.build / f"baselines{('_' + tag) if tag else ''}.json")
        for r in d or []:
            if r["method"].startswith("RSA"):
                floor = ("RSA alone (floor)", r, "control")
    if floor:
        rows.append(floor)

    width = max((len(r[0]) for r in rows), default=30) + 2
    print(f"{'Method':<{width}} {'AUC':>6} {'AUPRC':>7} {'TPR':>6} {'FPR':>6} {'MCC':>6}")
    print("-" * (width + 36))

    for label, r, kind in sorted(rows, key=lambda x: -x[1]["AUC"]):
        mark = "  <-" if kind == "ours" else ""
        print(f"{label:<{width}} {r['AUC']:>6.3f} {r['AUPRC']:>7.3f} "
              f"{r['TPR']:>6.3f} {r['FPR']:>6.3f} {r['MCC']:>6.3f}{mark}")

    print("-" * (width + 36))
    for name, m in evaluate.BENCHMARK.items():
        print(f"{name + ' (published)':<{width}} {m['AUC']:>6.2f} {m['AUPRC']:>7.2f} "
              f"{m['TPR']:>6.2f} {m['FPR']:>6.2f} {'':>6}")

    # The comparison that answers "did the expensive features pay for themselves"
    a = _load(args.build / "modal_result_tierA.json") or legacy
    b = _load(args.build / "modal_result_ensemble.json")
    if a and b:
        ra, rb = a["result"], b["result"]
        print("\nensemble features, delta on TabICL:")
        for k in ("AUC", "AUPRC", "TPR", "MCC"):
            d = rb[k] - ra[k]
            print(f"  {k:<6} {ra[k]:.3f} -> {rb[k]:.3f}   {d:+.3f}")

        sa = {s["stratum"]: s for s in a.get("stratified", [])}
        sb = {s["stratum"]: s for s in b.get("stratified", [])}
        if sa and sb:
            print("\nby stratum (AUC):")
            for k in sa:
                if k in sb:
                    print(f"  {k:<28} {sa[k]['AUC']:.3f} -> {sb[k]['AUC']:.3f}   "
                          f"{sb[k]['AUC'] - sa[k]['AUC']:+.3f}")


if __name__ == "__main__":
    main()
