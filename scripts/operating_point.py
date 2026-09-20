"""Select a decision rule on the CV folds, then score the test split once.

THE PROBLEM
    FPR, MCC and F1 are decided by how a score becomes a call, not by how good
    the ranking is. The incumbent rule is one global cutoff for all 231 test
    proteins: swept to a mean per-protein FPR of 0.05 on the CV folds, it lands
    at 0.071 on test. RESULTS.md section 8 lists that gap as unmeasured.

TWO COMMANDS, AND WHY THEY ARE SEPARATE
    `--select` reads ONLY the CV folds and writes `operating_point_frozen.json`.
    `--apply` refuses to run until that file exists, then scores test once.

    They are separate so the freeze can be COMMITTED before test is touched, and
    the commit timestamp is the audit trail. This matters more than usual here:
    during scoping, the rank rule's q was swept directly on test (q ~ 0.08 gives
    MCC 0.3804). That number is an ORACLE, not a result, and knowing it is
    exactly the kind of contamination pre-registration exists to contain.

WHAT IS AND IS NOT DECIDED HERE
    The headline rule is fixed in advance: `global_threshold` stays the headline,
    because pLM-NN used a single constant threshold and the comparison must stay
    protocol-matched. What `--select` decides is which PER-PROTEIN rule (rank or
    zscore) is reported as the pre-registered secondary, on nested-CV MCC alone.

    Per-protein AUC and AUPRC are invariant under these transforms, so no rule
    here can move the 0.862 / 0.441 headline. The apply step prints them anyway,
    identical across rows, as visible proof.

USAGE
    python scripts/operating_point.py --select     # CV only; commit the output
    python scripts/operating_point.py --apply      # scores test exactly once
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config, evaluate, operating_point as op

CV_DIR = config.BUILD_DIR / "cv_scores"
FOLDS = ["train-fold-0", "train-fold-1", "train-fold-2", "train-fold-3"]
# Tracked, not under build/: build/ is gitignored as regenerable output, but the
# pre-registration is worthless unless its commit timestamp is in the history.
OUT_DIR = config.PROJECT_ROOT / "operating_point"
FROZEN = OUT_DIR / "frozen.json"
REPORT = OUT_DIR / "result.json"
MANIFEST = OUT_DIR / "cv_scores_manifest.json"
TEST_RESULT = config.BUILD_DIR / "modal_result_esm64.json"

# The per-protein rule with the best nested-CV MCC becomes the reported
# secondary, provided it does not buy that MCC with a blown FPR budget.
MAX_NESTED_FPR = 0.055


def _load(path: Path) -> Dict[str, np.ndarray]:
    d = json.loads(path.read_text())
    block = d["scores"] if "scores" in d else d
    return {
        "y": np.asarray(block["y"], dtype=int),
        "s": np.asarray(block["score"], dtype=float),
        "g": np.asarray(block["apo_id"]),
    }


def _load_folds() -> Dict[str, Dict[str, np.ndarray]]:
    missing = [f for f in FOLDS if not (CV_DIR / f"k64_{f}.json").exists()]
    if missing:
        raise SystemExit(
            f"missing CV scores for {missing} in {CV_DIR}.\n"
            "Stage them first:\n"
            "  modal volume get cryptobench-fanout-results "
            "run_20260910_124330/ build/cv_scores/"
        )
    return {f: _load(CV_DIR / f"k64_{f}.json") for f in FOLDS}


def _pool(folds: Dict[str, Dict[str, np.ndarray]], names: List[str]):
    return {k: np.concatenate([folds[n][k] for n in names]) for k in ("y", "s", "g")}


def select() -> dict:
    folds = _load_folds()
    print(f"CV folds (test prevalence for reference: 0.0528)")
    for name, d in folds.items():
        print(f"  {name}: n={len(d['y']):6d}  proteins={len(np.unique(d['g'])):3d}  "
              f"prevalence={d['y'].mean():.4f}")

    out: Dict[str, dict] = {}
    for rule in op.RULES:
        # 1. Median-of-4: the frozen value, same discipline as the incumbent.
        per_fold = []
        for name in FOLDS:
            d = folds[name]
            t = op.transform(rule, d["s"], d["g"])
            per_fold.append(evaluate.choose_threshold(d["y"], t, d["g"],
                                                      target_fpr=config.TARGET_FPR))
        frozen = float(np.median(per_fold))

        # 2. Nested leave-one-fold-out: select on 3 folds pooled, apply to the
        #    4th. This is the honest transfer estimate, and the only thing that
        #    decides which per-protein rule gets reported.
        nested = []
        for name in FOLDS:
            others = [f for f in FOLDS if f != name]
            tr, te = _pool(folds, others), folds[name]
            thr = evaluate.choose_threshold(
                tr["y"], op.transform(rule, tr["s"], tr["g"]), tr["g"],
                target_fpr=config.TARGET_FPR)
            t_te = op.transform(rule, te["s"], te["g"])
            nested.append(evaluate.summarise(te["y"], t_te, te["g"], thr, label=name))
        nested_mean = {
            k: float(np.mean([r[k] for r in nested]))
            for k in ("TPR", "FPR", "MCC", "F1", "F1_weighted")
        }

        out[rule] = {
            "fold_thresholds": [float(x) for x in per_fold],
            "frozen_threshold": frozen,
            "operating_point": op.describe(rule, frozen),
            "nested_cv_mean": nested_mean,
            "nested_cv_per_fold": nested,
        }
        print(f"\n{rule}: frozen={frozen:.4f}  ({op.describe(rule, frozen)})")
        print(f"  per-fold thresholds: "
              f"{', '.join(f'{x:.4f}' for x in per_fold)}")
        print(f"  nested CV: FPR {nested_mean['FPR']:.4f}  TPR {nested_mean['TPR']:.4f}  "
              f"MCC {nested_mean['MCC']:.4f}  F1w {nested_mean['F1_weighted']:.4f}")

    # Choose the secondary rule on nested CV alone. Test is not loaded here.
    candidates = [r for r in ("rank", "zscore")
                  if out[r]["nested_cv_mean"]["FPR"] <= MAX_NESTED_FPR]
    if not candidates:
        candidates = ["rank", "zscore"]
        note = f"no per-protein rule met FPR <= {MAX_NESTED_FPR}; chose on MCC alone"
    else:
        note = f"chosen on nested-CV MCC among rules with FPR <= {MAX_NESTED_FPR}"
    secondary = max(candidates, key=lambda r: (out[r]["nested_cv_mean"]["MCC"],
                                               out[r]["nested_cv_mean"]["F1_weighted"]))

    frozen_doc = {
        "created": datetime.date.today().isoformat(),
        "target_fpr": config.TARGET_FPR,
        "selection": "per-fold threshold at target mean per-protein FPR, frozen as "
                     "the median of the 4 folds (identical to the incumbent rule)",
        "headline_rule": "global_threshold",
        "headline_rationale": "pLM-NN used a single constant threshold; the headline "
                              "row stays protocol-matched to it",
        "secondary_rule": secondary,
        "secondary_selection": note,
        "max_nested_fpr": MAX_NESTED_FPR,
        "cv_source": json.loads(MANIFEST.read_text()),
        "rules": out,
        "commitment": "The test set is scored exactly once, by --apply, after this "
                      "file is committed. The frozen threshold is reported whatever "
                      "it gives: if it lands below the FPR target with a lower TPR, "
                      "that is the result.",
    }
    OUT_DIR.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps(frozen_doc, indent=2))
    print(f"\nheadline rule : global_threshold (fixed in advance)")
    print(f"secondary rule: {secondary}  ({note})")
    print(f"\nwrote {FROZEN}")
    print("COMMIT THIS FILE before running --apply.")
    return frozen_doc


def apply() -> dict:
    if not FROZEN.exists():
        raise SystemExit(f"{FROZEN} does not exist. Run --select first, and commit it.")
    frozen = json.loads(FROZEN.read_text())
    test = _load(TEST_RESULT)
    y, s, g = test["y"], test["s"], test["g"]

    rows = []
    for rule, spec in frozen["rules"].items():
        thr = spec["frozen_threshold"]
        t = op.transform(rule, s, g)
        row = evaluate.summarise(y, t, g, thr, label=rule)
        row["operating_point"] = op.describe(rule, thr)
        row["kind"] = ("headline" if rule == frozen["headline_rule"]
                       else "secondary" if rule == frozen["secondary_rule"]
                       else "reference")
        rows.append(row)

    # The oracle: swept on test. Published so the honesty gap is visible.
    for rule in ("global_threshold", "rank", "zscore"):
        t = op.transform(rule, s, g)
        thr = evaluate.choose_threshold(y, t, g, target_fpr=config.TARGET_FPR)
        row = evaluate.summarise(y, t, g, thr, label=f"{rule} [ORACLE]")
        row["operating_point"] = op.describe(rule, thr)
        row["kind"] = "ORACLE - NOT A RESULT (swept on test)"
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["method", "kind", "operating_point", "TPR", "FPR", "MCC", "F1",
            "F1_weighted", "AUC", "AUPRC"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nAUC/AUPRC are identical across every row by construction: these rules "
          "are monotone within each protein, and per-protein AUC/AUPRC are rank "
          "statistics. The headline ranking numbers cannot move.")

    REPORT.write_text(json.dumps(
        {"frozen": frozen["created"], "headline_rule": frozen["headline_rule"],
         "secondary_rule": frozen["secondary_rule"], "rows": rows}, indent=2))
    print(f"\nwrote {REPORT}")
    return {"rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select", action="store_true", help="CV only; writes the freeze")
    mode.add_argument("--apply", action="store_true", help="scores test exactly once")
    args = ap.parse_args()
    select() if args.select else apply()
