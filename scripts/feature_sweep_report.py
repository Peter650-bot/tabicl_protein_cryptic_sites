"""Turn a feature-sweep run into the tracked record, and gate it on determinism.

WHY A SEPARATE SCRIPT
    `deploy/feature_sweep.py` writes to `build/`, which is gitignored, so its
    output cannot be reviewed in a diff. This reads the per-job score files
    pulled from the Modal volume and writes `feature_sweep/screen.json`, which
    is tracked -- the same arrangement `operating_point/` and `knob_sweep/` use,
    and for the same reason.

    It also costs no GPU, so a mistake in the reporting is free to fix.

THE DETERMINISM GATE
    The `control` arm is the published 117 features, drawn from a table that has
    been rebuilt, re-merged and re-ordered since the headline run. Its fold-0
    scores must come back ELEMENTWISE identical to
    `build/cv_scores/k64_train-fold-0.json`. If they do not, something in the
    rebuild changed behaviour, every delta in the screen is measuring two things
    at once, and the run is void.

    This is not a formality. TabICL's default `feat_shuffle_method="latin"`
    assigns features to groups by POSITION, so a column reordering alone would
    move the scores while leaving every value in the table correct.

USAGE
    modal volume get cryptobench-feature-results <run-id>/ build/feature_scores/
    python scripts/feature_sweep_report.py --scores build/feature_scores/<run-id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptic_tabicl import config as C  # noqa: E402
from cryptic_tabicl import evaluate  # noqa: E402

GPU_HOURLY_USD = 0.59
KEYS = ["apo_id", "chain", "seq_id", "icode"]


def _align(job: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "apo_id": job["apo_id"],
        "icode": [str(i) for i in job["icode"]],
        "y": job["y"],
        "score": job["score"],
    })


def determinism_gate(control: dict, reference: Path) -> dict:
    """Compare the control arm's scores to the headline run's, elementwise."""
    if not reference.exists():
        return {"ran": False, "reason": f"{reference} not present"}
    ref = json.loads(reference.read_text())
    a = np.asarray(control["score"], dtype=float)
    b = np.asarray(ref["score"], dtype=float)
    if len(a) != len(b):
        return {"ran": True, "passed": False,
                "reason": f"{len(a)} scores vs {len(b)} in the reference"}
    same_rows = (list(control["apo_id"]) == list(ref["apo_id"]))
    diff = float(np.max(np.abs(a - b)))
    return {"ran": True, "passed": bool(same_rows and diff == 0.0),
            "max_abs_diff": diff, "row_order_identical": same_rows,
            "n": len(a), "reference": str(reference)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True,
                    help="directory of per-job JSONs pulled from the volume")
    ap.add_argument("--stage", default="screen")
    ap.add_argument("--reference", type=Path,
                    default=C.BUILD_DIR / "cv_scores" / "k64_train-fold-0.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(C.PROJECT_ROOT))
    from deploy.feature_grid import ARMS, MIN_DELTA_AUPRC, SEED

    files = sorted(args.scores.glob("*.json"))
    if not files:
        raise SystemExit(f"no job files under {args.scores}")
    jobs = [json.loads(f.read_text()) for f in files]
    print(f"{len(jobs)} job files from {args.scores}")

    rows: dict[str, dict] = {}
    for j in jobs:
        d = _align(j)
        m = evaluate.summarise(d["y"].to_numpy(), d["score"].to_numpy(),
                               d["apo_id"].to_numpy(), 0.5, label=j["name"])
        rows.setdefault(j["name"], {})[j["held"] or "final"] = {
            "AUC": m["AUC"], "AUPRC": m["AUPRC"],
            "n_features": j["n_features"], "seconds": j["seconds"],
            "n_rows": len(d),
        }

    gpu_seconds = sum(j["seconds"] for j in jobs)
    print(f"{gpu_seconds/3600:.2f} GPU-h  (~${gpu_seconds/3600*GPU_HOURLY_USD:.2f})")

    control_job = next((j for j in jobs
                        if j["name"] == "control" and j["held"] == "train-fold-0"), None)
    gate = ({"ran": False, "reason": "no control job on fold 0"} if control_job is None
            else determinism_gate(control_job, args.reference))
    print("\ndeterminism gate:", json.dumps(gate))
    if gate.get("ran") and not gate.get("passed"):
        print("  FAILED -- the rebuild changed behaviour; the screen is void.")

    folds = sorted({f for r in rows.values() for f in r})
    print()
    for fold in folds:
        base = rows.get("control", {}).get(fold, {}).get("AUPRC")
        print(f"--- {fold} "
              f"(control AUPRC {base:.4f})" if base else f"--- {fold}")
        print(f"{'arm':<24} {'feat':>5} {'AUPRC':>8} {'dAUPRC':>9} {'AUC':>8} "
              f"{'sec':>6} {'prior':>8}")
        print("-" * 78)
        ordered = sorted(
            (n for n in rows if fold in rows[n]),
            key=lambda n: -rows[n][fold]["AUPRC"])
        for name in ordered:
            r = rows[name][fold]
            d = r["AUPRC"] - base if base is not None else float("nan")
            flag = "PROMOTE" if (name != "control" and d >= MIN_DELTA_AUPRC) else ""
            prior = ARMS.get(name, ("", ""))[0]
            print(f"{name:<24} {r['n_features']:>5} {r['AUPRC']:>8.4f} {d:>+9.4f} "
                  f"{r['AUC']:>8.4f} {r['seconds']:>6.0f} {prior:>8}  {flag}")
        print()

    out = args.out or (C.PROJECT_ROOT / "feature_sweep" / f"{args.stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": args.stage, "seed": SEED, "bar": MIN_DELTA_AUPRC,
        "gpu_hours": gpu_seconds / 3600,
        "usd": gpu_seconds / 3600 * GPU_HOURLY_USD,
        "determinism_gate": gate,
        "rows": rows,
        "priors": {n: ARMS[n][0] for n in rows if n in ARMS},
        "rationale": {n: ARMS[n][1] for n in rows if n in ARMS},
        "features": {j["name"]: j["features"] for j in jobs},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
