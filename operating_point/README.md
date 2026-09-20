# Per-protein decision rules

The ranking is unchanged. Only the rule that turns a score into a call changes,
so **AUC 0.8618 and AUPRC 0.4413 are identical in every row below** — they are
per-protein rank statistics, and these rules are monotone within each protein.

Produced by `scripts/operating_point.py`, pre-registered in `frozen.json`
(committed before the test split was scored), significance by
`scripts/paired_bootstrap.py`. No GPU: it reuses the stored per-residue scores
from the run that produced the headline.

## The problem

`choose_threshold` sweeps to a mean per-protein FPR of 0.05 and hits exactly
0.0500 on every CV fold, but the frozen value lands at **0.0707** on test. One
global cutoff is shared by all 231 test proteins while every metric here is
computed per protein, so any between-protein shift in the score distribution
moves the realised FPR. RESULTS.md § 8 lists this as unmeasured.

## Selection (CV only — test not loaded)

Both per-protein rules are monotone per-protein transforms, so
`evaluate.choose_threshold` selects them unchanged: per-fold threshold at the
target FPR, frozen as the median of 4.

| rule | per-fold thresholds | frozen | operating point |
|---|---|---|---|
| global_threshold | 0.5896, 0.5433, 0.5825, 0.5586 | 0.5706 | `score >= 0.5706` |
| rank | 0.9264, 0.9193, 0.9215, 0.9237 | 0.9226 | top 7.74% per protein |
| zscore | 2.0450, 1.9016, 1.9631, 2.0162 | 1.9897 | `z >= 1.9897` within protein |

The global row reproduces the published 0.5706 exactly, which is the check that
this path matches the incumbent protocol.

Nested leave-one-fold-out (select on 3 folds pooled, apply to the 4th) is the
transfer estimate, and the only thing that chose the secondary rule:

| rule | FPR | TPR | MCC | F1 (weighted) |
|---|---|---|---|---|
| global_threshold | 0.0501 | 0.4386 | 0.3353 | 0.9194 |
| rank | 0.0501 | 0.4374 | 0.3422 | 0.9196 |
| **zscore** | 0.0500 | 0.4457 | **0.3456** | 0.9205 |

CV fold prevalence is 0.0516–0.0550 against test's 0.0528, so the folds are not
an unusually easy match.

## Test, scored once

| rule | | TPR | FPR | MCC | F1 | F1 (weighted) |
|---|---|---|---|---|---|---|
| global_threshold | headline | 0.5593 | 0.0707 | 0.3740 | 0.3793 | 0.9121 |
| rank | reference | 0.4794 | 0.0481 | 0.3772 | 0.3962 | 0.9219 |
| **zscore** | **secondary** | 0.4995 | **0.0502** | 0.3804 | 0.3945 | 0.9212 |
| pLM-NN (published) | | 0.48 | 0.05 | 0.39 | — | 0.92 |

**The FPR target transfers.** The frozen z-score rule lands at 0.0502 against a
0.05 target; the global cutoff overshoots to 0.0707. That is the § 8 gap closed,
and it is the point of the whole exercise: a within-protein rule cannot be moved
by a between-protein shift in score scale.

The honest-vs-oracle gap is essentially nil. Sweeping z directly on test gives
1.9923 against the CV-frozen 1.9897, for MCC 0.3805 vs 0.3804. Knowing the
oracle bought nothing, which is the outcome the pre-registration was there to
make checkable rather than assertable.

## Is the gain real?

Paired bootstrap, 10,000 resamples of the 231 test proteins, zscore minus
global_threshold (both arms score the same proteins, so the pairing cancels the
dominant "which proteins" variance):

| metric | Δ | 95% CI | |
|---|---|---|---|
| FPR | **−0.0205** | [−0.0251, −0.0159] | significant |
| F1 (positive) | **+0.0152** | [+0.0028, +0.0272] | significant |
| F1 (weighted) | **+0.0090** | [+0.0061, +0.0119] | significant |
| MCC | +0.0064 | [−0.0048, +0.0173] | **not** significant |
| TPR | **−0.0598** | [−0.0786, −0.0414] | significant, against us |
| AUC / AUPRC | +0.0000 | [+0.0000, +0.0000] | identical by construction |

**Reading: this is a trade, not a free win.** The rule buys a real FPR reduction
and real F1 gains, and pays for them with real TPR. That is what moving to a
stricter operating point does; the reason it is still worth reporting is that
the global rule was never *at* the intended operating point on test, so its
higher TPR was bought with an FPR overshoot nobody chose. MCC improves but not
significantly, so **MCC remains a loss against pLM-NN's 0.39** and should not be
claimed otherwise.

Against pLM-NN the secondary rule wins AUC, AUPRC, TPR and weighted F1, ties
FPR, and loses MCC — four wins, one tie, one loss, against the headline rule's
three and three.

## Why global_threshold is still the headline

pLM-NN sets a single constant threshold. Keeping a protocol-matched row means a
reader can compare against the published table without taking the deviation on
trust. The per-protein rule is reported beside it, named in advance in
`frozen.json`, not substituted for it.

## Files

| file | contents |
|---|---|
| `frozen.json` | the pre-registration: rules, target, selection procedure, nested-CV results, and the commitment that test is scored once |
| `result.json` | the test table above, including the oracle rows |
| `paired_bootstrap.json` | the paired CIs |
| `cv_scores_manifest.json` | provenance of the staged CV scores (run id, sha256 per file) |

## Reproduce

```bash
modal volume get cryptobench-fanout-results run_20260910_124330/ build/cv_scores/
python scripts/operating_point.py --select    # CV only; commit the result
python scripts/operating_point.py --apply     # scores test once
python scripts/paired_bootstrap.py
```
