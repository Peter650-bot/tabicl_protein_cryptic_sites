# TabICL constructor knobs: a null result

`cryptic_tabicl/model.py` pinned 16 of `TabICLClassifier`'s 22 arguments at
library defaults that had never been checked against this problem — 117
features, 5.3% prevalence, a 50,000-row context. Nine of them were swept.

**None of them helps.** The defaults are already the right settings here, which
is what the TabICLv2 paper claims for them ("we prioritize out-of-the-box
performance ... rather than hyperparameter tuning or fine-tuning").

Produced by `deploy/knob_sweep.py`, grid in `deploy/knob_grid.py`, run id
`ks_knobs1`. 10 jobs, 2.85 T4-hours, **$1.68**.

## Protocol

Every config is scored on **CV fold 0 only**. The test split was never touched.
Ranking nine configs against test and keeping the winner is nine looks at the
test set; the finalist was to be confirmed on folds 1–3 and scored on test once.

The context seed is 0 for every config, so the context draw is identical and
each comparison is **paired** — the 0.0015 AUPRC context-draw noise cancels
instead of swamping the effect.

**Determinism gate.** The `control` job's 75,800 scores are elementwise
identical (max abs diff 0.000e+00) to `k64_train-fold-0.json` from the run that
produced the headline. So surfacing the knobs changed nothing, and every delta
below is attributable to the knob alone.

## Result — CV fold 0, control AUPRC 0.3759

| config | AUPRC | ΔAUPRC | AUC | sec | |
|---|---|---|---|---|---|
| est32_norm5 | 0.3782 | **+0.0023** | 0.8166 | 2642 | best, but under the bar |
| kv_repr | 0.3759 | +0.0000 | 0.8148 | **416** | speed |
| control | 0.3759 | — | 0.8148 | 663 | baseline |
| outlier8 | 0.3759 | −0.0000 | 0.8149 | 693 | |
| batch8 | 0.3758 | −0.0001 | 0.8148 | 662 | speed |
| norm4 | 0.3755 | −0.0004 | 0.8153 | 655 | |
| avg_probs | 0.3754 | −0.0005 | 0.8148 | 653 | |
| est32 | 0.3751 | −0.0008 | 0.8158 | 2545 | |
| norm5 | 0.3747 | −0.0012 | 0.8147 | 677 | |
| shuffle_random | 0.3744 | −0.0015 | 0.8147 | 668 | |

**The pre-registered bar was ΔAUPRC > +0.003 to promote a finalist. Nothing met
it, so nothing was promoted and the test split was not scored.** The bar was set
at the single-draw noise band from RESULTS.md § 4.6 before the run.

## What the individual nulls rule out

- **Ensemble width does almost nothing.** `est32` quadruples the members for
  −0.0008 AUPRC and 4× the compute. The paper's "8 estimators unlocks most of
  the benefits of ensembling" holds here, and the remaining benefit is not
  merely small, it is absent.
- **Preprocessor diversity does not help, and slightly hurts.** Widening
  `norm_methods` from the default `['none','power']` to four or five methods
  costs nothing in compute (the member list is truncated to `n_estimators`
  either way) but moves AUPRC −0.0004 and −0.0012. The default pair is not an
  arbitrary choice.
- **The heavy-tail hypothesis is dead.** `outlier8` doubles the clipping
  threshold on the reasoning that 64 of the 117 features are ESM-2 principal
  components with heavy tails, and the default |z| > 4 clip might be deleting
  real signal. It moves AUPRC by −0.0000. Whatever is in those tails, the model
  was not using it.
- **Feature-shuffle scheme does not matter** (`shuffle_random`, −0.0015), nor
  does combining members in probability rather than logit space (`avg_probs`,
  −0.0005).

## The one thing worth keeping: `kv_cache="repr"` is 37% faster

416 s against the control's 663 s, for ΔAUPRC +0.0000. It caches the context
representation at `fit()` instead of re-forwarding all 50,000 context rows for
each of the ~6 protein chunks `predict_by_protein` produces.

**Caveat, and why it is not switched on by default.** It is *not*
bit-reproducible: max absolute score difference 3.793e-03 against the control,
because caching changes the order of float accumulation. The metrics do not move
at four decimal places, but enabling it would mean the published numbers no
longer reproduce exactly. Worth taking for future exploratory sweeps, not for
re-running anything that has to match the record.

## The honest read on est32_norm5

It is the paper's own TALENT configuration (32 estimators, 5 preprocessors), it
is the only config pointing the right way, and it has the best AUC in the table
(0.8166 vs 0.8148). But +0.0023 AUPRC is **inside** the single-draw noise band,
on **one** fold, for **4×** the compute — and its own 32-estimator sibling
without the extra preprocessors came out at −0.0008, which is not the behaviour
of a real effect. Under a pre-registered bar it does not promote, and reporting
it as a win after setting that bar would be exactly the selection the bar exists
to prevent.

If it is ever worth revisiting, the test is folds 1–3 plus seeds, at roughly $4,
and the prediction to beat is that it regresses to zero.

## Files

| file | contents |
|---|---|
| `screen.json` | per-config AUC/AUPRC/seconds on fold 0, plus the exact kwargs of each config |

## Reproduce

```bash
modal deploy deploy/knob_sweep.py     # deploy+spawn: this host drops .map() streams
modal run deploy/knob_sweep.py --stage screen --run-id <id> --spawn
modal run deploy/knob_sweep.py --stage screen --run-id <id> --collect
```
