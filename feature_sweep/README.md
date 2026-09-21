# The feature screen

Ten feature sets, screened on CV fold 0, paired on a fixed context seed.
Pre-registered in `deploy/feature_grid.py` (committed at `f01fe48`, before the
screen ran); arms defined in `cryptic_tabicl/feature_sets.py`; run by
`deploy/feature_sweep.py`.

**Every one of the ten arms is negative.** Nothing was promoted and the test
split was not scored. Run `fs_feat1`, 10 jobs, 1.88 T4-hours, **$1.11**.

## Why features, and why these features

Weights (fine-tuning, +0.002 with a CI crossing zero), inference knobs (nine,
all null) and the decision rule (a trade, no ranking metric moved) have all
been measured. Features are what is left, and they are where the measured
leverage is: 33 → 117 features moves AUPRC 0.343 → 0.441, while swapping the
model moves at most 0.061.

The *kind* of feature follows the evidence. arXiv 2609.13202 measures 26
feature-engineering conditions on TabICLv2 itself: numeric interactions and
group aggregation both come out at median **+0.00** macro-F1 (0 of 11 datasets
positive for TabICLv2), binning at −0.96, and feature selection is found to
harm consistently. Its conclusion is that the frontier has moved *"from
representation engineering toward information augmentation"*.

That is the distinction this screen is built on. A row here is one residue, and
the model cannot see the other residues of the same protein — so a neighbour's
conservation, or how this pocket ranks within this protein, is not a
rearrangement of the row's own columns. It is a fact the row did not contain.

## Protocol

Every arm is scored on **CV fold 0 and nothing else**. Ranking ten arms against
the test split and keeping the winner is ten looks at the test set; the
finalist is confirmed on folds 1–3 and scored on test exactly once.

The context seed is 0 for every arm, so the context draw is identical and each
comparison is **paired** — the 0.0015 AUPRC context-draw noise from RESULTS.md
§ 4.6 cancels rather than swamping the effect. That the draw really is
identical is not assumed: `build_context` samples on label and protein, never
on features, and `tests/test_feature_sets.py` pins it.

**The bar is ΔAUPRC > +0.003**, the single-draw noise band, the same bar the
knob sweep used, set before the run. A win is declared only if the four-fold
mean also clears it and all four fold deltas share a sign.

## The arms

| arm | features | prior | what it adds |
|---|---|---|---|
| `control` | 117 | baseline | the published set; also the determinism gate |
| `seqwin` | 130 | high | 5-residue backbone windows, on everything except `ens_*` |
| `esmnb` | 133 | high | 8 Å spatial mean of the 16 leading ESM components |
| `nbstat` | 132 | medium | neighbourhood spread and extreme, plus a 12 Å shell |
| `pctx` | 125 | medium | protein-level aggregates broadcast to each row |
| `prel` | 129 | **low** | within-protein rank and z |
| `esmsup32` | 85 | high | 32 fold-nested PLS components replace the 64 PCs |
| `esmsup32_seqwin_esmnb` | 114 | high | the supervised block plus both pooling families |
| `free_combo` | 150 | medium | `seqwin + pctx + prel` |
| `trim` | 93 | **low** | drop the 24 ESM PCs at \|AUC−0.5\| ≤ 0.02 on train |

Two arms carry an explicitly low prior and are expected to fail. `prel` because
within-protein normalisation does not add univariate signal here, and `trim`
because feature selection is the one thing arXiv 2609.13202 finds consistently
harmful and our own ladder has 117 features beating 85. They are screened
because they are free, and because reading a univariate table and concluding
"does not help" is the mistake RESULTS.md § 4.3 and § 4.3b record this project
making twice, in both directions.

## What was built, and what it cost

All five derived families are pure CPU: coordinates from the 1,135 cached
mmCIFs (~35 min, 4 workers), then pandas. The supervised block is a CPU Modal
job that reads the raw fp16 ESM-2-3B vectors already sitting in the
`cryptobench-esm` volume — no re-embedding, and no 1.95 GB download to a host
with 5 GB of RAM.

## Provenance gates

- **`derive.spatial_pool` reproduces every published `_nb8` column**
  elementwise over all 381,704 rows — 22 columns, max abs diff 5.7e-14. So the
  new pooling is the old pooling, and any delta below is the new columns rather
  than a changed definition of the old ones.
- **The `control` arm is the determinism gate.** Its fold-0 scores must equal
  `build/cv_scores/k64_train-fold-0.json` from the headline run. If they do
  not, something in the rebuild changed behaviour and the screen is void.
- **The supervised block is checked against the fold it is scored on.**
  `feature_sets.supervised_block` refuses a block whose manifest says it was
  fitted without a different fold — a projection scored on its own training
  rows would look like a large, clean win.

## The free proxy, and how wrong the priors were

Before spending anything, every arm was scored with logistic regression on each
of the four CV folds — `scripts/feature_proxy.py`, zero GPU, about a minute per
fold. This is **not** the screen and decides nothing: the model ladder has
TabICL ahead of logistic regression by 0.027–0.061 AUPRC at every feature
count, and the gap itself moves with the feature set, which is exactly the
quantity a proxy cannot see. A linear model also has the least possible use for
neighbour-pooled columns, because they are near-collinear with the parents it
already has.

ΔAUPRC against `control`, per fold:

| arm | fold 0 | fold 1 | fold 2 | fold 3 | mean | signs |
|---|---|---|---|---|---|---|
| `prel` | +0.0020 | +0.0020 | +0.0018 | +0.0020 | **+0.0019** | all same |
| `nbstat` | +0.0011 | +0.0005 | +0.0004 | +0.0011 | +0.0008 | all same |
| `free_combo` | +0.0007 | +0.0010 | +0.0028 | −0.0019 | +0.0006 | mixed |
| `esmnb` | −0.0035 | +0.0020 | +0.0017 | +0.0016 | +0.0004 | mixed |
| `pctx` | −0.0005 | +0.0011 | −0.0004 | +0.0006 | +0.0002 | mixed |
| `seqwin` | −0.0019 | −0.0019 | +0.0004 | −0.0007 | −0.0010 | mixed |
| `trim` | −0.0049 | −0.0022 | +0.0003 | −0.0037 | −0.0026 | mixed |
| `esmsup32` | −0.0163 | — | — | — | −0.0163 | — |
| `esmsup32_seqwin_esmnb` | −0.0160 | — | — | — | −0.0160 | — |

Three things worth saying out loud, because they are all corrections:

1. **The priors were wrong in both directions.** `prel` was entered with an
   explicitly LOW prior, on the grounds that within-protein normalisation does
   not add univariate signal — and it is the most consistent positive here, the
   only arm positive on all four folds at a non-trivial size. `seqwin` and
   `esmnb` were entered HIGH and are not. That is the third time this project
   has been misled by a univariate reading (RESULTS.md § 4.3, § 4.3b), in the
   third different direction.

2. **`esmsup32` fails under the model that should flatter it most.** PLS
   optimises its components for a *linear* predictor of the label, so logistic
   regression on those components is the best case for the method, and it loses
   0.016 AUPRC to the unsupervised basis it replaces. The likely reading is
   that supervision throws away embedding structure a non-linear model can
   still use — variance that is not linearly predictive is not therefore
   useless. This is evidence against the arm, gathered for free, and it is
   reported here rather than quietly dropped from the grid.

3. **Fold 0 is not a safe place to screen one of these arms.** `esmnb` is
   −0.0035 on fold 0 and positive on the other three. The pre-registration
   commits to fold 0, so fold 0 is what was run — but if the TabICL screen
   returns a null for `esmnb` specifically, this table is the reason not to
   read that as settled.

Nothing in the proxy clears the +0.003 bar, on any fold, for any arm.

## Result — CV fold 0, control AUPRC 0.3759

| arm | feat | AUPRC | ΔAUPRC | AUC | sec | prior |
|---|---|---|---|---|---|---|
| **control** | 117 | **0.3759** | — | 0.8148 | 660 | baseline |
| `trim` | 93 | 0.3751 | −0.0008 | 0.8144 | 550 | low |
| `pctx` | 125 | 0.3746 | −0.0013 | 0.8126 | 689 | medium |
| `seqwin` | 130 | 0.3709 | −0.0050 | 0.8153 | 715 | **high** |
| `esmnb` | 133 | 0.3708 | −0.0051 | 0.8123 | 716 | **high** |
| `nbstat` | 132 | 0.3697 | −0.0062 | 0.8140 | 716 | medium |
| `prel` | 129 | 0.3693 | −0.0066 | 0.8157 | 704 | low |
| `free_combo` | 150 | 0.3689 | −0.0070 | 0.8135 | 848 | medium |
| `esmsup32_seqwin_esmnb` | 114 | 0.3642 | −0.0117 | 0.8060 | 651 | **high** |
| `esmsup32` | 85 | 0.3574 | −0.0185 | 0.8029 | 515 | **high** |

**The determinism gate passed exactly**: the control's 75,800 fold-0 scores are
elementwise identical to `build/cv_scores/k64_train-fold-0.json`, max abs diff
**0.000e+00**, row order identical. Its 0.3759 / 0.8148 also reproduces the knob
sweep's independent control. So the rebuilt table is the published table.

`trim` and `pctx` are inside the 0.0015 single-draw noise band. Everything from
`seqwin` down is 3–12× it, and these are paired comparisons in which the
context-draw noise has already cancelled, so those losses are real.

## What the result actually says

**The feature-count story is dead.** The obvious reading — that we crossed
TabICLv2's 100-feature pretraining maximum — has the wrong sign. The correlation
between feature count and ΔAUPRC is **+0.33**: more features, *less* damage. The
smallest arm (`esmsup32`, 85) is the worst; `free_combo` at 150 is mid-table.

**A better feature made the model worse.** On the fold it was never fitted on,
`esm_sup00` reaches AUC **0.7353** against `esm_pc00`'s 0.6729 — the strongest
single ESM column this project has produced. Its arm loses 0.0185 AUPRC. The
medians explain it: PLS *concentrates* label-covariance, so the median of its 32
components is 0.5101, near noise; PCA *spreads* what it has, and the median of
its 64 is 0.5235, higher than PLS's despite a much weaker best. TabICL ensembles
over feature groups and evidently prefers many moderately-informative columns to
one excellent column and a tail of noise. Supervision optimised the wrong
objective — "predictive on its own, linearly" rather than "predictive".

**Pooling is saturated, not missing.** The published 117 already carry 25
neighbour-pooled columns at 8 Å. A second radius, a second statistic, and
pooling the one family that had never been pooled all lose. The marginal view of
a neighbourhood the model already sees costs more in redundancy than it pays.

## Files

| file | contents |
|---|---|
| `screen.json` | per-arm AUC/AUPRC/seconds on fold 0, with each arm's exact column list |
| `confirm.json` | the finalist on folds 1–3 |

## Reproduce

```bash
python scripts/build_coords.py
python scripts/build_derived.py
modal run deploy/esm_supervised_modal.py
python scripts/merge_derived.py --extra build/derived_features.parquet \
                                        build/esm_supervised.parquet
cp build/residues_derived.parquet build/esm_supervised_manifest.json \
   ~/cryptobench_stage/tables/

modal deploy deploy/feature_sweep.py          # this host drops .map() streams
modal run deploy/feature_sweep.py --stage screen --run-id <id> --spawn
modal run deploy/feature_sweep.py --stage screen --run-id <id> --collect
```
