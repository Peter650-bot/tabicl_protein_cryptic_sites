# Results

Per-residue cryptic binding site prediction on CryptoBench, zero-shot TabICLv2.

All numbers below are **measured**, on CryptoBench's own test split (231 apo structures,
76,636 residues, 5.28% positive), with metrics computed **per protein then averaged** —
the same procedure as CryptoBench's `Protein.py`, so they are comparable to the
published benchmark. The decision threshold is always selected on training folds and
frozen before the test set is touched.

Reproduce with `python scripts/compare_variants.py`.

> **Re-measured after the insertion-code fix (§ 7).** Residue keys previously dropped
> the PDB insertion code, so `B_60A` and `B_60` collapsed onto one key. Every number in
> this document was regenerated from a rebuilt table afterwards. The test split was
> unaffected — 0 of 76,636 test rows changed label — so the corrections reach the
> results only through the in-context training table, and the headline moved by less
> than the context-draw noise (AUC 0.882 → 0.882, AUPRC 0.480 → 0.476). The one
> conclusion that did not survive is in § 5.

---

## 1. Headline

**TabICLv2, zero-shot, 118 apo-only features (54 hand-built + 64 ESM-2 PCs):**

| metric | 118 feat (+ESM) | 54 feat (no pLM) |
|---|---|---|
| AUC | **0.882** | 0.867 |
| AUPRC | **0.476** | 0.450 |
| TPR | 0.591 | 0.565 |
| FPR | 0.062 | 0.064 |
| MCC | 0.410 | 0.389 |
| F1 (weighted — the published convention) | 0.921 | 0.918 |
| F1 (positive class) | 0.415 | 0.397 |

No training of any kind: the labelled context is a single forward pass, and the ESM
features are frozen embeddings, not a fine-tuned network. No holo structure at
prediction time.

**Both rows are worth keeping.** The 54-feature row is the stronger claim — it beats
pLM-NN on CB-full using no protein language model at all. The 118-feature row is the
better predictor. They answer different questions and neither supersedes the other.

---

## 2. Against the published benchmark

CryptoBench reports on several subsets. **Only CB-full is comparable to us** — we
evaluate every test protein, whereas the P2Rank and PocketMiner figures are on subsets
those tools could run on, and the P2Rank "holo" row is not the apo prediction task at
all.

| Method | Dataset | AUC | AUPRC | TPR | FPR | MCC | F1 (weighted) |
|---|---|---|---|---|---|---|---|
| **TabICLv2 +ESM (this work)** | **CB-full** | **0.882** | **0.476** | **0.591** | 0.062 | 0.410 | 0.921 |
| TabICLv2, no pLM (this work) | CB-full | 0.867 | 0.450 | 0.565 | 0.064 | 0.389 | 0.918 |
| pLM-NN (benchmark) | CB-full | 0.86 | 0.36 | 0.48 | 0.05 | 0.39 | 0.92 |
| **TabICLv2 +ESM (this work)** | **CB-P2RANK-apo** | **0.885** | **0.501** | **0.602** | 0.062 | 0.425 | 0.922 |
| TabICLv2, no pLM (this work) | CB-P2RANK-apo | 0.868 | 0.471 | 0.571 | 0.063 | 0.405 | 0.919 |
| pLM-NN | CB-P2RANK-apo | 0.88 | 0.42 | 0.51 | 0.04 | 0.43 | 0.93 |
| P2Rank | CB-P2RANK-apo | 0.81 | 0.21 | 0.62 | 0.14 | 0.27 | 0.81 |
| pLM-NN | CB-PM | 0.88 | 0.43 | 0.52 | 0.04 | 0.44 | 0.93 |
| PocketMiner | CB-PM | 0.76 | 0.19 | 0.51 | 0.16 | 0.22 | 0.78 |
| P2Rank | CB-P2RANK-**holo** | 0.89 | 0.34 | 0.84 | 0.15 | 0.38 | 0.81 |

The F1 column is the published one — a **weighted average over both classes**, computed
here exactly as the paper computes it (per structure, weighted, then averaged) so the
rows are directly comparable. See § *Metric conventions* below for why the positive-class
F1 must not be placed in this column. Published values are from
`external/CryptoBench/README.md`; ours regenerate with `python scripts/f1_by_subset.py`.

On CB-full the ESM variant exceeds pLM-NN on AUC (+0.022), AUPRC (+0.116), TPR (+0.111)
and MCC (+0.020), at a slightly looser FPR (0.062 vs 0.05). The FPR gap is threshold
transfer — the threshold was frozen from CV targeting 0.05 and landed at 0.062 on test.
**Read these margins against § 2.1**: the AUPRC and TPR gaps survive test-set
resampling, the MCC gap does not, and the FPR gap runs against us significantly.

CB-P2RANK-apo (the 184 single-chain test structures) is reconstructible from the shipped
dataset, so we report it too. Without a language model we trailed pLM-NN on AUC (−0.012)
and MCC (−0.025); with ESM we lead on AUC (+0.005), AUPRC (+0.081) and TPR (+0.092) and
are level on MCC (0.425 vs 0.43). Note this is the subset where our evaluation unit and
pLM-NN's are identical by construction, so it is the cleanest of the two rows. **The
+0.005 AUC margin here does not survive resampling (§ 2.1)** — only the AUPRC and TPR
margins do.

**F1 separates almost nothing here, and that is the finding.** Every pLM-based row —
ours and pLM-NN's, on both subsets — sits at 0.92–0.93; the column's only real
discrimination is against the structure-only tools (P2Rank 0.81, PocketMiner 0.78). At
5.28 % prevalence a weighted F1 is governed by the 95 % of residues that are trivially
negative, so it largely measures whether a method avoided flooding the negative class.
Two consequences for reporting: our 0.921 on CB-full is **not** a win over pLM-NN's 0.92
(their value is published to 2 dp, at which precision the two are equal), and on
CB-P2RANK-apo pLM-NN is genuinely ahead, 0.93 vs 0.922. Quote AUPRC, not F1.

**CB-PM is not reconstructible** from the shipped files — it is defined by which 60
structures PocketMiner failed on, which is a property of that tool's run, not of the
dataset. Getting that row requires asking the authors for the structure list.

### 2.1 Confidence intervals, and which claims survive them

Every number above is a point estimate on one test set. Because each metric is a **mean
over per-protein values**, a CI comes free: resample the test proteins with replacement
and re-average the stored per-protein table — no re-scoring, no GPU. 10,000 resamples,
percentile interval, `python scripts/bootstrap_test.py`.

**TabICLv2 + ESM (118 feat)**

| metric | CB-full (n=231) | 95 % CI | pLM-NN | CB-P2RANK-apo (n=184) | 95 % CI | pLM-NN |
|---|---|---|---|---|---|---|
| AUC | 0.882 | [0.863, 0.901] | 0.86 ✔ | 0.885 | [0.862, 0.906] | 0.88 ✗ |
| AUPRC | 0.476 | [0.439, 0.514] | 0.36 ✔ | 0.501 | [0.459, 0.543] | 0.42 ✔ |
| TPR | 0.591 | [0.546, 0.635] | 0.48 ✔ | 0.602 | [0.552, 0.651] | 0.51 ✔ |
| FPR | 0.062 | [0.057, 0.068] | 0.05 ✗† | 0.062 | [0.056, 0.068] | 0.04 ✗† |
| MCC | 0.410 | [0.377, 0.443] | 0.39 ✗ | 0.425 | [0.386, 0.460] | 0.43 ✗ |
| F1 (weighted) | 0.921 | [0.915, 0.927] | 0.92 ✗ | 0.922 | [0.914, 0.929] | 0.93 ✗† |

**TabICLv2, no pLM (54 feat)**

| metric | CB-full (n=231) | 95 % CI | pLM-NN | CB-P2RANK-apo (n=184) | 95 % CI | pLM-NN |
|---|---|---|---|---|---|---|
| AUC | 0.867 | [0.847, 0.886] | 0.86 ✗ | 0.868 | [0.844, 0.890] | 0.88 ✗ |
| AUPRC | 0.450 | [0.413, 0.487] | 0.36 ✔ | 0.471 | [0.431, 0.513] | 0.42 ✔ |
| TPR | 0.565 | [0.520, 0.609] | 0.48 ✔ | 0.571 | [0.521, 0.620] | 0.51 ✔ |
| FPR | 0.064 | [0.059, 0.070] | 0.05 ✗† | 0.063 | [0.056, 0.070] | 0.04 ✗† |
| MCC | 0.389 | [0.356, 0.422] | 0.39 ✗ | 0.405 | [0.367, 0.442] | 0.43 ✗ |
| F1 (weighted) | 0.918 | [0.911, 0.924] | 0.92 ✗ | 0.919 | [0.910, 0.926] | 0.93 ✗† |

✔ = pLM-NN's published value falls **outside** our interval, on the losing side — the
gap survives test-set resampling.  ✗ = it falls **inside**, so the difference is not
resolved by this test set.  † = outside the interval but on the side that favours
pLM-NN (lower FPR; higher F1 on CB-P2RANK-apo), i.e. a gap against us.

**What this changes.**

- **AUPRC and TPR are the real result.** pLM-NN sits below our interval on both metrics
  and both subsets. These are the claims to lead with.
- **The AUC lead does not survive.** On CB-P2RANK-apo the +0.005 margin is inside a
  ±0.022 interval; on CB-full only the 118-feature variant clears it, and the
  54-feature variant does not. Do not report "+0.005 AUC" as a lead.
- **MCC is not resolved anywhere** — pLM-NN's value is inside every interval. The
  earlier reading "level on MCC" is right; "+0.020 on CB-full" is not.
- **Two gaps run against us and are significant**: our FPR is genuinely higher (the
  threshold-transfer effect, 0.062 vs 0.05), and on CB-P2RANK-apo pLM-NN's weighted F1
  of 0.93 sits above our interval.

**What the interval does NOT cover.** Two caveats, both real:

1. It is a CI on *our* number, not on the *difference*. CryptoBench publishes point
   estimates only — without their per-protein scores a paired interval on the gap is
   not computable, so containment is the strongest available check.
2. It captures test-set sampling only, not the 50k context draw. That second source is
   far smaller, and it has now been measured for TabICL directly (§ 4.6, 12 context
   seeds on the 118-feature run): sd 0.0006 AUC and 0.0030 AUPRC, roughly 20–30×
   tighter than the intervals above, so it is a correction rather than a competing
   source. Seed 0, the one every headline run uses, is mid-pack there (AUPRC 0.4800
   against a mean of 0.4788 and a max of 0.4846). A cheaper stand-in agrees: over 20
   seeds with logistic regression on 54 features
   (`python scripts/context_seed_variance.py`) the draw moves AUC by sd 0.0004 and
   AUPRC by sd 0.0012 — though there seed 0 *was* the highest-AUPRC draw of the 20.


### Metric conventions, verified against the paper and `Protein.py`

| convention | CryptoBench | this work | comparable? |
|---|---|---|---|
| aggregation | per structure, then mean | per structure, then mean | yes |
| AUPRC estimator | `metrics.auc(recall, precision)` | same | yes |
| F1 | **weighted average over both classes** | positive-class | **no — see below** |
| decision threshold | 0.95, fixed on a validation set | selected on train folds for FPR 0.05 | per-method by design |
| ACC | reported | reported | meaningless at 5% prevalence |

The paper states F1 was "computed using a weighted average", which at ~5% prevalence is
dominated by the negative class — that is why their F1 (0.92) tracks their ACC (0.93)
rather than their TPR (0.48). `evaluate.py` now emits **both**: our `F1_weighted` is
**0.918** for the 54-feature variant and **0.921** with ESM, i.e. indistinguishable from
their 0.92, and our positive-class `F1` is **0.397** / **0.415**, which is the number
that actually carries information. Only `F1_weighted` may be placed next to the
published column.

The threshold is not a benchmark constant: pLM-NN used 0.95, PocketMiner 0.7 (its
authors' recommendation), and P2Rank emits binary calls with no threshold at all. Each
method sets its own, so selecting ours on training folds is the same procedure, not a
deviation. Applying their 0.95 to TabICL's scores would measure calibration rather than
ranking — only 1.15% of test residues reach 0.95, collapsing TPR to 0.121.

---

## 3. All model variants

Every row below is on the **corrected table and the fixed sampler**, so the comparison
isolates model and feature set. (The pre-fix document mixed two sampler eras in this
table; that is no longer the case, and the old-sampler rows are kept separately at the
bottom for the record.)

| Model | feat | negatives | AUC | AUPRC | TPR | FPR | MCC |
|---|---|---|---|---|---|---|---|
| **TabICLv2 zero-shot + ESM-64** | 118 | 50k context | **0.882** | **0.476** | 0.591 | 0.062 | 0.410 |
| TabICLv2 zero-shot + ESM-32 | 86 | 50k context | 0.880 | 0.472 | 0.594 | 0.064 | 0.408 |
| LogisticRegression + ESM-64 | 118 | 50k context | 0.875 | 0.472 | 0.503 | 0.044 | 0.393 |
| LogisticRegression + ESM-64 | 118 | full 305k | 0.875 | 0.470 | 0.526 | 0.054 | 0.406 |
| TabICLv2 zero-shot + ESM-16 | 70 | 50k context | 0.874 | 0.464 | 0.576 | 0.064 | 0.399 |
| LogisticRegression + ESM-32 | 86 | 50k context | 0.872 | 0.469 | 0.497 | 0.044 | 0.388 |
| LogisticRegression + ESM-32 | 86 | full 305k | 0.872 | 0.468 | 0.519 | 0.052 | 0.400 |
| **TabICLv2 zero-shot** | 54 | 50k context | **0.867** | **0.450** | 0.565 | 0.064 | 0.389 |
| LogisticRegression + ESM-16 | 70 | 50k context | 0.864 | 0.455 | 0.471 | 0.042 | 0.376 |
| LogisticRegression + ESM-16 | 70 | full 305k | 0.863 | 0.452 | 0.488 | 0.049 | 0.385 |
| RandomForest + ESM-64 | 118 | full 305k | 0.863 | 0.448 | 0.193 | 0.009 | 0.232 |
| TabICLv2 zero-shot | 48 | 50k context | 0.862 | 0.430 | 0.543 | 0.065 | 0.366 |
| RandomForest + ESM-64 | 118 | 50k context | 0.861 | 0.446 | 0.732 | 0.159 | 0.354 |
| RandomForest + ESM-32 | 86 | full 305k | 0.860 | 0.451 | 0.221 | 0.011 | 0.251 |
| RandomForest + ESM-16 | 70 | full 305k | 0.857 | 0.442 | 0.234 | 0.011 | 0.266 |
| **TabICLv2 zero-shot** | 34 | 50k context | **0.856** | **0.433** | 0.543 | 0.063 | 0.369 |
| RandomForest | 54 | 50k context | 0.855 | 0.432 | 0.682 | 0.137 | 0.350 |
| LogisticRegression | 54 | 50k context | 0.854 | 0.439 | 0.455 | 0.042 | 0.361 |
| LogisticRegression | 54 | full 305k | 0.854 | 0.437 | 0.477 | 0.051 | 0.373 |
| RandomForest | 54 | full 305k | 0.853 | 0.434 | 0.257 | 0.014 | 0.278 |
| LogisticRegression | 48 | 50k context | 0.847 | 0.423 | 0.443 | 0.043 | 0.339 |
| LogisticRegression | 48 | full 305k | 0.847 | 0.418 | 0.451 | 0.050 | 0.342 |
| RandomForest | 48 | 50k context | 0.846 | 0.401 | 0.662 | 0.139 | 0.326 |
| RandomForest | 34 | 50k context | 0.841 | 0.417 | 0.638 | 0.126 | 0.329 |
| RandomForest | 48 | full 305k | 0.839 | 0.398 | 0.249 | 0.014 | 0.257 |
| RandomForest | 34 | full 305k | 0.838 | 0.415 | 0.257 | 0.015 | 0.257 |
| LogisticRegression | 34 | 50k context | 0.835 | 0.414 | 0.443 | 0.044 | 0.335 |
| LogisticRegression | 34 | full 305k | 0.834 | 0.413 | 0.456 | 0.050 | 0.347 |
| RSA alone (floor) | 1 | — | 0.499 | 0.089 | 0.876 | 0.832 | 0.034 |

Three readings hold up:

1. **TabICL leads at every feature count**, including against models that see all 305k
   training rows — by ~0.005–0.010 AUC and ~0.004–0.020 AUPRC. Real, consistent, and
   small next to what the features do.
2. **The features carry most of the result.** Holding model and sampler fixed, TabICL
   goes 34 → 48 → 54 → 118 features for AUPRC 0.433 → 0.430 → 0.450 → 0.476. Swapping
   the model at fixed features moves it by ~0.02.
3. **The task is multivariate.** RSA alone is 0.499 — chance.

Note the 48-feature row is the one place where a feature addition did *not* help TabICL
(0.433 → 0.430 AUPRC, though AUC rose 0.856 → 0.862). The ensemble features' small
contribution is discussed in § 4.3; at this sample size a −0.003 AUPRC step is inside
the context-draw band.

### Superseded: old-sampler runs

Kept because § 4.1 quantifies the sampler bug and these are its evidence. Do not compare
them against the table above — the negative sampler differs, which is a larger effect
than anything else in this document.

| Model | feat | AUC | AUPRC |
|---|---|---|---|
| TabICLv2 zero-shot ×5 contexts | 48 | 0.840 | 0.327 |
| TabICLv2 zero-shot | 48 | 0.839 | 0.323 |
| TabICLv2 zero-shot | 34 | 0.830 | 0.324 |
| LogisticRegression | 48 | 0.827 | 0.333 |
| RandomForest | 48 | 0.822 | 0.299 |
| RandomForest | 34 | 0.820 | 0.312 |
| LogisticRegression | 34 | 0.813 | 0.324 |

The RSA floor at 0.499 confirms no single descriptor solves this — the signal is
genuinely multivariate.

### Cross-validation (fixed sampler)

Inner CV over CryptoBench's train folds, used only to freeze the threshold:

| fold | 54 feat AUC / AUPRC | 118 feat AUC / AUPRC |
|---|---|---|
| train-fold-0 | 0.835 / 0.404 | 0.845 / 0.431 |
| train-fold-1 | 0.863 / 0.479 | 0.878 / 0.506 |
| train-fold-2 | 0.844 / 0.441 | 0.854 / 0.460 |
| train-fold-3 | 0.838 / 0.433 | 0.850 / 0.456 |

Frozen thresholds: 0.5527 (54 feat) and 0.5677 (118 feat). ESM helps on every fold, by
+0.023 to +0.027 AUPRC. Spread of ~0.03 AUC across folds — no single fold carries the
result.

---

## 4. What actually moved the number

Ranked by effect on AUPRC, which is the honest summary at 5% prevalence.

| change | ΔAUPRC | cost |
|---|---|---|
| Fixing the negative sampler | **+0.113** | one line |
| Fixing PDB column alignment | +0.144 *(on ensemble features alone)* | one line |
| Adding 64 ESM-2 principal components | +0.026 | 0.06 A10G-hours (~$0.07) |
| Adding 6 conservation features | +0.017 | ~2,300 API calls |
| Adding 14 ensemble pocket features | +0.014 | ~28,000 fpocket runs |
| Averaging over 5 contexts | +0.004 | 5× GPU inference |
| Fixing the insertion-code key (§ 7) | −0.004 *(noise; 0 test rows changed)* | three lines |

Three of the five largest entries are bugs in code I wrote, and the deliberate
capability additions are the smaller ones. The insertion-code fix is in the table for
completeness rather than for its effect: it corrected 32 mislabelled residues and
restored a quarantined protein, all of them in the training split, and the resulting
movement is smaller than the context-draw noise.

By measured compute ESM-2 was the best value of the three additions — the largest gain
for a few minutes of A10G time. The real cost there was baking an 11 GB checkpoint into
the image, not running it.

### 4.1 The negative sampler (+0.113 AUPRC)

The sampler reserved 40% of the negative budget for residues lining already-open
pockets, on the reasoning that those are the hard negatives. Isolating each choice with
logistic regression at fixed ~49k context:

| context | AUC | AUPRC |
|---|---|---|
| capped, no forced share | 0.847 | **0.420** |
| uncapped, no forced share | 0.847 | 0.417 |
| uncapped, 40% forced share | 0.819 | 0.317 |
| capped + 40% forced share *(old default)* | 0.827 | 0.333 |
| all 288k negatives *(the target)* | 0.847 | 0.417 |

Only ~17k residues in the dataset line a non-cryptic pocket, so a 40% quota over ~33k
slots over-represents that stratum by an order of magnitude. The per-protein cap was
harmless; the quota cost 0.087 AUPRC for logistic regression and 0.113 for TabICL.

**Consequence: a 50k context already matches training on all 288k negatives (0.422 vs
0.417). There is no context-size ceiling at this dataset size.** Every earlier
conclusion about TabICL being handicapped by its context window was an artifact of this
quota.

Setting the share to `0.0` was *also* wrong: splitting the pool and taking zero from one
side excludes pocket-lining negatives entirely — the same distortion mirrored. The
stratification must be skipped, not zeroed. A test now pins the observed rate to the
natural rate.

### 4.2 Context ensembling (+0.004 AUPRC — the hypothesis that failed)

Five contexts, each a different 11.4% sample of the negative pool (pairwise Jaccard
0.18, union covering 37.8%), combined three ways:

| K | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| AUPRC | 0.323 | 0.324 | 0.327 | 0.327 | 0.327 |

Predicted +0.084, delivered +0.004, saturating by K=3. All three combination rules
(mean-probability, mean-logit, mean-rank) agree. This is what falsified the
context-ceiling hypothesis and led to finding the sampler bug.

Hard-negative *mining* was also tried — rank negatives by a cheap model, take the most
cryptic-looking. It loses to plain proportional sampling (0.398–0.410 across mixtures),
and pure mining collapses to 0.057 because it removes every example of an easy negative.

### 4.3 Ensemble pocket features (+0.014 AUPRC)

CryptoSite's five strongest descriptors are all ensemble pocket statistics, led by mean
fpocket pocket score over MD snapshots (−log₁₀P = 138.6). We approximate its
100k-snapshot AllosMod ensemble with 24 ANM normal-mode conformations per protein and
run fpocket on each (~28k invocations, on Modal).

Ablation, logistic regression on full train, pooled test metrics:

| feature set | n | AUC | AUPRC |
|---|---|---|---|
| Tier A only | 34 | 0.864 | 0.366 |
| ensemble only | 14 | 0.766 | 0.162 |
| Tier A + ensemble | 48 | **0.876** | **0.380** |

They are **not redundant** — maximum absolute correlation with any Tier A feature is
0.21 — there is simply little incremental information. Three of the combined model's
fourteen largest weights are ensemble features.

Measured end-to-end on TabICL against the corrected table, the effect is smaller still
and not uniformly positive: 34 → 48 features gives AUC 0.856 → 0.862 (+0.006) but AUPRC
0.433 → 0.430 (−0.004), and on the hard stratum only +0.005. The ablation above is
logistic regression on pooled metrics, which is a different and more favourable
measurement; both are reported rather than only the flattering one.

Why so much weaker than CryptoSite's result: its −log₁₀P ranking is **univariate**,
measuring each descriptor alone against the null rather than its increment over the
other 57. CryptoSite's *final* model kept only three features, so its MD pocket score
dominated a small set. Our 34 geometric descriptors already encode much of what makes a
region pocket-like.

### 4.3b Conservation features (+0.017 AUPRC)

Sequence conservation measures how strongly evolution resists substitution at a
position: align a protein against its homologues and a column that never varies is
doing something the organism cannot afford to change. Functional sites are under
selection and show up conserved.

Sourced from the **PDBe Graph API**, which serves precomputed per-residue conservation.
The CryptoBench authors used the same API to analyse their own dataset
(`src/F-statistics/conservation/`), so the scores are consistent with the benchmark
rather than a different pipeline's numbers. It also returns full per-amino-acid
probability distributions, so we compute proper Shannon entropy — CryptoSite's actual
definition — rather than relying on the binned integer grade.

Six features: `cons_score`, `cons_entropy`, `cons_prob_self`, plus their 8 Å
neighbourhood means. Coverage 1,136/1,136 structures, zero quarantined, median join
match rate 1.000.

| feature | test AUC (sign-corrected) |
|---|---|
| `cons_prob_self_nb8` | 0.673 |
| `cons_score_nb8` | 0.668 |
| `cons_entropy_nb8` | 0.659 |
| `cons_score` | 0.635 |
| `cons_entropy` | 0.631 |
| `cons_prob_self` | 0.627 |

These are the strongest individual features added in the project. Ablation with
logistic regression on full train (pooled test metrics):

| feature set | n | AUC | AUPRC |
|---|---|---|---|
| 48 features | 48 | 0.876 | 0.380 |
| conservation only | 6 | 0.694 | 0.134 |
| **54 features** | 54 | **0.889** | **0.394** |

Redundancy with geometry is moderate — maximum |correlation| with any of the 48 is
0.31–0.45, against protrusion and half-sphere exposure. Higher than the ensemble
features' 0.21, but still substantially independent.

**A caveat on the reasoning that selected this feature.** Conservation was chosen on the
argument that it should crack the hard stratum specifically: a cryptic binding site is
functional and under selection, while an incidental open pocket is not. Judged
univariately that prediction failed — in the hard stratum the conservation features
score 0.585–0.606, indistinguishable from `ens_mean_nb8` (0.609) and `convexity`
(0.604). Judged in combination it partly succeeded: the stratum moved 0.681 → 0.699,
the largest shift it has seen. Reading univariate AUCs and concluding "this does not
help here" was the same mistake as trusting CryptoSite's univariate ranking, in the
opposite direction.

### 4.4 The PDB fixed-column bug

PDB is a fixed-column format and fpocket parses it positionally. The first PDB writer
omitted the `altLoc` column, shifting `resName`, `chainID`, `resSeq` and all three
coordinates one place left. fpocket still ran and still reported pockets — it simply
typed the atoms wrongly. Same 24 proteins, same amplitude:

| feature | malformed | correct |
|---|---|---|
| `ens_mean_nb8` | 0.550 | **0.694** |
| `ens_opening` | 0.506 | **0.653** |
| combined (grouped CV) | 0.628 | **0.755** |

With the bug, `ens_opening` sat at chance and the honest reading was "ANM is a poor
surrogate for MD". That conclusion was entirely an artifact. A tolerant downstream
parser turns a format error into a quiet loss of signal, which is indistinguishable
from a genuine negative result.

### 4.5 ANM amplitude

Displacements are rescaled to a target Cα RMSD. Swept out-of-sample, grouped by protein:

| target RMSD | mean pocket opening | combined AUC |
|---|---|---|
| **1.5 Å** | 0.105 | **0.705** |
| 3.0 Å | 0.176 | 0.668 |
| 5.0 Å | 0.281 | 0.680 |

Larger amplitudes open more pockets but open them *everywhere*. Selectivity, not
magnitude, carries the signal.

---

### 4.6 Context-draw variance (12 seeds)

Measured on the **pre-fix** table; not re-run after § 7 (it costs ~$6.60 of T4 time and
the corrected headline sits inside the band below). § 5.1 reuses its stored per-residue
scores to put an error bar on the per-stratum AUCs, which is a question about noise
magnitude rather than about the corrected point estimate.

| metric | mean | sd | min – max | seed 0 (reported) |
|---|---|---|---|---|
| AUC | 0.8822 | 0.0006 | 0.8814 – 0.8832 | 0.8819 |
| AUPRC | 0.4788 | 0.0030 | 0.4752 – 0.4846 | 0.4800 |
| TPR | 0.5974 | 0.0042 | 0.5912 – 0.6047 | 0.5952 |
| FPR | 0.0635 | 0.0011 | 0.0617 – 0.0653 | 0.0617 |
| MCC | 0.4115 | 0.0029 | 0.4062 – 0.4153 | 0.4142 |
| F1 (positive class) | 0.4163 | 0.0029 | 0.4113 – 0.4206 | 0.4185 |
| F1 (weighted) | 0.9209 | 0.0005 | 0.9201 – 0.9221 | 0.9221 |
| Accuracy | 0.9130 | 0.0008 | 0.9119 – 0.9147 | 0.9147 |

---

## 5. Where the model is weak

Test set stratified by whether a residue lines an already-open (non-cryptic) pocket,
118-feature run:

| stratum | residues | prevalence | AUC | AUPRC | TPR |
|---|---|---|---|---|---|
| bulk surface | 73,296 | 3.5% | 0.879 | 0.257 | 0.482 |
| lines a non-cryptic pocket | 3,340 | 44.0% | **0.714** | 0.635 | 0.942 |

0.714 against 0.879 remains the limiting factor: telling a cryptic site from a pocket
that is *already open* is the discrimination the task is actually about, and it is
where essentially all of the remaining headroom sits.

**How each addition moved it:**

All rows on the corrected table and the fixed sampler, so this is a like-for-like
series for the first time:

| run | feat | hard stratum | bulk surface | gap |
|---|---|---|---|---|
| Tier A | 34 | 0.667 | 0.824 | 0.157 |
| + ensemble pocket features | 48 | 0.672 | 0.838 | 0.166 |
| + conservation | 54 | 0.690 | 0.852 | 0.163 |
| + ESM-16 | 70 | 0.705 | 0.867 | 0.162 |
| + ESM-32 | 86 | 0.710 | 0.876 | 0.166 |
| **+ ESM-64** | **118** | **0.714** | **0.879** | **0.165** |

### 5.1 A conclusion that did not survive re-measurement

The previous version of this section read: *"ESM did not close the gap — it widened it.
The bulk surface improved by +0.027 while the hard stratum improved by +0.015, so the
spread went from 0.152 to 0.164. A protein language model makes the model better at the
part it was already good at."*

**That claim does not hold.** On the corrected table the two strata improve by almost
the same amount — bulk +0.027 (0.852 → 0.879), hard +0.024 (0.690 → 0.714) — and the
gap is flat across k at 0.162–0.166.

The reason the earlier reading was unsafe is measurable. Recomputing per-stratum AUC
for each of the 12 context seeds in § 4.6 (the scores are stored, so this costs
nothing):

| stratum | mean | sd | min – max |
|---|---|---|---|
| bulk surface | 0.8783 | 0.0006 | 0.8777 – 0.8794 |
| lines a non-cryptic pocket | 0.7132 | **0.0045** | 0.7073 – 0.7233 |
| gap | 0.1650 | **0.0043** | 0.1558 – 0.1708 |

The hard stratum holds 3,340 residues against the bulk stratum's 73,296, and its AUC is
**seven times noisier** under a change of context draw — sd 0.0045 against 0.0006. The
gap inherits that noise. So a +0.012 movement in the gap, read off two single draws, was
never distinguishable from resampling; both the old 0.152 and the new 0.163 sit inside
the seed band, and so does every value of k.

The general lesson is the same one § 4.3b records in the opposite direction: a
difference measured on a small stratum needs its own error bar before it carries an
interpretation. The stratum is 4.4% of the test set, and nothing about the headline
AUC's tightness (sd 0.0006) transfers to it.

**What is still true, and is the finding:** the gap sits at 0.157–0.166 across every
feature set tried, and nothing has closed it. Each addition lifts both strata together —
34 → 118 features moves the hard stratum +0.047 and bulk surface +0.055 — so the
descriptors are getting better at pocket-likeness in general rather than at crypticity
in particular. Ensemble pocket features are the weakest of the three additions here
(+0.005 hard), which is the same conclusion § 4.3 reaches from the headline metrics:
harmonic collective motion cannot produce the side-chain rotamer flips and loop
crossings that plausibly separate a cryptic site from an already-open one, and no
amount of resampling substitutes for that.

---

## 6. Dataset and integrity

| | |
|---|---|
| Rows | 381,189 residues |
| Proteins | 1,136 apo structures (0 quarantined) |
| Prevalence | 5.34% positive |
| Features | 118 (54 without the pLM), all computed from the **apo structure only** |
| Splits | CryptoBench's own sequence-identity folds, group = apo structure |
| Label match rate | median 1.000, min 0.800 |
| Ensemble feature coverage | 100% of residues and proteins |
| Tests | 85 passing |

Four guards enforced by tests:

1. **No holo-derived feature reaches X.** A cryptic site is *defined* by the apo→holo
   change, so any holo-derived quantity reconstructs the label and is unavailable at
   prediction time. `tests/test_no_leakage.py` fails the build on a holo-named column.
2. **No protein appears in two splits.** Residues within a protein are correlated and
   homologues leak across a random split; we use CryptoBench's shipped folds.
3. **Labels are the union over all holo partners.** 5,493 pairings for 1,136 apo
   structures (~5 each); labelling from one partner would mark true cryptic residues as
   negatives.
4. **The residue key carries its insertion code.** `tests/test_icode_keys.py` pins the
   key format, the build/record spellings agreeing, and all three feature merges joining
   on the same four-part key. See § 7.

---

## 7. The insertion-code key collapse

Author residue numbering is not dense. Proteases and immunoglobulins insert residues
without renumbering the rest of the chain, so 3bv9 chain B holds **ten** residues at
position 60 — 60, 60A, 60B … 60I — and six at 149. CryptoBench keys them the way the
PDB spells them, `B_60` and `B_60A`. This project keyed residues on `(chain, seq_id)`
alone, which collapsed all ten onto `B_60`.

That failed in both directions at once, which is why it survived so long:

- an annotated `B_60A` matched no observed key, so a true positive was scored negative;
- an annotated `B_149` matched all six residues at 149, so five residues were handed a
  positive label they never earned.

The second is the more common failure. Measured on the built table:

| | wrongly labelled 1 | wrongly labelled 0 |
|---|---|---|
| cryptic label | **19** | 1 |
| `in_noncryptic_pocket` | 9 | 3 |

32 residues of 380,930, across six structures (`1fdpA`, `1pfzC`, `3psgA`, `3bn9A`,
`3oy6U`, `3vhoB`), plus `3bv9B` quarantined outright — 6 of its 29 annotated keys
carried an insertion code, giving a match rate of 0.793 against the 0.80 threshold.
Only 9 of 20,387 annotated cryptic keys carry a code at all, which is why exactly one
structure crossed the line and the rest went unnoticed.

**Every affected residue was in the training split. Zero of 76,636 test rows changed.**
That is luck rather than design, and it is the reason the correction moves the headline
so little: the corrected labels reach the model only through the in-context table, where
they amount to 11 changed rows out of 50,000.

| | before | after |
|---|---|---|
| rows / proteins | 380,930 / 1,135 | 381,189 / 1,136 |
| quarantined | 1 | **0** |
| 3bv9B match rate | 0.793 | **1.000** |
| context positives / proteins | 16,282 / 904 | 16,293 / 905 |
| **AUC (118 feat)** | 0.882 | **0.882** |
| **AUPRC (118 feat)** | 0.480 | **0.476** |

The AUPRC movement is a context-draw effect, not a label effect — 0.476 sits inside the
12-seed range of § 4.6 (0.4752 – 0.4846). Every conclusion in § 2.1 survives unchanged;
the one that did not is in § 5.1.

### What was fixed

`RESIDUE_KEY_FMT` now spells the key `{chain}_{seq_id}{icode}`, and the three places
that build or consume it agree. The same collision existed a second time downstream:
`merge_ensemble` and `merge_conservation` joined on the three-part key and called
`drop_duplicates`, so residues 60A–60I received whichever row survived — the features of
residue 60. Both now join on the four-part key under `validate="1:1"`, and a duplicate
key is an error rather than something to deduplicate.

Two related defects were fixed in passing:

- **The ANM seed was not reproducible.** `ensemble_modal` seeded each protein's ensemble
  with `abs(hash(apo_id))`, and `hash()` on a `str` is salted per interpreter, so every
  run drew different conformations. It is now blake2b of the id.
- **`evaluate_modal` could not regenerate `modal_result_esm0.json`** — `esm0` is
  `tabicl_fanout`'s name for k=0, which no variant in the serial script mapped to. Both
  names now resolve to the 54-feature table.

The ensemble and conservation features were **backfilled** with their insertion codes
rather than recomputed, so their values are bit-identical to the runs behind the earlier
numbers and this comparison isolates the key fix from an ANM reseed. Both producers emit
one row per residue record in record order, so the code is recoverable positionally;
alignment was verified `(chain, seq_id)` position by position for all 1,136 structures,
and independently cross-checked against the ESM feature table, which had carried its own
`icode` from a separate code path all along — the two agree on 381,189 of 381,189 rows.

---

## 8. Not measured

- Fine-tuned TabICL (`tabicl[finetune]`) — deliberately out of scope; it would forfeit
  the zero-training property that makes the comparison interesting.
- Fragment docking (CryptoSite's #3) — needs PatchDock + RankScore.
- The CryptoSite 14-pair external test set — `scripts/cryptosite_testset.py` builds it
  (none of its apo structures appears in CryptoBench), but scoring has not been run.
- Context-draw variance for the corrected table — § 4.6's 12 seeds are from the
  pre-fix run. Re-running costs ~$6.60 of T4 time and was deliberately skipped; the
  strata analysis in § 5.1 reuses its stored scores, which is valid because it asks
  about noise magnitude rather than about the corrected point estimate.
