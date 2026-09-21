# Results

Per-residue cryptic binding site prediction on CryptoBench, zero-shot TabICLv2.

All numbers below are **measured**, on CryptoBench's own test split (231 apo structures,
76,753 residues, 5.28% positive), with metrics computed **per protein then averaged** —
the same procedure as CryptoBench's `Protein.py`, so they are comparable to the
published benchmark. The decision threshold is always selected on training folds and
frozen before the test set is touched.

Reproduce with `python scripts/compare_variants.py`.

> **Re-measured after removing a holo-derived feature (§ 9).** `in_noncryptic_pocket`
> was an input to the model. CryptoBench derives that annotation by comparing apo to
> holo (`apo_pocket_rms < MAX_POCKET_RMSD`) — the same comparison that makes the label,
> with the inequality flipped — so it cannot be computed on a novel apo structure, and
> a model using it cannot be run on the task this project is about. It is now metadata,
> used only to stratify the evaluation. **Every number in this document was regenerated
> without it**, along with three smaller corrections (§ 9).
>
> The headline fell: **AUC 0.882 → 0.862, AUPRC 0.476 → 0.441**. Two claims did not
> survive and have been withdrawn — see § 2.1 and § 9.1. Two got *stronger*: the margin
> over the sklearn controls roughly doubled, and the per-stratum AUCs rose slightly,
> because the feature was shifting the model's prior rather than helping it rank.

---

## 1. Headline

**TabICLv2, zero-shot, 117 apo-only features (53 hand-built + 64 ESM-2 PCs):**

| metric | 117 feat (+ESM) | 53 feat (no pLM) |
|---|---|---|
| AUC | **0.862** | 0.848 |
| AUPRC | **0.441** | 0.388 |
| TPR | 0.559 | 0.515 |
| FPR | 0.071 | 0.073 |
| MCC | 0.374 | 0.336 |
| F1 (weighted — the published convention) | 0.912 | 0.906 |
| F1 (positive class) | 0.379 | 0.345 |

No training of any kind: the labelled context is a single forward pass, and the ESM
features are frozen embeddings, not a fine-tuned network. No holo structure at
prediction time — and, since § 9, no quantity *derived* from one either.

**The two rows no longer answer the same question they used to.** The 53-feature row
was previously described here as the stronger claim, on the grounds that it beat
pLM-NN without a protein language model. That is no longer true: at 53 features AUPRC
0.388 [0.356, 0.420] **contains** pLM-NN's 0.36, so it is a tie. Only the 117-feature
row clears the benchmark, and only on AUPRC and TPR (§ 2.1). The 53-feature row is
still worth keeping as the no-pLM reference point; it is not a win.

---

## 2. Against the published benchmark

CryptoBench reports on several subsets. **Only CB-full is comparable to us** — we
evaluate every test protein, whereas the P2Rank and PocketMiner figures are on subsets
those tools could run on, and the P2Rank "holo" row is not the apo prediction task at
all.

| Method | Dataset | AUC | AUPRC | TPR | FPR | MCC | F1 (weighted) |
|---|---|---|---|---|---|---|---|
| **TabICLv2 +ESM (this work)** | **CB-full** | 0.862 | **0.441** | **0.559** | 0.071 | 0.374 | 0.912 |
| TabICLv2, no pLM (this work) | CB-full | 0.848 | 0.388 | 0.515 | 0.073 | 0.336 | 0.906 |
| pLM-NN (benchmark) | CB-full | 0.86 | 0.36 | 0.48 | 0.05 | 0.39 | 0.92 |
| **TabICLv2 +ESM (this work)** | **CB-P2RANK-apo** | 0.861 | **0.469** | **0.569** | 0.068 | 0.392 | 0.914 |
| TabICLv2, no pLM (this work) | CB-P2RANK-apo | 0.845 | 0.411 | 0.512 | 0.068 | 0.350 | 0.910 |
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

On CB-full the ESM variant exceeds pLM-NN on AUPRC (+0.081) and TPR (+0.079), is level
on AUC (0.862 vs 0.86) and MCC (0.374 vs 0.39), and is **worse** on FPR (0.071 vs 0.05)
and weighted F1 (0.912 vs 0.92). Only two of those six are wins, and § 2.1 confirms it:
the AUPRC and TPR gaps clear resampling, AUC and MCC contain pLM-NN's value, and FPR and
F1 run against us with the benchmark outside our interval.

The FPR overshoot is threshold transfer, and it is now cleanly attributable. The
threshold is swept on the CV folds to a **mean per-protein FPR of 0.05** — the statistic
CryptoBench publishes — and every fold lands at exactly 0.0500. It arrives at 0.071 on
the held-out test split purely through distribution shift across sequence-identity
folds. (Before § 9.3 the sweep targeted the *pooled* FPR, a quantity this document never
reports, so part of the old gap was a definitional mismatch. That component is gone; what
remains is transfer.)

CB-P2RANK-apo (the 184 single-chain test structures) is reconstructible from the shipped
dataset, so we report it too. With ESM we lead on AUPRC (+0.049) and TPR (+0.059), trail
on AUC (0.861 vs 0.88), MCC (0.392 vs 0.43), FPR (0.068 vs 0.04) and F1 (0.914 vs 0.93).
Without a language model nothing clears. This is the subset where our evaluation unit and
pLM-NN's are identical by construction, so it is the cleaner of the two rows — and on it
our advantage is narrower than on CB-full, not wider.

**F1 separates almost nothing here, and that is the finding.** Every pLM-based row —
ours and pLM-NN's, on both subsets — sits at 0.92–0.93; the column's only real
discrimination is against the structure-only tools (P2Rank 0.81, PocketMiner 0.78). At
5.28 % prevalence a weighted F1 is governed by the 95 % of residues that are trivially
negative, so it largely measures whether a method avoided flooding the negative class.
Two consequences for reporting: our 0.912 on CB-full is now **behind** pLM-NN's 0.92
rather than level with it, and on CB-P2RANK-apo pLM-NN is ahead too, 0.93 vs 0.914.
Quote AUPRC, not F1 — but do not quote F1 as a tie either.

**CB-PM is not reconstructible** from the shipped files — it is defined by which 60
structures PocketMiner failed on, which is a property of that tool's run, not of the
dataset. Getting that row requires asking the authors for the structure list.

### 2.1 Confidence intervals, and which claims survive them

Every number above is a point estimate on one test set. Because each metric is a **mean
over per-protein values**, a CI comes free: resample the test proteins with replacement
and re-average the stored per-protein table — no re-scoring, no GPU. 10,000 resamples,
percentile interval, `python scripts/bootstrap_test.py`.

**TabICLv2 + ESM (117 feat)**

| metric | CB-full (n=231) | 95 % CI | pLM-NN | CB-P2RANK-apo (n=184) | 95 % CI | pLM-NN |
|---|---|---|---|---|---|---|
| AUC | 0.862 | [0.840, 0.882] | 0.86 ✗ | 0.861 | [0.835, 0.885] | 0.88 ✗ |
| AUPRC | 0.441 | [0.405, 0.478] | 0.36 ✔ | 0.469 | [0.427, 0.510] | 0.42 ✔ |
| TPR | 0.559 | [0.514, 0.604] | 0.48 ✔ | 0.569 | [0.517, 0.619] | 0.51 ✔ |
| FPR | 0.071 | [0.065, 0.076] | 0.05 ✗† | 0.068 | [0.063, 0.074] | 0.04 ✗† |
| MCC | 0.374 | [0.342, 0.406] | 0.39 ✗ | 0.392 | [0.354, 0.427] | 0.43 ✗† |
| F1 (weighted) | 0.912 | [0.905, 0.918] | 0.92 ✗† | 0.914 | [0.906, 0.921] | 0.93 ✗† |

**TabICLv2, no pLM (53 feat)**

| metric | CB-full (n=231) | 95 % CI | pLM-NN | CB-P2RANK-apo (n=184) | 95 % CI | pLM-NN |
|---|---|---|---|---|---|---|
| AUC | 0.848 | [0.828, 0.867] | 0.86 ✗ | 0.845 | [0.820, 0.868] | 0.88 ✗† |
| AUPRC | 0.388 | [0.356, 0.420] | 0.36 ✗ | 0.411 | [0.374, 0.448] | 0.42 ✗ |
| TPR | 0.515 | [0.474, 0.556] | 0.48 ✗ | 0.512 | [0.463, 0.559] | 0.51 ✗ |
| FPR | 0.073 | [0.068, 0.079] | 0.05 ✗† | 0.068 | [0.062, 0.074] | 0.04 ✗† |
| MCC | 0.336 | [0.307, 0.365] | 0.39 ✗† | 0.350 | [0.315, 0.383] | 0.43 ✗† |
| F1 (weighted) | 0.906 | [0.899, 0.913] | 0.92 ✗† | 0.910 | [0.902, 0.917] | 0.93 ✗† |

✔ = pLM-NN's published value falls **outside** our interval, on the losing side — the
gap survives test-set resampling.  ✗ = it falls **inside**, so the difference is not
resolved by this test set.  † = outside the interval but on the side that favours
pLM-NN (lower FPR; higher F1 on CB-P2RANK-apo), i.e. a gap against us.

**What this changes.**

- **AUPRC and TPR are the whole result, and only with ESM.** pLM-NN sits below our
  interval on both metrics and both subsets — but *only* for the 117-feature variant.
  These are the two claims to lead with, and there are no others.
- **The AUC lead is gone entirely.** It survived resampling on CB-full in the previous
  (leaky) measurement. It does not now: 0.862 [0.840, 0.882] contains pLM-NN's 0.86.
  Report AUC as a tie.
- **The "no language model needed" claim is withdrawn.** This was the document's
  strongest-sounding claim. At 53 features AUPRC 0.388 [0.356, 0.420] contains 0.36 and
  TPR 0.515 [0.474, 0.556] contains 0.48 — ties, not wins — and MCC 0.336 now sits
  *below* pLM-NN's 0.39 with 0.39 outside our interval. Without a protein language
  model this work does not beat the benchmark on anything.
- **MCC has moved from "unresolved" to "a loss" in three of four cells.** Only CB-full
  at 117 features still contains pLM-NN's value.
- **Four gaps now run against us significantly**: FPR on both subsets, weighted F1 on
  both, MCC on three cells of four, and AUC on CB-P2RANK-apo at 53 features.

The honest one-line summary: *with ESM-2 we rank residues better than pLM-NN (AUPRC,
TPR) while operating at a looser threshold (FPR, F1, MCC); without ESM-2 we match it.*

**What the interval does NOT cover.** Two caveats, both real:

1. It is a CI on *our* number, not on the *difference*. CryptoBench publishes point
   estimates only — without their per-protein scores a paired interval on the gap is
   not computable, so containment is the strongest available check.
2. It captures test-set sampling only, not the 50k context draw. That second source is
   far smaller and is measured separately in § 4.6 — 12 TabICL seeds on this same table
   move AUC by sd 0.0006 and AUPRC by sd 0.0015, so the interval above is ~24× wider.
   It is a correction to these numbers rather than a competing source of doubt.


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

Every row below is on the **corrected table, the fixed sampler, and the leak-free
feature set**, so the comparison isolates model and feature set. Feature counts dropped
by one against earlier versions of this document (34→33, 48→47, 54→53, 70→69, 86→85,
118→117): `in_noncryptic_pocket` is no longer an input (§ 9.1).

The old-sampler rows are kept separately at the bottom for the record. They are now
**doubly** superseded — old sampler *and* holo-derived feature — so they are evidence
for § 4.1 only and must not be read against the table below.

| Model | feat | negatives | AUC | AUPRC | TPR | FPR | MCC |
|---|---|---|---|---|---|---|---|
| **TabICLv2 zero-shot + ESM-64** | 117 | 50k context | **0.862** | **0.441** | 0.559 | 0.071 | 0.374 |
| **TabICLv2 zero-shot + ESM-32** | 85 | 50k context | **0.858** | **0.434** | 0.557 | 0.072 | 0.365 |
| LogisticRegression + ESM-64 | 117 | 50k context | 0.856 | 0.417 | 0.425 | 0.044 | 0.339 |
| LogisticRegression + ESM-64 | 117 | full 305k | 0.856 | 0.416 | 0.412 | 0.046 | 0.343 |
| LogisticRegression + ESM-32 | 85 | 50k context | 0.853 | 0.410 | 0.412 | 0.044 | 0.327 |
| LogisticRegression + ESM-32 | 85 | full 305k | 0.852 | 0.410 | 0.397 | 0.044 | 0.333 |
| **TabICLv2 zero-shot + ESM-16** | 69 | 50k context | **0.851** | **0.411** | 0.526 | 0.070 | 0.350 |
| **TabICLv2 zero-shot** | 53 | 50k context | **0.848** | **0.388** | 0.515 | 0.073 | 0.336 |
| LogisticRegression + ESM-16 | 69 | 50k context | 0.842 | 0.388 | 0.389 | 0.043 | 0.312 |
| LogisticRegression + ESM-16 | 69 | full 305k | 0.840 | 0.387 | 0.360 | 0.044 | 0.305 |
| **TabICLv2 zero-shot** | 47 | 50k context | **0.839** | **0.352** | 0.490 | 0.076 | 0.311 |
| RandomForest + ESM-64 | 117 | 50k context | 0.838 | 0.380 | 0.676 | 0.164 | 0.315 |
| RandomForest + ESM-64 | 117 | full 305k | 0.837 | 0.381 | 0.564 | 0.104 | 0.337 |
| RandomForest + ESM-32 | 85 | 50k context | 0.837 | 0.381 | 0.669 | 0.156 | 0.320 |
| RandomForest + ESM-32 | 85 | full 305k | 0.836 | 0.382 | 0.558 | 0.101 | 0.337 |
| RandomForest + ESM-16 | 69 | 50k context | 0.833 | 0.375 | 0.655 | 0.149 | 0.317 |
| LogisticRegression | 53 | 50k context | 0.831 | 0.362 | 0.358 | 0.044 | 0.284 |
| RandomForest + ESM-16 | 69 | full 305k | 0.830 | 0.371 | 0.534 | 0.096 | 0.327 |
| LogisticRegression | 53 | full 305k | 0.829 | 0.358 | 0.335 | 0.046 | 0.277 |
| RandomForest | 53 | 50k context | 0.829 | 0.358 | 0.636 | 0.148 | 0.309 |
| **TabICLv2 zero-shot** | 33 | 50k context | **0.826** | **0.343** | 0.477 | 0.083 | 0.286 |
| RandomForest | 53 | full 305k | 0.824 | 0.346 | 0.508 | 0.097 | 0.305 |
| LogisticRegression | 47 | 50k context | 0.822 | 0.329 | 0.324 | 0.046 | 0.241 |
| RandomForest | 47 | 50k context | 0.821 | 0.324 | 0.612 | 0.150 | 0.280 |
| LogisticRegression | 47 | full 305k | 0.821 | 0.327 | 0.294 | 0.045 | 0.231 |
| RandomForest | 47 | full 305k | 0.814 | 0.311 | 0.491 | 0.101 | 0.281 |
| RandomForest | 33 | 50k context | 0.809 | 0.313 | 0.591 | 0.161 | 0.257 |
| RandomForest | 33 | full 305k | 0.805 | 0.313 | 0.494 | 0.106 | 0.275 |
| LogisticRegression | 33 | 50k context | 0.799 | 0.297 | 0.300 | 0.045 | 0.226 |
| LogisticRegression | 33 | full 305k | 0.798 | 0.298 | 0.288 | 0.045 | 0.228 |
| RSA alone (floor) | 1 | — | 0.499 | 0.089 | 0.877 | 0.832 | 0.035 |

Three readings hold up, and two are **stronger** than before § 9:

1. **TabICL leads at every feature count** — all six, without exception, including
   against models that see all 305k training rows. The within-condition spread is
   0.027–0.061 AUPRC. That margin roughly **doubled** when the holo-derived feature was
   removed (it was 0.003–0.030): the leak was flattering the sklearn controls more than
   it flattered TabICL, which had been masking the in-context model's advantage.
2. **The features still carry most of the result.** Holding model and sampler fixed,
   TabICL goes 33 → 47 → 53 → 69 → 85 → 117 features for AUPRC 0.343 → 0.352 → 0.388 →
   0.411 → 0.434 → 0.441. The feature effect (0.098) remains larger than the model
   effect (≤0.061), but the two are now much closer than the previous measurement
   suggested.
3. **The task is multivariate.** RSA alone is 0.499 — chance.

The ladder is now **monotonic in AUPRC at every step**, which it was not before: the
48-feature row used to be the one place a feature addition made TabICL slightly worse
(0.433 → 0.430). That inversion was an artifact of the leak and is gone. The ensemble
features remain the weakest addition (+0.009, § 4.3).

### Superseded: old-sampler runs

Kept because § 4.1 quantifies the sampler bug and these are its evidence. **Do not
compare them against the table above.** Two things differ, not one: the negative sampler,
and the fact that these runs still had the holo-derived feature. Their feature counts are
written as 34/48 because that is what those runs actually saw.

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

| fold | 53 feat AUC / AUPRC | 117 feat AUC / AUPRC |
|---|---|---|
| train-fold-0 | 0.802 / 0.334 | 0.815 / 0.376 |
| train-fold-1 | 0.837 / 0.404 | 0.860 / 0.442 |
| train-fold-2 | 0.820 / 0.386 | 0.834 / 0.411 |
| train-fold-3 | 0.814 / 0.371 | 0.829 / 0.405 |

Frozen thresholds: 0.5667 (53 feat) and 0.5706 (117 feat), each the median of the four
fold thresholds. ESM helps on every fold, by +0.026 to +0.042 AUPRC. Spread of ~0.06 AUC
across folds — no single fold carries the result.

Every fold lands at a mean per-protein FPR of **exactly 0.0500** by construction
(§ 9.3): the sweep now targets the statistic this document reports. The gap to the
0.071 seen on test is therefore train→test distribution shift alone.

---

## 4. What actually moved the number

Ranked by effect on AUPRC, which is the honest summary at 5% prevalence.

| change | ΔAUPRC | cost |
|---|---|---|
| Fixing PDB column alignment | +0.144 *(on ensemble features alone)* | one line |
| Fixing the negative sampler | **+0.113** | one line |
| Adding 64 ESM-2 principal components | +0.053 | 0.07 A10G-hours (~$0.08) |
| **Removing the holo-derived feature (§ 9.1)** | **−0.035** | one line — a correction |
| Adding 6 conservation features | +0.036 | ~2,300 API calls |
| Adding 14 ensemble pocket features | +0.009 | ~28,000 fpocket runs |
| Averaging over 5 contexts | +0.004 | 5× GPU inference |
| Fixing the insertion-code key (§ 7) | −0.004 *(noise; 0 test rows changed)* | three lines |

Four of the entries are bugs in code I wrote, and they are four of the five largest.
The deliberate capability additions are the smaller ones.

**The −0.035 entry is the important one and it points the other way.** Every other row
is a number going up. That one is a number given back: a feature that could never have
been computed on a novel apo structure, so 0.035 AUPRC of the previous headline was
unearned. It belongs in a table titled "what actually moved the number" precisely
because omitting it would repeat the mistake the feature itself was.

The insertion-code fix is here for completeness rather than effect: it corrected 32
mislabelled residues and restored a quarantined protein, all in the training split, and
the movement is smaller than the context-draw noise.

By measured compute ESM-2 is decisively the best value of the three additions — the
largest gain, +0.053, for about five minutes of A10G time (~$0.08). Conservation is
second at +0.036 for ~2,300 cached API calls. The ensemble pocket features are last:
+0.009 for ~28,000 fpocket runs and half an hour of fanned-out Modal CPU, which is the
worst ratio in the project by a wide margin.

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

### 4.3 Ensemble pocket features (+0.009 AUPRC)

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

### 4.3b Conservation features (+0.036 AUPRC)

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

Re-run on the corrected, leak-free table (§ 9): 60 T4 jobs, 10.7 GPU-h, ~$6.33. Every
draw keeps all 16,330 positives in full; only the 33,670 negatives that fill the 50k
context change. § 5.1 reuses the stored per-residue scores to put an error bar on the
per-stratum AUCs.

| metric | mean | sd | min – max | seed 0 (reported) |
|---|---|---|---|---|
| AUC | 0.8623 | 0.0006 | 0.8617 – 0.8635 | 0.8618 |
| AUPRC | 0.4405 | 0.0015 | 0.4374 – 0.4425 | 0.4413 |
| TPR | 0.5605 | 0.0049 | 0.5551 – 0.5735 | 0.5591 |
| FPR | 0.0707 | 0.0007 | 0.0697 – 0.0717 | 0.0707 |
| MCC | 0.3760 | 0.0035 | 0.3704 – 0.3846 | 0.3738 |
| F1 (positive class) | 0.3814 | 0.0033 | 0.3756 – 0.3893 | 0.3792 |
| F1 (weighted) | 0.9127 | 0.0005 | 0.9116 – 0.9133 | 0.9121 |
| Accuracy | 0.9026 | 0.0006 | 0.9014 – 0.9035 | 0.9023 |

Seed 0 is the draw behind every headline in this document, and it sits inside the band on
every metric — it is an ordinary draw, not a lucky one. The context is a far smaller
source of uncertainty than the test set: sd 0.0015 on AUPRC against the test-set
interval of 0.405 – 0.478 (§ 2.1), whose half-width is ~24× larger. Any AUPRC difference
in this document smaller than ~0.003 is inside the noise of a single draw.

---

## 5. Where the model is weak

Test set stratified by whether a residue lines an already-open (non-cryptic) pocket,
117-feature run.

**This section is only meaningful now.** `in_noncryptic_pocket` *defines* these strata,
and until § 9.1 it was also an input to the model — so the earlier version of this table
scored a model that had been told which stratum each residue was in. It responded exactly
as you would expect: it fired positive on **94.2 %** of the hard stratum, because the
feature announced a 44 % base rate there. That is now 66.3 %, and the model has to earn
it.

| stratum | residues | prevalence | AUC | AUPRC | TPR |
|---|---|---|---|---|---|
| bulk surface | 73,409 | 3.5% | 0.882 | 0.262 | 0.580 |
| lines a non-cryptic pocket | 3,344 | 43.9% | **0.723** | 0.646 | 0.663 |

0.723 against 0.882 remains the limiting factor: telling a cryptic site from a pocket
that is *already open* is the discrimination the task is actually about, and it is
where essentially all of the remaining headroom sits.

**Both strata scored slightly HIGHER after the leak was removed** (0.714 → 0.723 hard,
0.879 → 0.882 bulk). That is the tell: the holo-derived feature was never helping the
model *rank* residues within a stratum, it was shifting its prior between strata. The
headline AUPRC fell because the model lost cheap precision on the 36 % of test positives
that feature flagged — not because it lost discrimination.

**How each addition moved it:**

All rows on the corrected table, the fixed sampler and the leak-free feature set:

| run | feat | hard stratum | bulk surface | gap |
|---|---|---|---|---|
| Tier A | 33 | 0.679 | 0.825 | 0.146 |
| + ensemble pocket features | 47 | 0.692 | 0.842 | 0.149 |
| + conservation | 53 | 0.700 | 0.856 | 0.156 |
| + ESM-16 | 69 | 0.709 | 0.871 | 0.162 |
| + ESM-32 | 85 | 0.722 | 0.878 | 0.156 |
| **+ ESM-64** | **117** | **0.723** | **0.882** | **0.160** |

### 5.1 A conclusion that did not survive re-measurement

The previous version of this section read: *"ESM did not close the gap — it widened it.
The bulk surface improved by +0.027 while the hard stratum improved by +0.015, so the
spread went from 0.152 to 0.164. A protein language model makes the model better at the
part it was already good at."*

**The ESM-specific claim does not hold.** On the corrected, leak-free table ESM lifts the
hard stratum +0.023 (0.700 → 0.723) and the bulk surface +0.026 (0.856 → 0.882). It
widens the gap by **+0.003**, not +0.012 — and +0.003 is 1.7 sd of the noise on a gap
difference, so ESM's effect on the gap is not resolvable at all.

The error bar comes from recomputing per-stratum AUC for each of the 12 context seeds in
§ 4.6 (the scores are stored, so this costs nothing — `scripts/stratum_seed_noise.py`):

| stratum | mean | sd | min – max |
|---|---|---|---|
| bulk surface (73,409 residues) | 0.8825 | 0.0004 | 0.8818 – 0.8832 |
| lines a non-cryptic pocket (3,344) | 0.7219 | **0.0013** | 0.7194 – 0.7241 |
| gap | 0.1606 | **0.0014** | 0.1586 – 0.1632 |

The hard stratum is 4.4 % of the test set and its AUC is 3.3× noisier under a change of
context draw — sd 0.0013 against 0.0004. A difference of two gaps carries √2 × 0.0014 ≈
0.0020, which is the yardstick used above and below.

**Removing the leak cut this noise by a factor of three.** On the leaky table the same
three rows read 0.0006 / 0.0045 / 0.0043, and the hard stratum was *seven* times noisier
than the bulk. That is not a coincidence: `in_noncryptic_pocket` **defines** this
stratum, so handing it to the model made its scores here hostage to how many flagged
negatives a given draw happened to include. The old error bar was itself partly an
artifact of the leak it was being used to reason about — which is why § 9.1 is ordered
first among the corrections.

**What the sharper error bar does resolve.** Reading the ladder above against it, the
gap is not flat. From Tier A to the full set the gap widens **+0.013** (0.146 → 0.160)
— 6.7 sd, and the one movement here comfortably outside the noise. 33 → 117 features
moves the hard
stratum +0.044 and the bulk surface +0.057. So the *cumulative* direction of the original
claim survives: descriptors added to this pipeline help the easy stratum somewhat more
than the hard one. What does not survive is attributing that to ESM, or to any single
addition — no individual step clears 2 sd, and the ladder is not monotonic (+ESM-32 moves
the gap *down* 0.005).

**What is still true, and is the finding:** the gap sits at 0.146 – 0.162 across every
feature set tried, and nothing has closed it. Each addition lifts both strata together.
The descriptors are getting better at pocket-likeness in general rather than at
crypticity in particular.

One earlier reading in this section has to be withdrawn too. It claimed the ensemble
pocket features were "the weakest of the three additions here (+0.005 hard)", and used
that to argue that harmonic collective motion cannot produce the side-chain rotamer flips
and loop crossings that separate a cryptic site from an already-open one. On the
leak-free numbers the ensemble features contribute **+0.013** to the hard stratum, *more*
than conservation's +0.008, so the hard stratum does not single them out. They remain the
worst value per unit of compute by headline AUPRC (+0.009 for ~28,000 fpocket runs,
§ 4.3), which is a claim about cost, not about what the hard stratum shows. The
mechanistic argument about rotamer flips may still be right; these numbers do not
evidence it.

---

## 6. Dataset and integrity

| | |
|---|---|
| Rows | 381,704 residues |
| Proteins | 1,136 apo structures (0 quarantined) |
| Prevalence | 5.34% positive |
| Features | 117 (53 without the pLM), all computed from the **apo structure only** |
| Splits | CryptoBench's own sequence-identity folds, group = apo structure |
| Label match rate | median 1.000, min 0.800 |
| Annotated residues matched | 20,383 of 20,387 (99.98%) |
| Ensemble feature coverage | 100% of residues and proteins |
| Conservation coverage | 99.47% of residues, 100% of proteins |
| Tests | 124 passing |

Four guards enforced by tests:

1. **No holo-derived feature reaches X.** A cryptic site is *defined* by the apo→holo
   change, so any holo-derived quantity reconstructs the label and is unavailable at
   prediction time. The guard now has **two halves**, because a substring blacklist
   alone was not enough: it catches columns *named* after a holo quantity, and it
   separately rejects the columns named in `config.HOLO_DERIVED_METADATA_COLS`.
   `in_noncryptic_pocket` matched none of the forbidden substrings and reached the model
   for the whole first measurement era (§ 9.1). `tests/test_no_leakage.py` fails the
   build on either.
2. **No protein appears in two splits.** Residues within a protein are correlated and
   homologues leak across a random split; we use CryptoBench's shipped folds.
3. **Labels are the union over all holo partners.** 5,493 pairings for 1,136 apo
   structures (~5 each); labelling from one partner would mark true cryptic residues as
   negatives.
4. **The residue key carries its insertion code.** `tests/test_icode_keys.py` pins the
   key format, the build/record spellings agreeing, and all three feature merges joining
   on the same four-part key. See § 7.
5. **Modified residues resolve to their parent, and stay in the structure.**
   `tests/test_modified_residues.py` pins MSE→MET, SEP→SER, LLP→LYS and the rest through
   gemmi's component table, and pins what stays excluded (nucleic acids, `UNK`, waters,
   ligands). Dropping them used to remove their atoms from the surface entirely, not
   just from the prediction set. See § 9.4.

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
12-seed range measured on that same leaky table, 0.4752 – 0.4846. (§ 4.6 now reports the
12 seeds re-run on the corrected, leak-free table, so its band is the post-§ 9 one and is
not the comparison for this row.) Every conclusion in § 2.1 survives unchanged;
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

- ~~Fine-tuned TabICL~~: measured on 2026-09-16 and reported separately in
  `fine_tuned_tabicl/README.md`, so the zero-shot record here is unchanged.
  - Result: AUPRC 0.444 against 0.441 for a paired zero-shot control, a gain of +0.002
    with a 95% CI of [−0.002, +0.007]. AUC is 0.863.
  - The hard stratum does not move.
- ~~A threshold that actually lands at 5 % FPR on test.~~ Measured on 2026-09-21 and
  reported in § 10. A per-protein decision rule lands at **FPR 0.0502** without any
  re-tuning on test; the global cutoff's 0.071 was a property of the *rule*, not an
  irreducible transfer loss.
- ~~TabICL's unset constructor arguments.~~ Swept on 2026-09-21, § 11. All null.
- Fragment docking (CryptoSite's #3) — needs PatchDock + RankScore.
- The CryptoSite 14-pair external test set — `scripts/cryptosite_testset.py` builds it
  (none of its apo structures appears in CryptoBench), but scoring has not been run.

---

## 9. Four corrections, 2026-09-10

Four defects found by review, all in code written for this project, all fixed and every
number in this document re-measured afterwards. They are ordered by how much they matter.

### 9.1 A holo-derived feature was an input to the model

**What it was.** `in_noncryptic_pocket` marks residues lining a pocket that is *already
open* in the apo form. CryptoBench ships that annotation, and builds it in
`src/E-add-noncryptic-pockets/main.py`:

```python
additional_noncryptic = additional_noncryptic[
    additional_noncryptic['apo_pocket_rms'] < filter_utils.MAX_POCKET_RMSD]
```

`apo_pocket_rms` is how far the pocket moved **between apo and holo**. That is the same
measurement that makes the label, with the inequality flipped: the label says "this
pocket changes shape when a ligand binds", this says "this one does not". Both require
the holo partner. At prediction time on a novel apo structure there is no holo partner,
so the column cannot be computed — and a model that needs it cannot be run on the task
this project is about.

**How big a hint.** Measured on the built table:

| | rows | positive rate |
|---|---|---|
| flag = 1 | 3,344 (4.4 % of test) | **43.9 %** |
| flag = 0 | 73,409 | 3.5 % |

A 12× enrichment, and **36 % of every cryptic residue in the test set** sits inside that
4.4 %. Used alone as the entire score it reaches AUC 0.652 and average precision 0.251
against a 5.3 % base rate — better than any single real feature in the table (RSA alone:
AUC 0.499, AP 0.089).

**Why the guard missed it.** `build_table.validate()` blacklisted *substrings* —
`holo`, `rmsd`, `delta_drug` and so on. `in_noncryptic_pocket` contains none of them.
The tripwire was looking for the word, not the provenance. It now also rejects columns
by name via `config.HOLO_DERIVED_METADATA_COLS`, and two tests pin both halves.

**The fix.** One line: the column moved into `NON_FEATURE_COLS`. It is still in the
parquet, because `evaluate.stratified` needs it — grouping an evaluation by a
ground-truth-adjacent property is legitimate; feeding it to the model is not.

**What it cost, and what it bought back.**

| | with leak | without |
|---|---|---|
| AUPRC (117 feat) | 0.476 | **0.441** |
| AUC | 0.882 | **0.862** |
| TabICL margin over sklearn | 0.003–0.030 | **0.027–0.061** |
| hard-stratum AUC | 0.714 | **0.723** |
| hard-stratum TPR | 0.942 | **0.663** |

The headline fell by 0.035 AUPRC. Three things improved. The margin over the sklearn
controls roughly doubled, because the leak flattered them more than it flattered TabICL.
Both per-stratum AUCs rose. And the operating point stopped being absurd: the model had
been firing positive on 94 % of the hard stratum because the feature announced a 44 %
base rate there.

A sklearn ablation run before the re-measurement predicted −0.066 AUPRC; TabICL actually
lost 0.035, i.e. it was leaning on the leak about half as hard as the linear and tree
models were. That difference is the whole reason the margin widened.

### 9.2 `_add_neighbourhood` was skipped on the fallback path

`compute_ensemble_features` falls back to the static crystal score when a protein is too
small or too fragmented for an ANM (fewer than 20 Cα). It did so by `return`-ing early,
which skipped the 8 Å neighbourhood averaging — so those proteins would come out missing
7 of their 14 ensemble columns. `validate` only rejects columns that are *entirely* NaN,
so a partially-NaN column would have passed silently.

**Dormant, not active**: 0 of 1,136 CryptoBench proteins take that path (the smallest has
52 residues), and there are 0 NaN `_nb8` values in the built table, so this never touched
a published number. It is a landmine for any smaller input — the CryptoSite external set,
a peptide, a fragmented chain. Fixed in three lines, with two tests that fail without it.

### 9.3 The threshold targeted a statistic this document never reports

`choose_threshold` used `sklearn.roc_curve`, which pools every residue of every protein
into one bucket — so a 1,138-residue protein counted twenty times more than a 57-residue
one. But every number here is a **mean over per-protein values**, matching CryptoBench.
The sweep was tuning against one quantity and the document reported another.

CryptoBench's convention is unambiguous. `src/H-prediction-evaluation/pocketminer_stats.py`
writes its headline row as `sum(avg_fpr) / counter` over per-protein `Protein.get_FPR()`
values, and emits a pooled `overall` row separately. Worth noting: CryptoBench never
sweeps to a target FPR at all — it fixes a constant and reports whatever FPR follows, and
its own sources disagree on the constant (the paper says 0.95 chosen on a validation set;
`src/I-final/benchmark-example.py` hardcodes `DECISION_THRESHOLD = 0.75`). Sweeping is
this project's choice; the defect was sweeping the wrong statistic.

`choose_threshold` now targets the mean per-protein FPR, computed exactly in one pass by
weighting each negative residue by `1 / (n_proteins × n_negatives_in_its_protein)`. Every
CV fold now lands at 0.0500. Measured effect on the old scores: pooled FPR 0.0658 vs mean
per-protein 0.0621, and hitting a true 0.05 moved the threshold 0.568 → 0.636 and TPR
0.591 → 0.544. AUC and AUPRC are threshold-free and were never affected.

### 9.4 Modified residues were deleted from the structure

`residue_records` kept only the literal 20 standard amino-acid names. Anything else in
the polymer — overwhelmingly **MSE**, selenomethionine, which crystallographers
substitute for methionine to phase a structure — was dropped *entirely*, not merely
excluded from the prediction set. Its atoms left the structure, so the SASA calculation,
the neighbour counts, the depth measure and fpocket itself all saw a protein with
methionine-shaped holes punched in it.

Two costs, both measured:

1. **Lost ground truth.** 47 of 20,387 annotated cryptic residues (0.23 %) never matched
   a record. `4yt8`'s match rate of 0.857 was exactly three MSE residues — `A_243`,
   `A_283`, `B_214` — all three annotated cryptic by CryptoBench.
2. **Distorted geometry for the residues that remain.** Over 200 sampled structures, 8 %
   contain MSE and 14.5 % contain at least one dropped polymer residue.

**The fix.** Modified residues are kept under their **parent** standard name, resolved
from gemmi's chemical-component table rather than a hand-maintained list: a modified
amino acid carries its parent's one-letter code in lower case (MSE→`m`, LLP→`k`,
PCA→`e`), so the mapping falls out for free and cannot silently miss a rare code.
Selenomethionine additionally has its `SE` presented as a methionine `SD`/sulfur —
selenium is isosteric with sulfur by design, which is why it is used — so fpocket's
druggability calibration stays on ground it was calibrated for. That aliasing
deliberately does **not** generalise: `LLP` is a lysine carrying a whole pyridoxal
phosphate, so every other modified residue keeps its real atoms, because that bulk
genuinely occupies space in the apo structure.

Result: **+515 residues, +43 recovered annotations** (20,340 → 20,383 of 20,387), 31
structures improved, none regressed.

Still excluded, deliberately: **nucleic acids** (a modelling choice — the task is defined
on the apo protein, so a protein crystallised with its DNA has a computed surface that is
not the one in the complex), `UNK` and anything gemmi gives no amino-acid parent. The 4
annotations still unmatched are `2dfpA/A_200` (`MIS`), `4fl8A-B-C-D/C_263` and `/D_363`
(`IL0`), and `7alfA/A_1` (`SNM`) — components gemmi does not class as amino acids at all,
so there is no honest `MAX_ASA` or hydropathy to give them.

### 9.5 Two infrastructure fixes found while re-running

Not scientific defects, but both could have corrupted results silently:

- **`esm_modal.py` mixed runs.** The reducer globbed `/vol/emb_*.npz` — every shard the
  Volume had ever held — and shards are named by batch index. A pilot over 8 structures
  wrote `emb_0000` and then read the 28 surviving shards of the previous full run,
  producing a 372,521-row table belonging to no single run. It matched neither era's row
  count, which is the only reason it was caught. Shards now live under a per-run prefix,
  and three assertions (shard count, row count, duplicate keys) make the failure loud.
- **`tabicl_fanout.py` and `context_bootstrap.py` lost GPU work on a client drop.** Both
  aggregate in the local entrypoint, so a gRPC disconnect discarded every GPU-second
  already spent — which happened twice on this host. Each job now persists its scores to
  a Volume, is skipped for free if re-run with the same `--run-id`, and `--collect`
  replays the aggregation without touching a GPU.

  That fix is necessary but **not sufficient**, which the § 4.6 re-run then demonstrated:
  a job only reaches the Volume when it *finishes*, so a drop still cancels everything
  in flight. `modal run --detach` does not save them either — it keeps only the last
  triggered function alive, not a 60-way fan-out — and a third drop duly cancelled 42
  running jobs. What works is removing the need for a long-lived stream at all: deploy
  the app once (`modal deploy deploy/context_bootstrap.py`) and `spawn()` each job, so
  the client makes one short submit RPC per job and exits while the work runs
  server-side. All 60 jobs were submitted first try and ran to completion unattended.
  The `.map()` entrypoint is kept for interactive use; the deploy + spawn path is the
  one to use on a connection that cannot hold for an hour.

---

## 10. The decision rule, 2026-09-21

§ 8 listed "a threshold that actually lands at 5 % FPR on test" as not measured. It is
now measured, and the 0.071 turns out to have been a property of the decision *rule*
rather than an irreducible train→test loss.

**The diagnosis.** Every metric in this document is computed per protein and then
averaged, but the rule was a single global cutoff shared by all 231 test proteins. Any
between-protein shift in the score distribution therefore moves the realised FPR. Sweeping
harder cannot fix that; the statistic being controlled and the statistic being reported
are different objects.

**The change.** Two per-protein rules, both expressed as a monotone per-protein
*transform* of the score, after which the existing machinery applies unchanged —
`choose_threshold` still selects on the CV folds, `summarise` still scores:

- **rank** — `1 − (within-protein descending rank)/n`. A cutoff `t` on this is exactly
  "flag the top `1 − t` of every protein".
- **zscore** — `(s − mean_i)/sd_i` within each protein.

No new selection code, no grid approximation: selection stays the same exact one-pass
algorithm, and the global-threshold path reproduces the published 0.5706 to the digit,
which is the check that this route matches the incumbent protocol.

**What cannot move.** Per-protein AUC and AUPRC are rank statistics computed inside one
protein, so they are invariant under any strictly monotone per-protein transform. The
headline is arithmetically untouched — and it is, byte for byte, 0.8618362991984249 and
0.44130501771474306 in every row below. `tests/test_operating_point.py` asserts this per
protein rather than trusting the argument.

### 10.1 Selection, on the CV folds only

Per-fold threshold at the target mean per-protein FPR, frozen as the median of 4 —
identical discipline to the incumbent.

| rule | per-fold thresholds | frozen | operating point |
|---|---|---|---|
| global_threshold | 0.5896, 0.5433, 0.5825, 0.5586 | 0.5706 | `score ≥ 0.5706` |
| rank | 0.9264, 0.9193, 0.9215, 0.9237 | 0.9226 | top 7.74 % per protein |
| zscore | 2.0450, 1.9016, 1.9631, 2.0162 | 1.9897 | `z ≥ 1.9897` within protein |

Nested leave-one-fold-out (select on 3 folds pooled, apply to the 4th) is the transfer
estimate, and the only thing that chose which per-protein rule to report:

| rule | FPR | TPR | MCC | F1 (weighted) |
|---|---|---|---|---|
| global_threshold | 0.0501 | 0.4386 | 0.3353 | 0.9194 |
| rank | 0.0501 | 0.4374 | 0.3422 | 0.9196 |
| **zscore** | 0.0500 | 0.4457 | **0.3456** | 0.9205 |

CV fold prevalence is 0.0516–0.0550 against test's 0.0528, so the folds are not an
unusually flattering match.

### 10.2 Test, scored once

| rule | | AUC | AUPRC | TPR | FPR | MCC | F1 | F1 (weighted) |
|---|---|---|---|---|---|---|---|---|
| global_threshold | headline | 0.862 | 0.441 | 0.559 | 0.071 | 0.374 | 0.379 | 0.912 |
| rank | reference | 0.862 | 0.441 | 0.479 | 0.048 | 0.377 | 0.396 | 0.922 |
| **zscore** | **secondary** | 0.862 | 0.441 | 0.499 | **0.050** | 0.380 | 0.395 | 0.921 |
| pLM-NN | published | 0.86 | 0.36 | 0.48 | 0.05 | 0.39 | — | 0.92 |

Full precision for the selected rule (`z ≥ 1.9896689080896004`): AUC 0.8618362991984249,
AUPRC 0.44130501771474306, TPR 0.4994631502359641, FPR 0.05021823247775074,
MCC 0.3803835194661635, F1 0.3945394161246145, F1-weighted 0.9211807704541123,
ACC 0.916545166240146.

**The target transfers.** 0.0502 against a 0.05 target, where the global cutoff
overshoots to 0.0707.

**Knowing the oracle bought nothing.** During scoping the rank rule's `q` was swept
directly on test, which is a contamination the pre-registration exists to contain.
Sweeping `z` on test gives 1.9923 against the CV-frozen 1.9897, for MCC 0.3805 vs 0.3804.
The oracle rows are kept in `operating_point/result.json`, labelled as not results.

### 10.3 Is it real? Paired bootstrap, 10,000 resamples

Both arms score the same 231 proteins from the same scores, so the pairing cancels the
dominant "which proteins are in the test set" variance. zscore minus global_threshold:

| metric | Δ | 95 % CI | |
|---|---|---|---|
| FPR | **−0.0205** | [−0.0251, −0.0159] | significant |
| F1 (positive) | **+0.0152** | [+0.0028, +0.0272] | significant |
| F1 (weighted) | **+0.0090** | [+0.0061, +0.0119] | significant |
| MCC | +0.0064 | [−0.0048, +0.0173] | **not** significant |
| TPR | **−0.0598** | [−0.0786, −0.0414] | significant, against us |
| AUC / AUPRC | +0.0000 | [+0.0000, +0.0000] | identical by construction |

**This is a trade, not a free win.** The rule buys a real FPR reduction and real F1 gains
and pays for them with real TPR. What makes it worth reporting is that the global rule was
never *at* its intended operating point on test, so its higher TPR was bought with an FPR
overshoot nobody chose. **MCC improves but not significantly, so it remains a loss against
pLM-NN's 0.39 and is not claimed otherwise.**

Against pLM-NN the secondary rule wins AUC, AUPRC, TPR and weighted F1, ties FPR and loses
MCC — four, one and one, against the headline rule's three and three.

**global_threshold stays the headline** so a reader can compare against the published
table without taking the deviation on trust. The per-protein rule is named in advance in
`operating_point/frozen.json`, committed before test was scored, and reported beside it —
not substituted for it.

Zero GPU: it reuses the stored per-residue scores from the run that produced the headline.
Detail in `operating_point/README.md`.

---

## 11. TabICL's unset constructor arguments, 2026-09-21 — all null

`model.py` passed 6 of `TabICLClassifier`'s 22 arguments; the other 16 sat at library
defaults never checked against 117 features at 5.28 % prevalence. Nine were swept.
**None helps.**

Screened on **CV fold 0 only** — ranking nine configs against test and keeping the best is
nine looks at the test set. The context seed is fixed at 0, so every comparison is paired
and the § 4.6 context noise cancels rather than swamping the effect. The finalist was to
be confirmed on folds 1–3 and scored on test once.

**Determinism gate.** The control job's 75,800 fold-0 scores are elementwise identical
(max abs diff 0.000e+00) to the run that produced the headline, so surfacing the knobs
changed nothing and every delta is attributable to the knob alone.

| config | AUPRC | ΔAUPRC | AUC | sec |
|---|---|---|---|---|
| est32_norm5 | 0.3782 | **+0.0023** | 0.8166 | 2642 |
| kv_repr | 0.3759 | +0.0000 | 0.8148 | **416** |
| control | 0.3759 | — | 0.8148 | 663 |
| outlier8 | 0.3759 | −0.0000 | 0.8149 | 693 |
| batch8 | 0.3758 | −0.0001 | 0.8148 | 662 |
| norm4 | 0.3755 | −0.0004 | 0.8153 | 655 |
| avg_probs | 0.3754 | −0.0005 | 0.8148 | 653 |
| est32 | 0.3751 | −0.0008 | 0.8158 | 2545 |
| norm5 | 0.3747 | −0.0012 | 0.8147 | 677 |
| shuffle_random | 0.3744 | −0.0015 | 0.8147 | 668 |

**The pre-registered bar was ΔAUPRC > +0.003 — the § 4.6 single-draw noise band, set
before the run. Nothing met it, so nothing was promoted and the test split was never
scored.** 10 jobs, 2.85 T4-hours, $1.68.

What the nulls rule out:

- **Ensemble width.** `n_estimators=32` gives −0.0008 for 4× the compute. The paper's
  "8 estimators unlocks most of the benefits of ensembling" holds here, and the residue is
  not small but absent.
- **Preprocessor diversity.** Widening `norm_methods` from the default `['none','power']`
  costs *nothing* (the member list is truncated to `n_estimators` either way) and still
  loses: −0.0004 at four methods, −0.0012 at five. The default pair is not arbitrary.
- **The heavy-tail hypothesis.** `outlier_threshold=8.0` was the most promising guess:
  TabICL clips every feature at |z| > 4, and 64 of the 117 features are ESM-2 principal
  components with heavy tails. It moves AUPRC by −0.0000. Whatever is out in those tails,
  the model was not using it.
- **Feature-shuffle scheme** (−0.0015) and **probability- vs logit-space averaging**
  (−0.0005).

**One keeper.** `kv_cache="repr"` is **37 % faster** (416 s vs 663 s) at unchanged
metrics, by caching the context representation at `fit()` instead of re-forwarding all
50,000 context rows for each of the ~6 protein chunks. It is **not** enabled by default:
it is not bit-reproducible (max score drift 3.793e-03, from float accumulation order), so
it must not be used to re-run anything that has to match the published record.

**On est32_norm5.** It is the paper's own TALENT configuration, the only config pointing
the right way, and it has the best AUC in the table. But +0.0023 is inside the noise band,
on one fold, at 4× compute — and its 32-estimator sibling without the extra preprocessors
came out at −0.0008, which is not how a real effect behaves. Under a bar set in advance it
does not promote, and reporting it as a win after setting that bar is the selection the
bar exists to prevent. If revisited, the test is folds 1–3 plus seeds at roughly $4, and
the prediction to beat is that it regresses to zero.

Detail in `knob_sweep/README.md`; grid in `deploy/knob_grid.py`.
