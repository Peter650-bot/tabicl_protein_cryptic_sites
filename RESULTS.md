# Results

Per-residue cryptic binding site prediction on CryptoBench, zero-shot TabICLv2.

All numbers below are **measured**, on CryptoBench's own test split (231 apo structures,
76,636 residues, 5.28% positive), with metrics computed **per protein then averaged** —
the same procedure as CryptoBench's `Protein.py`, so they are comparable to the
published benchmark. The decision threshold is always selected on training folds and
frozen before the test set is touched.

Reproduce with `python scripts/compare_variants.py`.

---

## 1. Headline

**TabICLv2, zero-shot, 118 apo-only features (54 hand-built + 64 ESM-2 PCs):**

| metric | 118 feat (+ESM) | 54 feat (no pLM) |
|---|---|---|
| AUC | **0.882** | 0.868 |
| AUPRC | **0.480** | 0.453 |
| TPR | 0.595 | 0.567 |
| FPR | 0.062 | 0.063 |
| MCC | 0.414 | 0.391 |
| F1 (weighted — the published convention) | 0.922 | 0.919 |
| F1 (positive class) | 0.418 | 0.399 |

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
| **TabICLv2 +ESM (this work)** | **CB-full** | **0.882** | **0.480** | **0.595** | 0.062 | 0.414 | 0.922 |
| TabICLv2, no pLM (this work) | CB-full | 0.868 | 0.453 | 0.567 | 0.063 | 0.391 | 0.919 |
| pLM-NN (benchmark) | CB-full | 0.86 | 0.36 | 0.48 | 0.05 | 0.39 | 0.92 |
| **TabICLv2 +ESM (this work)** | **CB-P2RANK-apo** | **0.884** | **0.504** | **0.606** | 0.061 | 0.429 | 0.923 |
| TabICLv2, no pLM (this work) | CB-P2RANK-apo | 0.867 | 0.473 | 0.570 | 0.061 | 0.404 | 0.919 |
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

On CB-full the ESM variant exceeds pLM-NN on AUC (+0.022), AUPRC (+0.120), TPR (+0.115)
and MCC (+0.024), at a slightly looser FPR (0.062 vs 0.05). The FPR gap is threshold
transfer — the threshold was frozen from CV targeting 0.05 and landed at 0.062 on test.
**Read these margins against § 2.1**: the AUPRC and TPR gaps survive test-set
resampling, the MCC gap does not, and the FPR gap runs against us significantly.

CB-P2RANK-apo (the 184 single-chain test structures) is reconstructible from the shipped
dataset, so we report it too. Without a language model we trailed pLM-NN on AUC (−0.013)
and MCC (−0.026); with ESM we lead on AUC (+0.004), AUPRC (+0.084) and TPR (+0.096) and
are level on MCC (0.429 vs 0.43). Note this is the subset where our evaluation unit and
pLM-NN's are identical by construction, so it is the cleanest of the two rows. **The
+0.004 AUC margin here does not survive resampling (§ 2.1)** — only the AUPRC and TPR
margins do.

**F1 separates almost nothing here, and that is the finding.** Every pLM-based row —
ours and pLM-NN's, on both subsets — sits at 0.92–0.93; the column's only real
discrimination is against the structure-only tools (P2Rank 0.81, PocketMiner 0.78). At
5.28 % prevalence a weighted F1 is governed by the 95 % of residues that are trivially
negative, so it largely measures whether a method avoided flooding the negative class.
Two consequences for reporting: our 0.922 on CB-full is **not** a win over pLM-NN's 0.92
(their value is published to 2 dp, at which precision the two are equal), and on
CB-P2RANK-apo pLM-NN is genuinely ahead, 0.93 vs 0.923. Quote AUPRC, not F1.

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
| AUC | 0.882 | [0.862, 0.900] | 0.86 ✔ | 0.884 | [0.861, 0.906] | 0.88 ✗ |
| AUPRC | 0.480 | [0.443, 0.518] | 0.36 ✔ | 0.504 | [0.461, 0.546] | 0.42 ✔ |
| TPR | 0.595 | [0.550, 0.639] | 0.48 ✔ | 0.606 | [0.556, 0.654] | 0.51 ✔ |
| FPR | 0.062 | [0.056, 0.067] | 0.05 ✗† | 0.061 | [0.055, 0.067] | 0.04 ✗† |
| MCC | 0.414 | [0.381, 0.447] | 0.39 ✗ | 0.429 | [0.391, 0.465] | 0.43 ✗ |
| F1 (weighted) | 0.922 | [0.916, 0.928] | 0.92 ✗ | 0.923 | [0.915, 0.929] | 0.93 ✗† |

**TabICLv2, no pLM (54 feat)**

| metric | CB-full (n=231) | 95 % CI | pLM-NN | CB-P2RANK-apo (n=184) | 95 % CI | pLM-NN |
|---|---|---|---|---|---|---|
| AUC | 0.868 | [0.848, 0.886] | 0.86 ✗ | 0.867 | [0.844, 0.890] | 0.88 ✗ |
| AUPRC | 0.453 | [0.417, 0.490] | 0.36 ✔ | 0.473 | [0.432, 0.514] | 0.42 ✔ |
| TPR | 0.567 | [0.522, 0.610] | 0.48 ✔ | 0.570 | [0.519, 0.620] | 0.51 ✔ |
| FPR | 0.063 | [0.058, 0.069] | 0.05 ✗† | 0.061 | [0.055, 0.068] | 0.04 ✗† |
| MCC | 0.391 | [0.358, 0.424] | 0.39 ✗ | 0.404 | [0.366, 0.442] | 0.43 ✗ |
| F1 (weighted) | 0.919 | [0.911, 0.925] | 0.92 ✗ | 0.919 | [0.911, 0.927] | 0.93 ✗† |

✔ = pLM-NN's published value falls **outside** our interval, on the losing side — the
gap survives test-set resampling.  ✗ = it falls **inside**, so the difference is not
resolved by this test set.  † = outside the interval but on the side that favours
pLM-NN (lower FPR; higher F1 on CB-P2RANK-apo), i.e. a gap against us.

**What this changes.**

- **AUPRC and TPR are the real result.** pLM-NN sits below our interval on both metrics
  and both subsets. These are the claims to lead with.
- **The AUC lead does not survive.** On CB-P2RANK-apo the +0.004 margin is inside a
  ±0.022 interval; on CB-full only the 118-feature variant clears it, and the
  54-feature variant does not. Do not report "+0.004 AUC" as a lead.
- **MCC is not resolved anywhere** — pLM-NN's value is inside every interval. The
  earlier reading "level on MCC" is right; "+0.024 on CB-full" is not.
- **Two gaps run against us and are significant**: our FPR is genuinely higher (the
  threshold-transfer effect, 0.062 vs 0.05), and on CB-P2RANK-apo pLM-NN's weighted F1
  of 0.93 sits above our interval.

**What the interval does NOT cover.** Two caveats, both real:

1. It is a CI on *our* number, not on the *difference*. CryptoBench publishes point
   estimates only — without their per-protein scores a paired interval on the gap is
   not computable, so containment is the strongest available check.
2. It captures test-set sampling only, not the 50k context draw. That second source is
   far smaller: over 20 context seeds (`python scripts/context_seed_variance.py`, 54
   features, LogisticRegression as a CPU-cheap stand-in) the draw moves AUC by sd 0.0004
   and AUPRC by sd 0.0012 — roughly 30× tighter than the intervals above, so it is a
   correction rather than a competing source. It has **not** been measured for TabICL
   itself, where the context *is* the model; the only indirect evidence is § 4.2. Note
   also that seed 0, the one every headline run uses, was the highest-AUPRC draw of
   those 20.


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
**0.919** for the 54-feature variant and **0.922** with ESM, i.e. indistinguishable from
their 0.92, and our positive-class `F1` is **0.399** / **0.418**, which is the number
that actually carries information. Only `F1_weighted` may be placed next to the
published column.

The threshold is not a benchmark constant: pLM-NN used 0.95, PocketMiner 0.7 (its
authors' recommendation), and P2Rank emits binary calls with no threshold at all. Each
method sets its own, so selecting ours on training folds is the same procedure, not a
deviation. Applying their 0.95 to TabICL's scores would measure calibration rather than
ranking — only 0.98% of test residues reach 0.95, collapsing TPR to 0.104.

---

## 3. All model variants

| Model | feat | negatives | AUC | AUPRC | TPR | FPR | MCC |
|---|---|---|---|---|---|---|---|
| **TabICLv2 zero-shot + ESM-64** | 118 | fixed sampler, 50k | **0.882** | **0.480** | 0.595 | 0.062 | 0.414 |
| TabICLv2 zero-shot + ESM-32 | 86 | fixed sampler, 50k | 0.879 | 0.480 | 0.592 | 0.063 | 0.405 |
| LogisticRegression + ESM-64 | 118 | full 304k | 0.875 | 0.471 | — | — | — |
| TabICLv2 zero-shot + ESM-16 | 70 | fixed sampler, 50k | 0.874 | 0.469 | 0.574 | 0.063 | 0.397 |
| LogisticRegression + ESM-32 | 86 | full 304k | 0.872 | 0.468 | — | — | — |
| TabICLv2 zero-shot | 54 | fixed sampler, 50k | 0.868 | 0.453 | 0.567 | 0.063 | 0.391 |
| LogisticRegression + ESM-16 | 70 | full 304k | 0.863 | 0.452 | — | — | — |
| RandomForest + ESM-64 | 118 | full 304k | 0.862 | 0.449 | — | — | — |
| LogisticRegression | 54 | fixed sampler, 50k | 0.855 | 0.441 | 0.457 | 0.041 | — |
| RandomForest | 54 | fixed sampler, 50k | 0.856 | 0.435 | 0.685 | 0.141 | — |
| LogisticRegression | 54 | full 304k | 0.853 | 0.437 | 0.474 | 0.051 | — |
| RandomForest | 54 | full 304k | 0.853 | 0.432 | 0.245 | 0.013 | — |
| TabICLv2 zero-shot | 48 | fixed sampler, 50k | 0.862 | 0.436 | 0.549 | 0.066 | 0.371 |
| LogisticRegression | 48 | fixed sampler, 50k | 0.848 | 0.422 | — | — | — |
| LogisticRegression | 48 | full 304k | 0.847 | 0.417 | 0.451 | 0.050 | 0.341 |
| RandomForest | 34 | full 304k | 0.840 | 0.418 | 0.258 | 0.015 | 0.258 |
| RandomForest | 48 | full 304k | 0.839 | 0.396 | 0.229 | 0.013 | 0.244 |
| LogisticRegression | 34 | full 304k | 0.834 | 0.413 | 0.455 | 0.050 | 0.346 |
| TabICLv2 zero-shot ×5 contexts | 48 | old sampler, 50k | 0.840 | 0.327 | 0.472 | 0.081 | 0.291 |
| TabICLv2 zero-shot | 48 | old sampler, 50k | 0.839 | 0.323 | 0.443 | 0.074 | 0.286 |
| TabICLv2 zero-shot | 34 | old sampler, 50k | 0.830 | 0.324 | 0.441 | 0.077 | 0.277 |
| LogisticRegression | 48 | old sampler, 50k | 0.827 | 0.333 | 0.198 | 0.022 | 0.194 |
| RandomForest | 48 | old sampler, 50k | 0.822 | 0.299 | 0.624 | 0.156 | 0.283 |
| RandomForest | 34 | old sampler, 50k | 0.820 | 0.312 | 0.582 | 0.140 | 0.275 |
| LogisticRegression | 34 | old sampler, 50k | 0.813 | 0.324 | 0.207 | 0.023 | 0.205 |
| RSA alone (floor) | 1 | — | 0.499 | 0.089 | 0.876 | 0.832 | 0.034 |

The RSA floor at 0.499 confirms no single descriptor solves this — the signal is
genuinely multivariate.

### Cross-validation (fixed sampler)

Inner CV over CryptoBench's train folds, used only to freeze the threshold:

| fold | 48 feat AUC / AUPRC | 54 feat AUC / AUPRC |
|---|---|---|
| train-fold-0 | 0.830 / 0.391 | 0.835 / 0.402 |
| train-fold-1 | 0.859 / 0.463 | 0.862 / 0.474 |
| train-fold-2 | 0.838 / 0.421 | 0.845 / 0.440 |
| train-fold-3 | 0.834 / 0.424 | 0.838 / 0.433 |

Conservation helps on every fold, by +0.009 to +0.019 AUPRC. Spread of ~0.03 AUC
across folds — no single fold carries the result.

---

## 4. What actually moved the number

Ranked by effect on AUPRC, which is the honest summary at 5% prevalence.

| change | ΔAUPRC | cost |
|---|---|---|
| Fixing the negative sampler | **+0.113** | one line |
| Fixing PDB column alignment | +0.144 *(on ensemble features alone)* | one line |
| Adding 6 conservation features | +0.017 | ~2,300 API calls |
| Adding 14 ensemble pocket features | +0.014 | ~28,000 fpocket runs |
| Averaging over 5 contexts | +0.004 | 5× GPU inference |

The two largest effects were both bugs in code I wrote. The three deliberate capability
additions were the three smallest. Conservation was the best value per unit of effort:
comparable gain to the ensemble features for roughly 1% of the compute.

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

Test set stratified by whether a residue lines an already-open (non-cryptic) pocket:

| stratum | residues | prevalence | AUC | AUPRC | TPR |
|---|---|---|---|---|---|
| bulk surface | 73,296 | 3.5% | 0.851 | 0.236 | 0.433 |
| lines a non-cryptic pocket | 3,340 | 44.0% | **0.699** | 0.612 | 0.953 |

**This stratum resisted three interventions and then moved slightly on the fourth:**

| run | hard-stratum AUC | bulk-surface AUC | gap |
|---|---|---|---|
| Tier A, old sampler | 0.677 | — | — |
| + ensemble features, old sampler | 0.680 | — | — |
| + ensemble features, fixed sampler | 0.681 | — | — |
| + conservation, fixed sampler | 0.699 | 0.851 | 0.152 |
| + ESM-16 | 0.707 | 0.866 | 0.159 |
| + ESM-32 | 0.714 | 0.875 | 0.161 |
| **+ ESM-64** | **0.714** | **0.878** | **0.164** |

Conservation produced the first real movement (+0.018). ESM added another +0.015 and
then saturated between k=32 and k=64.

**But ESM did not close the gap — it widened it.** The bulk surface improved by +0.027
while the hard stratum improved by +0.015, so the spread went from 0.152 to 0.164. A
protein language model makes the model better at the part it was already good at. That
is evidence against the hope that crypticity is legible in sequence: if it were, ESM is
exactly the feature that should have found it, and it preferentially did not.

0.714 against 0.878 remains the limiting factor: telling a cryptic site from a pocket
that is *already open* is the discrimination the task is actually about.

What has NOT helped here: 28,000 fpocket runs over conformational ensembles (+0.001),
the sampler fix (+0.001), and context averaging (0.000). Harmonic collective motion
cannot produce the side-chain rotamer flips and loop crossings that plausibly separate
these cases, and no amount of resampling substitutes for that. ESM's failure to close
the gap points the same way: what distinguishes these two classes appears to be a
property of motion, not of sequence or of static shape.

---

## 6. Dataset and integrity

| | |
|---|---|
| Rows | 380,930 residues |
| Proteins | 1,135 apo structures (1 quarantined of 1,136) |
| Prevalence | 5.34% positive |
| Features | 48, all computed from the **apo structure only** |
| Splits | CryptoBench's own sequence-identity folds, group = apo structure |
| Label match rate | median 1.000, min 0.800 |
| Ensemble feature coverage | 100% of residues and proteins |
| Tests | 32 passing |

Three guards enforced by tests:

1. **No holo-derived feature reaches X.** A cryptic site is *defined* by the apo→holo
   change, so any holo-derived quantity reconstructs the label and is unavailable at
   prediction time. `tests/test_no_leakage.py` fails the build on a holo-named column.
2. **No protein appears in two splits.** Residues within a protein are correlated and
   homologues leak across a random split; we use CryptoBench's shipped folds.
3. **Labels are the union over all holo partners.** 5,493 pairings for 1,136 apo
   structures (~5 each); labelling from one partner would mark true cryptic residues as
   negatives.

---

## 7. Not measured

- Fine-tuned TabICL (`tabicl[finetune]`) — deliberately out of scope; it would forfeit
  the zero-training property that makes the comparison interesting.
- Sequence conservation (CryptoSite's #2 descriptor) — needs MSAs.
- ESM-2 embeddings — what pLM-NN itself uses.
- Fragment docking (CryptoSite's #3) — needs PatchDock + RankScore.
