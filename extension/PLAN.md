# Extension: NVIDIA Kumo Tabular on PAM's features

Plan written 2026-10-18, before any Kumo job ran. As in `finetuned_esm/PLAN.md`, each step
has a **gate**. The next step starts only once the gate passes, and a failed gate is recorded
here rather than worked around.

## Why

PAM (`ft117`, run `fe1`, `finetuned_esm/PLAN.md` step 10) is this project's best model. It
uses 53 apo-structure features plus 64 PCs of the clean fine-tuned ESM-2 3B embedding
(`cleantest_k` fold models), classified by zero-shot TabICLv2 over four cross-fitted fold
contexts whose predictions are averaged.

| PAM, CB-full (231) | value |
|---|---|
| per-protein AUPRC | 0.4859 [0.446, 0.525] |
| 12-seed mean (step 12) | 0.4856 ± 0.0012 |
| per-protein AUC | 0.872 |
| frozen threshold | 0.3407 |
| test FPR | 0.046 |

NVIDIA released **Kumo Tabular** in September 2026. It ships as `sdm.models.KumoTabular` in
`structured-data-models` (Apache-2.0 code, OpenMDW weights on HF `nvidia/Kumo-Tabular`). Like
TabICL, it is a zero-shot in-context tabular foundation model, pretrained on synthetic SCM
tables only. By NVIDIA's own report it ranks first on TabArena.

**Question:** if the classifier is the only thing that changes, does Kumo Tabular (large) beat
TabICLv2 on PAM's features, contexts, threshold rule and test set?

## Design

Everything except the classifier is fe1's. That makes the comparison paired by protein and by
context draw.

- **Table:** `residues_ftemb.parquet`, sha256 `4afaaa22…e13f`.
- **Rows:** `select` (a verbatim copy of `deploy/ftemb_stack.py`'s, enforced by a test) and
  `sampling.build_context` at seed 0 (all positives, ≤ 120 negatives per protein). That gives
  contexts of about 31k rows for test jobs and about 23k for sub-fold jobs.
- **Columns:** `feature_sets.resolve("ft117")` gives 117 columns, selected through
  `model.feature_matrix`.
- **Thresholds and test score:** the threshold for each fold context k is the one giving mean
  per-protein FPR 0.05 on its four protein sub-folds. The frozen threshold is the median of
  the four. The test score is the mean of the four k predictions.

### What differs, and why

- **Classifier:** `KumoTabularPredictor` (`extension/kumo_model.py`), size `large` (~215M) with
  `num_estimators=16`. Both are the user's choice, and 16 is NVIDIA's benchmark-adapter
  default. It runs under fp16 autocast with a seeded CUDA generator, using fit/predict with a
  cached context and `estimator_batch_size=1`, as in NVIDIA's adapter.
- **Categoricals:** `resname` and `ss` are typed `categorical`. Kumo's recipe aligns them to the
  context and passes them to the model as ordinal codes with a categorical mask. NaNs are
  passed through.
- **Library and GPU:** sdm 0.2.1, torch 2.7.1 + cu126, Python 3.11, pandas 2.3.3, numpy 2.2.6 (the local versions; sdm's pyarrow needs numpy 2, unlike fe1's image), A100-80GB. Kumo's model
  card supports only Ampere, Hopper and Blackwell GPUs, so fe1's T4 cannot be used.

### Known stretch

117 columns is more than Kumo's pretraining maximum of 100. TabICL had the same limit and
coped. The recipe's `SelectColumns(500)` keeps all 117, and a gate asserts this. The context
sizes are inside its range, since stage 3 of pretraining used tables of up to 60k rows.

## Steps

### 1. Smoke
`modal run extension/kumo_stack.py --stage smoke` runs the model card's breast-cancer
quick-start with Kumo large and 16 estimators.

**Gate:**
- AUC > 0.98.
- The output columns are `["0", "1"]`.
- The sdm, torch and CUDA versions, the GPU, and the sha256 of both checkpoints are recorded in
  `extension/smoke.json`. The Kumo revision is pinned in sdm at `v1.0.1`.

### 2. Harness check
Run sdm's `TabICLv2`, with PAM's checkpoint `tabicl-classifier-v2-20260212` and 8 estimators,
through this exact code on k=0 / test with fe1's context.

**Gate:**
- Mean per-protein AUPRC on that block is within ±0.01 of `fe1/ft117_k0_test.json`.
- Score Spearman ρ > 0.97.
- The same `n_context` and the same 117 features in the same order.

If it passes, a Kumo − PAM gap is the model and not the data path (column order,
categoricals, NaNs, probability column, row alignment). It is not expected to be bit-identical,
because sdm reimplements TabICL's preprocessing.

### 3. Determinism and cost pilot
Run Kumo on k=0 / sub0 twice (rep 0 and rep 1).

**Gate:**
- max |Δp| ≤ 1e-3 between the two runs (fp16 GPU noise).
- `n_context` equals fe1's.
- 117 columns reach the tensor, identical to fe1's list.
- The 20-job cost, projected from the measured fit and predict seconds, is ≤ $30 at $2.50/h.
  If it is higher, **stop and ask the user**.

`python extension/kumo_report.py --gates --run-id ku1` writes `extension/gates.json`.

### 4. Pre-register
Write `extension/frozen.json` before any CV or test job, and pin its sha256 in
`extension/kumo_report.py`. It records:

- **Primary claim:** kumo117 − PAM on CB-full per-protein AUPRC, paired 95% bootstrap with
  10,000 protein resamples at seed 0. The test is two-sided: *better* if the lower bound > 0,
  *worse* if the upper bound < 0, otherwise a *tie*.
- **Context, not claimed:**
  - per-protein AUC
  - pooled AUC and AUPRC
  - CB-P2RANK-apo (184)
  - TPR, FPR, MCC, F1 and F1w at each model's own CV-frozen threshold
  - test FPR transfer
  - per-k single-context AUPRC
  - a fixed 50/50 within-protein rank blend of Kumo and PAM (exploratory)
- **Not covered:** a single context-draw seed (0), paired with fe1. PAM's seed SD of 0.0012
  is quoted beside the result, and Kumo's seed variance is not measured.

### 5. Run
Run `--stage cv` (16 jobs) and `--stage test` (4 jobs) under run id `ku1`, then `--collect`.

### 6. Report
`python extension/kumo_report.py --run-id ku1` writes `kumo_thresholds.json` before it reads
any test file, then writes `kumo_result.json`. It asserts that the PAM column reproduces
step 10's 0.4859 exactly. The result is logged below as SCORED ONCE and written up in
`extension/README.md`.

### 7. Before touching anything shared
Ask the user before committing, and before editing `RESULTS.md` or the root `README.md`.

## Expectation, written before any result

A tie within about ±0.01 per-protein AUPRC. Both models see the same features and contexts,
TabArena margins between leading tabular foundation models are small, and a supervised
projection and every TabICL knob have already proved null on this table (RESULTS.md §§ 11–12).
A Kumo loss is plausible too. Its 100-column pretraining ceiling is a little further from 117
than its row ceiling is from 31k, and PAM's TabICL is the model this protocol was tuned
around (the context sampling and the 120-per-protein cap).

## Log

### Step 1 — PASSED, 2026-10-18
`extension/smoke.json`:
- The quick-start gives AUC 0.9969 on 269 query rows, and the output columns are `["0", "1"]`.
- The model has 213,668,250 parameters, confirming that `large` loaded.
- Environment: sdm 0.2.1, torch 2.7.1+cu126, A100 80GB.
- Kumo `large/classifier.pt` at v1.0.1 has sha256 `cc214843…ec613`.
- The TabICL checkpoint sha256 is `bdc7dbd5…d4ed0`, identical to the local tabicl 2.1.1 copy, so
  the harness check uses PAM's exact weights.

The first image build failed (pyarrow 26 needs numpy 2, and the build had pinned `numpy<2`).
The image now pins the local environment: pandas 2.3.3, numpy 2.2.6, pyarrow 25.0.1.

### Step 2 — PASSED, 2026-10-18 (`extension/gates.json`)
sdm TabICLv2 on k=0 / test, with 30,309 context rows, the same as fe1:

| | per-protein AUPRC |
|---|---|
| sdm TabICLv2 | 0.4520 |
| fe1 (tabicl 2.1.1) | 0.4508 |
| Δ | +0.0012 (tolerance ±0.01) |

- Spearman ρ = 0.987 (> 0.97).
- The 117 features are identical and in the same order.
- The job took 20 s.

### Step 3 — PASSED, 2026-10-18
Kumo on k=0 / sub0, run twice. The two runs landed on different A100 SKUs (PCIe and SXM).

- **Determinism:** max |Δp| = **0.0**, bit-identical.
- **Inputs match fe1:** `n_context` = 22,698, the same as fe1, and 117 columns reach the tensor,
  in fe1's list.
- **Time and memory:** fit 28 s, predict 14 s, peak 5.2 GB.
- **Cost:** 20 jobs projected at 1,107 s, about **$0.77**, against the $30 ceiling.

**Disclosed:** the pilot exposed one CV block before pre-registration. On k=0 / sub0, Kumo's
per-protein AUPRC is 0.3935 against fe1's 0.4006. The expectation above predates the pilot and
was not edited. This is recorded in `frozen.json`.

### Step 4 — FROZEN, 2026-10-18
`extension/frozen.json` has sha256 `3b1f78fc…c0693`, which is pinned in `extension/kumo_report.py`.
No cv or test job had run when it was written.

### Steps 5–6 — SCORED ONCE, 2026-10-18 (run `ku1`)

The 20 jobs (16 cv, 4 test) used 976 GPU-seconds, about **$0.68**. Each test job took about
88 s, and peak memory was 5.7 GB. The report checked `frozen.json` and the gates, wrote
`kumo_thresholds.json` before reading test, and reproduced PAM's step-10 value exactly
(0.4859192…).

**Thresholds** (CV only):
- Kumo: k = 0.2263 / 0.2523 / 0.2565 / 0.2660, frozen at **0.2544**.
- PAM keeps 0.3407.
- Kumo's CV AUPRC per k is 0.391 / 0.455 / 0.415 / 0.417. PAM's is 0.398 / 0.465 / 0.414 / 0.416.

**CB-full (231)**

| | Kumo | PAM | Kumo − PAM [95% CI] |
|---|---|---|---|
| per-protein AUPRC | 0.477 | **0.486** | **−0.009 [−0.016, −0.002]** |
| per-protein AUC | 0.866 | **0.872** | −0.006 [−0.010, −0.003] |
| pooled AUPRC | 0.416 | 0.424 | −0.007 [−0.021, +0.007] |
| pooled AUC | 0.900 | 0.903 | −0.003 [−0.007, +0.001] |
| TPR / FPR at frozen threshold | 0.475 / 0.041 | 0.515 / 0.046 | |
| MCC / F1 / F1w | 0.389 / 0.398 / 0.927 | 0.398 / 0.405 / 0.926 | |
| single-context AUPRC, k=0..3 | 0.434 0.449 0.448 0.436 | 0.451 0.447 0.455 0.450 | |

CB-P2RANK-apo (184): per-protein AUPRC 0.512 vs 0.520, a difference of −0.008 [−0.016, −0.000].
Pooled AUPRC is 0.482 vs 0.488.

**Primary claim: WORSE.** kumo117 − PAM = −0.009 [−0.016, −0.002]. The upper bound is below 0.

**Reading**

- **Size of the loss.** The pre-registered "tie within ±0.01" was right about the size and wrong
  about significance: the loss is small but consistent. Kumo also loses per-protein AUC on both
  subsets. It is behind on three of four single contexts and ties on k=1.
- **Comparison with seed noise.** The gap is about 7× PAM's context-draw SD of 0.0012.
  Kumo's own seed variance was not measured.
- **Threshold transfer.** Kumo's threshold transfers conservatively: test FPR is 0.041 against
  the 0.05 target. It buys that with 4 points of TPR, so MCC and F1 are slightly lower and
  weighted F1 is level.
- **Rank blend (exploratory).** The fixed 50/50 rank blend ties PAM per protein
  (−0.001 [−0.004, +0.003]), so Kumo adds nothing PAM lacks. The blend's pooled numbers are not
  meaningful, because within-protein ranks are not comparable across proteins.
- **Harness.** The step-2 harness check rules out the data path: the same code with TabICLv2
  reproduces fe1 to within +0.001. The loss belongs to the model, on this table: 117 columns
  (past Kumo's 100-column pretraining limit), 7:1 class imbalance, and a protocol tuned around
  TabICL.
