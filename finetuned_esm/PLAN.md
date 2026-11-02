# The published fine-tuned ESM-2 3B, scored on our table

Plan written 2026-10-06, before any code. Each step has a **gate**; the next step does not
start until the gate passes, and a failed gate is recorded here, not worked around.

## Why

Two papers from the CryptoBench group report the current state of the art on our exact
benchmark, and neither was known when this project was run:

- Škrhák, Gamouh, Hoksza, *Hidden in protein sequences: Predicting cryptic binding sites*,
  ACM BCB 2025. ESM-2 3B fine-tuned on CryptoBench, with a flexibility side task and
  LIGYSIS data added.
- Škrhák, Polák, Novotný, Hoksza, *Seq2Pocket*, bioRxiv 2026. Reuses that model unchanged
  for cryptic sites, so its residue-level AUC and AUPRC are the same numbers.

Both report **pooled** metrics only, on 184 single-chain test structures (= our
CB-P2RANK-apo): AUC 0.894, AUPRC 0.475, MCC 0.472. On the same subset we are pooled
AUC 0.883 [0.858, 0.905], AUPRC 0.398 [0.349, 0.452], MCC 0.408. Their per-protein numbers
are unknown, and per-protein is the convention this project reports.

## What we know about their model (from their code, not the paper)

- `MultitaskFinetunedEsmModel`: ESM-2 3B with three linear heads on the full 2,560-dim
  last hidden state. Head 1 is the cryptic-site logit; the other two are training-only
  side tasks. **No PCA anywhere.**
- Input: every residue of the chain with peptide backbone atoms, in order (biotite,
  model 1). Labels are 0-based positions in that string.
- Their loader **skips chains longer than 1,024**, so long chains are missing from their
  evaluation. We will chunk them as `esm_embed.chunk_spans` already does.
- Weights ship inside `seq2pocket.tar.gz` as a pickled whole model (`torch.load`,
  `weights_only=False`), so their class definitions are needed to load it.
- Their MCC threshold (0.70) was chosen by sweeping on the test file. AUC and AUPRC are
  unaffected.

## Steps

### 1. Get the model onto Modal
Download the archive inside a Modal function into a Volume (it times out from this
machine), list its contents, and extract the cryptic-site model and any annotation files.
**Gate:** contents listed and recorded below; the model loads with their class
definitions; the parameter shapes are ESM-2 3B plus three 2,560→1 heads; sha256 recorded.

### 2. Overlap audit
If their train/val/test annotation files are in the archive: confirm their test set is our
184 CB-P2RANK-apo structures, and that **none of our 231 test structures appear in their
training or validation files**.
**Gate:** zero overlap. If the training files are absent, record that the condition rests
on the paper's stated split and continue, flagged.

### 3. Residue mapping (CPU, tested)
Rebuild their input sequence for every apo chain in our table from our residue records,
keep the (apo_id, chain, seq_id, icode) key beside every position, and compare against
their sequence strings where we have them.
**Gate:** unit tests pass; mismatches against their strings enumerated and explained
(expected source: modified residues, which § 9.4 keeps and they may not).

### 4. Inference on Modal
Score every apo chain, train and test, with head 1 (sigmoid). Separately, score their own
test file exactly as shipped.
**Gate (reproduction):** their own test file, pooled, gives AUC 0.8935, AUPRC 0.4752 and
MCC@0.70 0.4720 to 3 dp. Our mapping on CB-P2RANK-apo lands close to that. If either
misses, stop and find out why before any result is read.

### 5. Step 0 result: their model, our evaluation
`evaluate.summarise` on CB-full (231) and CB-P2RANK-apo (184): per-protein and pooled,
10,000-resample bootstrap CIs. Then a **paired** per-protein bootstrap against TabICLv2-117,
now possible because both per-protein tables exist.

### 6. Memorisation check
Their model on our training proteins (in-sample for it) vs test. Measures how much an
in-sample score would mislead a TabICL context; decides how much step 8 matters.

### 7. Late fusion, pre-registered
Two blends of their score with the TabICLv2-117 test scores, fixed in `frozen.json` and
committed **before** the blend is scored: (a) mean of within-protein percentile ranks,
(b) mean of raw probabilities. Nothing fitted. Threshold-free metrics are the result;
threshold metrics reported as approximate.

### 8. Decision gate, not an implementation step
Out-of-fold stacking (four fine-tunes of a 3B model) is proposed only if a blend beats
their model alone with a paired CI that excludes zero. It needs an explicit go-ahead for
the GPU spend.

## Log

### Step 1 — PASSED, 2026-10-06
- The CESNET tarball host accepts no connections (from this machine or from Modal; curl sat
  at 0 bytes for 10+ min). The owncloud link in cryptic-finetuning's README returns 404.
- Same authors publish the weights on HuggingFace: `skrhakv/seq2pocket`, pinned to commit
  `610a4f8c…` (2026-08-26). `cbs-model.pt` is the filename their CryptoBench evaluation
  notebook loads. 5.69 GB, sha256 `d2db206d…0dac` verified against the LFS pointer.
- Pickle references `finetuning_utils.MultitaskFinetunedEsmModel`. Classes vendored
  verbatim as `deploy/s2p_models.py` (AST-identical to upstream).
- 36 layers, hidden 2,560, 2.84 B params, fp16; three 2,560→1 heads. A 66-residue forward
  returns three (1, 68, 1) outputs (BOS + 66 + EOS). Detail: `checkpoint_inspection.json`.
- Not in the HF repo: their train/val/test annotation files. Step 2 adapts (below); the
  step 4 reproduction gate becomes the end-to-end check of our rebuild of their inputs.

### Step 2 — PASSED (one part inferred), 2026-10-06
Their training files are unavailable, so the audit uses what their code fixes: CBS
training = CryptoBench train folds 0–2, validation = fold 3 (`generate-sequences.ipynb`),
plus LIGYSIS minus a hard-coded list of 46 UniProt IDs.
- **CryptoBench part: verified.** test.json (222 entries) vs train folds 0–3 (885): zero
  shared apo IDs, zero shared UniProt IDs (222 vs 888). All 231 of our test structures
  come from test.json.
- **LIGYSIS part: inferred, not verified — and the inference was WRONG (see 9b).** Their
  actual LIGYSIS training list still contains 28 CryptoBench test proteins by UniProt.
  Original text kept below for the record.
- **LIGYSIS part: inferred, not verified.** The 46 excluded IDs are all human; 9 are exact
  CryptoBench test UniProts (P18669, O75530, P50053, P14207, P17900, P12004, O43924,
  Q9BYC5, P15121) and the rest are in test-set families (AKR1, enolase, LDH, pyruvate kinase,
  MDH, UCH). That is the signature of a sequence-similarity screen of LIGYSIS against the
  test set. Confirming it needs their LIGYSIS training list.

### Step 3 — PASSED, 2026-10-06
`cryptic_tabicl/s2p_inputs.py` replicates their extraction (biotite 0.41.2, their
`three_to_one`, their label loop); `scripts/s2p_build_inputs.py` builds it for all 1,136
apo structures. 8 tests in `tests/test_s2p_inputs.py` pass, including a real structure
with insertion codes. Detail: `step3_mapping.json`.
- **Their test file reproduced exactly:** 184 structures, 55,572 residues, 3,158 binding —
  the paper's Table 1 to the residue. Their 184 = our CB-P2RANK-apo 184, same set.
- **Every one of our 381,704 residues has a position in their sequence** (join on author
  keys, no alignment). Zero label disagreements on joined residues.
- Their sequences carry 53 residues we do not: free amino-acid *ligands* filed under the
  protein's chain (Gly/Trp/Pro… numbered 401, 701, 1001 at the C-terminus), plus
  non-standard residues we exclude. Kept — it is the input their model saw — and never
  joined to an evaluated row. Their one extra positive is `2dfpA/A_200` (MIS), already
  listed in RESULTS.md § 9.4 as unplaceable.
- 2 duplicate keys (4z0y B/D 252, modelled as both HIS and HS8); neither is in our table.
- 8 residues get a different letter (e.g. CSS→X in theirs, C in ours). Their input, kept.
- 5 chains exceed 1,024 residues, all in **train**; they are chunked like our ESM run.
  None are in their test file or ours.

### Step 4 — PASSED, 2026-10-06
Run `s2p_20261006`: 1,309 chains, 381,757 residues, 78 s on one A10G. ESM dropout
asserted 0.0, so their notebook's missing `.eval()` changes nothing. Pilot first (3 chains,
one chunked at 1,094 residues). Detail: `step4_reproduction.json`.

| | AUC | AUPRC | MCC@0.7 | F1 (weighted) | ACC |
|---|---|---|---|---|---|
| published (seq2pocket notebook) | 0.8935 | 0.4752 | 0.4724 | 0.9390 | 93.51 % |
| **their file + their labels, our run** | **0.8935** | **0.4753** | **0.4721** | **0.9389** | **93.50 %** |
| our table + our labels, CB-P2RANK-apo | 0.8935 | 0.4751 | 0.4719 | 0.9389 | 93.50 % |

Every delta ≤ 0.0003. Our MCC equals their *Hidden* threshold notebook's 0.4720 exactly; the
0.4724 is a second run of theirs (fp16 batch padding). The weights, the input rebuild and
the key join are all confirmed. `build/s2p_scores_keyed.parquet` holds the joined scores.

### Step 5 — DONE, 2026-10-06 (result, not a gate)
`scripts/s2p_evaluate.py` → `step5_result.json`. Checks: the TabICL rows reproduce the
headline exactly (0.862 / 0.441, pooled 0.885 / 0.360); s2p pooled on CB-P2RANK-apo
reproduces step 4. Paired bootstrap, s2p − TabICL-117 (10,000 resamples per-protein,
2,000 pooled):

| | CB-full (231) | CB-P2RANK-apo (184) |
|---|---|---|
| per-protein AUC | 0.839 vs **0.862**, −0.023 [−0.039, −0.006] | 0.848 vs 0.861, −0.013 [−0.030, +0.003] |
| per-protein AUPRC | 0.446 vs 0.441, +0.005 [−0.018, +0.027] | 0.480 vs 0.469, +0.012 [−0.014, +0.037] |
| pooled AUC | 0.882 vs 0.885, −0.003 [−0.020, +0.011] | 0.893 vs 0.883, +0.011 [−0.006, +0.024] |
| pooled AUPRC | 0.397 vs 0.360, +0.036 [−0.006, +0.080] | **0.475** vs 0.398, +0.078 [+0.034, +0.120] |

**Per protein the two models are tied on AUPRC, and TabICL is ahead on AUC (significant on
CB-full).** The fine-tuned model's only significant lead is pooled AUPRC on the 184 subset —
the one number its papers report. It loses almost nothing to pooling (0.480 → 0.475);
TabICL loses 0.071 (0.469 → 0.398). Its scores are comparable across proteins; TabICL ranks
well within a protein but its score level shifts between proteins (cf. § 10). The earlier
expectation that its per-protein AUPRC would sit well above 0.475 was wrong.

### Step 6 — DONE, 2026-10-06
Their model's AUPRC by our fold and by chain type (per-protein / pooled):

| proteins | n | per-protein AUPRC | pooled AUPRC |
|---|---|---|---|
| test, single-chain | 193 | 0.479 | 0.470 |
| train folds 0–2, single-chain | 581 | 0.632–0.668 | 0.654–0.709 |
| train fold 3, single-chain | 201 | 0.622 | 0.666 |
| test, multi-chain | 38 | 0.278 | 0.174 |
| train, multi-chain (all folds) | 123 | 0.190–0.311 | 0.137–0.259 |

- **Strong memorisation:** single-chain training proteins score ~0.15–0.19 per-protein AUPRC
  above test. In-sample scores in a TabICL context would overstate this feature badly, so
  step 8 needs out-of-fold scores — there is no shortcut.
- **Fold 3 is in-sample too.** Their notebook uses fold 3 as validation, but the released
  model scores it like folds 0–2, so it was trained on it.
- **Multi-chain proteins were never trained on** (their extraction skips them) and look
  like test. It is also their model's weak spot: on the 38 multi-chain test proteins,
  per-protein AUPRC 0.278 vs TabICL 0.301 (−0.023 [−0.068, +0.027]), AUC 0.807 vs 0.853.
  On the 193 single-chain: AUPRC 0.479 vs 0.469 (tie), AUC −0.018 [−0.035, −0.002], TabICL
  ahead.

### Step 7 — FROZEN, not yet scored (2026-10-06)
`frozen.json` fixes both blends (rank = primary, prob = secondary; 0.5/0.5, nothing
fitted), the metrics and the step 8 decision rule. sha256
`9aa0ebe8630b5b1f81e630075a01aa8005ae51fbe69f0a02bb14c0cfc8e2622f`. Waiting on a commit of
the pre-registration before any blend is computed. Test suite: 191 passed, 1 skipped.

### Step 7 — SCORED ONCE, 2026-10-06
`scripts/s2p_blend.py` → `step7_result.json`; frozen.json hash checked before scoring.
The s2p and TabICL arms reproduce step 5 exactly.

| | s2p | TabICL-117 | rank blend | prob blend |
|---|---|---|---|---|
| CB-full per-protein AUC | 0.839 | 0.862 | 0.865 | 0.867 |
| CB-full per-protein AUPRC | 0.446 | 0.441 | **0.477** | **0.480** |
| CB-full pooled AUC | 0.882 | 0.885 | 0.884 | **0.903** |
| CB-full pooled AUPRC | 0.397 | 0.360 | 0.367 | 0.417 |
| 184 per-protein AUPRC | 0.480 | 0.469 | **0.508** | **0.512** |
| 184 pooled AUC | 0.893 | 0.883 | 0.889 | **0.906** |
| 184 pooled AUPRC | 0.475 | 0.398 | 0.416 | 0.480 |

**Decision rule:**
- **Primary passes.** Rank blend − s2p, per-protein AUPRC, CB-full: +0.031 [+0.018, +0.044].
  Rank blend − TabICL: +0.036 [+0.022, +0.049]. The two models carry complementary signal.
- **Secondary fails.** Prob blend − s2p, pooled AUPRC on the 184: +0.005 [−0.022, +0.031].
  On the papers' own headline number, a fixed 50/50 blend does not beat their model.
- Two tests were run. The primary survives a Bonferroni correction: its interval implies
  SE ≈ 0.0066, so +0.031 sits ≈ 4.7 SE from zero against the 2.24 a two-test 95 % needs.
- Not pre-registered, so observation only: the prob blend is at least as good as the rank
  blend everywhere and significantly raises pooled AUC over both components (CB-full
  +0.021 vs s2p, +0.017 vs TabICL).
- Approximate threshold metrics, top 7.74 % per protein, CB-full: MCC rank 0.406 / prob
  0.403 vs s2p 0.375, TabICL 0.377.

**Step 8 is therefore proposed**, pending a go-ahead (see below).

## Step 8 — out-of-fold stacking, option 1 (plain fine-tune), approved 2026-10-06

The released model's multitask + LIGYSIS recipe needs data only published on the dead
CESNET share; the user has emailed the authors. Meanwhile, the **plain fine-tune** from
their `train.ipynb` ("Finetuning", pooled 0.888 / 0.446 on the 184 in their paper) needs
only CryptoBench, which we have.

Their recipe, exactly (`train.ipynb` cell 9, `finetuning_utils.py`): `FinetunedEsmModel`
(ESM-2 3B + one 2,560→1 linear head), cast `.half()`; `bnb.optim.AdamW8bit(lr=1e-4,
eps=1e-4)`, bitsandbytes 0.45.5; `BCEWithLogitsLoss(pos_weight=8.7481)`; batch 1, shuffled,
`torch.manual_seed(42)`; 3 epochs, backbone frozen in epochs 0–1 and unfrozen in epoch 2;
single-chain entries only, sequences > 1,024 skipped. Their per-epoch test pass is
monitoring only, feeds nothing, and is omitted. Their `train.txt` = all four train folds
(step 6: the released model memorised fold 3).

Five fine-tunes: model `all` on folds 0–3 (= their setup), and model `h` on the three folds
other than h, for h = 0..3.

- **8a. Training files.** Their annotation format, per fold, from `s2p_inputs`.
  *Gate:* single-chain/length filters applied as theirs; counts per fold recorded.
- **8b. Pilot.** One small job (≈20 proteins, all three phases) on the target GPU.
  *Gate:* runs to completion, loss falls, peak memory and s/protein measured. If the
  projected cost exceeds ~2× the $15 estimate, stop and report before spending it.
- **8c. Model `all` + reproduction.** Score the 184 pooled. *Gate (fixed now, before
  training):* within ±0.015 AUC and ±0.03 AUPRC of their 0.888 / 0.446 — one training run,
  so seed noise is allowed for, but a recipe error would miss by more.
- **8d. Four fold models.** Each scores its held-out fold (all proteins, multi-chain
  included, long chains chunked). *Gate:* every held-out fold scores like unseen data
  (per-protein AUPRC in the test range, not the ~0.63 of memorised proteins).
- **8e. Pre-register the stacking arms and decision rule;** commit before scoring test.
- **8f. TabICL with the out-of-fold score as a feature**, CV threshold as usual, test once.

### 8a — PASSED, 2026-10-06
`scripts/s2p_build_train.py` → `build/s2p_train/train-fold-{0..3}.txt`, built by the same
`s2p_inputs.their_annotation_rows` that rebuilt their test file (that file re-verified
byte-identical after the refactor, sha256 `a534584c…`). Detail: `step8a_train_files.json`.
- Single-chain entries over the four folds: **762 structures, 233,435 residues, 13,342
  binding — exactly the Seq2Pocket Table 1 "CryptoBench train subset"**, positive rate
  0.057155, exactly the rate their hard-coded class weights (0.5303, 8.7481) imply. So
  their `train.txt` is all four folds' single-chain entries, and model `all` is their setup.
- Their loader then drops 3 chains over 1,024 residues (fold 1: 1, fold 3: 2): 759
  proteins, 230,263 residues are actually trained on. Per fold: 183 / 192 / 191 / 193.

### 8b — PASSED, 2026-10-06
`deploy/s2p_finetune_modal.py` (their loop line for line; two immaterial differences: our
labels are float32 where their list→tensor conversion makes float64, and the shuffle uses
a seeded generator rather than the global RNG). Pilot: 20 proteins, all three phases, on
A100-80GB. Losses 1.067 → 1.045 → 1.012 (their logged epoch-0 loss: 1.065), none
non-finite; peak memory 5.8 GB frozen, 21.4 GB full; 0.34 s/protein in the full epoch.
Scores for 3 chains (one chunked) finite. Projected ≈ 10 min and ≈ $0.40 per job, ≈ $2
for all five — well under the $15 estimate.

### 8c — PASSED, 2026-10-06
Model `all` (run `s2p_ft_20261006`; 759 proteins; 6.5 min, $0.27 on A100-80GB). Mean loss
0.964 → 0.873 → 0.705, none non-finite; peak 24.9 GB. Detail: `step8_check_all.json`.
- **Their test file, their labels, pooled: AUC 0.8869, AUPRC 0.4402** vs their published
  0.888 / 0.446 (Δ −0.001 / −0.006; tolerance ±0.015 / ±0.03 fixed before training).
  Their recipe is reproduced from scratch.
- Same memorisation signature as the released model: single-chain training proteins
  0.598–0.642 per-protein AUPRC vs 0.458 on single-chain test.

### 8d — PASSED, 2026-10-06
Four fold models (576 / 567 / 568 / 566 training proteins = the 8a per-fold sums; none
non-finite; $0.71 together). Single-chain per-protein AUPRC (`step8_check_{0..3}.json`):

| model | its held-out fold | its training folds | test |
|---|---|---|---|
| 0 | **0.405** | 0.609–0.639 | 0.455 |
| 1 | **0.446** | 0.592–0.616 | 0.450 |
| 2 | **0.412** | 0.614–0.640 | 0.454 |
| 3 | **0.416** | 0.593–0.621 | 0.458 |

Every held-out fold scores like unseen data. The held-out folds sit slightly below test,
as TabICL's own CV folds do (0.376–0.442 vs 0.441 on test): fold difficulty, not leakage.

### 8e — FROZEN, not yet run (2026-10-06)
- **Out-of-fold column** (`scripts/s2p_build_stack_table.py` → `build/residues_s2p.parquet`,
  staged): fold h rows from model h, test rows from model `all`; all 381,704 rows scored;
  pooled AUPRC 0.315 / 0.366 / 0.357 / 0.353 on folds 0–3 and 0.371 on test, the same
  unseen-data range everywhere (`step8_stack_table.json`).
- **Arms registered** in `cryptic_tabicl/feature_sets.py`: family `S2P_OOF` excluded from
  the base explicitly; `s2p_117` (118 features) and `s2p_53` (54). Tests added; suite 194
  passed, 1 skipped.
- **Harness** `deploy/s2p_stack.py` (sibling of feature_sweep.py, own app and volume).
  **Determinism gate passed:** control on train-fold-0 from the new table equals
  `build/cv_scores/k64_train-fold-0.json` exactly — max |diff| 0.0 over 75,800 scores,
  identical row order (`step8_determinism_gate.json`).
- **Pre-registration** `frozen_step8.json`, sha256 `205e34cb4f2e77bd9a6f78fa8b84343a673183e2b4b63caa8fe8ec4c6f0ac8e2`: arms, threshold protocol,
  references (TabICL-117, the plain model alone, the fixed rank blend of the two), metrics,
  and four claims with their criteria. Waiting on a commit before the CV and test stages run.

### 8f — SCORED ONCE, 2026-10-06
`deploy/s2p_stack.py` run `st1` (10 T4 jobs, ≈ 1.5 GPU-h, ≈ $0.90), reported by
`scripts/s2p_stack_report.py` → `step8_thresholds.json` (written before test was read)
and `step8_result.json`. Checks: the TabICL-117 column reproduces the headline exactly;
the plain model alone reproduces 8c (pooled 0.887 / 0.440 on the 184).

| CB-full (231) | s2p_117 | s2p_53 | TabICL-117 | plain alone | fixed blend |
|---|---|---|---|---|---|
| per-protein AUC | **0.869** | 0.870 | 0.862 | 0.838 | 0.864 |
| per-protein AUPRC | **0.461** | 0.454 | 0.441 | 0.424 | 0.464 |
| pooled AUC | **0.898** | 0.896 | 0.885 | 0.876 | 0.882 |
| pooled AUPRC | **0.396** | 0.381 | 0.360 | 0.371 | 0.359 |

| CB-P2RANK-apo (184) | s2p_117 | s2p_53 | TabICL-117 | plain alone | fixed blend |
|---|---|---|---|---|---|
| per-protein AUPRC | **0.496** | 0.490 | 0.469 | 0.454 | 0.493 |
| pooled AUC | **0.900** | 0.899 | 0.883 | 0.887 | 0.886 |
| pooled AUPRC | **0.447** | 0.439 | 0.398 | 0.440 | 0.404 |

**Pre-registered claims** (per-protein AUPRC, CB-full, paired 95 % CI):
- **Primary PASSES:** s2p_117 − plain alone **+0.036 [+0.021, +0.052]**. Stacking beats the
  fine-tuned model it stacks.
- **Secondary 1 PASSES:** s2p_117 − TabICL-117 **+0.019 [+0.007, +0.031]**.
- **Secondary 2 does not pass:** s2p_117 − fixed rank blend −0.004 [−0.014, +0.007]. Per
  protein, learning the combination is no better than the fixed 50/50. Reported, not
  claimed: on *pooled* metrics the stack does beat the fixed blend (AUPRC +0.037
  [+0.010, +0.064], AUC +0.015 [+0.010, +0.020]) — a within-protein rank blend discards
  cross-protein scale by construction.
- **Secondary 3 does not pass:** s2p_117 − s2p_53 +0.006 [−0.001, +0.013]. Once the score
  is present, the 64 frozen ESM PCs add nothing resolvable per protein (pooled AUPRC on
  CB-full: +0.015 [+0.004, +0.026], not the claim metric).
- Against the papers' number (pooled AUPRC, 184): the stack 0.447 vs the plain model 0.440,
  +0.007 [−0.022, +0.035]; the released multitask+LIGYSIS model scored 0.475 (step 5).

**Threshold transfer failed.** Every CV fold lands at FPR 0.0500; test lands at **0.098**
(TPR 0.661, MCC 0.390 vs TabICL-117's 0.374 at FPR 0.071). Cause, measured: model `all`'s
scores on test sit higher than the fold models' scores on their held-out folds (negatives'
mean 0.296 vs 0.20–0.27; 21.5 % of negatives > 0.5 vs 11–17 %; positives 0.743 vs
0.58–0.68). TabICL learned the feature's scale from fold-model context, so its test
outputs shift up (negatives' mean 0.187 vs 0.11–0.13 in CV). Whether the shift comes from
the fourth fold of data or from run-to-run calibration of a single fp16 fine-tune can't be
separated — the four fold models already differ among themselves. AUC/AUPRC are threshold-
free and unaffected by the operating point; the scale mismatch may make the ranking gains
slightly conservative rather than inflated.

## Step 9 — the full recipe (multitask + LIGYSIS), 2026-10-06

The authors replied with their data: `cbs-finetuning-data.tar.gz` (26.1 GB, owncloud share
from Vít Škrhák). The user chose to go straight to the full recipe (skipping the 8f
threshold fix) and to commit at the end; pre-registrations are therefore pinned by
sha256 in this log before anything they govern is scored.

- **9a. Fetch and inventory.** Download inside Modal; record sha256 and the member list.
  *Gate:* the files their notebooks name are present (CryptoBench train/test, LIGYSIS
  train, pLDDT, residue distances), or the gap is named.
- **9b. Verify their data against ours.** Their CryptoBench train/test files must match
  our rebuilds (8a, step 3) residue for residue. Re-run step 2's LIGYSIS check with their
  actual LIGYSIS training list. *Gate:* exact match, or every difference explained; no
  CryptoBench test protein in their training data.
- **9c. Recipe.** Read the exact multitask + LIGYSIS training procedure from their code.
  *Gate:* every hyperparameter and stage traced to a line of theirs.
- **9d. Pilot, then reproduction.** Model `all` with the full recipe must reproduce the
  released model: pooled 0.8935 / 0.4752 on the 184. Tolerance fixed before training:
  ±0.015 AUC, ±0.03 AUPRC.
- **9e. Four fold models**; held-out folds must score like unseen data.
- **9f. Stack**, pre-registered as in step 8 (same arms, references, claims), scored once.

### 9a — PASSED, 2026-10-06
`cbs-finetuning-data.tar.gz`: 28.07 GB, 15,097 members, sha256 `12b06f2c…ffd41`
(`full_archive_manifest.json`). Holds CryptoBench `train.txt` / `test.txt` with pLDDT and
residue distances (946 proteins), LIGYSIS `train.txt` with pLDDT (3,007), and their trained
checkpoints. Their `multitask-finetuned-model-with-ligysis.pt` has sha256 `d2db206d…0dac`
— byte-identical to the HuggingFace `cbs-model.pt` reproduced in step 4.

### 9b — data verified; a test leak found in their recipe, 2026-10-06
- **Their CryptoBench files equal our rebuilds exactly**: test 184 entries / 55,572
  residues / 3,158 positives, train 762 / 233,435 / 13,342 — every sequence and label.
- **LIGYSIS vs CryptoBench** (`scripts/s2p_ligysis_overlap.py`; their loader's filters
  applied first, 2,614 LIGYSIS proteins trained on; 5-mer screen then BLOSUM62 local
  alignment, coverage ≥ 50 %):

  | | proteins | exact UniProt in LIGYSIS | ≥ 90 % identity | ≥ 40 % identity |
  |---|---|---|---|---|
  | test | 231 | 28 | 25 | 38 |
  | train folds 0–3 | 223–228 each | 36–42 | 36–42 | 54–62 |

- **This is test-label leakage in the released model.** For the 25 near-identical test
  proteins, 238 of 377 cryptic residues (63 %) are labelled binding in LIGYSIS training;
  for 16 of the 25 essentially all are (`step9b_test_label_leak.csv`). Both datasets take
  sites from ligand-bound PDB entries, so these labels are the test labels.
- **Measured impact is small** (`step9b_leak_impact.json`). Per-protein AUPRC, released vs
  plain fine-tune (which never saw LIGYSIS): leaked 25 — 0.475 vs 0.475, −0.000
  [−0.054, +0.057]; clean 206 — 0.442 vs 0.418, +0.024 [+0.011, +0.039]. LIGYSIS's benefit
  is real and sits on clean proteins; n = 25 cannot exclude a ~0.05 leak effect. The
  leaked proteins are easy for the fine-tuned models and hard for TabICL: dropping them,
  pooled AUPRC on the 184 goes released 0.475 → 0.461, TabICL 0.398 → 0.411, and the gap
  from 0.077 to 0.050.
- **Consequence for step 9.** Two versions of the full recipe:
  - *exact* — their LIGYSIS list as-is: the 9d reproduction gate only.
  - *clean* — each model's LIGYSIS stage also drops every LIGYSIS protein ≥ 40 % identical
    (coverage ≥ 50 %) to any protein that model scores held-out: model `all` → test,
    fold model h → fold h. Used for stacking. Same rule for context and test, and no
    model ever saw a protein it scores.

### 9c — DONE, 9d pilot PASSED, 2026-10-06
`deploy/s2p_finetune_modal.py::finetune_full` runs cell 16 of their `train.ipynb` with
their `finetuning_utils.py` vendored verbatim (`deploy/finetuning_utils.py`, commit
`7be0c676`). Traced: stage A LIGYSIS, 4 epochs, `llm` frozen in epochs 0–1, BCE + pLDDT MSE;
stage B CryptoBench, new optimizer, 3 epochs, `llm`/`classifier`/`plDDT` frozen in epochs
0–1 (only the distance head trains), BCE + distance MSE + pLDDT MSE, scalers refit on the
job's train file. Pilot (20 + 20 proteins, all 7 epochs): losses fall in both stages
(1.17 → 0.94; 1.35 → 1.20), none non-finite, peak 24.9 GB, LIGYSIS filter applied
(18 of 20 kept). Projected ≈ 40 min, ≈ $1.70 per job. LIGYSIS exclusion lists for the clean
variant: `ligysis_exclusions.json` (test 57; folds 80–97 of 2,614).

### 9f — PRE-REGISTERED, 2026-10-06 (before any step-9 model finished training)
`frozen_step9.json`, sha256 `c61241d2746aeeabf59783c74c40dd88ec13ad84f717d117ad1e3fd83049d656`
(the user waived commit-before-score; the hash is pinned here and in
`scripts/s2p_stack_report.py`, which refuses to report if the file changes). Arms
`s2pfull_117` (primary) / `s2pfull_53`, registered in `feature_sets.py` (tests: 195 passed).
References: TabICL-117, the clean full model alone, the fixed rank blend, step 8f's
`s2p_117`; the released model as context only, on the 206 non-leaked proteins.
Conditional on the 9d and 9e gates. The step-8 table is left untouched; step 9 writes
`residues_s2pfull.parquet`.

### 9d — PASSED, 2026-10-06
Run `s2p_full_20261006`, A100-80GB, ≈ 51 min and $2.14 per job. LIGYSIS trained on 2,614
(exact) / 2,557 (clean: 57 excluded) proteins, CryptoBench on 759; losses fall in both
stages (LIGYSIS 0.94 → 0.48; CryptoBench 0.85 → 0.70), none non-finite.

| their test file, pooled (184) | AUC | AUPRC |
|---|---|---|
| released model (published) | 0.8935 | 0.4752 |
| **exact recipe, our run (gate)** | **0.8928** | **0.4993** |
| clean recipe, our run | 0.8922 | 0.4847 |

- Gate passes (Δ −0.0007 AUC, +0.024 AUPRC; tolerance ±0.015 / ±0.03). AUC is exact; AUPRC
  lands above theirs, within tolerance but near its edge: run-to-run variance of a single
  fp16 fine-tune, in our favour this time. `step9_check_exact_all.json`.
- **Removing the leaked LIGYSIS proteins costs nothing visible**: the clean model scores
  0.485 pooled AUPRC on the 184, above the released model's 0.475.

### 9e — PASSED, 2026-10-06
Four clean fold models (LIGYSIS 2,517 / 2,521 / 2,534 / 2,520 = 2,614 minus each fold's
exclusions; CryptoBench 576 / 567 / 568 / 566; none non-finite; $6.45 together).
Single-chain per-protein AUPRC (`step9_check_clean_{0..3}.json`):

| model | its held-out fold | its training folds | test |
|---|---|---|---|
| clean_0 | **0.425** | 0.602–0.641 | 0.465 |
| clean_1 | **0.453** | 0.603–0.620 | 0.481 |
| clean_2 | **0.442** | 0.645–0.673 | 0.483 |
| clean_3 | **0.420** | 0.669–0.685 | 0.467 |

### 9f — determinism gate PASSED, 2026-10-06
`build/residues_s2pfull.parquet` (sha256 `f333c4dd…bcf73`; `step9_stack_table.json`): every
row scored; pooled AUPRC 0.309 / 0.389 / 0.394 / 0.364 on folds 0–3 and 0.411 on test.
Harness redeployed with `S2P_STACK_TABLE=residues_s2pfull.parquet`; control on train-fold-0
equals `build/cv_scores/k64_train-fold-0.json` exactly (max |diff| 0.0, 75,800 rows).

### 9f — SCORED ONCE, 2026-10-06
Run `st2` (10 T4 jobs, ≈ $0.90); `scripts/s2p_stack_report.py --step 9` →
`step9_thresholds.json` (written before test was read), `step9_result.json`. Checks:
TabICL-117 reproduces the headline; `s2p_117` reproduces 8f exactly; the clean full model
alone reproduces 9d (pooled 0.892 / 0.485 on the 184). The refactored report reproduces
`step8_result.json` byte-for-byte.

| CB-full (231) | **s2pfull_117** | s2pfull_53 | TabICL-117 | full alone | fixed blend | s2p_117 |
|---|---|---|---|---|---|---|
| per-protein AUC | **0.876** | 0.875 | 0.862 | 0.843 | 0.868 | 0.869 |
| per-protein AUPRC | **0.482** | 0.478 | 0.441 | 0.450 | 0.481 | 0.461 |
| pooled AUC | **0.905** | 0.902 | 0.885 | 0.881 | 0.886 | 0.898 |
| pooled AUPRC | **0.421** | 0.408 | 0.360 | 0.411 | 0.367 | 0.396 |

| CB-P2RANK-apo (184) | **s2pfull_117** | s2pfull_53 | TabICL-117 | full alone | fixed blend | s2p_117 |
|---|---|---|---|---|---|---|
| per-protein AUPRC | **0.517** | 0.512 | 0.469 | 0.481 | 0.510 | 0.496 |
| pooled AUC | **0.907** | 0.903 | 0.883 | 0.892 | 0.890 | 0.900 |
| pooled AUPRC | 0.476 | 0.471 | 0.398 | **0.485** | 0.413 | 0.447 |

**Pre-registered claims** (per-protein AUPRC, CB-full, paired 95 % CI):
- **Primary PASSES:** s2pfull_117 − full model alone **+0.032 [+0.018, +0.047]**.
- **Secondary 1 PASSES:** − TabICL-117 **+0.041 [+0.027, +0.055]**.
- **Secondary 2 does not pass:** − fixed rank blend +0.001 [−0.008, +0.010]. As in step 8:
  per protein, a fixed 50/50 is as good. On pooled metrics the stack beats it (AUPRC
  +0.054 [+0.029, +0.079]; AUC +0.019 [+0.013, +0.025]); per-protein AUC too
  (+0.008 [+0.002, +0.015]).
- **Secondary 3 does not pass:** − s2pfull_53 +0.005 [−0.002, +0.012]; the 64 frozen ESM PCs
  are redundant per protein once the score is in (pooled AUPRC +0.013 [+0.003, +0.024]).
- **Secondary 4 PASSES:** − s2p_117 **+0.022 [+0.013, +0.031]**. The stronger component makes
  the stronger stack.
- Not claimed: on the papers' metric (pooled AUPRC, 184) the stack ties its own component,
  0.476 vs 0.485 (−0.009 [−0.033, +0.019]).

**Threshold transfer is much better than in 8f:** test FPR **0.073** (TPR 0.616, MCC 0.411;
TabICL-117: 0.071 / 0.559 / 0.374). The clean `all` model's test scores sit closer to its
fold models' held-out scores than the plain model's did. Not claimed — it was not fixed.

**Context, not claimed** (`scripts/s2p_released_context.py` → `step9_released_context.json`):
on the 206 test proteins the released model never saw LIGYSIS labels for, per-protein /
pooled AUPRC: **s2pfull_117 0.480 / 0.424**; released 0.442 / 0.382; TabICL-117 0.443 /
0.372; clean full model alone 0.446 / 0.404. s2pfull_117 − released: per-protein
**+0.038 [+0.021, +0.055]**, pooled **+0.043 [+0.012, +0.072]**.

Cost of step 9: data fetch/extract ≈ $1; six full-recipe fine-tunes ≈ $10.7; TabICL ≈ $1.

## Step 10 — the fine-tuned model's embeddings as TabICL features, 2026-10-08

**Question (the user's):** instead of the fine-tuned model's *score* (step 9), feed TabICL its
*final embeddings* — the 2,560-dim last hidden state the cryptic head reads.

**Two constraints fix the design.**
- In-sample leak: a fine-tuned model's embeddings of its own training proteins are fit to their
  labels, so TabICL's context must come from proteins the embedding model never trained on.
- Each fine-tuned model has its own embedding space (a score means the same thing across fold
  models; a hidden state does not), so context and query must be embedded by the **same** model.

**Design — cross-fitted contexts.** Fold model *k* never saw train fold *k*: context = fold *k*,
query = test, both embedded by model *k*; 64 PCs fit on fold-*k* rows only (unsupervised, as the
frozen PCs). k = 0..3 give four test predictions, averaged. Raw 2,560 columns are far outside
TabICL's pretraining range, hence the PCA.

**New fold models, `cleantest`** (the user's choice over reusing `clean_0..3`, which dropped only
their own fold's LIGYSIS homologs and so saw labels for ~25 test proteins): LIGYSIS exclusions =
`exclude[train-fold-k] ∪ exclude[test]`. Same recipe, same code (`finetune_full`); the job also
dumps hidden states for fold *k* + test and the PCA (`_embed_and_reduce`).
Nothing already computed is recomputed (the user's instruction): steps 8–9 tables, scores and
the 117-feature CV/test results are reused as they are.

- **10a gate:** each cleantest_k's held-out-fold single-chain per-protein AUPRC within ±0.03 of
  clean_k (0.425 / 0.453 / 0.442 / 0.420); losses finite.
- **10b gate (alignment):** the cryptic head re-applied to the stored fp16 hidden states
  reproduces the job's own scores (max |Δp| < 1e-2) — row *i* is residue *i*.
- **10c–e:** table `residues_ftemb.parquet`; arms `ctrl117` (paired control, same fold-*k*
  contexts), `ft117` (53 + 64 fine-tuned PCs, primary), `ft64only`, `score118` (117 + model-*k*
  score); harness `deploy/ftemb_stack.py`; claims pre-registered in `frozen_step10.json` before
  the test stage.

**Expectation, written before any result:** supervising the ESM projection hurt before (§ PLS in
RESULTS.md), and the score is already one exact linear direction of this embedding that PCA may
dilute. A null is plausible.

**Pre-registered** `frozen_step10.json`, sha256 `707f56c46d301f30699dbdb3dacd3dfa3c1c4c1e8446561e30c0333e7c1b6d19` (pinned in `scripts/ftemb_report.py`), written while the fine-tunes trained and before any step-10 TabICL job.

### 10a — PASSED, 2026-10-08
Run `s2p_full_cleantest_20261008`, A100-80GB, 31–44 min and $1.28–1.81 per job ($5.72 together,
embedding included). LIGYSIS extra exclusions 152 / 140 / 135 / 149 (fold ∪ test). Losses finite.
Single-chain per-protein AUPRC (`step10_check_cleantest_{0..3}.json`):

| model | its held-out fold | clean_k (step 9) | its training folds | test |
|---|---|---|---|---|
| cleantest_0 | **0.419** | 0.425 | 0.633–0.665 | 0.466 |
| cleantest_1 | **0.460** | 0.453 | 0.610–0.622 | 0.476 |
| cleantest_2 | **0.437** | 0.442 | 0.620–0.655 | 0.468 |
| cleantest_3 | **0.420** | 0.420 | 0.664–0.686 | 0.471 |

### 10b — PASSED, 2026-10-08
Cryptic head on the stored fp16 hidden states vs the job's own scores: max |Δp| 1.02e-4 on all
four models (149–156 k residues each). PCA fit on exactly the fold-k residues (75,814 / 77,729 /
79,097 / 72,352); 64 PCs explain 45 / 43 / 49 / 43 % of variance (frozen ESM-2: see esm_modal).

### 10c — table built, 2026-10-08
`build/residues_ftemb.parquet` (381,704 rows, 386 columns, sha256 `4afaaa22…e13f`;
`step10_ftemb_table.json`). Every fold-k and test row has its block-k values. Local dry run of
`deploy/ftemb_stack.py::select`: no NaN in any used column, no context/query protein overlap;
fold-k contexts are ~31 k rows (226 proteins; the negative ratio binds, not the 50 k budget),
sub-fold contexts ~23 k.

### 10d — determinism gate PASSED, 2026-10-08
Control on train-fold-0 from `residues_ftemb.parquet` with the standard context equals
`build/cv_scores/k64_train-fold-0.json` exactly (max |diff| 0.0, 75,800 rows, same order).
CV (64 jobs) and test (16 jobs) spawned together as run `fe1`; the report computes and writes
thresholds from CV before reading any test score.

### 10e — SCORED ONCE, 2026-10-08
Run `fe1` (81 T4 jobs incl. the gate). `scripts/ftemb_report.py --run-id fe1` →
`step10_thresholds.json` (written from CV before test was read), `step10_result.json`.
Arms are the mean of four cross-fitted predictions (context = train fold k, embedded by
cleantest_k); `ft_alone` = mean of the four fold models' own scores.

| CB-full (231) | **ft117** | ft64only | score118 | control | ft_alone | TabICL-117 | s2pfull_117 |
|---|---|---|---|---|---|---|---|
| per-protein AUC | 0.872 | 0.847 | 0.873 | 0.860 | 0.841 | 0.862 | **0.876** |
| per-protein AUPRC | **0.486** | 0.451 | 0.485 | 0.438 | 0.454 | 0.441 | 0.482 |
| pooled AUC | 0.903 | 0.881 | **0.905** | 0.884 | 0.882 | 0.885 | **0.905** |
| pooled AUPRC | **0.424** | 0.393 | 0.420 | 0.349 | 0.409 | 0.360 | 0.421 |

| CB-P2RANK-apo (184) | **ft117** | ft64only | score118 | control | ft_alone | TabICL-117 | s2pfull_117 |
|---|---|---|---|---|---|---|---|
| per-protein AUPRC | **0.520** | 0.482 | **0.520** | 0.467 | 0.485 | 0.469 | 0.517 |
| pooled AUC | 0.906 | 0.893 | **0.908** | 0.882 | 0.894 | 0.883 | 0.907 |
| pooled AUPRC | 0.488 | 0.468 | 0.485 | 0.390 | **0.489** | 0.398 | 0.476 |

**Pre-registered claims** (per-protein AUPRC, CB-full, paired 95 % CI):
- **Primary PASSES:** ft117 − control **+0.048 [+0.033, +0.062]**. Replacing the frozen ESM-2
  PCs with 64 PCs of the fine-tuned model's own embedding is a large gain (pooled AUPRC
  +0.074 [+0.048, +0.099]; per-protein AUC +0.012 [+0.005, +0.019]).
- **Secondary 1 does not pass:** ft117 − score118 +0.001 [−0.005, +0.008]. The embedding
  carries no more than the score does: 64 PCs of it and the one score column are tied on every
  metric (pooled AUC favours the score by 0.002).
- **Secondary 2 does not pass:** ft64only − ft_alone −0.004 [−0.011, +0.004]. TabICL on the
  embedding alone only matches the fine-tuned head it was taken from (per-protein AUC +0.005
  [+0.001, +0.010]; pooled AUPRC −0.017 [−0.034, +0.000]). It is the structural features that
  add, not TabICL re-reading the embedding.

**Context, not claimed.**
- ft117 vs step 9's s2pfull_117: +0.003 [−0.005, +0.012] per-protein AUPRC — a tie. 0.486 is
  the highest per-protein AUPRC this project has measured, but not distinguishably so.
- The cross-fitted design costs nothing: control (four 31 k fold contexts, averaged) equals
  the 50 k headline, −0.003 [−0.010, +0.004] per protein. Each single fold context alone is
  worse (control 0.406–0.416, ft117 0.447–0.455); averaging four is worth ~+0.03, i.e. the
  four-context ensemble recovers what the smaller contexts lose.
- On the papers' metric (pooled AUPRC, 184): ft117 0.488, ft_alone 0.489, released model 0.475
  (published), all clean of the LIGYSIS leak except the last.
- Threshold metrics (CV-frozen, secondary): ft117 TPR 0.515 at FPR 0.046, MCC 0.398;
  control 0.450 / 0.044 / 0.353.

**Reading.** The fine-tuned model's information reaches TabICL equally well as one score
column or as 64 PCs of its hidden state; whichever form, it adds ~+0.045 per-protein AUPRC
over the frozen-PC model, and the result lands where step 9's stack already was. The embedding
route is not a better way in, it is an equivalent one, at the price of four fold models with
test-clean LIGYSIS and a cross-fitted context.

Cost of step 10: fine-tunes + embedding $5.72; TabICL 81 T4 jobs, ≈ 3.0 GPU-h of scoring (≈ $1.8, plus container start-up). Total ≈ $8.

### Final comparison figure, 2026-10-08
`scripts/final_comparison.py` → `final_comparison.json`; `scripts/final_comparison_figure.py`
→ `../Stacked_Figures/final_comparison.{pdf,png}`. Released model, fixed rank blend, step-9
stack, ft117 on CB-P2RANK-apo (184), all 7 metrics pooled and per protein, from scores
already computed. Gate: the released model on their own file rounds to every number in
the Hidden paper (0.894 / 0.475 / 0.472) and is within 0.0005 of their notebook. On these 184
it gives 0.8935 / 0.4751 / 0.4719 / F1w 0.9389 / ACC 0.9350; the residual is the 5 residues
our table lacks. The three other rows reproduce their step-9/10 values.

## Step 11 — PLS instead of PCA for the fine-tuned embedding, 2026-10-09

**Question.** Step 10 gave TabICL 64 PCA components of fold model *k*'s last hidden state.
Does a supervised projection, PLS on the residue label, serve it better? This was asked
for two models, 53 features + 64 components and the 64 components alone, plus a sweep
over the number of components.

**Design.** Pre-registered in `frozen_step11.json` (sha256 `aacccb7b…e48f`, pinned in
`scripts/ftpls_report.py`) before anything was fit.
- Inputs: step 10's stored hidden states (`hidden_cleantest_k.npy` on the `cryptobench-s2p`
  volume), reused; nothing was re-embedded.
- Estimator: `PLSRegression(scale=True)`, as in RESULTS § 12.
- Fits for the test jobs: fit `k{k}_test` on every train-fold-*k* table row, 128
  components. PLS components are nested (checked: identical to a d-component fit to
  1e-14), so the d-dim arm takes the first d.
- Fits for the CV threshold jobs: fit `k{k}_sub{s}` on fold *k* minus sub-fold *s*,
  64 components.
- Leak checks: each fit's manifest lists its proteins. A job asserts that no query
  protein is among them and that every context protein is.
- Arms: `pls{d}_53` and `pls{d}_only` (`cryptic_tabicl/feature_sets.py`), with
  d ∈ {2, 4, 8, 16, 32, 64, 128}. d = 64 is the headline; the other dimensions are
  descriptive.
- Claims: `pls64_53 − ft117` and `pls64_only − ft64only`, per-protein AUPRC, CB-full,
  10k paired bootstrap.

**Code.**
- `scripts/ftpls_build_index.py` → `build/ftpls_rows.parquet`: the table-row → hidden-row
  map, checked by requiring the PCs it points at to equal the table's `ftk{k}_pc`.
- `deploy/ftpls_stack.py`: `fit_pls` (CPU) and `score_one` (step 10's body and `select`).
- `scripts/ftpls_report.py` → `step11_thresholds.json`, `step11_result.json`.

### 11a — gates PASSED
- Mapping: on every k, PCA(64) refit on Modal from the gathered hidden-state rows matches
  step 10's PCs (min |corr| 0.99999998; max |Δ| 2.4e-4).
- Determinism: `ft117` k=0 through the new harness equals `fe1/ft117_k0_test.json`
  exactly (max |Δ| 0.0).
- Fit checks: no held protein in any fit. PLS component 0 has held-out AUC 0.86–0.89; it
  is sign-flipped on k=1, which is harmless.

### 11b — SCORED ONCE (run `fp1`)
CV-frozen thresholds: `pls64_53` 0.3551, `pls64_only` 0.4026.

| per-protein AUPRC | d=2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| CB-full, 53 + d | 0.473 | 0.480 | 0.480 | 0.470 | 0.472 | 0.470 | 0.470 |
| CB-full, d only | 0.444 | 0.441 | 0.437 | 0.427 | 0.428 | 0.429 | 0.430 |
| 184, 53 + d | 0.505 | 0.516 | 0.512 | 0.503 | 0.504 | 0.503 | 0.503 |
| 184, d only | 0.474 | 0.473 | 0.469 | 0.458 | 0.462 | 0.462 | 0.462 |

- Primary, `pls64_53 − ft117`: −0.016 [−0.027, −0.005]. **Does not pass**; PLS is
  significantly worse.
- Secondary, `pls64_only − ft64only`: −0.022 [−0.032, −0.012]. **Does not pass**.
- Context: `pls64_53 − control` +0.031 [+0.014, +0.049]; `pls64_53 − score118` −0.015
  [−0.026, −0.005].
- The pre-registered expectation (null or negative) held. Write-up: RESULTS § 13.9.

### Final comparison figure, updated 2026-10-09
- Rows added: `pls117` = `pls64_53` and `pls64only` = `pls64_only`, each at its own
  step-11 CV threshold. The gate also checks their recorded step-11 values.
- On the 184 proteins, per protein vs the released model:
  - PLS 53+64: AUC +0.021, AUPRC +0.022, both significant.
  - PLS 64-only: AUPRC −0.018, significant.
- New figure `scripts/pls_dims_figure.py` → `../Stacked_Figures/pls_dimension_sweep.{pdf,png}`:
  per-protein and pooled AUPRC vs d on the 184 proteins, with step 10's PCA and the released
  model as references.

Cost of step 11: about $3 (PLS CPU about $0.3; 89 T4 jobs, 4.6 GPU-hours, about $2.7).


## Step 12 — context-draw variance of ft117 (12 seeds), 2026-10-14

RESULTS § 4.6 redone for the best model. Only `build_context`'s seed varies (which
non-cryptic residues fill each fold-k context, and their order). The fine-tuned models,
embeddings, PCA, sub-fold split and TabICL's `random_state` are fixed. Each seed reruns the
whole protocol: 16 sub-fold jobs give its own threshold (median of four), and 4 test jobs are
averaged. Seed 0 is run `fe1`; seeds 1–11 are run `fe1seeds` (`deploy/ftemb_stack.py --stage
seeds`, 220 T4 jobs, 7.65 GPU-h of scoring, about $4.5). Report: `scripts/ftemb_seed_report.py`
→ `step12_seed_variance.json`.

- Gate: a seed-0 job rerun on the redeployed harness matches `fe1` on 18,688 of 18,694 scores,
  max |Δp| 2.2e-4, every metric equal to 6 decimals (GPU float noise, not bit-exact as before).
  The report reproduces `step10_result.json` exactly from the `fe1` files.
- Contexts per seed stay at 30,309–31,128 rows: the 120-per-protein cap binds, so every draw
  takes 120 negatives from each protein and only *which* 120 changes.
- Overlap (contexts rebuilt locally, sizes checked against every job's `n_context`): a seed's
  four fold contexts hold 106,923 negatives; two seeds share 51,751 on average (Jaccard
  31.9 %, sd 0.1 %), the most-similar pair (seeds 6, 7) 52,115 (32.2 %). TabICL-117's 50 k
  draws shared ~8 %. The 12 draws together cover 276,089 of 288,621 training negatives
  (95.7 %); 9,548 appear in all 12.

| CB-full (231), own threshold per seed | mean | sd | min – max | seed 0 |
|---|---|---|---|---|
| per-protein AUC | 0.8727 | 0.0004 | 0.8720 – 0.8734 | 0.8722 |
| per-protein AUPRC | 0.4856 | 0.0012 | 0.4836 – 0.4871 | 0.4859 |
| pooled AUC | 0.9030 | 0.0002 | 0.9028 – 0.9034 | 0.9030 |
| pooled AUPRC | 0.4210 | 0.0018 | 0.4185 – 0.4242 | 0.4235 |
| TPR | 0.5163 | 0.0035 | 0.5105 – 0.5222 | 0.5148 |
| FPR | 0.0467 | 0.0005 | 0.0461 – 0.0480 | 0.0463 |
| MCC | 0.3977 | 0.0021 | 0.3927 – 0.4004 | 0.3976 |
| threshold | 0.3414 | 0.0021 | 0.3376 – 0.3451 | 0.3407 |

| CB-P2RANK-apo (184), own threshold | mean | sd | min – max | seed 0 |
|---|---|---|---|---|
| per-protein AUPRC | 0.5196 | 0.0015 | 0.5161 – 0.5222 | 0.5199 |
| pooled AUC | 0.9059 | 0.0002 | 0.9056 – 0.9062 | 0.9058 |
| pooled AUPRC | 0.4847 | 0.0023 | 0.4824 – 0.4897 | 0.4877 |
| MCC | 0.4266 | 0.0025 | 0.4207 – 0.4297 | 0.4256 |

- The draw is a small source of variance, as for TabICL-117 (per-protein AUPRC sd 0.0012 vs
  0.0015). Seed 0 is ordinary on per-protein metrics (rank 6–7 of 12) but second-best of 12
  on pooled AUPRC, about 1.4 sd above the mean (CB-full 0.4235 vs 0.4210; 184: 0.4877 vs
  0.4847). Quote the 12-seed mean for pooled AUPRC where it matters.
- Keeping seed 0's threshold (0.341) instead of re-choosing it changes nothing material
  (CB-full TPR 0.5169 ± 0.0023, MCC 0.3979 ± 0.0020).
- Unchanged conclusions: every step-10 difference called a tie is still within the draw noise,
  and ft117 − control (+0.048) is ~40× the draw sd.

## Step 13 — PAM's feature-family ablation, 2026-10-25

Nested arms on PAM's own protocol (fold-k contexts, seed 0, CV-frozen threshold, test = mean
of the four cross-fitted predictions): `fam33` geometry/physico-chemistry → `fam47` + 14
ensemble pocket → `fam53` + 6 conservation (no pLM) → `ft69` / `ft85` / `ft117` + the first
16 / 32 / 64 PCs of the **fine-tuned** (full recipe, clean, fold) ESM-2 3B embedding, not the
frozen one. `ft117` is PAM (`fe1`, reused). Arms in `cryptic_tabicl/feature_sets.py`, stages
`abgate` / `ablation` in `deploy/ftemb_stack.py` (run `ab1`, 101 T4 jobs, 2.51 GPU-h),
report `scripts/ftemb_ablation_report.py`, scored on CryptoBench's unit (CB-full 262 chains,
CB-P2RANK-184).

### 13a — gate PASSED
`ft117` k0 test rerun on the redeployed harness equals `fe1` (max |Δp| 0.0, n_context 30,309).
Feature counts 33/47/53/69/85/117 checked on the real table; tests pass (200, 1 skipped).

### 13b — SCORED ONCE (run `ab1`), CIs 2026-10-26
Thresholds (`step13_thresholds.json`, from CV sub-folds before test was read): 0.352, 0.330,
0.346, 0.345, 0.348, 0.341 (ft117 = step 10). Point estimates in `step13_ablation_points.json`.

| CB-full 262, per protein / pooled | AUC | AUPRC | MCC | pooled AUC | pooled AUPRC |
|---|---|---|---|---|---|
| fam33 (33) | 0.826 | 0.340 | 0.271 | 0.831 | 0.269 |
| fam47 (47) | 0.836 | 0.342 | 0.263 | 0.849 | 0.273 |
| fam53 (53) | 0.843 | 0.381 | 0.292 | 0.864 | 0.317 |
| ft69 (69)  | 0.866 | 0.471 | 0.374 | 0.904 | 0.434 |
| ft85 (85)  | 0.866 | 0.472 | 0.380 | 0.905 | 0.435 |
| ft117 (117)| 0.864 | 0.469 | 0.379 | 0.905 | 0.434 |

- Reading so far: the ensemble block adds AUC but no AUPRC; conservation adds +0.04 AUPRC;
  the first 16 fine-tuned PCs add +0.09 and carry the whole pLM gain; 32 and 64 add nothing.
- Paired bootstrap (10k structure resamples, seed 0; `--workers 4`, draws drawn in order
  first, so identical to a serial run: checked on 40 draws) → `step13_ablation.json`,
  `step13_ablation.log`. Steps, CB-full per protein, AUPRC [95 % CI]:
  ensemble +0.002 [−0.013, +0.018]; conservation +0.039 [+0.023, +0.056];
  16 FT PCs +0.090 [+0.069, +0.111]; 32 PCs +0.001 [−0.004, +0.005];
  64 PCs −0.003 [−0.008, +0.002]. Pooled AUC: ensemble +0.017 [+0.010, +0.025] is the
  ensemble block's only significant gain; same pattern on the 184.
- ft69 and ft85 tie PAM on every ranking metric (|Δ| ≤ 0.003, all CIs span 0); ft69's
  CB-full per-protein TPR (−0.008) and MCC (−0.005) are just below PAM's, at the edge of
  their CIs. PAM keeps 64 PCs; 16 would do.
