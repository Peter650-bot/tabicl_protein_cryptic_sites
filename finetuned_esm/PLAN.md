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
