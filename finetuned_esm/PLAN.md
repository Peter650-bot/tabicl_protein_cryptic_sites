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
