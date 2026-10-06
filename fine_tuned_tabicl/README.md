# Fine-tuned TabICLv2 (117 features)

This is a ceiling measurement, reported next to the zero-shot headline and not in place of it.
Nothing outside this folder was re-measured or overwritten.

Produced by `deploy/finetune_modal.py`, run id `ft_20260916`. The run used Modal
A100-80GB GPUs for 1.25 GPU-h (about $3.14).

## Result — CryptoBench test split, per-protein means

| model | subset | AUC | AUPRC | TPR | FPR | MCC | F1 (pos.) | F1 (weighted) |
|---|---|---|---|---|---|---|---|---|
| **TabICLv2 + ESM, fine-tuned** | CB-full (231) | **0.863** | **0.444** | **0.571** | 0.072 | **0.378** | 0.383 | 0.912 |
| TabICLv2 + ESM, zero-shot (paired control) | CB-full (231) | 0.862 | 0.441 | 0.559 | 0.071 | 0.374 | 0.379 | 0.912 |
| **TabICLv2 + ESM, fine-tuned** | CB-P2RANK-apo (184) | **0.862** | **0.472** | **0.583** | 0.070 | **0.397** | 0.403 | 0.915 |
| TabICLv2 + ESM, zero-shot (paired control) | CB-P2RANK-apo (184) | 0.861 | 0.468 | 0.569 | 0.068 | 0.391 | 0.398 | 0.914 |
| pLM-NN (published) | CB-full | 0.86 | 0.36 | 0.48 | 0.05 | 0.39 | — | 0.92 |

Frozen thresholds (median of the four CV folds):

| model | threshold | per-fold thresholds |
|---|---|---|
| fine-tuned | 0.5627 | 0.577, 0.544, 0.572, 0.553 |
| control | 0.5704 | 0.590, 0.544, 0.582, 0.559 |

Pooled over all residues:

| model | AUC | AUPRC |
|---|---|---|
| fine-tuned | 0.887 | 0.367 |
| control | 0.885 | 0.361 |

**The control reproduces the published zero-shot run.** It scores AUPRC 0.4412 against the
headline's 0.4413 (`build/modal_result_esm64.json`), with identical TPR, FPR and MCC to 3 dp.
It ran on tabicl 2.2.0 and the headline ran on 2.1.1, so the library upgrade moves nothing.
Every difference in the table therefore comes from the fine-tuned weights.

## Is the gain real?

It is barely there. The table below is a paired bootstrap over the 231 test proteins
(5,000 resamples, fine-tuned minus control):

| metric | Δ | 95% CI | proteins better / worse |
|---|---|---|---|
| AUC | +0.0014 | [−0.0006, +0.0035] | 130 / 101 |
| AUPRC | +0.0024 | [−0.0023, +0.0070] | 130 / 101 |
| MCC | +0.0045 | [−0.0014, +0.0113] | 100 / 97 |
| TPR | +0.0113 | [+0.0040, +0.0189] | 59 / 30 |

- **Ranking metrics (AUC, AUPRC):** the confidence intervals include zero.
- **TPR:** the rise is significant, but it comes partly from a slightly lower frozen threshold (0.563 vs 0.570). FPR rose with it, by +0.0014.
- **Scale of the gain:** the AUPRC gain (+0.002) is about the size of the zero-shot context-draw noise (sd 0.0015 over 12 seeds; RESULTS.md). It is also about 40× smaller than the 33 → 117 feature gain.
- **CV folds:**

  | | fold 0 | fold 1 | fold 2 | fold 3 |
  |---|---|---|---|---|
  | AUPRC change | +0.003 | +0.002 | +0.003 | −0.002 |
  | AUC change | ≤ +0.002 | ≤ +0.002 | ≤ +0.002 | ≤ +0.002 |

- **Strata:**

  | stratum | AUC before → after |
  |---|---|
  | already-open-pocket (hard) | 0.723 → 0.721 |
  | "elsewhere" | 0.882 → 0.884 |

  Fine-tuning does nothing for the hard stratum, which is the project's open problem.

**Reading.** Adapting the weights to this table buys roughly as much as re-drawing the
context. The zero-shot model was already close to what this recipe can extract from these
117 features. Fine-tuning gives up the no-training property, and in exchange it does not
change any conclusion in RESULTS.md.

## Training curves (`training.json`)

Validation AUC is on the early-stopping proteins, and the model is saved at its best epoch.

| job | trained on | epochs run | val AUC pretrained → best | best epoch | wall-clock (s) |
|---|---|---|---|---|---|
| final (scores test) | all 4 train folds | 15 | 0.8694 → 0.8713 | 7 | 760 |
| CV, fold 0 held out | folds 1–3 | 21 | 0.8694 → 0.8739 | 13 | 938 |
| CV, fold 1 held out | folds 0, 2, 3 | 23 | 0.8638 → 0.8660 | 15 | 964 |
| CV, fold 2 held out | folds 0, 1, 3 | 23 | 0.8732 → 0.8767 | 15 | 1136 |
| CV, fold 3 held out | folds 0–2 | 12 | 0.8728 → 0.8730 | 4 | 718 |

Early stopping (patience 8) ended every job well before the 30-epoch limit. Training loss
falls steadily in every job, but validation AUC moves by only 0.0002–0.0045. That is the
pattern of a model fitting its fine-tuning chunks rather than learning something that
transfers to new proteins.

## Protocol

The zero-shot protocol is kept everywhere except for which weights are loaded.

1. **Five jobs, each with its own fine-tune.**
   - *CV job for train fold h:* fine-tunes on the other three train folds and scores fold h.
   - *Final job:* fine-tunes on all four train folds and scores the test split.
   - No set of weights ever sees the proteins it scores.
2. **Early-stopping set.** 8% of the *proteins* in each job's training pool are held out, which
   is 72–76 proteins. The library default holds out 10% of *rows*, which would split proteins
   across both sides.
3. **Fine-tuning data.** The rest of the pool goes through `sampling.build_context`
   (50k rows: all positives plus proportional negatives), so training sees the class
   balance that inference sees.
4. **Hyperparameters.** These are the `FinetunedTabICLClassifier` defaults (tabicl 2.2.0):
   - optimiser: AdamW, learning rate 1e-5, cosine schedule with warmup
   - `max_data_size` 10,000, i.e. five 8k-context / 2k-query chunks per epoch
   - ensembles: 2 members for training, 2 for validation
   - early stopping on validation ROC-AUC, patience 8, up to 30 epochs
   - FP16 automatic mixed precision
   - changed from the defaults: `random_state=0`, and a 30-minute training cap (never reached)
5. **Scoring.** The job's full pool goes through the same seed-0 50k `build_context` as the
   zero-shot run. The target is then scored with `CrypticSitePredictor(n_estimators=8)`
   twice: once with `model_path=best.ckpt` and once with the pretrained checkpoint (the
   control).
6. **Threshold.** The threshold is set on the CV folds only, at a mean per-protein FPR of 0.05
   (median of the four folds). It is then applied unchanged to the test split. CB-P2RANK-apo
   is rebuilt as in `scripts/f1_by_subset.py`.

**Hardware.** A 10k-row chunk with 117 features peaks at 66.6 GB during training, and an
L4 (22 GB) ran out of memory. Checkpoints (`best.ckpt` per job) are kept in the Modal volume
`cryptobench-finetune-results` under `ft_20260916/`.

## Files

| file | contents |
|---|---|
| `result_finetuned_esm64.json` | Same schema as `build/modal_result_*.json`, plus `result_cb_p2rank_apo` and `cv_thresholds`. Includes per-residue test scores. |
| `result_pretrained_control_esm64.json` | The paired zero-shot control, same schema. |
| `summary.csv` | The headline table above. |
| `training.json` | Per-job settings, timings, peak GPU memory and per-epoch history. |

## Reproduce

```bash
modal deploy deploy/finetune_modal.py
modal run deploy/finetune_modal.py --spawn --held all --run-id <id>
modal volume ls cryptobench-finetune-results <id>      # wait for 5 json files
modal run deploy/finetune_modal.py --collect --run-id <id>
python3 figures/make_benchmark_figure.py
```
