# Extension: NVIDIA Kumo Tabular on PAM's features

PAM (`ft117`) is this project's best model: 53 apo-structure features plus 64 PCs of the clean
fine-tuned ESM-2 3B embedding, classified by zero-shot TabICLv2. This extension swaps TabICLv2 for
**NVIDIA Kumo Tabular** (large, ~214M parameters, `structured-data-models` 0.2.1, released September
2026). Everything else is held fixed: the 117 columns, the four cross-fitted fold contexts drawn at
seed 0, the threshold rule and the 231-protein test set. That makes the comparison paired, protein
by protein and context by context.

The plan, gates, pre-registration and log are in [`PLAN.md`](PLAN.md).

## Result (run `ku1`, pre-registered in `frozen.json`, scored once)

Kumo Tabular is **significantly worse than PAM, by a small margin.**

| CB-full (231), per protein | Kumo Tabular | PAM (TabICLv2) | Kumo − PAM, paired 95% CI |
|---|---|---|---|
| AUPRC | 0.477 | **0.486** | **−0.009 [−0.016, −0.002]** |
| AUC | 0.866 | **0.872** | −0.006 [−0.010, −0.003] |
| pooled AUPRC | 0.416 | 0.424 | −0.007 [−0.021, +0.007] |
| pooled AUC | 0.900 | 0.903 | −0.003 [−0.007, +0.001] |
| TPR / FPR (each model's CV-frozen threshold) | 0.475 / 0.041 | 0.515 / 0.046 | |
| MCC | 0.389 | 0.398 | |

- **Same picture on the 184 single-chain structures.** CB-P2RANK-apo per-protein AUPRC is
  0.512 vs 0.520 (−0.008 [−0.016, −0.000]).
- **Kumo behind in three of four contexts.** Single-context per-protein AUPRC, k = 0..3: Kumo
  0.434 / 0.449 / 0.448 / 0.436, PAM 0.451 / 0.447 / 0.455 / 0.450. Kumo trails in three and
  ties in one.
- **Blending adds nothing.** A fixed 50/50 within-protein rank blend of the two ties PAM
  (−0.001 [−0.004, +0.003]), so Kumo carries no signal PAM lacks.
- **Expectation partly right.** The pre-registered expectation was a tie within ±0.01. The size
  was right; the CI excludes zero.

## Why the loss is the model, not the plumbing

The same code path was first run with sdm's **TabICLv2**, which uses PAM's exact checkpoint
(sha256 `bdc7dbd5…`), on fold context 0 against test. It reproduced PAM's tabicl 2.1.1 run to
**+0.0012** per-protein AUPRC (Spearman 0.987), with identical contexts and columns. Kumo's two
pilot runs were bit-identical.

What remains is Kumo on this table. Three things may contribute:

- 117 columns is past its 100-column pretraining maximum.
- The classes are imbalanced, at about 1:7 in the context.
- The protocol (context sampling, the 120-negatives-per-protein cap) was tuned around TabICL.

None of these was tested separately.

## Not covered

- **One context draw.** Only seed 0 was run, paired with PAM's own draw. PAM's 12-seed SD is
  0.0012, and the gap is about 7× that, but Kumo's own seed variance was not measured.
- **One configuration.** Only `large` with 16 estimators was run. Small and medium, other
  ensemble sizes, Kumo fine-tuning and Kumo-specific context sizes were not tried.
- **CV exposure before pre-registration.** The determinism pilot showed one CV block, k0/sub0
  (Kumo 0.393 vs 0.401), before `frozen.json` was written. This is disclosed in that file. No
  test score existed at the time.

## Files

| file | |
|---|---|
| `kumo_model.py` | `KumoTabularPredictor`, a drop-in for `CrypticSitePredictor` (fit / predict_proba / predict_by_protein / config) |
| `kumo_stack.py` | Modal harness (A100-80GB): stages `smoke`, `harness`, `gate`, `cv`, `test`; `select` copied verbatim from `deploy/ftemb_stack.py` |
| `kumo_report.py` | `--gates` → `gates.json`; default → `kumo_thresholds.json` (from CV, before test) then `kumo_result.json` |
| `frozen.json` | pre-registration (sha256 pinned in the report) |
| `smoke.json`, `gates.json`, `kumo_thresholds.json`, `kumo_result.json` | records |
| `tests/test_kumo_extension.py` | local tests, no GPU needed: `select` matches fe1's, all 20 contexts and queries reproduce fe1's, and the model sees fe1's 117 columns |

## Reproduce

```
pytest extension/tests -q
modal run    extension/kumo_stack.py --stage smoke
modal deploy extension/kumo_stack.py
modal run    extension/kumo_stack.py --stage harness --run-id ku1
modal run    extension/kumo_stack.py --stage gate    --run-id ku1
modal run    extension/kumo_stack.py --stage cv      --run-id ku1
modal run    extension/kumo_stack.py --stage test    --run-id ku1
modal run    extension/kumo_stack.py --collect       --run-id ku1
python extension/kumo_report.py --gates --run-id ku1
python extension/kumo_report.py --run-id ku1
```

Cost of the whole run: about $0.70 of A100 time. The 20 scoring jobs took 976 s, about 88 s per
test job, with peak memory 5.7 GB.
