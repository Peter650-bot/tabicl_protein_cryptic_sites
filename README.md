# PAM: cryptic binding site prediction from apo structures

Per-residue cryptic binding site prediction on [CryptoBench](https://osf.io/pz4a9/). Given an
**apo** (unbound) protein structure, PAM predicts for every residue whether it belongs to a
cryptic site: a site that is closed or malformed in the unbound state and opens when a ligand
binds.

## The model

PAM classifies each residue with **zero-shot TabICLv2**, an in-context tabular foundation
model. TabICLv2 is not trained or fine-tuned here. It reads a labelled context table from the
training folds and scores the query residues in one forward pass. Each residue has 117
features:

- **53 structural features** computed from the apo structure only:
  - 33 geometry and physicochemistry features (packing, SASA, depth, protrusion, convexity,
    half-sphere exposure, B-factor, secondary structure, hydropathy, charge), each with its
    8 Å neighbourhood mean;
  - 14 pocket features from 24 ANM normal-mode conformations scored with fpocket;
  - 6 PDBe conservation features.
- **64 PCs of a fine-tuned ESM-2 3B's last hidden state.** The fine-tuned model is Škrhák
  et al.'s full recipe (multitask plus LIGYSIS pre-training), retrained here once per
  training fold. Each fold model excludes every LIGYSIS protein ≥ 40 % identical to its
  held-out fold or to the test set.

**Cross-fitting.** A fine-tuned model's embeddings of its own training proteins encode their
labels. So fold model *k* embeds only fold *k*, which it never saw and which becomes TabICL's
context, and the test set. The 64 PCs are fit on fold *k* only, and the four test predictions
(*k* = 0..3) are averaged.

The decision threshold (0.341) is chosen on the training folds, for a mean per-protein FPR of
0.05, and frozen before the test set is scored. No holo structure, and no quantity derived
from one, is used at prediction time.

## Results

PAM is compared with the strongest published model on CryptoBench: **Škrhák et al.'s
fine-tuned ESM-2 3B** (*Hidden in protein sequences*, ACM BCB 2025; *Seq2Pocket*, bioRxiv
2026). Its released checkpoint was rerun here, and it reproduces their published pooled
numbers on the 184 single-chain structures (AUC 0.8935, AUPRC 0.4751, MCC 0.4719).

How to read the tables:

- **per protein** is CryptoBench's convention: each metric is computed within each protein,
  then averaged. **pooled** is the convention of Škrhák et al.'s papers: all residues form
  one vector.
- **Thresholds:** PAM uses its CV-frozen threshold. Their model uses their own threshold of
  0.7, which they chose on their test set.
- **Significance:** "significant" below means the paired 95 % bootstrap CI of the difference
  excludes zero (2,000 protein resamples; all CIs in `finetuned_esm/readme_comparison.json`).
- **Leakage in their model:** their LIGYSIS pre-training set contains 28 CryptoBench test
  proteins, so their model saw binding-site labels for part of the test set. PAM's fine-tuned
  models exclude them.
- **Units:** the three tables below score both models per test record (231 records for
  222 structures). CryptoBench's own published numbers use a different unit, so they are
  compared separately, on that unit, in the last table.

### CB-full: all 231 test proteins

76,753 residues, 4,053 cryptic.

| model | convention | AUC | AUPRC | ACC | TPR | FPR | MCC | F1 (weighted) |
|---|---|---|---|---|---|---|---|---|
| **PAM** | per protein | **0.872** | **0.486** | **0.924** | **0.515** | **0.046** | **0.398** | **0.926** |
| Škrhák et al. fine-tuned ESM-2 | per protein | 0.839 | 0.446 | 0.918 | 0.496 | 0.059 | 0.381 | 0.924 |
| **PAM** | pooled | **0.903** | **0.424** | **0.929** | **0.577** | **0.051** | **0.437** | **0.936** |
| Škrhák et al. fine-tuned ESM-2 | pooled | 0.882 | 0.397 | 0.927 | 0.556 | 0.052 | 0.418 | 0.934 |

### CB-P2RANK-apo: the 184 single-chain structures

55,567 residues, 3,157 cryptic.

| model | convention | AUC | AUPRC | ACC | TPR | FPR | MCC | F1 (weighted) |
|---|---|---|---|---|---|---|---|---|
| **PAM** | per protein | **0.874** | **0.520** | **0.929** | **0.533** | **0.043** | **0.426** | 0.930 |
| Škrhák et al. fine-tuned ESM-2 | per protein | 0.848 | 0.480 | 0.927 | 0.509 | 0.050 | 0.408 | 0.930 |
| **PAM** | pooled | **0.906** | **0.488** | 0.935 | **0.598** | 0.045 | **0.483** | 0.939 |
| Škrhák et al. fine-tuned ESM-2 | pooled | 0.893 | 0.475 | 0.935 | 0.576 | **0.043** | 0.472 | 0.939 |

### CB-full minus CB-P2RANK-apo: the 47 multi-chain structures

21,186 residues, 896 cryptic.

| model | convention | AUC | AUPRC | ACC | TPR | FPR | MCC | F1 (weighted) |
|---|---|---|---|---|---|---|---|---|
| **PAM** | per protein | **0.866** | **0.353** | **0.902** | 0.444 | **0.061** | **0.288** | **0.906** |
| Škrhák et al. fine-tuned ESM-2 | per protein | 0.805 | 0.310 | 0.883 | 0.444 | 0.095 | 0.274 | 0.901 |
| **PAM** | pooled | **0.895** | **0.249** | **0.915** | **0.503** | **0.067** | **0.314** | **0.928** |
| Škrhák et al. fine-tuned ESM-2 | pooled | 0.844 | 0.197 | 0.906 | 0.485 | 0.075 | 0.285 | 0.922 |

- **PAM is ahead per protein on all three sets.** The AUPRC gain is significant on every set:
  +0.040 on CB-full, +0.039 on the 184, and +0.043 on the 47. The AUC gain is significant too,
  and PAM has a lower FPR at an equal or higher TPR.
- **The pooled AUC gain is significant on every set.** On the 184 single-chain structures
  that their papers report, PAM leads on pooled AUPRC (0.488 vs 0.475), but that difference
  is not significant.
- **The 47 multi-chain structures are the hardest set for both models.** That is also where
  the gap is largest: PAM's per-protein AUC is +0.061 higher, and its FPR is 0.061 against
  their 0.095.
- PAM's per-protein AUPRC varies very little with the random draw of its context: 0.4856 ±
  0.0012 over 12 seeds.

### Against CryptoBench's published baselines

CryptoBench's published per-protein numbers average over **chains**. Its evaluation code
(`skrhakv/apolo`, branch `cryptobench-v2`) merges each structure's cryptic pockets, splits
the structure into chains, skips chains with no cryptic residue, and drops 7ndl. PAM is
rescored on exactly those units here (`scripts/published_comparison.py`). pLM-NN,
PocketMiner and P2Rank are the published values (CryptoBench Table 5), on the subsets each
tool could run on.

| set | model | AUC | AUPRC | ACC | TPR | FPR | MCC | F1 (weighted) |
|---|---|---|---|---|---|---|---|---|
| CB-full, 262 chains | **PAM** | 0.864 | **0.469** | 0.921 | 0.501 | 0.048 | 0.379 | 0.924 |
| | pLM-NN | 0.86 | 0.36 | 0.93 | 0.48 | 0.05 | 0.39 | 0.92 |
| CB-PM, 162 structures | **PAM** | 0.877 | **0.528** | 0.929 | 0.534 | 0.043 | 0.430 | 0.930 |
| | pLM-NN | 0.88 | 0.43 | 0.93 | 0.52 | 0.04 | 0.44 | 0.93 |
| | PocketMiner | 0.76 | 0.19 | 0.82 | 0.51 | 0.16 | 0.22 | 0.78 |
| CB-P2RANK-apo, 183 structures | **PAM** | 0.876 | **0.523** | 0.930 | 0.536 | 0.042 | 0.428 | 0.930 |
| | pLM-NN | 0.88 | 0.42 | 0.93 | 0.51 | 0.04 | 0.43 | 0.93 |
| | P2Rank | 0.81 | 0.21 | 0.85 | 0.62 | 0.14 | 0.27 | 0.81 |

- **Against pLM-NN, PAM's AUPRC lead is the only gap outside PAM's 95 % CI**, on every set
  (structure-level bootstrap, 10,000 draws). Its other metrics tie pLM-NN, except ACC on
  CB-full, where pLM-NN is higher (0.93 vs 0.921).
- PAM is ahead of PocketMiner and P2Rank on every metric except P2Rank's TPR, which comes
  at an FPR of 0.14.
- CB-PM is the single-chain structures minus the 21 on which PocketMiner failed (from
  CryptoBench's own logs) and minus 7ndl. PocketMiner was scored with 7ndl (163 structures)
  and P2Rank on all 184; PAM moves by about 0.003 on those sets.

## Reproduce

### Requirements

- Python ≥ 3.10 with `numpy<2`, `pandas`, `pyarrow`, `scipy`, `scikit-learn`, `gemmi`, `requests`
  and `pytest`.
- A [Modal](https://modal.com) account (`pip install modal && modal setup`). Every GPU step
  runs there, and each script builds its own image (torch 2.6.0, `tabicl==2.1.1`,
  `transformers`, fpocket).
- Steps that rebuild Škrhák et al.'s inputs also need `biotite==0.41.2` and Biopython on
  `PYTHONPATH` (see each script's docstring).
- Data is staged in `~/cryptobench_stage/` (override with `CRYPTOBENCH_STAGE`), and
  intermediate tables go to `build/`.

### 1. The 53 structural features

```bash
python scripts/fetch_cryptobench.py                              # annotations and folds
python -m cryptic_tabicl.build_table --out build/residues.parquet    # 33 geometry features
modal run --detach deploy/ensemble_modal.py                      # 24 ANM conformations x fpocket
python scripts/merge_ensemble.py                                 # + 14 ensemble-pocket features
python scripts/fetch_conservation.py                             # PDBe conservation (cached)
python scripts/merge_conservation.py                             # + 6 conservation features = 53
modal run --detach deploy/esm_modal.py                           # frozen ESM-2 (the base table's
python scripts/merge_esm.py --n-components 64                    #   column order; PAM drops these)
```

### 2. The fine-tuned ESM-2 fold models

Škrhák et al.'s full recipe (multitask plus LIGYSIS pre-training), with their code vendored
in `deploy/s2p_models.py` and `deploy/finetuning_utils.py`. It is retrained once per
training fold *k*, without fold *k*, and without any LIGYSIS protein that is ≥ 40 % identical
to fold *k* or to the test set. Each fold model embeds fold *k* and the test set, then fits
64 PCs on fold *k* only.

```bash
PYTHONPATH=<biotite 0.41.2 + numpy 1.26>:. python scripts/s2p_build_inputs.py   # their sequences
PYTHONPATH=<biotite 0.41.2 + numpy 1.26>:. python scripts/s2p_build_train.py    # their per-fold training files
modal deploy deploy/s2p_finetune_modal.py
modal run deploy/s2p_finetune_modal.py --fetch && modal run deploy/s2p_finetune_modal.py --extract
modal run deploy/s2p_finetune_modal.py --recipe full --variant cleantest \
    --jobs 0,1,2,3 --run-id s2p_full_cleantest_20261008                 # 4 x A100, about $6
modal volume get cryptobench-s2p ft/s2p_full_cleantest_20261008/ build/s2p_ft/
python scripts/ftemb_build_table.py --run-id s2p_full_cleantest_20261008
```

The LIGYSIS exclusion lists are committed in `finetuned_esm/ligysis_exclusions.json`
(produced by `scripts/s2p_ligysis_overlap.py`).

### 3. PAM: TabICLv2 on the cross-fitted contexts

```bash
modal deploy deploy/ftemb_stack.py
modal run deploy/ftemb_stack.py --stage cv   --run-id fe1    # 4 folds x 4 sub-folds: thresholds
modal run deploy/ftemb_stack.py --stage test --run-id fe1    # 4 fold contexts -> test
modal run deploy/ftemb_stack.py --collect    --run-id fe1    # -> build/ftemb_stack/fe1/
python scripts/ftemb_report.py --run-id fe1                  # threshold frozen from CV, then test
```

### 4. Their released model, and the comparison above

Their checkpoint (`cbs-model.pt` from Hugging Face `skrhakv/seq2pocket`) is pinned by commit
and sha256 and scored unchanged on every chain:

```bash
modal deploy deploy/finetuned_esm_modal.py
modal run deploy/finetuned_esm_modal.py::score --run-id s2p_20261006
modal volume get cryptobench-s2p scores/s2p_20261006.parquet build/
python scripts/s2p_reproduce.py          # reproduces their published numbers, keys scores to our rows
python scripts/readme_comparison.py      # -> finetuned_esm/readme_comparison.json
python scripts/published_comparison.py   # CryptoBench's units -> finetuned_esm/published_comparison.json
pytest tests/ extension/tests -q
```

## Repository

| path | contents |
|---|---|
| `cryptic_tabicl/` | features, sampling, TabICL wrapper, per-protein evaluation |
| `deploy/` | Modal jobs: fpocket ensembles, ESM-2, fine-tuning, TabICL scoring |
| `scripts/` | data fetching, table merges, reports |
| `finetuned_esm/` | the fine-tuned ESM-2 work: log (`PLAN.md`), pre-registrations, results |
| `extension/` | NVIDIA Kumo Tabular in place of TabICLv2 (worse: −0.009 AUPRC) |
| `tests/` | leakage guards, label logic, sampler and feature-set invariants |

Every other model variant, ablation and correction is recorded in [RESULTS.md](RESULTS.md).
