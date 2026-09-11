# cryptic_tabicl

Per-residue cryptic binding site prediction on [CryptoBench](https://osf.io/pz4a9/),
using **zero-shot TabICLv2** as the classifier.

Given an **apo** (unbound) protein structure, predict for every residue whether it
belongs to a cryptic binding site — a site that is malformed or inaccessible in the
unbound state but opens on ligand binding.

This project is self-contained and shares no code with the PPO/HOP project.

Full measurements, ablations and confidence intervals: **[RESULTS.md](RESULTS.md)**.
This file is the summary; RESULTS.md is the record.

---

## The task

| | |
|---|---|
| Unit of prediction | one **residue** of one apo chain |
| Label | residue appears in `apo_pocket_selection` for any holo partner |
| Rows | 381,704 residues across 1,136 apo structures |
| Prevalence | 5.34% positive (5.28% on the test split) |
| Features | 117 = 33 geometric + 14 ensemble-pocket + 6 conservation + 64 ESM-2 PCs |
| Model | TabICLv2, zero-shot (no fitting, no fine-tuning) |
| Splits | CryptoBench's own sequence-identity folds |

## Results

Test set (231 proteins, 76,753 residues), metrics computed per protein then averaged,
matching CryptoBench's `Protein.py`. The decision threshold is selected on the training
folds and frozen before the test set is touched.

| metric | 117 feat (+ESM) | 53 feat (no pLM) | pLM-NN (published, CB-full) |
|---|---|---|---|
| AUC | 0.862 | 0.848 | 0.86 |
| AUPRC | **0.441** | 0.388 | 0.36 |
| TPR | **0.559** | 0.515 | 0.48 |
| FPR | 0.071 | 0.073 | **0.05** |
| MCC | 0.374 | 0.336 | **0.39** |
| F1 (weighted — the published convention) | 0.912 | 0.906 | **0.92** |
| F1 (positive class) | 0.379 | 0.345 | — |

No training of any kind: the labelled context is a single forward pass, and the ESM
features are frozen embeddings, not a fine-tuned network. No holo structure at
prediction time — and, since the 2026-09-10 corrections, no quantity *derived* from one
either (RESULTS.md § 9.1).

**Two wins, two ties, two losses.** Against pLM-NN on CB-full we beat it on AUPRC and
TPR by margins that survive resampling the test proteins; we tie on AUC and MCC; and we
are genuinely worse on FPR and weighted F1 — we run at a looser operating point. Bolded
cells above are the winner of each row.

**The 53-feature row is no longer a win.** An earlier version of this file claimed it
"beats pLM-NN on CB-full using no protein language model at all". That was true only
while a holo-derived feature was reaching the model. Without it, AUPRC 0.388
[0.356, 0.420] contains pLM-NN's 0.36 — a tie — and MCC falls below theirs. Only the
117-feature row clears the benchmark, and only on two metrics.

Only the CB-full row is comparable: our evaluation covers all 231 test proteins, whereas
the published P2Rank and PocketMiner figures are on subsets those tools could run on.
CB-P2RANK-apo (the 184 single-chain test structures) *is* reconstructible and we report
it too — see RESULTS.md § 2. CB-PM is not: it is defined by which structures PocketMiner
happened to run on.

### Which margins survive resampling

Each metric is a mean over per-protein values, so a CI comes free — resample the 231
test proteins with replacement and re-average (`python scripts/bootstrap_test.py`,
10,000 draws). Doing that changes what may be claimed:

- **AUPRC and TPR are the result, and only with ESM.** pLM-NN's published values sit
  below our interval on both metrics and both subsets — for the 117-feature run only.
- **The AUC lead is gone.** 0.862 [0.840, 0.882] contains pLM-NN's 0.86. It survived
  resampling in the previous, leaky measurement; it does not now. Report AUC as a tie.
- **The "no language model needed" claim is withdrawn.** At 53 features AUPRC
  0.388 [0.356, 0.420] contains 0.36, TPR contains 0.48, and MCC 0.336 sits *below*
  their 0.39. Without ESM-2 this work beats the benchmark on nothing.
- **Four gaps run against us significantly**: FPR on both subsets, weighted F1 on both,
  MCC on three cells of four, and AUC on CB-P2RANK-apo without ESM.

The FPR overshoot is now cleanly attributable. The threshold is swept to a **mean
per-protein FPR of 0.05** — the statistic CryptoBench publishes — and hits exactly 0.0500
on every CV fold; the 0.071 on test is train→test shift across sequence-identity folds.
It previously targeted the *pooled* FPR, a quantity this project never reports
(RESULTS.md § 9.3).

The context draw is a much smaller source of uncertainty than the test set: over 12
TabICL seeds on the corrected table the draw moves AUC by sd 0.0006 and AUPRC by sd
0.0015 — the test-set interval is ~24× wider (RESULTS.md § 4.6).

Inner CV on the train folds, used only to freeze the threshold (at 0.5706 for the
117-feature run, 0.5667 for the 53-feature run):

| fold | 53 feat AUC / AUPRC | 117 feat AUC / AUPRC |
|---|---|---|
| train-fold-0 | 0.802 / 0.334 | 0.815 / 0.376 |
| train-fold-1 | 0.837 / 0.404 | 0.860 / 0.442 |
| train-fold-2 | 0.820 / 0.386 | 0.834 / 0.411 |
| train-fold-3 | 0.814 / 0.371 | 0.829 / 0.405 |

Fold spread is ~0.06 AUC, so no single fold carries the result.

Reproduce the full table with `python scripts/compare_variants.py`.

### Control: the same features with conventional models

Everything held fixed except the model — same table, same folds, same threshold rule,
same per-protein averaging:

| Method | features | context | AUC | AUPRC |
|---|---|---|---|---|
| **TabICLv2 zero-shot** | 117 | 50k subsample | **0.862** | **0.441** |
| LogisticRegression | 117 | 50k subsample | 0.856 | 0.417 |
| LogisticRegression | 117 | full 305k train | 0.856 | 0.416 |
| RandomForest | 117 | full 305k train | 0.837 | 0.381 |
| **TabICLv2 zero-shot** | 53 | 50k subsample | **0.848** | **0.388** |
| LogisticRegression | 53 | same 50k subsample | 0.831 | 0.362 |
| RandomForest | 53 | same 50k subsample | 0.829 | 0.358 |
| LogisticRegression | 53 | full 305k train | 0.829 | 0.358 |
| RandomForest | 53 | full 305k train | 0.824 | 0.346 |
| RSA alone (floor) | 1 | — | 0.499 | 0.089 |

Read honestly, this says three things:

1. **TabICL is ahead in every one of the six feature conditions**, without exception,
   including against models that see all 305k training rows. The within-condition spread
   is 0.027–0.061 AUPRC. That margin roughly **doubled** when the holo-derived feature
   was removed (it was 0.003–0.030): the leak flattered the sklearn controls more than it
   flattered TabICL, masking the in-context model's advantage.

2. **The features still carry most of the result, but less lopsidedly.** Holding model
   and sampler fixed, TabICL goes 33 → 47 → 53 → 69 → 85 → 117 features for AUPRC
   0.343 → 0.352 → 0.388 → 0.411 → 0.434 → 0.441. The feature effect (0.098) is still
   larger than the model effect (≤0.061), but the two are now much closer.

3. **The task is genuinely multivariate.** RSA alone scores 0.499 — chance. No single
   descriptor solves this.

An earlier version of this README concluded the opposite — that TabICL's context
ceiling cost real accuracy and a RandomForest on full data beat it. That reading was an
artifact of a sampler bug; see *The context ceiling that wasn't* below.

### Where it is weak

Stratifying the test set by whether a residue lines an already-open pocket
(117-feature run):

| stratum | residues | prevalence | AUC | AUPRC | TPR |
|---|---|---|---|---|---|
| bulk surface | 73,409 | 3.5% | 0.882 | 0.262 | 0.580 |
| lines a non-cryptic pocket | 3,344 | 43.9% | **0.723** | 0.646 | 0.663 |

**This split is only meaningful now.** `in_noncryptic_pocket` *defines* these strata,
and until the 2026-09-10 corrections it was also an input to the model — so the earlier
version of this table scored a model that had been told which stratum each residue was
in. It fired positive on **94.2 %** of the hard stratum, because the feature announced a
44 % base rate there. That is now 66.3 %.

The model separates cryptic residues from bulk surface well. On residues that already
line an open pocket it drops to 0.723, and **that is the discrimination the task is
actually about**. The stratum has resisted almost everything thrown at it:

| run | feat | hard stratum | bulk surface | gap |
|---|---|---|---|---|
| Tier A | 33 | 0.679 | 0.825 | 0.146 |
| + ensemble pocket features | 47 | 0.692 | 0.842 | 0.149 |
| + conservation | 53 | 0.700 | 0.856 | 0.156 |
| + ESM-16 | 69 | 0.709 | 0.871 | 0.162 |
| + ESM-32 | 85 | 0.722 | 0.878 | 0.156 |
| **+ ESM-64** | **117** | **0.723** | **0.882** | **0.160** |

Every addition lifts both strata together — 33 → 117 features moves the hard stratum
+0.044 and bulk surface +0.057 — so the gap sits at 0.146–0.162 throughout and nothing
has closed it. The descriptors are getting better at pocket-likeness in general rather
than at crypticity in particular. On this stratum the ensemble pocket features are not
the weakest of the three additions — they add +0.013 against conservation's +0.008 — even
though they are the worst value per unit of compute on the headline metrics.

**Both strata scored slightly higher once the leak was removed** (0.714 → 0.723 hard,
0.879 → 0.882 bulk). That is the tell: the holo-derived feature was shifting the model's
prior between strata, not helping it rank within one. See RESULTS.md § 5.1 and § 9.1.

---

## What actually moved the number

Ranked by effect on AUPRC (RESULTS.md § 4):

| change | ΔAUPRC | cost |
|---|---|---|
| Fixing PDB column alignment | +0.144 *(on ensemble features alone)* | one line |
| Fixing the negative sampler | **+0.113** | one line |
| Adding 64 ESM-2 principal components | +0.053 | 0.07 A10G-hours (~$0.08) |
| **Removing the holo-derived feature** | **−0.035** | one line — a correction |
| Adding 6 conservation features | +0.036 | ~2,300 API calls |
| Adding 14 ensemble pocket features | +0.009 | ~28,000 fpocket runs |
| Averaging over 5 contexts | +0.004 | 5× GPU inference |
| Fixing the insertion-code key | −0.004 *(noise; 0 test rows changed)* | three lines |

Four of the entries are bugs in code written here, and they are four of the five
largest; the deliberate capability additions are the smaller ones.

**The −0.035 row points the other way and is the important one.** Every other entry is a
number going up. That one is a number given back: `in_noncryptic_pocket` is derived by
comparing apo to holo, so it could never have been computed on a novel apo structure, and
0.035 AUPRC of the previous headline was unearned. It is listed here precisely because
leaving it out would repeat the mistake the feature itself was. See RESULTS.md § 9.1.

ESM-2 is decisively the best value per unit of compute — +0.053 for about five minutes of
A10G time (~$0.08). Conservation is second at +0.036 for ~2,300 cached API calls. The
ensemble pocket features are last: +0.009 for ~28,000 fpocket runs, the worst ratio in
the project.

The insertion-code fix is listed for completeness rather than for its effect: it
corrected 32 mislabelled residues and un-quarantined a protein, all of them in the
training split, so the measured movement is smaller than the context-draw noise. See
*Insertion codes* below.

---

## Three things that would silently inflate these numbers

**1. Holo-derived features.** A cryptic site is *defined* by the apo→holo change, so
any feature computed from the holo structure (Δdruggability, Δvolume, `pRMSD`)
reconstructs the label almost perfectly — and is unavailable at prediction time, when
only an apo structure exists.

An earlier version of this section warned that the guard was name-based and so would
miss "a holo-derived quantity under an innocuous name". **That is exactly what
happened.** `in_noncryptic_pocket` — CryptoBench's own annotation, built by thresholding
the apo↔holo pocket RMSD — matched none of the forbidden substrings and was an input to
the model for the whole first measurement era. Removing it cost 0.035 AUPRC of the
headline (RESULTS.md § 9.1).

The guard now has two halves: the substring blacklist, plus an explicit by-name set,
`config.HOLO_DERIVED_METADATA_COLS`, for columns that are holo-derived without saying so.
`tests/test_no_leakage.py` fails the build on either. The lesson generalises past this
one column: **provenance is not visible in a name**, and any feature adopted from a
benchmark's auxiliary data needs its construction read before it is trusted.

**2. Random splits.** Residues within a protein are highly correlated and homologous
proteins leak across a random split. We use CryptoBench's shipped folds, which are
built on sequence-identity clustering, and assert that no apo structure appears in two
splits.

**3. Accuracy.** At 5.3% prevalence, predicting "never cryptic" scores 94.7%. We report
AUC, AUPRC and TPR at a fixed low FPR instead. The decision threshold is chosen on
training folds and frozen before the test set is touched.

A fourth, subtler one: one apo structure maps to **5,493 holo pairings across 1,136
structures** — about five each. Labelling from a single holo partner would mark genuine
cryptic residues as negatives. Labels are the union over all partners.

---

## Features

117 total, in four tiers. Every numeric feature in tiers A–C additionally gets its
**8 Å spatial-neighbourhood mean** — CryptoSite found the neighbour-averaged variants
rank among its most discriminative, because a cryptic site is a property of a surface
patch rather than a lone residue.

**Tier A — geometry and physicochemistry (33).** Where CryptoSite gives an exact
definition we use it verbatim:

- `packing` — atoms of other residues within 4 Å, divided by the residue's atom count
- `n_neighbors_{4,6,8}A` — distinct residues within that distance
- `depth_to_surface` — distance to the nearest atom with SASA > 2 Å²

Plus: `sasa`, `rsa` (Tien et al. maxima), `protrusion` (empty fraction of a 10 Å
sphere), `convexity` (outward bulge relative to the 12 Å neighbourhood), half-sphere
exposure, B-factor (raw and per-chain z), secondary structure from backbone dihedrals,
Kyte–Doolittle hydropathy, formal charge, relative sequence position and chain length.

The table also carries `in_noncryptic_pocket`, a flag for residues lining an
already-open pocket — but it is **metadata, not a feature**. CryptoBench derives it from
apo↔holo pocket RMSD, so it cannot be computed at prediction time; it is used only to
stratify the evaluation. It *was* a Tier A feature until 2026-09-10, which is why the
tier is 33 and not 34 (RESULTS.md § 9.1).

Modified residues (MSE, SEP, PTR, LLP, …) are kept under their parent standard name,
resolved from gemmi's component table. Dropping them, as this code used to, removed
their atoms from the structure entirely — so the SASA, neighbour counts and fpocket all
saw a protein with holes in it, and 47 annotated cryptic residues were lost (§ 9.4).
Nucleic acids, `UNK` and ligands remain excluded by design.

Solvent accessibility is a vectorised Shrake–Rupley implemented here (`freesasa` needs
a C toolchain); ~0.5 s per structure.

**Tier C — ensemble pocket scores (14).** 24 ANM normal-mode conformations per protein,
fpocket on each, then CryptoSite's residue pocket score ("the maximum druggability
score among the alpha spheres within 5 Å of the residue"). See *Tier C* below.

**Conservation (6).** `cons_score`, `cons_entropy`, `cons_prob_self` and their
neighbourhood means, from the PDBe Graph API — the same source the CryptoBench authors
used for their own analysis. The join is on `label_seq_id`, not `auth_seq_id`; getting
that wrong yields plausible, wrong values for a subset of structures.

**ESM-2 (64).** `facebook/esm2_t36_3B_UR50D` — the same model pLM-NN uses — reduced from
2,560 dims by PCA. The projection is fit on **train rows only**, so no held-out protein
influences the basis.

---

## Layout

```
cryptic_tabicl/
├── config.py              paths, constants, the forbidden-feature list
├── dataset.py             CryptoBench JSON → labels, chains, folds, hard negatives
├── structures.py          mmCIF → per-residue records (gemmi) + Shrake–Rupley
├── features.py            Tier A features + neighbourhood aggregates
├── anm.py                 ANM normal modes + conformation sampling
├── fpocket.py             fpocket subprocess + CryptoSite's residue pocket score
├── ensemble_features.py   Tier C ensemble pocket statistics
├── conservation.py        PDBe Graph API conservation, joined on label_seq_id
├── esm_embed.py           chain sequences + chunk bookkeeping for ESM-2
├── build_table.py         orchestration → parquet, with join instrumentation
├── sampling.py            negative subsampling under TabICL's context budget
├── model.py               zero-shot TabICL wrapper
└── evaluate.py            per-protein metrics, comparable to the paper

deploy/
├── evaluate_modal.py         GPU evaluation on Modal
├── ensemble_modal.py         ~28k fpocket runs, fanned out
├── esm_modal.py              ESM-2 embeddings + train-only PCA
├── context_bootstrap.py      context-seed variance for TabICL
└── context_ensemble_modal.py multi-context averaging

scripts/
├── fetch_cryptobench.py   annotations
├── fetch_conservation.py  PDBe conservation, cached
├── merge_{ensemble,conservation,esm}.py   staged joins onto the residue table
├── baseline.py            sklearn controls on the same table
├── compare_variants.py    every run in one table
├── bootstrap_test.py      test-set CIs
├── f1_by_subset.py        CB-full / CB-P2RANK-apo, with the published F1 convention
└── cryptosite_testset.py  the 14-pair CryptoSite external test set

figures/
├── make_benchmark_figure.py   the published-benchmark bars  → benchmark_comparison.{png,pdf}
├── make_model_ladder_figure.py  every model configuration    → model_ladder.{png,pdf}
├── make_slide_figures.py      the nine slide panels          → slides/*.png
├── make_deck.py               assembles them                → ../Cryptic_site_prediction_pipeline.pptx
└── benchmark_comparison.html  interactive version; carries its OWN copy of the
                               benchmark values and of the model ladder

tests/                     leakage guards, label logic, sampler invariants,
                           fpocket parsing, conservation join, ESM index bookkeeping
```

## Usage

```bash
# 1. annotations (~51 MB; skips the 1.1 GB CIF archive — structures come from RCSB)
python scripts/fetch_cryptobench.py

# 2. build the residue table (~35 min, downloads ~1,100 mmCIFs)
python -m cryptic_tabicl.build_table --out build/residues.parquet   # (33 features)

# 3. Tier C: ensemble pocket features (~28k fpocket runs on Modal)
modal run --detach deploy/ensemble_modal.py
python scripts/merge_ensemble.py                      # → residues_ens.parquet   (47)

# 4. conservation (~2,300 PDBe calls, local, cached)
python scripts/fetch_conservation.py
python scripts/merge_conservation.py                  # → residues_cons.parquet  (53)

# 5. ESM-2 embeddings + train-only PCA
modal run --detach deploy/esm_modal.py
python scripts/merge_esm.py --n-components 64         # → residues_esm64.parquet (117)

# 6. evaluate on a Modal GPU
modal run --detach deploy/evaluate_modal.py --smoke                  # pilot
CRYPTIC_ONLY_VARIANT=tierA modal run --detach \
    deploy/evaluate_modal.py --variant tierA                         # one tier
modal run --detach deploy/tabicl_fanout.py --variants 0,16,32,64     # the ESM sweep

# CRYPTIC_ONLY_VARIANT trims the image to the one table a run needs; mounting all
# six stages ~600 MB before a container starts. It changes what is MOUNTED, never
# what is computed.
#
# Both GPU scripts persist each job to a Modal Volume, so a client disconnect costs
# nothing — re-aggregate for free with the run id they print:
#   modal run deploy/tabicl_fanout.py --collect --run-id <id> --variants 0,16,32,64

# 7. report
python scripts/compare_variants.py
python scripts/bootstrap_test.py
python scripts/f1_by_subset.py

# 8. figures and deck — rebuild these whenever a number moves
python scripts/model_ladder.py            # build/model_ladder.json, the ladder's data
python figures/make_benchmark_figure.py
python figures/make_model_ladder_figure.py
python figures/make_slide_figures.py
python figures/make_deck.py

pytest tests/ -q      # 124 tests
```

`figures/benchmark_comparison.html` is the one artifact with no build step: it is
hand-maintained and holds its own copies of the benchmark table and the model ladder,
so it has to be edited alongside the scripts. Its ladder block is regenerated from
`build/model_ladder.json` and both blocks are checked back against the artifacts — the
values in it are not transcribed by eye.

Data is staged at `~/cryptobench_stage/` on local disk rather than under OneDrive,
which would otherwise try to sync ~200 MB of structures.

---

## The context ceiling that wasn't

For most of this project the headline problem looked like TabICL's in-context row
limit. A 50k context scored AUPRC 0.333 where training on all 304k rows scored 0.417,
so ~7/8 of the negatives were apparently being thrown away at real cost.

Two experiments dismantled that reading.

**Averaging over independent contexts does nothing.** Five contexts, each a different
11.4% sample of the negative pool (pairwise Jaccard 0.18, union covering 37.8%),
averaged three different ways:

| K | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| AUPRC | 0.323 | 0.324 | 0.327 | 0.327 | 0.327 |

+0.004 against a predicted +0.084, saturating by K=3. If context size were the
binding constraint, seeing 3.3× the negatives should have moved more than that.

**The real cause was a line I wrote myself.** The sampler reserved 40% of the negative
budget for residues lining already-open pockets, on the reasoning that those are the
hard negatives worth over-weighting. Isolating each choice at fixed context size:

| context | AUC | AUPRC |
|---|---|---|
| capped, no forced share | 0.847 | **0.420** |
| uncapped, no forced share | 0.847 | 0.417 |
| uncapped, 40% forced share | 0.819 | 0.317 |
| capped + 40% forced share *(old default)* | 0.827 | 0.333 |
| all 288k negatives *(the target)* | 0.847 | 0.417 |

Only ~17k residues in the entire dataset line a non-cryptic pocket, so a 40% quota
over ~33k negative slots over-represents that stratum by roughly an order of
magnitude. The per-protein cap was harmless. The quota cost **0.087 AUPRC** for
logistic regression and 0.113 for TabICL.

With it removed, a 50k context scores AUPRC 0.422 against full-data 0.417 — **there is
no context-size ceiling at this dataset size at all.** Every "TabICL is handicapped by
its context window" conclusion earlier in this project was an artifact of that quota.

Two smaller lessons came out of the fix. Setting the share to `0.0` was itself wrong:
splitting the pool and taking zero from one side *excludes* pocket-lining negatives
entirely, the same distortion mirrored — the stratification has to be skipped, not
zeroed, and a test now pins the observed rate to the natural rate. And hard-negative
*mining* (rank negatives by a cheap model, take the most cryptic-looking) does not beat
plain proportional sampling: 0.398–0.410 across mixtures, and 0.057 for pure mining,
which removes every example of an easy negative and collapses the boundary.

## Why TabICL still needs a subsampled context

TabICL is *in-context*: the labelled table is pushed through the network on every
forward pass, so its size is an inference-time memory cost, not a one-off training
cost. 304k training residues will not fit, and v2 is pretrained for contexts up to
roughly 60k rows. So `sampling.py` keeps **every positive** (they are the entire
signal) and subsamples negatives under a 50k budget, in proportion to how they
actually occur. A per-protein cap keeps context composition from tracking protein
size.

With ~16k positives, the 50k budget binds before the nominal 1:8 ratio, giving ~1:2.

## Tier C: ensemble pocket scores

CryptoSite's five strongest descriptors are all ensemble pocket statistics, led by
the mean fpocket pocket score across MD snapshots (−log₁₀P = 138.6, roughly double
the best static feature). We approximate its 100k-snapshot AllosMod ensemble with 24
ANM normal-mode conformations per protein — one eigendecomposition instead of
nanoseconds of dynamics — and run fpocket on each (~28k invocations, fanned out on
Modal). Features follow CryptoSite's definitions where they are given, including the
residue pocket score as "the maximum druggability score among the alpha spheres
within 5 Å of the residue".

### A fixed-column bug that hid the entire signal

PDB is a fixed-column format and fpocket parses it positionally. The first version of
the PDB writer omitted the `altLoc` column, shifting `resName`, `chainID`, `resSeq`
and all three coordinates one place left. fpocket still ran and still reported
pockets — it simply typed the atoms wrongly. Measured on the same 24 proteins:

| feature | malformed columns | correct columns |
|---|---|---|
| `ens_mean_nb8` | 0.550 | **0.694** |
| `ens_opening` | 0.506 | **0.653** |
| combined (grouped CV) | 0.628 | **0.755** |

With the bug, `ens_opening` sat at chance and the honest reading was "ANM is a poor
surrogate for MD". That conclusion was entirely an artifact. The lesson generalises:
a tolerant parser downstream turns a format error into a quiet loss of signal rather
than a crash, which is indistinguishable from a negative result.

### Amplitude

The ANM displacement is rescaled to a target Cα RMSD. Sweeping it (out-of-sample,
grouped by protein) selects the *smallest* setting tried:

| target RMSD | mean opening | combined AUC |
|---|---|---|
| **1.5 Å** | 0.105 | **0.705** |
| 3.0 Å | 0.176 | 0.668 |
| 5.0 Å | 0.281 | 0.680 |

Larger amplitudes open more pockets but open them *everywhere*, so the feature stops
discriminating. Selectivity, not magnitude, is what carries the signal.

### Did they pay for themselves?

Ablation with logistic regression on the full training set (pooled test metrics):

| feature set | n | AUC | AUPRC |
|---|---|---|---|
| Tier A only | 34 | 0.864 | 0.366 |
| ensemble only | 14 | 0.766 | 0.162 |
| Tier A + ensemble | 48 | **0.876** | **0.380** |

A real but small gain: +0.012 AUC, +0.014 AUPRC. Two observations make that
number interpretable rather than merely disappointing.

**They are not redundant.** The maximum absolute correlation between any ensemble
feature and any Tier A feature is 0.21. The information is genuinely new — there is
simply not very much of it, and three of the combined model's fourteen largest
weights are ensemble features.

**CryptoSite's ranking is univariate.** Its −log₁₀P values measure each descriptor
alone against the null, not its increment over the other 57. That distinction matters
here: CryptoSite's *final* model kept only three features, so its MD pocket score
dominated a small set. Our 34 geometric descriptors already encode much of what makes
a region pocket-like, so an ensemble pocket score arrives with most of its variance
already explained.

The remaining gap to CryptoSite's ensemble signal is most likely sampling: 24 ANM
conformations against 100,000 MD snapshots, and harmonic collective modes that cannot
produce the side-chain and loop rearrangements many cryptic sites require.

## Insertion codes are part of a residue's identity

Author residue numbering is not dense. Proteases and immunoglobulins insert residues
without renumbering the chain, so 3bv9 chain B holds **ten** residues at position 60 —
60, 60A, 60B … 60I — and six at 149. CryptoBench keys them the way the PDB spells them:
`B_60`, `B_60A`. This project keyed on `(chain, seq_id)` alone, collapsing all ten onto
`B_60`, and that failed in both directions at once:

- an annotated `B_60A` matched no observed key, so a true positive was scored negative;
- an annotated `B_149` matched all six residues at 149, so five were handed a positive
  label they never earned — the more common of the two.

19 residues were wrongly positive and 1 wrongly negative, plus 9 and 3 for the
`in_noncryptic_pocket` feature: 32 residues across six structures, and `3bv9B`
quarantined outright at match rate 0.793 because 6 of its 29 annotated keys carried a
code. Only 9 of 20,387 annotated cryptic keys carry one at all, which is why exactly one
structure crossed the quarantine line.

**Every affected residue was in the training split — 0 of 76,636 test rows changed.**
That is luck, not design. The corrections reach the model only through the in-context
table, where they amount to 11 changed rows out of 50,000, so the headline moved by less
than the context-draw noise (AUC 0.882 → 0.882, AUPRC 0.480 → 0.476) while the
quarantine went 1 → 0.

*(Those figures are from the 2026-09-09 measurement era, before the holo-derived feature
was removed and before modified residues were restored. They are the correct record of
what that fix did at the time; the current headline is 0.862 / 0.441 on a 76,753-row test
split. Both later corrections are in RESULTS.md § 9.)*

The key is now `{chain}_{seq_id}{icode}`, and all three feature merges join on the same
four-part key under `validate="1:1"` — previously two of them joined on three parts and
called `drop_duplicates`, handing residues 60A–60I the features of residue 60. Two
related defects were fixed alongside: the ANM ensemble seed was `hash(apo_id)`, which is
salted per interpreter and so drew different conformations every run, and
`evaluate_modal` had no variant name that resolved to the conservation-tier table. Full
accounting in RESULTS.md § 7.

## Four corrections, 2026-09-10

Found by review, all in code written here, all fixed with every number re-measured
afterwards. Full accounting in RESULTS.md § 9.

| # | defect | effect |
|---|---|---|
| 1 | `in_noncryptic_pocket`, a **holo-derived** annotation, was an input to the model | −0.035 AUPRC; two claims withdrawn, TabICL's margin over sklearn doubled |
| 2 | `_add_neighbourhood` skipped on the no-ANM fallback path | dormant — 0 of 1,136 proteins take that path; a landmine for smaller inputs |
| 3 | the threshold swept **pooled** FPR while the document reports **per-protein** FPR | every CV fold now lands at exactly 0.0500 |
| 4 | modified residues (MSE, …) were deleted from the structure, not just from the prediction set | +515 residues, +43 recovered annotations |

Two infrastructure fixes came out of the re-run: `esm_modal.py` could silently mix
shards from two different runs, and both GPU scripts lost all completed work when the
Modal client dropped its connection. Both are fixed and described in § 9.5.

## Deferred

- **Fine-tuning** — `tabicl[finetune]` exists, but it forfeits the zero-training
  property; if run, report beside the zero-shot number, never instead of it
- **CryptoSite external test set** — `scripts/cryptosite_testset.py` builds the 14-pair
  held-out set (none of its apo structures appears in CryptoBench); scoring has not
  been run
- **Side-chain and loop motion** — the one thing that plausibly moves the hard stratum,
  and the one thing harmonic ANM modes cannot produce
