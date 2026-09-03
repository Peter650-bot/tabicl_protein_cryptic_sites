# cryptic_tabicl

Per-residue cryptic binding site prediction on [CryptoBench](https://osf.io/pz4a9/),
using **zero-shot TabICLv2** as the classifier.

Given an **apo** (unbound) protein structure, predict for every residue whether it
belongs to a cryptic binding site — a site that is malformed or inaccessible in the
unbound state but opens on ligand binding.

This project is self-contained and shares no code with the PPO/HOP project.

---

## The task

| | |
|---|---|
| Unit of prediction | one **residue** of one apo chain |
| Label | residue appears in `apo_pocket_selection` for any holo partner |
| Rows | 380,930 residues across 1,135 apo structures |
| Prevalence | 5.34% positive |
| Features | 34, all computed from the **apo structure only** |
| Model | TabICLv2, zero-shot (no fitting, no fine-tuning) |
| Splits | CryptoBench's own sequence-identity folds |

## Results

Test set (231 proteins, 76,636 residues), metrics computed per protein then averaged,
matching CryptoBench's `Protein.py`:

| Method | features | AUC | AUPRC | TPR | FPR | MCC |
|---|---|---|---|---|---|---|
| **TabICLv2 zero-shot** | 48 | **0.862** | **0.436** | 0.549 | 0.066 | 0.371 |
| LogisticRegression, full train | 48 | 0.847 | 0.417 | 0.451 | 0.050 | 0.341 |
| RandomForest, full train | 34 | 0.840 | 0.418 | 0.258 | 0.015 | 0.258 |
| RSA alone (floor) | 1 | 0.499 | 0.089 | — | — | — |
| *pLM-NN, CB-full (published)* | — | *0.86* | *0.36* | *0.48* | *0.05* | *0.39* |
| *P2Rank apo, subset (published)* | — | *0.81* | *0.21* | *0.62* | *0.14* | *0.27* |
| *PocketMiner, subset (published)* | — | *0.76* | *0.19* | *0.51* | *0.16* | *0.22* |

Zero-shot TabICL matches pLM-NN on AUC (0.862 vs 0.86) and exceeds it on AUPRC
(0.436 vs 0.36) and TPR (0.549 vs 0.48), at a somewhat looser FPR (0.066 vs 0.05) and
slightly lower MCC (0.371 vs 0.39). It does so with 48 hand-crafted apo-only structural
descriptors, no protein language model, and no training of any kind.

Only the CB-full row is comparable: our evaluation covers all 231 test proteins, whereas
the published P2Rank and PocketMiner figures are on subsets those tools could run on.
Do not compare ACC or F1 across the two tables — the published F1 of 0.92 cannot be
positive-class F1 at 5% prevalence (that would be ~0.40), so it is weighted or macro,
while ours is positive-class.

Pooled (rather than per-protein) AUC 0.879, AUPRC 0.380.

Inner CV on the train folds, used only to freeze the threshold at 0.5425:

| fold | AUC | AUPRC |
|---|---|---|
| train-fold-0 | 0.830 | 0.391 |
| train-fold-1 | 0.859 | 0.463 |
| train-fold-2 | 0.838 | 0.421 |
| train-fold-3 | 0.834 | 0.424 |

Reproduce the table with `python scripts/compare_variants.py`.

pLM-NN uses protein-language-model embeddings and is trained. This uses 34
hand-crafted geometric and physicochemical descriptors and performs no training —
the labelled context is a single forward pass.

Inner 4-fold CV over the train folds, used only to pick the decision threshold
(frozen at 0.5157 before touching test):

| fold | AUC | AUPRC |
|---|---|---|
| train-fold-0 | 0.805 | 0.320 |
| train-fold-1 | 0.839 | 0.379 |
| train-fold-2 | 0.814 | 0.346 |
| train-fold-3 | 0.823 | 0.351 |

Fold spread is narrow (AUC 0.805–0.839), so no single fold carries the result.
The frozen threshold gives FPR 0.08 on test against the 0.05 it was tuned for —
the honest cost of not tuning on the test set.

### Control: the same features with conventional models

Everything held fixed except the model — same table, same folds, same threshold rule,
same per-protein averaging:

| Method | context | AUC | AUPRC |
|---|---|---|---|
| RandomForest | full 304k train | **0.841** | **0.419** |
| LogisticRegression | full 304k train | 0.834 | 0.413 |
| **TabICLv2 zero-shot** | 50k subsample | 0.830 | 0.320 |
| RandomForest | same 50k subsample | 0.819 | 0.315 |
| LogisticRegression | same 50k subsample | 0.813 | 0.324 |
| RSA alone | — | 0.499 | 0.089 |

Read honestly, this says three things:

1. **The features carry the result, not the model.** At equal context TabICL (0.830)
   is barely ahead of a RandomForest (0.819). The gap over P2Rank and PocketMiner is
   attributable to the descriptors, not to in-context learning.

2. **TabICL's context ceiling costs real accuracy here.** A RandomForest that can see
   all 304k training rows reaches AUPRC 0.419 — above both TabICL's 0.320 and the
   published pLM-NN's 0.36. Subsampling to fit the context is discarding usable signal.
   This is a structural limitation of applying an in-context model to a dataset with
   ~300k rows, not a tuning problem.

3. **The task is genuinely multivariate.** RSA alone scores 0.499 — chance. No single
   descriptor solves this.

So the defensible claim is not "TabICL is the right model for cryptic site prediction".
It is "34 apo-only structural descriptors are competitive with a protein language model
on CryptoBench, and TabICL reaches that level with no training at all — while a plain
RandomForest with full data exceeds it."

### Where it is weak

Stratifying the test set by whether a residue lines an already-open pocket:

| stratum | residues | prevalence | AUC | AUPRC | TPR |
|---|---|---|---|---|---|
| elsewhere | 73,296 | 3.5% | 0.820 | 0.188 | 0.474 |
| lines a non-cryptic pocket | 3,340 | 44.0% | 0.677 | 0.600 | 0.484 |

The model is clearly separating cryptic residues from bulk surface, but on residues
that already line an open pocket it drops to AUC 0.677. That is the discrimination
the task is actually about, and it is where the remaining headroom is — plausibly
what CryptoSite's conformational-ensemble features (its single strongest signal)
would address.

---

## Three things that would silently inflate these numbers

**1. Holo-derived features.** A cryptic site is *defined* by the apo→holo change, so
any feature computed from the holo structure (Δdruggability, Δvolume, `pRMSD`)
reconstructs the label almost perfectly — and is unavailable at prediction time, when
only an apo structure exists. Every feature here comes from the apo structure alone,
and `tests/test_no_leakage.py` fails the build if a holo-named column reaches `X`.

**2. Random splits.** Residues within a protein are highly correlated and homologous
proteins leak across a random split. We use CryptoBench's shipped folds, which are
built on sequence-identity clustering, and assert that no apo structure appears in two
splits.

**3. Accuracy.** At 5.3% prevalence, predicting "never cryptic" scores 94.7%. We report
AUC, AUPRC and TPR at a fixed low FPR instead. The decision threshold is chosen on
training folds and frozen before touching the test set.

A fourth, subtler one: one apo structure maps to **5,493 holo pairings across 1,136
structures** — about five each. Labelling from a single holo partner would mark genuine
cryptic residues as negatives. Labels are the union over all partners.

---

## Features (Tier A)

All per-residue, all apo-only. Where CryptoSite gives an exact definition we use it
verbatim:

- `packing` — atoms of other residues within 4 Å, divided by the residue's atom count
- `n_neighbors_{4,6,8}A` — distinct residues within that distance
- `depth_to_surface` — distance to the nearest atom with SASA > 2 Å²

Plus: `sasa`, `rsa` (Tien et al. maxima), `protrusion` (empty fraction of a 10 Å
sphere), `convexity` (outward bulge relative to the 12 Å neighbourhood), half-sphere
exposure, B-factor (raw and per-chain z), secondary structure from backbone dihedrals,
Kyte–Doolittle hydropathy, formal charge, relative sequence position, chain length, and
a flag for residues lining an already-open (non-cryptic) pocket.

Every numeric feature additionally gets its **8 Å spatial-neighbourhood mean** —
CryptoSite found the neighbour-averaged variants rank among its most discriminative,
because a cryptic site is a property of a surface patch rather than a lone residue.

Solvent accessibility is a vectorised Shrake–Rupley implemented here (`freesasa` needs
a C toolchain); ~0.5 s per structure.

---

## Layout

```
cryptic_tabicl/
├── config.py       paths, constants, the forbidden-feature list
├── dataset.py      CryptoBench JSON → labels, chains, folds, hard negatives
├── structures.py   mmCIF → per-residue records (gemmi) + Shrake–Rupley
├── features.py     Tier A features + neighbourhood aggregates
├── build_table.py  orchestration → parquet, with join instrumentation
├── sampling.py     negative subsampling under TabICL's context budget
├── model.py        zero-shot TabICL wrapper
└── evaluate.py     per-protein metrics, comparable to the paper
deploy/evaluate_modal.py   GPU evaluation on Modal
tests/                     leakage guards, label logic, sampler invariants
```

## Usage

```bash
# 1. annotations (~51 MB; skips the 1.1 GB CIF archive — structures come from RCSB)
python scripts/fetch_cryptobench.py

# 2. build the residue table (~35 min, downloads ~1,100 mmCIFs)
python -m cryptic_tabicl.build_table --out build/residues.parquet

# 3. evaluate on a Modal GPU
modal run --detach deploy/evaluate_modal.py --smoke   # pilot
modal run --detach deploy/evaluate_modal.py           # full

pytest tests/ -q
```

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
magnitude. The per-protein cap was harmless. The quota cost **0.087 AUPRC**.

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
signal) and subsamples negatives under a 50k budget, reserving a share for residues
lining already-open pockets — the hard negatives that separate "found a pocket" from
"found a *cryptic* pocket". A per-protein cap keeps context composition from tracking
protein size.

With ~12k positives, the 50k budget binds before the nominal 1:8 ratio, giving ~1:3.

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

On TabICL the end-to-end effect is the same small, consistent shift:

| | Tier A | + ensemble | Δ |
|---|---|---|---|
| AUC | 0.830 | 0.839 | +0.009 |
| AUPRC | 0.324 | 0.323 | −0.001 |
| MCC | 0.277 | 0.286 | +0.009 |
| AUC, bulk surface | 0.820 | 0.831 | +0.010 |
| AUC, lines an open pocket | 0.677 | 0.680 | +0.003 |

The last row is the disappointing one. The hard stratum — telling a cryptic site from
a pocket that is already open — is where ensemble features were supposed to help most,
and it barely moved. Whatever distinguishes those two cases, 24 harmonic conformations
do not capture it.

## Deferred

- **Conservation** (CryptoSite's #2 feature) — needs MSAs
- **ESM-2 embeddings** — what pLM-NN uses; PCA to 32–64 dims to stay near TabICL's
  feature sweet spot
- **Fine-tuning** — `tabicl[finetune]` exists, but it forfeits the zero-training
  property; if run, report beside the zero-shot number, never instead of it
