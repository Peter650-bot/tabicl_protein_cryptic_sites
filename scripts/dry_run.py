"""Dry run of the full model path on the smoke table.

Not a result: the split here is mechanical (first 15 vs last 10 proteins), not
CryptoBench's fold split. It exists to prove the plumbing -- feature typing,
context construction, TabICL fit/predict, threshold selection, metrics -- before
spending time on the real evaluation.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from cryptic_tabicl import config as C, sampling, model, evaluate

t = pd.read_parquet("build/smoke.parquet")
print(f"smoke table: {len(t):,} rows, {t[C.GROUP_COL].nunique()} proteins, "
      f"{t[C.LABEL_COL].mean():.2%} positive", flush=True)

prots = sorted(t[C.GROUP_COL].unique())
tr = t[t[C.GROUP_COL].isin(prots[:15])]
te = t[t[C.GROUP_COL].isin(prots[15:])]
print(f"dry-run split: train {len(tr):,} rows / 15 prot, test {len(te):,} rows / {len(prots)-15} prot", flush=True)

ctx, plan = sampling.build_context(tr, budget=6000, negative_ratio=8)
print(plan, flush=True)

t0 = time.time()
p = model.CrypticSitePredictor(n_estimators=4, batch_size=2).fit(ctx)
print(f"fit: {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
scores = p.predict_by_protein(te)
print(f"predict {len(te):,} rows: {time.time()-t0:.1f}s", flush=True)

# threshold chosen on the context only, never on test
ctx_scores = p.predict_by_protein(ctx)
thr = evaluate.choose_threshold(ctx[C.LABEL_COL].to_numpy(), ctx_scores,
                                ctx[C.GROUP_COL].to_numpy())
print(f"threshold @FPR<={C.TARGET_FPR}: {thr:.4f}", flush=True)

res = evaluate.summarise(te[C.LABEL_COL].to_numpy(), scores,
                         te[C.GROUP_COL].to_numpy(), thr,
                         label="TabICL dry-run")
print()
print(evaluate.comparison_table(res))
print(f"\npooled AUC {res['AUC_pooled']:.3f} | AUPRC {res['AUPRC_pooled']:.3f}")
