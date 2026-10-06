"""Their plain ESM-2 3B fine-tune, retrained per fold for out-of-fold scores (PLAN.md step 8).

WHY RETRAIN AT ALL
    TabICL can only use a score as a feature if the score exists, honestly, on its
    context rows -- the training proteins. The released model trained on every one of
    them (step 6: +0.15-0.19 AUPRC in-sample), so its scores there are memorised. Each
    fold model below never sees the fold it scores.

THE RECIPE IS THEIRS, LINE FOR LINE
    skrhakv/cryptic-finetuning, src/train.ipynb cell 9 ("Finetuning ESM2 model"):
      FinetunedEsmModel(esm2_t36_3B).half()        (deploy/s2p_models.py, verbatim)
      bnb.optim.AdamW8bit(all params, lr=1e-4, eps=1e-4)   bitsandbytes 0.45.5
      BCEWithLogitsLoss(pos_weight=8.7481)                 their precomputed weight
      batch 1, shuffled, torch.manual_seed(42)
      3 epochs; backbone frozen in epochs 0-1, unfrozen from epoch 2 (`if epoch > 1`)
      loss on positions whose label != -100 (BOS/EOS excluded)
      sequences longer than 1,024 skipped (their loader)
    Omitted: their per-epoch pass over the TEST set. It is monitoring only -- nothing
    is selected from it -- and evaluating on test during training is not something
    this project does.

FIVE JOBS
    all     train folds 0-3      = their setup; scores test (reproduction gate 8c)
    h=0..3  the three other folds; scores fold h (out-of-fold)
    Every job scores EVERY chain in build/s2p_inputs.parquet afterwards (~80 s), so
    in-sample vs held-out behaviour can be checked from the outputs.

USAGE
    modal deploy deploy/s2p_finetune_modal.py
    modal run deploy/s2p_finetune_modal.py --pilot                     # 8b
    modal run deploy/s2p_finetune_modal.py --spawn --jobs all --run-id ...
    modal run deploy/s2p_finetune_modal.py --spawn --jobs 0,1,2,3 --run-id ...
    modal volume get cryptobench-s2p ft/<run-id>/ build/s2p_ft/
"""
from __future__ import annotations

import modal

ESM_ID = "facebook/esm2_t36_3B_UR50D"
POS_WEIGHT = 8.7481
LR, EPS, EPOCHS, SEED = 1e-4, 1e-4, 3, 42
UNFREEZE_AFTER_EPOCH = 1          # their `if epoch > 1`
MAX_LENGTH = 1024
GPU = "A100-80GB"
GPU_HOURLY_USD = 2.50
VOL_ROOT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.48.1", "bitsandbytes==0.45.5", "numpy==1.26.4",
                 "pandas", "pyarrow", "huggingface_hub")
    # Bake ESM-2 3B into the image: per-container downloads are slow and fragile.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download as d; "
        f"d(repo_id='{ESM_ID}', allow_patterns=['*.json','*.txt','*.safetensors'])\"")
    .add_local_file("deploy/s2p_models.py", "/app/s2p_models.py")
    .add_local_file("cryptic_tabicl/esm_embed.py", "/app/esm_embed.py")
)

vol = modal.Volume.from_name("cryptobench-s2p", create_if_missing=True)
app = modal.App("cryptobench-s2p-finetune")


def parse_annotation(text: str) -> list:
    """Their loader (`process_sequence_dataset`): (id, sequence, 0/1 labels), > 1,024 skipped."""
    import numpy as np

    out = []
    for row in text.splitlines():
        pdb_id, chain, _, lab, seq = row.split(";")
        if len(seq) > MAX_LENGTH:
            continue
        y = np.zeros(len(seq), dtype=np.float32)
        if lab:
            y[[int(r[1:]) for r in lab.split(" ")]] = 1.0
        out.append((pdb_id.lower() + chain, seq, y))
    return out


@app.function(image=image, gpu=GPU, timeout=6 * 3600, retries=0, volumes={VOL_ROOT: vol})
def finetune(job: dict) -> dict:
    """Train one model on `job['train_text']`, then score every chain in `job['items']`."""
    import os
    import sys
    import time

    import bitsandbytes as bnb
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer

    sys.path.insert(0, "/app")
    from esm_embed import chunk_spans
    from s2p_models import FinetunedEsmModel

    t0 = time.time()
    name, run_id = job["name"], job["run_id"]
    out_dir = f"{VOL_ROOT}/ft/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    data = parse_annotation(job["train_text"])
    if job.get("limit"):
        data = data[: job["limit"]]
    print(f"[{name}] {len(data)} training proteins", flush=True)

    torch.manual_seed(SEED)
    model = FinetunedEsmModel(ESM_ID).half().cuda()
    tok = AutoTokenizer.from_pretrained(ESM_ID)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, eps=EPS)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device="cuda"))
    for pname, p in model.named_parameters():
        if pname.startswith("llm"):
            p.requires_grad = False

    log = []
    order_gen = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        if epoch > UNFREEZE_AFTER_EPOCH:
            for p in model.parameters():
                p.requires_grad = True
        model.train()
        torch.cuda.reset_peak_memory_stats()
        te, losses = time.time(), []
        for i in torch.randperm(len(data), generator=order_gen).tolist():
            _, seq, y = data[i]
            enc = tok(seq, return_tensors="pt")
            labels = torch.tensor(np.concatenate([[-100.0], y, [-100.0]]),
                                  dtype=torch.float32, device="cuda")
            output = model({k: v.cuda() for k, v in enc.items()})
            mask = labels != -100
            logits = output.flatten()[mask]
            loss = loss_fn(logits, labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        rec = {"epoch": epoch, "backbone_trainable": epoch > UNFREEZE_AFTER_EPOCH,
               "mean_loss": float(np.mean(losses)), "first_q_loss": float(np.mean(losses[: max(1, len(losses)//4)])),
               "last_q_loss": float(np.mean(losses[-max(1, len(losses)//4):])),
               "seconds": time.time() - te,
               "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
               "nonfinite_losses": int(np.sum(~np.isfinite(losses)))}
        log.append(rec)
        print(f"[{name}] {rec}", flush=True)
    train_seconds = time.time() - t0

    # ---- score every chain we were given (held-out fold / test / everything)
    model.eval()
    frames = []
    with torch.no_grad():
        for apo_id, chain, seq in job["items"]:
            probs = []
            for start, end, ks, ke in chunk_spans(len(seq)):
                enc = tok(seq[start:end], return_tensors="pt")
                lg = model({k: v.cuda() for k, v in enc.items()})[0, 1:-1, 0]
                assert lg.shape[0] == end - start
                probs.append(torch.sigmoid(lg.float()).cpu().numpy()[ks - start: ke - start])
            p = np.concatenate(probs)
            assert len(p) == len(seq)
            frames.append(pd.DataFrame({"apo_id": apo_id, "chain": chain,
                                        "pos": range(len(seq)), "score": p}))
    scores = pd.concat(frames, ignore_index=True)
    scores.to_parquet(f"{out_dir}/scores_{name}.parquet", index=False)
    if job.get("save_weights"):
        torch.save(model.state_dict(), f"{out_dir}/model_{name}.pt")

    meta = {"name": name, "run_id": run_id, "n_train_proteins": len(data),
            "epochs": log, "train_seconds": train_seconds,
            "total_seconds": time.time() - t0,
            "cost_usd": (time.time() - t0) / 3600 * GPU_HOURLY_USD, "gpu": GPU,
            "n_scored_chains": len(job["items"]), "n_scored_residues": len(scores),
            "torch": torch.__version__, "bitsandbytes": bnb.__version__,
            "recipe": {"lr": LR, "eps": EPS, "epochs": EPOCHS, "seed": SEED,
                       "pos_weight": POS_WEIGHT, "unfreeze_after_epoch": UNFREEZE_AFTER_EPOCH}}
    pd.Series(meta).to_json(f"{out_dir}/meta_{name}.json")
    vol.commit()
    print(f"[{name}] done in {meta['total_seconds']:.0f}s (~${meta['cost_usd']:.2f})", flush=True)
    return meta


def _jobs(names: list, run_id: str, items: list, limit: int = 0) -> list:
    from pathlib import Path

    folds = {h: Path(f"build/s2p_train/train-fold-{h}.txt").read_text() for h in range(4)}
    jobs = []
    for name in names:
        train = [h for h in range(4) if name == "all" or h != int(name)]
        jobs.append({"name": name, "run_id": run_id, "limit": limit, "items": items,
                     "train_folds": train, "save_weights": not limit,
                     "train_text": "\n".join(folds[h] for h in train)})
    return jobs


@app.local_entrypoint()
def main(pilot: bool = False, spawn: bool = False, jobs: str = "", run_id: str = ""):
    import pandas as pd

    inputs = pd.read_parquet("build/s2p_inputs.parquet").sort_values(["apo_id", "chain", "pos"])
    items = [(a, c, "".join(g["aa"])) for (a, c), g in inputs.groupby(["apo_id", "chain"])]
    f = modal.Function.from_name("cryptobench-s2p-finetune", "finetune")
    if pilot:
        pilot_items = [it for it in items if it[0] in ("1a4uB", "1dpjA", "2po4A")]
        job = _jobs(["0"], "pilot", pilot_items, limit=20)[0]
        call = f.spawn(job)
        print(f"pilot spawned: {call.object_id} -> cryptobench-s2p:ft/pilot/")
    elif spawn:
        assert run_id and jobs, "--spawn needs --jobs and --run-id"
        for job in _jobs(jobs.split(","), run_id, items):
            call = f.spawn(job)
            print(f"spawned {job['name']} (train folds {job['train_folds']}): {call.object_id}")
    else:
        raise SystemExit("pass --pilot or --spawn")
