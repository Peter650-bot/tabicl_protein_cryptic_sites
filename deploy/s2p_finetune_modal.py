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
                 "pandas", "pyarrow", "huggingface_hub", "datasets==3.5.0", "scikit-learn")
    # Bake ESM-2 3B into the image: per-container downloads are slow and fragile.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download as d; "
        f"d(repo_id='{ESM_ID}', allow_patterns=['*.json','*.txt','*.safetensors'])\"")
    .add_local_file("deploy/s2p_models.py", "/app/s2p_models.py")
    .add_local_file("deploy/finetuning_utils.py", "/app/finetuning_utils.py")
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


DATA_URL = "https://owncloud.cesnet.cz/index.php/s/w29gFI3GEzCNlKc/download"
DATA_TAR = f"{VOL_ROOT}/full/cbs-finetuning-data.tar.gz"


@app.function(image=modal.Image.debian_slim(python_version="3.11").apt_install("curl"),
              timeout=6 * 3600, cpu=4.0, memory=8192, volumes={VOL_ROOT: vol})
def fetch_full_data() -> dict:
    """Step 9a: the authors' training data (26.1 GB), downloaded once, then inventoried."""
    import hashlib
    import os
    import subprocess
    import tarfile
    import time

    os.makedirs(os.path.dirname(DATA_TAR), exist_ok=True)
    t0 = time.time()
    if not os.path.exists(DATA_TAR + ".done"):
        subprocess.run(["curl", "-fL", "--retry", "8", "--retry-delay", "10", "-C", "-",
                        "-sS", "-o", DATA_TAR, DATA_URL], check=True)
        open(DATA_TAR + ".done", "w").close()
        vol.commit()
    size = os.path.getsize(DATA_TAR)
    print(f"downloaded {size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    h = hashlib.sha256()
    with open(DATA_TAR, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    members = []
    with tarfile.open(DATA_TAR, "r:gz") as tf:
        for m in tf:
            members.append((m.name, m.size, m.isdir()))
    return {"bytes": size, "sha256": h.hexdigest(), "members": members,
            "seconds": time.time() - t0}


FULL_DIR = f"{VOL_ROOT}/full"
SKIP_PREFIXES = ("final-data/cryptobench-with-unobserved/embeddings",
                 "final-data/cryptobench/embeddings", "final-data/cryptobench/distance-matrices")


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=3 * 3600,
              cpu=4.0, memory=8192, volumes={VOL_ROOT: vol})
def extract_full_data() -> dict:
    """Step 9b: unpack what the recipe needs; hash their trained checkpoints on the way."""
    import hashlib
    import os
    import tarfile

    hashes, n = {}, 0
    with tarfile.open(DATA_TAR, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or m.name.startswith(SKIP_PREFIXES):
                continue
            if m.name.startswith("final-data/trained-models/"):
                h, f = hashlib.sha256(), tf.extractfile(m)
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
                hashes[os.path.basename(m.name)] = {"bytes": m.size, "sha256": h.hexdigest()}
                continue
            tf.extract(m, FULL_DIR)
            n += 1
    vol.commit()
    texts = {}
    for rel in ("final-data/cryptobench/train.txt", "final-data/cryptobench/test.txt",
                "final-data/ligysis/train.txt"):
        texts[rel] = open(f"{FULL_DIR}/{rel}").read()
    return {"n_extracted": n, "checkpoints": hashes, "texts": texts}


CB_DIR = f"{FULL_DIR}/final-data/cryptobench"
LIG_DIR = f"{FULL_DIR}/final-data/ligysis"
LIG_EPOCHS, CB_EPOCHS = 4, 3


def _score_items(model, tok, items, head_is_tuple: bool):
    """Per-residue sigmoid(cryptic head) for every chain; long chains chunked."""
    import numpy as np
    import pandas as pd
    import torch
    from esm_embed import chunk_spans

    model.eval()
    frames = []
    with torch.no_grad():
        for apo_id, chain, seq in items:
            probs = []
            for start, end, ks, ke in chunk_spans(len(seq)):
                enc = tok(seq[start:end], return_tensors="pt")
                out = model({k: v.cuda() for k, v in enc.items()})
                lg = (out[0] if head_is_tuple else out)[0, 1:-1, 0]
                assert lg.shape[0] == end - start
                probs.append(torch.sigmoid(lg.float()).cpu().numpy()[ks - start: ke - start])
            p = np.concatenate(probs)
            assert len(p) == len(seq)
            frames.append(pd.DataFrame({"apo_id": apo_id, "chain": chain,
                                        "pos": range(len(seq)), "score": p}))
    return pd.concat(frames, ignore_index=True)


@app.function(image=image, gpu=GPU, timeout=8 * 3600, retries=0, volumes={VOL_ROOT: vol})
def finetune_full(job: dict) -> dict:
    """Their multitask + LIGYSIS recipe (train.ipynb cell 16), both stages, one session.

    Stage A, LIGYSIS: MultitaskFinetunedEsmModel.half(); AdamW8bit(lr 1e-4, eps 1e-4);
      4 epochs; backbone frozen in epochs 0-1; loss = BCE(pos_weight 8.7481) +
      MSE(sigmoid(pLDDT head), scaled pLDDT). Their loader drops their 46 IDs and
      sequences > 1,024. The `clean` variant additionally drops job['ligysis_exclude'].
    Stage B, CryptoBench: a NEW AdamW8bit; 3 epochs; in epochs 0-1 every parameter whose
      name starts with llm / classifier / plDDT is frozen (only the distance head trains);
      loss = BCE + MSE(distance) + MSE(pLDDT); scalers refit on this job's train file.
    Every loader, scaler and collate call is their `finetuning_utils`, unmodified.
    """
    import functools
    import os
    import sys
    import time

    import bitsandbytes as bnb
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    sys.path.insert(0, "/app")
    import finetuning_utils as fu

    t0 = time.time()
    name, run_id = job["name"], job["run_id"]
    out_dir = f"{VOL_ROOT}/ft/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    vol.reload()

    # ---- this job's training files (their format), written beside the outputs
    lig_lines = open(f"{LIG_DIR}/train.txt").read().splitlines()
    excl = set(job.get("ligysis_exclude", []))
    lig_lines = [l for l in lig_lines if l.split(";")[0] not in excl]
    if job.get("limit"):
        lig_lines = lig_lines[: job["limit"]]
    lig_file = f"{out_dir}/ligysis_train_{name}.txt"
    open(lig_file, "w").write("\n".join(lig_lines))
    cb_lines = [l for l in job["train_text"].splitlines() if l.strip()]
    if job.get("limit"):
        cb_lines = cb_lines[: job["limit"]]
    cb_file = f"{out_dir}/cryptobench_train_{name}.txt"
    open(cb_file, "w").write("\n".join(cb_lines))

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(ESM_ID)
    collate = functools.partial(fu.collate_fn, tokenizer=tok)
    model = fu.MultitaskFinetunedEsmModel(ESM_ID).half().to("cuda")
    cbs_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device="cuda"))
    mse = nn.MSELoss()
    log = []

    def run_epochs(stage, loader, epochs, frozen_prefixes, use_distance):
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, eps=EPS)
        for pname, p in model.named_parameters():
            if pname.startswith(frozen_prefixes):
                p.requires_grad = False
        for epoch in range(epochs):
            if epoch > UNFREEZE_AFTER_EPOCH:
                for p in model.parameters():
                    p.requires_grad = True
            model.train()
            torch.cuda.reset_peak_memory_stats()
            te, losses = time.time(), []
            for batch in loader:
                o1, o2, o3 = model(batch)
                labels = batch["labels"].to("cuda")
                mask = labels.flatten() != -100
                loss = cbs_loss_fn(o1.flatten()[mask], labels.flatten()[mask])
                loss = loss + mse(torch.sigmoid(o2.flatten()[mask]),
                                  batch["plDDTs"].to("cuda").flatten()[mask].half())
                if use_distance:
                    loss = loss + mse(torch.sigmoid(o3.flatten()[mask]),
                                      batch["distances"].to("cuda").flatten()[mask].half())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            rec = {"stage": stage, "epoch": epoch, "n": len(losses),
                   "mean_loss": float(np.mean(losses)), "seconds": time.time() - te,
                   "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "nonfinite_losses": int(np.sum(~np.isfinite(losses)))}
            log.append(rec)
            print(f"[{name}] {rec}", flush=True)

    # ---- stage A: LIGYSIS
    lig_pl = fu.train_scaler(lig_file, plDDT_path=f"{LIG_DIR}/plDDT", uniprot_ids=True)
    lig_ds = fu.process_sequence_dataset(lig_file, tok, plDDT_scaler=lig_pl,
                                         plDDT_path=f"{LIG_DIR}/plDDT", uniprot_ids=True)
    run_epochs("ligysis", DataLoader(lig_ds, batch_size=1, shuffle=True, collate_fn=collate),
               LIG_EPOCHS, ("llm",), use_distance=False)

    # ---- stage B: CryptoBench
    cb_pl = fu.train_scaler(cb_file, plDDT_path=f"{CB_DIR}/plDDT")
    cb_di = fu.train_scaler(cb_file, distances_path=f"{CB_DIR}/residue-distances")
    cb_ds = fu.process_sequence_dataset(cb_file, tok, plDDT_scaler=cb_pl, plDDT_path=f"{CB_DIR}/plDDT",
                                        distances_scaler=cb_di,
                                        distances_path=f"{CB_DIR}/residue-distances")
    run_epochs("cryptobench", DataLoader(cb_ds, batch_size=1, shuffle=True, collate_fn=collate),
               CB_EPOCHS, ("llm", "classifier", "plDDT"), use_distance=True)
    train_seconds = time.time() - t0

    scores = _score_items(model, tok, job["items"], head_is_tuple=True)
    scores.to_parquet(f"{out_dir}/scores_{name}.parquet", index=False)
    if job.get("save_weights"):
        torch.save(model.state_dict(), f"{out_dir}/model_{name}.pt")
    meta = {"name": name, "run_id": run_id, "recipe": "full", "variant": job["variant"],
            "n_ligysis_lines": len(lig_lines), "n_ligysis_trained": lig_ds.num_rows,
            "n_ligysis_excluded_extra": len(excl), "n_cb_lines": len(cb_lines),
            "n_cb_trained": cb_ds.num_rows, "epochs": log, "train_seconds": train_seconds,
            "total_seconds": time.time() - t0,
            "cost_usd": (time.time() - t0) / 3600 * GPU_HOURLY_USD, "gpu": GPU,
            "n_scored_residues": len(scores), "torch": torch.__version__,
            "bitsandbytes": bnb.__version__}
    pd.Series(meta).to_json(f"{out_dir}/meta_{name}.json")
    vol.commit()
    print(f"[{name}] done in {meta['total_seconds']:.0f}s (~${meta['cost_usd']:.2f})", flush=True)
    return meta


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
def main(pilot: bool = False, spawn: bool = False, jobs: str = "", run_id: str = "",
         fetch: bool = False, extract: bool = False, recipe: str = "plain",
         variant: str = "clean"):
    import pandas as pd

    if fetch:
        call = modal.Function.from_name("cryptobench-s2p-finetune", "fetch_full_data").spawn()
        print(f"fetch spawned: {call.object_id}")
        return
    if extract:
        call = modal.Function.from_name("cryptobench-s2p-finetune", "extract_full_data").spawn()
        print(f"extract spawned: {call.object_id}")
        return

    inputs = pd.read_parquet("build/s2p_inputs.parquet").sort_values(["apo_id", "chain", "pos"])
    items = [(a, c, "".join(g["aa"])) for (a, c), g in inputs.groupby(["apo_id", "chain"])]
    f = modal.Function.from_name("cryptobench-s2p-finetune", "finetune")
    pilot_items = [it for it in items if it[0] in ("1a4uB", "1dpjA", "2po4A")]
    if pilot and recipe == "plain":
        job = _jobs(["0"], "pilot", pilot_items, limit=20)[0]
        call = f.spawn(job)
        print(f"pilot spawned: {call.object_id} -> cryptobench-s2p:ft/pilot/")
    elif recipe == "full":
        import json
        from pathlib import Path

        ff = modal.Function.from_name("cryptobench-s2p-finetune", "finetune_full")
        excl = json.loads(Path("finetuned_esm/ligysis_exclusions.json").read_text())["exclude"]
        names = ["0"] if pilot else jobs.split(",")
        for job in _jobs(names, "pilot_full" if pilot else run_id,
                         pilot_items if pilot else items, limit=20 if pilot else 0):
            held = "test" if job["name"] == "all" else f"train-fold-{job['name']}"
            job["variant"] = variant
            job["ligysis_exclude"] = excl[held] if variant == "clean" else []
            job["name"] = f"{variant}_{job['name']}"
            call = ff.spawn(job)
            print(f"spawned full/{job['name']} (CB folds {job['train_folds']}, "
                  f"{len(job['ligysis_exclude'])} extra LIGYSIS exclusions): {call.object_id}")
    elif spawn:
        assert run_id and jobs, "--spawn needs --jobs and --run-id"
        for job in _jobs(jobs.split(","), run_id, items):
            call = f.spawn(job)
            print(f"spawned {job['name']} (train folds {job['train_folds']}): {call.object_id}")
    else:
        raise SystemExit("pass --pilot or --spawn")
