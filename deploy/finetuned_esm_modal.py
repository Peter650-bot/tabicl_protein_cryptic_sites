"""The published fine-tuned ESM-2 3B (Hidden / Seq2Pocket), run on Modal.

WHY THIS RUN EXISTS
    Škrhák et al. fine-tuned ESM-2 3B on CryptoBench and report pooled AUC 0.894 /
    AUPRC 0.475 on the 184 single-chain test structures. They never report the
    per-protein numbers this project is measured in. This job runs THEIR released
    weights, unchanged, so their model can be scored by our `evaluate.py`.
    Plan and gates: finetuned_esm/PLAN.md.

WHERE THE WEIGHTS COME FROM
    The papers point at a CESNET S3 tarball; that host accepts no connections
    (from here or from Modal, 2026-10-06). The same authors publish the files on
    HuggingFace as skrhakv/seq2pocket. `cbs-model.pt` there is the file their
    CryptoBench evaluation notebook loads (`data/models/cbs-model.pt`). It is
    pinned to one commit and its sha256 is checked against the LFS pointer, so a
    later re-upload cannot silently change what we score.

STEP 1: fetch and inspect
    The checkpoint is a whole pickled nn.Module. Before unpickling, the module
    and class names it references are read straight out of `data.pkl`, so the
    shim that makes their classes importable is chosen from evidence, not guessed.

STEP 4: score every chain
    One forward pass per chain, on the sequence THEIR extraction produces
    (`scripts/s2p_build_inputs.py`), head 1 through a sigmoid -- exactly what
    their evaluation notebook thresholds. Batch size 1: they padded the whole
    test set into one batch, but ESM masks padding, so that changes only fp16
    rounding. The 5 chains over 1,022 residues (all train) are chunked with the
    same `esm_embed.chunk_spans` our ESM run used; their loader skips them.

    The job writes its parquet to the Volume and is spawned, not streamed: long
    client streams die on this host, so nothing depends on the connection
    surviving the run.

USAGE
    modal run deploy/finetuned_esm_modal.py::inspect
    modal deploy deploy/finetuned_esm_modal.py
    modal run deploy/finetuned_esm_modal.py::score --run-id s2p_YYYYMMDD   # spawns
    modal volume get cryptobench-s2p scores/s2p_YYYYMMDD.parquet build/
"""
from __future__ import annotations

import modal

HF_REPO = "skrhakv/seq2pocket"
HF_REVISION = "610a4f8ce0d0465444ad6edc1bc783b3ff8f049a"   # 2026-08-26
CBS_FILE = "cbs-model.pt"
CBS_SHA256 = "d2db206dd85254a5d2404d4ccf8a1b99aea9952bfb0baaad19428eac6c970dac"
ESM_ID = "facebook/esm2_t36_3B_UR50D"
VOL_ROOT = "/vol"

# Their pinned versions (seq2pocket/requirements.txt). The checkpoint pickles
# transformers' EsmModel itself, so a different transformers could unpickle into
# a module whose attributes no longer match its forward().
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.48.1", "numpy==1.26.4", "huggingface_hub",
                 "pandas", "pyarrow", "scikit-learn")
    .add_local_file("deploy/s2p_models.py", "/app/s2p_models.py")
    .add_local_file("cryptic_tabicl/esm_embed.py", "/app/esm_embed.py")
)
GPU = "A10G"

vol = modal.Volume.from_name("cryptobench-s2p", create_if_missing=True)
app = modal.App("cryptobench-finetuned-esm")


def _fetch_checkpoint() -> str:
    """Download once into the Volume; verify the hash every time it is used."""
    import hashlib
    import os

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, CBS_FILE, revision=HF_REVISION,
                           cache_dir=f"{VOL_ROOT}/hf")
    vol.commit()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != CBS_SHA256:
        raise RuntimeError(f"{CBS_FILE} sha256 {h.hexdigest()} != pinned {CBS_SHA256}")
    print(f"{CBS_FILE}: {os.path.getsize(path)/1e9:.2f} GB, sha256 verified", flush=True)
    return path


def _pickle_globals(path: str) -> list[str]:
    """Every `module.Name` the checkpoint's pickle stream imports."""
    import pickletools
    import zipfile

    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        data = z.read(pkl)
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            found.add(f"{strings[-2]}.{strings[-1]}")
        elif op.name == "GLOBAL":
            found.add(arg.replace(" ", "."))
    return sorted(found)


def _load_model(path: str, device: str):
    """Unpickle with their classes registered under the name the pickle records."""
    import sys

    import torch

    sys.path.insert(0, "/app")
    import s2p_models

    for name in {g.rsplit(".", 1)[0] for g in _pickle_globals(path)}:
        if name.split(".")[0] in ("torch", "transformers", "collections", "builtins",
                                  "numpy", "tokenizers"):
            continue
        sys.modules[name] = s2p_models   # their module name -> verbatim copy
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model


@app.function(image=image, timeout=3600, cpu=4.0, memory=32768,
              volumes={VOL_ROOT: vol})
def inspect_checkpoint() -> dict:
    """Gate 1: hash, pickle contents, class, parameter shapes, one forward pass."""
    import torch
    from transformers import AutoTokenizer

    path = _fetch_checkpoint()
    globs = _pickle_globals(path)
    model = _load_model(path, "cpu")

    shapes = {n: list(p.shape) for n, p in model.named_parameters()
              if not n.startswith("llm.encoder.layer.")}
    n_params = sum(p.numel() for p in model.parameters())
    dtypes = sorted({str(p.dtype) for p in model.parameters()})

    # Smoke forward on CPU in fp32: shapes only, not a result.
    tok = AutoTokenizer.from_pretrained(ESM_ID)
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    enc = tok(seq, return_tensors="pt")
    with torch.no_grad():
        out = model.float()({"input_ids": enc["input_ids"],
                             "attention_mask": enc["attention_mask"]})
    heads = out if isinstance(out, tuple) else (out,)
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "pickle_globals": globs,
        "n_params": n_params,
        "dtypes": dtypes,
        "non_layer_param_shapes": shapes,
        "n_encoder_layers": len(model.llm.encoder.layer),
        "hidden_size": model.llm.config.hidden_size,
        "forward_head_shapes": [list(h.shape) for h in heads],
        "seq_len": len(seq),
        "head0_probs_first5": torch.sigmoid(heads[0].flatten()[1:6]).tolist(),
    }


@app.function(image=image, gpu=GPU, timeout=3 * 3600, memory=32768,
              volumes={VOL_ROOT: vol})
def score_chains(items: list, run_id: str) -> dict:
    """Per-residue sigmoid(head 1) for every (apo_id, chain, sequence) in `items`."""
    import io
    import os
    import sys
    import time

    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, "/app")
    from esm_embed import chunk_spans

    t0 = time.time()
    path = _fetch_checkpoint()
    model = _load_model(path, "cuda")             # fp16, as pickled
    cfg = model.llm.config
    # They never called .eval(). Their scores equal eval-mode scores only if
    # every dropout inside ESM is zero, so check it rather than assume it.
    dropout = {"hidden": cfg.hidden_dropout_prob,
               "attention": cfg.attention_probs_dropout_prob}
    assert all(p == 0.0 for p in dropout.values()), f"ESM dropout non-zero: {dropout}"
    tok = AutoTokenizer.from_pretrained(ESM_ID)

    frames = []
    for n, (apo_id, chain, seq) in enumerate(items):
        probs = []
        for start, end, keep_start, keep_end in chunk_spans(len(seq)):
            enc = tok(seq[start:end], return_tensors="pt")
            with torch.no_grad():
                out = model({k: v.cuda() for k, v in enc.items()})
            logits = (out[0] if isinstance(out, tuple) else out)[0, 1:-1, 0]
            assert logits.shape[0] == end - start, f"{apo_id}/{chain}: token count"
            p = torch.sigmoid(logits.float()).cpu().numpy()
            probs.append(p[keep_start - start: keep_end - start])
        p = np.concatenate(probs)
        assert len(p) == len(seq), f"{apo_id}/{chain}: {len(p)} scores for {len(seq)}"
        frames.append(pd.DataFrame({"apo_id": apo_id, "chain": chain,
                                    "pos": range(len(seq)), "s2p_score": p}))
        if n % 100 == 0:
            print(f"{n}/{len(items)} chains, {time.time()-t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(f"{VOL_ROOT}/scores", exist_ok=True)
    out = f"{VOL_ROOT}/scores/{run_id}.parquet"
    df.to_parquet(out, index=False)
    meta = {"run_id": run_id, "n_chains": len(items), "n_residues": len(df),
            "seconds": time.time() - t0, "gpu": GPU, "esm_dropout": dropout,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "hf_revision": HF_REVISION, "sha256": CBS_SHA256}
    pd.Series(meta).to_json(f"{VOL_ROOT}/scores/{run_id}.meta.json")
    vol.commit()
    print(f"wrote {out}: {len(df):,} residues in {time.time()-t0:.0f}s", flush=True)
    return meta


@app.local_entrypoint()
def score(run_id: str, pilot: bool = False):
    """Spawn the scoring job over every chain in build/s2p_inputs.parquet.

    --pilot scores three chains, one of them longer than 1,022 residues, so the
    chunked path runs before the full job does.
    """
    import pandas as pd

    inputs = pd.read_parquet("build/s2p_inputs.parquet").sort_values(["apo_id", "chain", "pos"])
    items = [(a, c, "".join(g["aa"])) for (a, c), g in inputs.groupby(["apo_id", "chain"])]
    if pilot:
        items = [it for it in items if it[0] in ("1a4uB", "1dpjA", "2po4A")]
    # Spawned on the DEPLOYED app: a call spawned from this ephemeral `modal run`
    # app would be killed the moment this entrypoint returns.
    f = modal.Function.from_name("cryptobench-finetuned-esm", "score_chains")
    call = f.spawn(items, run_id)
    print(f"spawned {len(items)} chains as {call.object_id}; "
          f"results -> cryptobench-s2p:scores/{run_id}.parquet")


@app.local_entrypoint()
def inspect():
    import json
    from pathlib import Path

    r = inspect_checkpoint.remote()
    out = Path("finetuned_esm/checkpoint_inspection.json")
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
