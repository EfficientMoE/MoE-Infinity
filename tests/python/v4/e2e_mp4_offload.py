import json
import os
import sys

import torch
import torch.distributed as dist
from safetensors import safe_open

sys.path.insert(0, "/workspace/official/inference")
import model as M
from model import ModelArgs, Transformer

sys.path.insert(0, "/workspace/moe")
from moe_infinity.models.deepseek_v4.official_offload_adapter import (
    OfficialExpertHostStore,
    patch_moe_with_offload,
)

ws = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
lr = int(os.environ["LOCAL_RANK"])
dist.init_process_group("nccl")
torch.cuda.set_device(lr)
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(33377335)

with open("/workspace/official/inference/config.json") as f:
    margs = ModelArgs(**json.load(f))
margs.max_batch_size = 1
margs.max_seq_len = 4096
dev = torch.device("cuda", lr)
CKPT = "/ckpt"
shard = os.path.join(CKPT, f"model{rank}-mp{ws}.safetensors")

# Construct with routed experts replaced by None (shared_experts stay real).
_OrigMoEInit = M.MoE.__init__


def _patched_moe_init(self, layer_id, args):
    import torch.nn as _nn

    _OrigExpertList = M.nn.ModuleList
    # Temporarily make routed Expert construction a no-op by patching Expert to None-list
    orig_expert = M.Expert

    class _NullExpert(_nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, x, weights=None):
            return x

    M.Expert = _NullExpert
    try:
        _OrigMoEInit(self, layer_id, args)
    finally:
        M.Expert = orig_expert
    # Replace the nulled routed experts with None to free even the empty modules
    for i in range(len(self.experts)):
        self.experts[i] = None
    # Rebuild shared_experts as a REAL expert (was nulled above)
    self.shared_experts = orig_expert(
        args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
    )


M.MoE.__init__ = _patched_moe_init
with torch.device("cuda", lr):
    model = Transformer(margs)
M.MoE.__init__ = _OrigMoEInit
torch.set_grad_enabled(False)

store = OfficialExpertHostStore(
    CKPT, dev, max_resident_experts=8, shard_file=shard
)
patch_moe_with_offload(model, store, M)

torch.cuda.empty_cache()

with safe_open(shard, framework="pt", device="cpu") as f:
    np_ = dict(model.named_parameters())
    nb_ = dict(model.named_buffers())
    loaded = 0
    for k in f.keys():
        if ".ffn.experts." in k:
            continue
        if k.endswith(".scale"):
            wp = np_.get(k[:-6] + ".weight")
            if wp is not None and getattr(wp, "scale", None) is not None:
                t = f.get_tensor(k).to(dev)
                if wp.scale.shape == t.shape:
                    wp.scale.copy_(t)
                    loaded += 1
            continue
        tgt = np_.get(k)
        if tgt is None:
            tgt = nb_.get(k)
        if tgt is None:
            continue
        t = f.get_tensor(k).to(dev)
        if tgt.shape == t.shape:
            tgt.copy_(t)
            loaded += 1
if rank == 0:
    print(
        "RESULT loaded_nonexpert",
        loaded,
        "mem_gb",
        round(torch.cuda.memory_allocated() / 1e9, 2),
        flush=True,
    )
torch.set_default_device(dev)

prompt_ids = [
    0,
    128803,
    3085,
    344,
    223,
    20,
    13,
    20,
    33,
    9361,
    24752,
    16,
    128804,
    128822,
]
toks = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
with torch.inference_mode():
    logits = model.forward(toks, 0)
    argmax1 = int(logits[0].argmax().item())
if rank == 0:
    topv, topi = logits[0].float().topk(5)
    print("RESULT argmax", argmax1, "expected 22", flush=True)
    print("RESULT top5", topi.tolist(), flush=True)
    print("RESULT resident_experts", len(store.resident_experts()), flush=True)
    assert argmax1 == 22, f"prefill argmax {argmax1} != 22"
    print("RESULT QA_D1 PASS", flush=True)
dist.barrier()
dist.destroy_process_group()
