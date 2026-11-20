# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess probe for test_multi_gpu_offload_decode.py: the calling
thread's current CUDA device must be the same before and after model init and
after a generate call, and at the entry of every model forward (engine calls
used to leave it on the last device they touched)."""

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
model_name, offload = sys.argv[2], sys.argv[3]

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from moe_infinity import MoE  # noqa: E402

torch.cuda.set_device(0)
torch.zeros(1, device="cuda:0")
before = torch.cuda.current_device()
tok = AutoTokenizer.from_pretrained(model_name)
moe = MoE(
    model_name,
    {
        "offload_path": offload,
        "device_memory_ratio": 0.15,
        "expert_eviction_policy": "lcnu",
    },
)
after_init = torch.cuda.current_device()
engine = moe._native_generation_engine
orig = engine._model_forward
seen = set()


def wrapped(*args, **kwargs):
    seen.add(torch.cuda.current_device())
    return orig(*args, **kwargs)


engine._model_forward = wrapped
embed = moe.model.get_input_embeddings().weight.device
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(embed)
with torch.no_grad():
    moe.generate(
        ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
    )
print(
    "RESULT "
    + json.dumps(
        {
            "before": before,
            "after_init": after_init,
            "after_generate": torch.cuda.current_device(),
            "forward_entry": sorted(seen),
        }
    ),
    flush=True,
)
sys.stdout.flush()
