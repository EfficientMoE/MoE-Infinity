# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""GPU-gated smoke: Path B offloaded V4.1-Flash greedy decode.

Run (host with >= 40 GB VRAM and the full checkpoint on local SSD):

    MOE_V41_FLASH_CKPT=/ckpt/DeepSeek-V4.1-Flash \
    MOE_V41_OFFICIAL_DIR=/path/to/official-v41/inference \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
        tests/python/v41/e2e_offload.py -q -s

The reference ``Transformer.forward(input_ids, start_pos)`` returns
``(output_ids, logits, main_hidden)`` and maintains KV caches through
``start_pos`` increments (verified reference API in
moe_store/wrappers/deepseek_v41/OFFICIAL_PIN.md); greedy decode here uses
the returned logits, not the internally sampled output_ids.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _greedy_decode(model, prompt_ids: torch.Tensor, steps: int):
    ids = prompt_ids.clone()
    _, logits, _ = model(ids, start_pos=0)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    out = [next_id]
    pos = ids.shape[1]
    for _ in range(steps - 1):
        _, logits, _ = model(next_id, start_pos=pos)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out.append(next_id)
        pos += 1
    return torch.cat(out, dim=-1)


def test_greedy_decode_produces_stable_tokens(
    v41_ckpt_dir, v41_official_module
):
    from moe_store.wrappers.deepseek_v41 import load_offloaded_v41_flash
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    official_dir = os.environ["MOE_V41_OFFICIAL_DIR"]
    tokenizer = AutoTokenizer.from_pretrained(v41_ckpt_dir)
    model, store = load_offloaded_v41_flash(
        v41_official_module,
        ckpt_path=v41_ckpt_dir,
        config_path=os.path.join(official_dir, "config.json"),
        device=device,
        max_resident_experts=16,
        text_only=True,
        tokenizer=tokenizer,
    )
    prompt = "The capital of France is"
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)], device=device, dtype=torch.long
    )
    outs = [_greedy_decode(model, prompt_ids, steps=16) for _ in range(2)]
    assert torch.equal(outs[0], outs[1])
    assert len(store.resident_experts()) <= 16
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"\ndecoded: {tokenizer.decode(outs[0][0].tolist())!r}")
    print(f"peak resident GiB: {peak:.1f}")
