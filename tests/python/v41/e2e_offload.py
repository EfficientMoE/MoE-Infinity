# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""GPU-gated smoke: Path B offloaded V4.1-Flash greedy decode.

Run (host with >= 40 GB VRAM and the full checkpoint on local SSD):

    MOE_V41_FLASH_CKPT=/ckpt/DeepSeek-V4.1-Flash \
    MOE_V41_OFFICIAL_DIR=/path/to/official-v41 \
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
    # Transformer.forward returns final-position logits only: [b, vocab].
    ids = prompt_ids.clone()
    _, logits, _ = model(ids, start_pos=0)
    next_id = logits.argmax(dim=-1, keepdim=True)
    out = [next_id]
    pos = ids.shape[1]
    for _ in range(steps - 1):
        _, logits, _ = model(next_id, start_pos=pos)
        next_id = logits.argmax(dim=-1, keepdim=True)
        out.append(next_id)
        pos += 1
    return torch.cat(out, dim=-1)


def test_greedy_decode_produces_stable_tokens(
    v41_ckpt_dir, v41_official_module
):
    from moe_store.wrappers.deepseek_v41 import load_offloaded_v41_flash
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    # Reference kernels allocate outputs with torch.get_default_dtype()
    # (e.g. fp8_gemm's C buffer); the official generate.py runs the whole
    # process under bf16, so the harness must too.
    torch.set_default_dtype(torch.bfloat16)
    official_dir = os.environ["MOE_V41_OFFICIAL_DIR"]
    tokenizer = AutoTokenizer.from_pretrained(v41_ckpt_dir)
    model, store = load_offloaded_v41_flash(
        v41_official_module,
        ckpt_path=v41_ckpt_dir,
        config_path=os.path.join(official_dir, "inference", "config.json"),
        device=device,
        max_resident_experts=16,
        text_only=True,
        tokenizer=tokenizer,
    )
    # Reference helpers (e.g. get_window_topk_idxs) build cached index
    # tensors on the default device; set it only AFTER load so the host
    # stores keep allocating on CPU.
    torch.set_default_device(device)
    prompt = "The capital of France is"
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)], device=device, dtype=torch.long
    )
    import time

    outs = []
    times = []
    for _ in range(2):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        outs.append(_greedy_decode(model, prompt_ids, steps=16))
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    assert torch.equal(outs[0], outs[1])
    decoded = tokenizer.decode(outs[0][0].tolist())
    assert "Paris" in decoded, f"degenerate greedy output: {decoded!r}"
    assert len(store.resident_experts()) <= 16
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"\ndecoded: {decoded!r}")
    print(f"peak resident GiB: {peak:.1f}")
    print(f"warm-run tok/s (16 greedy tokens): {16 / times[1]:.2f}")
