# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""GPU-gated end-to-end smoke for Hy3-FP8 offloaded serving.

Run manually (needs a GPU with >= 40 GB, the real checkpoint, and a
moe-store build with hy_v3 support, e.g.
PYTHONPATH=/path/to/moe-store with branch feat/hy3-support):

    HY3_FP8_CKPT=/path/to/Hy3-FP8 .venv/bin/python -m pytest \
        tests/python/hy3/test_e2e_offload.py -q -s
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.integration,
]


def test_offloaded_greedy_decode(hy3_ckpt_dir, tmp_path):
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    config = {
        "offload_path": os.path.join(str(tmp_path), "hy3-offload"),
        "device_memory_ratio": 0.75,
    }
    model = MoE(hy3_ckpt_dir, config)
    tokenizer = AutoTokenizer.from_pretrained(hy3_ckpt_dir)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=16,
        do_sample=False,
    )
    text = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[-1] :],
        skip_special_tokens=True,
    )
    print(f"decoded: {text!r}")
    # non-degenerate: some tokens, not a single repeated token
    new_tokens = output_ids[0][inputs.input_ids.shape[-1] :]
    assert new_tokens.numel() >= 8
    assert len(set(new_tokens.tolist())) > 1
    assert "Paris" in text
