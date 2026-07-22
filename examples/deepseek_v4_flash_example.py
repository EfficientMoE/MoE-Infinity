# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""DeepSeek-V4-Flash FP4 expert-offloading example (multi-GPU).

DeepSeek-V4-Flash routed experts are FP4 (E2M1) and total ~132 GB across
43 layers x 256 experts x 3 projections, which does not fit on a single GPU.
This example streams routed experts from pinned host RAM onto the GPU on
demand (LRU GPU cache + async copy-stream prefetch), so resident GPU memory
drops to ~5-6 GB/rank.

This path is different from the simple ``MoE(...)`` HuggingFace API used in
``examples/deepseek_v2_chat_example.py``. V4-Flash ships a DeepSeek-native
checkpoint (FP4 routed + FP8 shared/attention weights) that the HuggingFace
``DeepseekV4ForCausalLM`` cannot load, so it is driven through the official
``inference/model.py`` plus MoE-Infinity's ``load_offloaded_v4_flash`` adapter.

Suggested hardware / environment
--------------------------------
- 4x GPUs, tensor-parallel mp4 (16 attention heads/rank). The official sparse
  attention kernel exceeds single-GPU shared memory at mp1, so mp4 is required.
- >= ~140 GB pinned host RAM to hold the FP4 experts off-GPU.
- The ``v4flash`` docker image (tilelang for ``fp4_gemm``; on Blackwell/SM120
  the native ``moe_infinity._v4_fp4`` CUDA path is used automatically and is
  1.5-3.2x faster than tilelang).
- Checkpoint converted to the official mp-sharded format once:

      python convert.py --hf-ckpt-path <HF_CKPT> --save-path <OUT> \
          --n-experts 256 --model-parallel 4

Run with torchrun:

    torchrun --nproc-per-node 4 examples/deepseek_v4_flash_example.py \
        --ckpt-path <OUT> --config-path <OUT>/config.json \
        --max-resident-experts 16
"""

import argparse
import os

import torch

# The official DeepSeek-V4 inference modules ship with the converted
# checkpoint and must be importable (e.g. run from the inference/ directory
# or add it to PYTHONPATH).
import model as M  # official inference/model.py
from generate import generate  # official inference/generate.py

from moe_infinity.models.deepseek_v4 import load_offloaded_v4_flash


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4-Flash FP4 expert-offloading example (mp4)"
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Path to the official mp-sharded V4-Flash checkpoint directory",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the official model config.json",
    )
    parser.add_argument(
        "--max-resident-experts",
        type=int,
        default=16,
        help=(
            "GPU expert-cache size per rank. Must be >= the max distinct "
            "experts routed in any single layer per step (64 for full "
            "coverage; 16 is a good throughput/memory default)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Tokens to generate"
    )
    parser.add_argument(
        "--use-native",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "FP4 expert kernel: auto (native on Blackwell, else tilelang), "
            "true (force native, needs _v4_fp4), false (force tilelang)"
        ),
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    shard_file = os.path.join(args.ckpt_path, f"model{rank}-mp4.safetensors")
    use_native = {"auto": None, "true": True, "false": False}[args.use_native]

    model, store = load_offloaded_v4_flash(
        M,
        args.ckpt_path,
        args.config_path,
        device,
        shard_file,
        max_resident_experts=args.max_resident_experts,
        use_native=use_native,
    )

    prompt_ids = [0, 1, 2, 3]
    out = generate(
        model,
        [prompt_ids],
        max_new_tokens=args.max_new_tokens,
        eos_id=1,
        temperature=0.0,
    )
    if rank == 0:
        print(out)


if __name__ == "__main__":
    main()
