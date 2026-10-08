# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""No-download MiniMax-M3 indexer prefill microbenchmark (eager vs kernel).

Builds one BF16 ``MiniMaxM3VLAttention`` at production geometry (hidden 6144,
64 Q heads / 4 KV heads, head dim 128, index block 128 / top-k 16 / local 1)
with random weights -- no checkpoint download -- and times eager prefill
against the block-sparse kernel path at ``S in {2048, 8192, 16384}`` using CUDA
events. Only the ``S=8192`` row gates kernel landing: the kernel lands only if
``t_kernel <= 0.8 * t_eager``. The ``S=2048`` and ``S=16384`` rows are
record-only.

The sweep needs a GPU; ``--help`` and ``--enforce`` run on CPU. Manual GPU run:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/minimax_m3/bench_indexer_attention.py

The sweep prints a ready-to-run enforcement command populated from its S=8192
row; run it and paste the ``pass``/``fail`` result on the implementation PR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

try:
    from benchmarks.minimax_m3.gates import kernel_speedup_passes, speedup_row
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gates import kernel_speedup_passes, speedup_row

ENFORCED_SEQ_LEN = 8192
DEFAULT_SEQ_LENS = (2048, 8192, 16384)
SCRIPT_PATH = "benchmarks/minimax_m3/bench_indexer_attention.py"


def _build_attention(device: torch.device, dtype: torch.dtype):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLAttention,
    )

    config = MiniMaxM3VLTextConfig(
        num_hidden_layers=1, layer_types=["minimax_m3_sparse"]
    )
    config._attn_implementation = "eager"
    module = (
        MiniMaxM3VLAttention(config, 0).to(device=device, dtype=dtype).eval()
    )
    return config, module


def _build_inputs(config, seq_len: int, device: torch.device, dtype):
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLRotaryEmbedding,
    )

    hidden_states = torch.randn(
        1, seq_len, config.hidden_size, device=device, dtype=dtype
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    rotary = MiniMaxM3VLRotaryEmbedding(config).to(device)
    cos, sin = rotary(hidden_states, position_ids)
    return hidden_states, (cos, sin), position_ids


def _time_ms(run, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def run_sweep(args) -> None:
    from moe_infinity.kernel.minimax_m3_sparse_attention import (
        indexer_attention_forward,
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.manual_seed(0)
    config, module = _build_attention(device, dtype)

    enforce_row = None
    for seq_len in args.seq_lens:
        hidden_states, position_embeddings, position_ids = _build_inputs(
            config, seq_len, device, dtype
        )

        def run_eager():
            return module(
                hidden_states,
                position_embeddings,
                None,
                position_ids=position_ids,
            )

        def run_kernel():
            return indexer_attention_forward(
                module,
                hidden_states,
                position_embeddings,
                None,
                position_ids=position_ids,
            )

        with torch.no_grad():
            t_eager = _time_ms(run_eager, args.warmup, args.iters)
            t_kernel = _time_ms(run_kernel, args.warmup, args.iters)

        row = speedup_row(seq_len, t_eager, t_kernel)
        print(json.dumps(row))
        if seq_len == ENFORCED_SEQ_LEN:
            enforce_row = row

    if enforce_row is not None:
        print(
            f"ENFORCE_COMMAND: python {SCRIPT_PATH} --enforce "
            f"--t-eager-ms {enforce_row['t_eager_ms']} "
            f"--t-kernel-ms {enforce_row['t_kernel_ms']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MiniMax-M3 indexer prefill microbenchmark (eager vs kernel)"
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQ_LENS),
        help="Prefill sequence lengths to sweep",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations per path"
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="Timed iterations per path"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Evaluate the S=8192 speedup gate from supplied timings and exit",
    )
    parser.add_argument("--t-eager-ms", type=float, default=None)
    parser.add_argument("--t-kernel-ms", type=float, default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.enforce:
        if args.t_eager_ms is None or args.t_kernel_ms is None:
            build_parser().error(
                "--enforce requires --t-eager-ms and --t-kernel-ms"
            )
        passed = kernel_speedup_passes(args.t_eager_ms, args.t_kernel_ms)
        print("pass" if passed else "fail")
        sys.exit(0 if passed else 1)

    if not torch.cuda.is_available():
        print(
            "ERROR: the eager-vs-kernel sweep requires a CUDA device.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_sweep(args)


if __name__ == "__main__":
    main()
