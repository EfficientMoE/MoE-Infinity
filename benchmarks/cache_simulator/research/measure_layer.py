"""Measure the batch-1 decode GEMV time of one MoE layer's weights on a GPU.

Anchors ``timing_model``'s ``layer_compute_s`` (``compute_scale``) with a
measurement on the capture host instead of the bandwidth roofline alone. One
synthetic layer of random weights with the model's shapes is replayed in a
CUDA graph: the q / k / v / o attention projections, the router, and
``top_k`` experts (fused gate+up, then down, SiLU-gated). Attention over the
KV cache, norms and routing kernels are not included, so this is a LOWER
bound on a real engine's per-layer time with all experts resident.

FP8 layers are timed with BF16 weights of the same shapes and scaled by the
byte ratio (0.5), i.e. assuming the same achieved bandwidth.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m \
        benchmarks.cache_simulator.research.measure_layer
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

#: name -> (hidden, q heads, kv heads, head dim, moe intermediate, top_k,
#: experts, bytes per weight in the deployed checkpoint)
SHAPES = {
    "qwen3-30b-a3b": (2048, 32, 4, 128, 768, 8, 128, 2.0),
    "qwen3-235b-a22b-fp8": (4096, 64, 4, 128, 1536, 8, 128, 1.0),
}


def measure(name: str, repeats: int = 200) -> Dict[str, float]:
    import torch

    hidden, heads, kv_heads, head_dim, inter, top_k, experts, width = SHAPES[
        name
    ]
    dev, dt = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)

    def weight(rows: int, cols: int):
        return torch.randn(rows, cols, device=dev, dtype=dt, generator=gen)

    wq = weight(hidden, heads * head_dim)
    wk = weight(hidden, kv_heads * head_dim)
    wv = weight(hidden, kv_heads * head_dim)
    wo = weight(heads * head_dim, hidden)
    router = weight(hidden, experts)
    gate_up = [weight(hidden, 2 * inter) for _ in range(top_k)]
    down = [weight(inter, hidden) for _ in range(top_k)]
    x = torch.randn(1, hidden, device=dev, dtype=dt, generator=gen)

    def layer():
        q, k, v = x @ wq, x @ wk, x @ wv
        h = (q[:, : heads * head_dim]) @ wo + x
        _ = h @ router
        out = torch.zeros_like(h)
        for gu, dn in zip(gate_up, down):
            a = h @ gu
            out += (torch.nn.functional.silu(a[:, :inter]) * a[:, inter:]) @ dn
        return out, k, v

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    bf16_s = start.elapsed_time(end) / 1e3 / repeats
    params = (
        hidden * heads * head_dim * 2
        + 2 * hidden * kv_heads * head_dim
        + hidden * experts
        + top_k * 3 * hidden * inter
    )
    return {
        "bf16_layer_s": bf16_s,
        "bf16_bytes_per_s": params * 2.0 / bf16_s,
        "deployed_layer_s": bf16_s * width / 2.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(SHAPES))
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)

    import torch

    print(f"device: {torch.cuda.get_device_name()}, torch {torch.__version__}")
    print(
        "| Model | BF16 layer (us) | achieved GB/s "
        "| deployed-precision layer (us) |"
    )
    print("| --- | ---: | ---: | ---: |")
    for name in args.models:
        row = measure(name, args.repeats)
        print(
            f"| {name} | {row['bf16_layer_s'] * 1e6:.1f} "
            f"| {row['bf16_bytes_per_s'] / 1e9:.0f} "
            f"| {row['deployed_layer_s'] * 1e6:.1f} |"
        )


if __name__ == "__main__":
    main()
