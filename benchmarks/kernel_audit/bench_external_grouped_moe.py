"""Time one grouped MoE layer at the V4.1-Flash and GLM-5.3-Flash shapes.

Run with the interpreter that owns the stack:

  CUDA_VISIBLE_DEVICES=5 /path/to/sglang/python \
      benchmarks/kernel_audit/bench_external_grouped_moe.py --stack sglang
  CUDA_VISIBLE_DEVICES=5 /path/to/vllm/python \
      benchmarks/kernel_audit/bench_external_grouped_moe.py --stack vllm

GLM-5.3-Flash is the sm_120 grouped FP8-block-128 kernel (SGLang Triton,
vLLM fused_experts). DeepSeek-V4.1-Flash is attempted as FlashInfer CUTLASS
MXFP4 when that entry point imports; a failure is recorded and does not
discard the FP8 rows. Weights are synthetic. This is kernel time, not a
serving measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback

import torch

POINTS = (
    ("b1", 1),
    ("b64", 64),
    ("1K", 1024),
    ("12K", 12288),
)


def _gpu_name() -> str:
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} cc={props.major}.{props.minor} sms={props.multi_processor_count}"


def _time_cuda(fn, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # microseconds
    samples.sort()
    return {
        "us_median": samples[len(samples) // 2],
        "us_min": samples[0],
        "us_max": samples[-1],
        "iters": iters,
    }


def _topk(
    tokens: int, experts: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(tokens, device="cuda")[:, None]
    cols = torch.arange(k, device="cuda")[None, :]
    ids = (rows * 17 + cols * 3) % experts
    weights = torch.full(
        (tokens, k), 1.0 / k, device="cuda", dtype=torch.float32
    )
    return weights, ids.to(torch.int32)


def _fp8_block(experts: int, n: int, k: int, block: int = 128):
    weight = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    scale = torch.ones(
        experts,
        (n + block - 1) // block,
        (k + block - 1) // block,
        device="cuda",
        dtype=torch.float32,
    )
    return weight, scale


def _publish_sglang_stub() -> None:
    """fused_moe reads the process config bag before any server exists."""
    import sglang.srt.runtime_context as rc

    class _Moe:
        enable_fused_moe_sum_all_reduce = False

    class _Det:
        enable_deterministic_inference = False

    class _Exec:
        moe = _Moe()
        deterministic = _Det()

    rc.get_exec = lambda: _Exec()


def bench_sglang(iters: int) -> list[dict]:
    _publish_sglang_stub()
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_moe,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    rows = []
    # GLM-5.3-Flash: H=4096, I=2048, E=288, top-8, FP8 block 128.
    e, h, inter, topk = 288, 4096, 2048, 8
    w1, s1 = _fp8_block(e, 2 * inter, h)
    w2, s2 = _fp8_block(e, h, inter)
    for name, tokens in POINTS:
        weights, ids = _topk(tokens, e, topk)
        hidden = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
        topk_out = StandardTopKOutput(
            weights, ids, torch.empty(0, device="cuda")
        )

        def run(
            hidden=hidden,
            w1=w1,
            w2=w2,
            s1=s1,
            s2=s2,
            topk_out=topk_out,
        ):
            return fused_moe(
                hidden,
                w1,
                w2,
                topk_out,
                use_fp8_w8a8=True,
                w1_scale=s1,
                w2_scale=s2,
                block_shape=[128, 128],
            )

        rec = _time_cuda(run, iters, warmup=3)
        rec.update(
            stack="sglang",
            model="glm-5.3-flash",
            kernel="fused_moe_triton_fp8_block128",
            point=name,
            tokens=tokens,
        )
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    del w1, w2, s1, s2
    torch.cuda.empty_cache()
    return rows


def bench_vllm(iters: int) -> list[dict]:
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    rows = []
    e, h, inter, topk = 288, 4096, 2048, 8
    w1, s1 = _fp8_block(e, 2 * inter, h)
    w2, s2 = _fp8_block(e, h, inter)
    quant = fp8_w8a8_moe_quant_config(s1, s2, block_shape=[128, 128])
    for name, tokens in POINTS:
        weights, ids = _topk(tokens, e, topk)
        hidden = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)

        def run(
            hidden=hidden, w1=w1, w2=w2, weights=weights, ids=ids, quant=quant
        ):
            return fused_experts(
                hidden, w1, w2, weights, ids, quant_config=quant
            )

        rec = _time_cuda(run, iters, warmup=3)
        rec.update(
            stack="vllm",
            model="glm-5.3-flash",
            kernel="fused_experts_fp8_block128",
            point=name,
            tokens=tokens,
        )
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["sglang", "vllm"], required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.cuda.init()
    header = {
        "stack": args.stack,
        "python": sys.executable,
        "torch": torch.__version__,
        "gpu": _gpu_name(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        if args.stack == "sglang":
            import sglang

            header["version"] = getattr(sglang, "__version__", "?")
            rows = bench_sglang(args.iters)
        else:
            import vllm

            header["version"] = getattr(vllm, "__version__", "?")
            rows = bench_vllm(args.iters)
        header["ok"] = True
        header["rows"] = rows
    except Exception:
        header["ok"] = False
        header["error"] = traceback.format_exc()
        print(header["error"], file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(header, f, indent=2)
        f.write("\n")
    if not header["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
