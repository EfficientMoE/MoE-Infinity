"""Fair V4.1-Flash DeepGEMM layer: permute + two GEMMs + SwiGLU/requant + reduce.

Does not overwrite vllm_mxfp4_grouped.json. The two-GEMM-only rows are the
same lower bound as that file, retimed on this process for the audit routing
and for the published equal-expert padding.
"""

import json
import math
import traceback

import torch
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.utils.deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous

H, I, E, TOPK = 5120, 2304, 384, 6
OUT = (
    "/mnt/raid0nvme0/leyang/moeinf-wt/v41-glm53-audit/"
    "benchmarks/kernel_audit/results/framework_bench/vllm_mxfp4_full_layer.json"
)


def active_experts(tokens: int) -> int:
    return int(round(E * (1.0 - (1.0 - TOPK / E) ** tokens)))


def median_us(fn, warmup=3, iters=10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def dram_gbs() -> float:
    n = 4 << 30
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty(n, dtype=torch.uint8, device="cuda")
    a.fill_(1)
    torch.cuda.synchronize()

    def copy():
        b.copy_(a)

    us = median_us(copy, warmup=2, iters=8)
    del a, b
    torch.cuda.empty_cache()
    # copy reads n and writes n
    return (2.0 * n) / (us * 1e-6) / 1e9


def swiglu_quant(inp: torch.Tensor):
    m, two_i = inp.shape
    if m % 128 == 0 and m % 8 == 0 and two_i % 256 == 0:
        q, s = silu_mul_per_token_group_quant_fp8_colmajor(
            inp, use_ue8m0=False, group_size=128
        )
        if not s.is_contiguous():
            s = s.contiguous()
        return q, s
    half = two_i // 2
    h = torch.nn.functional.silu(inp[:, :half].float()) * inp[:, half:].float()
    group = 128
    hv = h.view(m, half // group, group)
    amax = hv.abs().amax(dim=-1).clamp(min=1e-10)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = amax / fp8_max
    q = (hv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(m, half)
    return q, scale


def gemm(a, a_s, w, s, d, layout):
    m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a, a_s),
        (w, s),
        d,
        layout,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )


def main():
    print("device", torch.cuda.get_device_name(0), flush=True)
    bw = dram_gbs()
    print("dram_copy_gbs", round(bw, 1), flush=True)
    w1 = torch.randint(0, 127, (E, 2 * I, H // 2), device="cuda", dtype=torch.int8)
    s1 = torch.ones(E, 2 * I, H // 32, device="cuda")
    w2 = torch.randint(0, 127, (E, H, I // 2), device="cuda", dtype=torch.int8)
    s2 = torch.ones(E, H, I // 32, device="cuda")
    rows = []
    points = (("b1", 1, 6), ("b64", 64, 244), ("1K", 1024, 384), ("12K", 12288, 384))
    for name, tokens, expect in points:
        got = active_experts(tokens)
        if got != expect:
            print("active mismatch", name, got, expect, flush=True)
        n_exp = got
        ids = (torch.arange(tokens * TOPK, device="cuda") % n_exp).view(tokens, TOPK)
        ids = ids.to(torch.int32)
        wts = torch.rand(tokens, TOPK, device="cuda")
        x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16)
        xq, xs = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
        if not xs.is_contiguous():
            xs = xs.contiguous()
        out = torch.empty(tokens, H, device="cuda", dtype=torch.bfloat16)
        # Shrink the weight's expert count by views: ids only touch 0..n_exp-1,
        # but the tensor stays E so the kernel sees the published E. Passing
        # the full E matches DeepGemmFP4Experts' local_num_experts.
        rec = {
            "stack": "vllm",
            "point": name,
            "tokens": tokens,
            "experts": n_exp,
            "E_weights": E,
            "dram_copy_gbs": bw,
        }
        try:
            probe_a, probe_s, probe_ids, _, _ = deepgemm_moe_permute(
                xq, xs, ids, E, None, None
            )
            m_sum = probe_a.shape[0]
            d1 = torch.empty(m_sum, 2 * I, device="cuda", dtype=torch.bfloat16)
            d2 = torch.empty(m_sum, H, device="cuda", dtype=torch.bfloat16)
            rec["m_sum_aligned"] = m_sum

            def full(xq=xq, xs=xs, ids=ids, wts=wts, d1=d1, d2=d2, out=out):
                aq, aq_s, layout, inv, _align = deepgemm_moe_permute(
                    xq, xs, ids, E, None, None
                )
                gemm(aq, aq_s, w1, s1, d1, layout)
                a2, a2s = swiglu_quant(d1)
                gemm(a2, a2s, w2, s2, d2, layout)
                deepgemm_unpermute_and_reduce(d2, ids, wts, inv, None, out)

            def two(aq=probe_a, aq_s=probe_s, layout=probe_ids, d1=d1, d2=d2):
                gemm(aq, aq_s, w1, s1, d1, layout)
                gemm(aq[:, :I].contiguous() if False else aq, aq_s, w2, s2, d2, layout)

            # two-GEMM lower bound on the permuted audit routing, gate+up then
            # a second GEMM on already-quantized down activations of shape
            # [m_sum, I]. Built outside the timer, same as the published cell.
            a2 = torch.randint(0, 255, (m_sum, I), device="cuda", dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            a2s = torch.ones(m_sum, I // 128, device="cuda")

            def two_gemm(aq=probe_a, aq_s=probe_s, layout=probe_ids, d1=d1, d2=d2, a2=a2, a2s=a2s):
                gemm(aq, aq_s, w1, s1, d1, layout)
                gemm(a2, a2s, w2, s2, d2, layout)

            rec["full_layer_us"] = median_us(full)
            rec["two_gemm_audit_routing_us"] = median_us(two_gemm)
            rec["full_ok"] = bool(torch.isfinite(out.float()).all().item())
            rec["ok"] = True
            del probe_a, probe_s, probe_ids, a2, a2s, d1, d2
        except Exception as exc:
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        torch.cuda.empty_cache()
        if not rec["ok"]:
            break
    # Published equal-expert padding, two GEMMs only, to confirm the old cells.
    for name, tokens in (("b1", 1), ("b64", 64), ("1K", 1024), ("12K", 12288)):
        m = tokens * TOPK
        per = max(1, (m + E - 1) // E)
        m_sum = per * E
        a = torch.randint(0, 255, (m_sum, H), device="cuda", dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        a_s = torch.ones(m_sum, H // 128, device="cuda")
        a2 = torch.randint(0, 255, (m_sum, I), device="cuda", dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        a2_s = torch.ones(m_sum, I // 128, device="cuda")
        d1 = torch.empty(m_sum, 2 * I, device="cuda", dtype=torch.bfloat16)
        d2 = torch.empty(m_sum, H, device="cuda", dtype=torch.bfloat16)
        layout = torch.arange(E, device="cuda", dtype=torch.int32).repeat_interleave(per)
        rec = {
            "stack": "vllm",
            "kernel": "deepgemm_fp8xfp4_gateup_and_down",
            "point": name,
            "tokens": tokens,
            "m_sum": m_sum,
            "rows_per_expert": per,
            "experts_with_rows": E,
        }
        try:

            def run(a=a, a_s=a_s, a2=a2, a2_s=a2_s, d1=d1, d2=d2, layout=layout):
                gemm(a, a_s, w1, s1, d1, layout)
                gemm(a2, a2_s, w2, s2, d2, layout)

            rec["us_median"] = median_us(run)
            rec["ok"] = True
        except Exception as exc:
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        del a, a2, d1, d2
        torch.cuda.empty_cache()
        if not rec["ok"]:
            break
    with open(OUT, "w") as f:
        json.dump({"dram_copy_gbs": bw, "rows": rows}, f, indent=2)
        f.write("\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
