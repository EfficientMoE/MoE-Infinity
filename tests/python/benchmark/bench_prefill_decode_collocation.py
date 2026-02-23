"""
bench_prefill_decode_collocation.py

Benchmark four strategies for collocating prefill and decode attention on
the same GPU time-slice.  Expert I/O dominates MoE-Infinity latency; this
benchmark shows how much attention wall-time can be saved by overlapping
decode and prefill attention kernels.

Modes
-----
0  serial          decode → prefill on default stream
1  varlen-fused    single flash_attn_varlen_func (continuous-batching)
2  dual-stream     two CUDA streams, no SM partition
3  green-ctx-sm    SM-partitioned green contexts (CUDA ≥ 12.4)
4  green-ctx-sm-wq SM + work-queue partition       (CUDA ≥ 13.1 / drv ≥ 575)
"""

import argparse
import ctypes
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# flash-attn imports (hard requirement)
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
except ImportError:
    sys.exit(
        "flash-attn is required: pip install flash-attn\n"
        "(set FLASH_ATTENTION_FORCE_BUILD=TRUE if installing from source)"
    )

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Prefill–decode attention collocation benchmark"
)
parser.add_argument("--n-decode", type=int, default=8, help="decode batch size")
parser.add_argument("--n-prefill", type=int, default=1, help="prefill batch size")
parser.add_argument(
    "--decode-ctx", type=int, default=512, help="KV-cache length per decode request"
)
parser.add_argument(
    "--prefill-len", type=int, default=512, help="sequence length per prefill request"
)
parser.add_argument("--num-heads", type=int, default=32, help="attention heads")
parser.add_argument("--head-dim", type=int, default=128, help="per-head dimension")
parser.add_argument("--warmup", type=int, default=20, help="warmup iterations")
parser.add_argument("--iters", type=int, default=100, help="timed iterations")
args = parser.parse_args()

N_DEC = args.n_decode
N_PRE = args.n_prefill
L_DEC = args.decode_ctx
L_PRE = args.prefill_len
H = args.num_heads
D = args.head_dim
WARMUP = args.warmup
ITERS = args.iters
DEVICE = "cuda"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------
# Decode: Q [N_DEC, 1, H, D],  KV-cache [N_DEC, L_DEC, H, D]
q_dec = torch.randn(N_DEC, 1, H, D, dtype=DTYPE, device=DEVICE)
k_cache = torch.randn(N_DEC, L_DEC, H, D, dtype=DTYPE, device=DEVICE)
v_cache = torch.randn(N_DEC, L_DEC, H, D, dtype=DTYPE, device=DEVICE)

# Prefill: Q/K/V packed as [N_PRE*L_PRE, H, D]
q_pre = torch.randn(N_PRE * L_PRE, H, D, dtype=DTYPE, device=DEVICE)
k_pre = torch.randn(N_PRE * L_PRE, H, D, dtype=DTYPE, device=DEVICE)
v_pre = torch.randn(N_PRE * L_PRE, H, D, dtype=DTYPE, device=DEVICE)

# cu_seqlens for prefill (one sequence of length L_PRE per request)
cu_seqlens_pre = torch.tensor(
    [i * L_PRE for i in range(N_PRE + 1)], dtype=torch.int32, device=DEVICE
)

# ---------------------------------------------------------------------------
# Mode 1: combined varlen tensors (decode + prefill packed together)
# ---------------------------------------------------------------------------
# q_combined: N_DEC decode tokens (q_len=1 each) + N_PRE*L_PRE prefill tokens
q_dec_flat = q_dec.reshape(N_DEC, H, D)  # [N_DEC, H, D]
k_dec_flat = k_cache.reshape(N_DEC * L_DEC, H, D)  # [N_DEC*L_DEC, H, D]
v_dec_flat = v_cache.reshape(N_DEC * L_DEC, H, D)

q_combined = torch.cat([q_dec_flat, q_pre], dim=0)
k_combined = torch.cat([k_dec_flat, k_pre], dim=0)
v_combined = torch.cat([v_dec_flat, v_pre], dim=0)

# cu_seqlens_q: 1 token per decode request, L_PRE per prefill request
cu_q_list = [0]
for _ in range(N_DEC):
    cu_q_list.append(cu_q_list[-1] + 1)
for _ in range(N_PRE):
    cu_q_list.append(cu_q_list[-1] + L_PRE)
cu_seqlens_q_combined = torch.tensor(cu_q_list, dtype=torch.int32, device=DEVICE)

# cu_seqlens_k: L_DEC per decode request, L_PRE per prefill request
cu_k_list = [0]
for _ in range(N_DEC):
    cu_k_list.append(cu_k_list[-1] + L_DEC)
for _ in range(N_PRE):
    cu_k_list.append(cu_k_list[-1] + L_PRE)
cu_seqlens_k_combined = torch.tensor(cu_k_list, dtype=torch.int32, device=DEVICE)

max_seqlen_q_combined = L_PRE  # max(1, L_PRE)
max_seqlen_k_combined = max(L_DEC, L_PRE)

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)  # ms → µs
    return np.median(times), np.percentile(times, 5), np.percentile(times, 95)


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


def run_serial():
    flash_attn_with_kvcache(q_dec, k_cache, v_cache, causal=True)
    flash_attn_varlen_func(
        q_pre,
        k_pre,
        v_pre,
        cu_seqlens_q=cu_seqlens_pre,
        cu_seqlens_k=cu_seqlens_pre,
        max_seqlen_q=L_PRE,
        max_seqlen_k=L_PRE,
        causal=True,
    )


def run_varlen_fused():
    flash_attn_varlen_func(
        q_combined,
        k_combined,
        v_combined,
        cu_seqlens_q=cu_seqlens_q_combined,
        cu_seqlens_k=cu_seqlens_k_combined,
        max_seqlen_q=max_seqlen_q_combined,
        max_seqlen_k=max_seqlen_k_combined,
        causal=True,
    )


def run_dual_stream(stream1, stream2):
    def _fn():
        with torch.cuda.stream(stream1):
            flash_attn_with_kvcache(q_dec, k_cache, v_cache, causal=True)
        with torch.cuda.stream(stream2):
            flash_attn_varlen_func(
                q_pre,
                k_pre,
                v_pre,
                cu_seqlens_q=cu_seqlens_pre,
                cu_seqlens_k=cu_seqlens_pre,
                max_seqlen_q=L_PRE,
                max_seqlen_k=L_PRE,
                causal=True,
            )
        torch.cuda.current_stream().wait_stream(stream1)
        torch.cuda.current_stream().wait_stream(stream2)

    return _fn


# ---------------------------------------------------------------------------
# Green context setup (modes 3 & 4)
# ---------------------------------------------------------------------------

# ctypes struct definitions matching cuda.h (CUDA 13.1 / driver 590.x)
#
# CUdevResource layout (CUDA 13.1):
#   offset  0: type (int, 4 bytes)
#   offset  4: _padding[92] (92 bytes)
#   offset 96: anonymous union (RESOURCE_ABI_BYTES = 40 bytes)
#                sm:      CUdevSmResource      (4 uints = 16 bytes)
#                wqConfig: CUdevWorkqueueConfigResource (3 fields = 12 bytes)
#                _oversize: unsigned char[40]
#   offset 136: nextResource (pointer, 8 bytes)
#   total: 144 bytes


class CUdevResourceSM(ctypes.Structure):
    """CUdevSmResource from CUDA 13.1 cuda.h — 4 unsigned int fields."""
    _fields_ = [
        ("smCount", ctypes.c_uint),
        ("minSmPartitionSize", ctypes.c_uint),
        ("smCoscheduledAlignment", ctypes.c_uint),
        ("flags", ctypes.c_uint),
    ]


class CUdevWorkqueueConfigResource(ctypes.Structure):
    """CUdevWorkqueueConfigResource from CUDA 13.1 cuda.h."""
    _fields_ = [
        ("device", ctypes.c_int),
        ("wqConcurrencyLimit", ctypes.c_uint),
        ("sharingScope", ctypes.c_int),  # CUdevWorkqueueConfigScope enum
    ]


class CUdevResourceUnion(ctypes.Union):
    """Anonymous union inside CUdevResource; RESOURCE_ABI_BYTES = 40."""
    _fields_ = [
        ("sm", CUdevResourceSM),
        ("wqConfig", CUdevWorkqueueConfigResource),
        ("_data", ctypes.c_uint8 * 40),
    ]


class CUdevResource(ctypes.Structure):
    """
    CUdevResource from CUDA 13.1 cuda.h.
    type(4) + _pad(92) + union(40) + nextResource(8) = 144 bytes.
    The union (containing sm / wqConfig) starts at offset 96.
    """
    _fields_ = [
        ("type", ctypes.c_int),              # offset 0
        ("_pad", ctypes.c_uint8 * 92),       # offset 4
        ("res", CUdevResourceUnion),         # offset 96, size 40
        ("nextResource", ctypes.c_void_p),   # offset 136, size 8
    ]


# CUdevResourceDesc is an opaque POINTER (typedef struct CUdevResourceDesc_st *)
# NOT a struct — treat as c_void_p.
CUdevResourceDesc = ctypes.c_void_p

CUgreenCtx = ctypes.c_void_p
CUstream = ctypes.c_void_p

CU_DEV_RESOURCE_TYPE_SM = 1
CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG = 1000   # WQ scope configuration
CU_GREEN_CTX_DEFAULT_STREAM = 1
CU_STREAM_NON_BLOCKING = 1                     # required by cuGreenCtxStreamCreate
CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED = 1      # avoid scheduling interference


def _load_libcuda():
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def setup_green_ctx_sm(libcuda):
    """
    Create two SM-partitioned green contexts.
    Returns (stream1, stream2, sm_counts) or raises RuntimeError.
    """
    device = ctypes.c_int()
    ret = libcuda.cuInit(0)
    if ret != 0:
        raise RuntimeError(f"cuInit failed: {ret}")
    ret = libcuda.cuDeviceGet(ctypes.byref(device), 0)
    if ret != 0:
        raise RuntimeError(f"cuDeviceGet failed: {ret}")

    # Get total SM resource
    sm_res = CUdevResource()
    ret = libcuda.cuDeviceGetDevResource(
        device, ctypes.byref(sm_res), CU_DEV_RESOURCE_TYPE_SM
    )
    if ret != 0:
        raise RuntimeError(f"cuDeviceGetDevResource(SM) failed: {ret}")

    total_sms = sm_res.res.sm.smCount

    # Split into 2 equal partitions.
    # minCount = total//2 gives each group half the SMs; minCount=1 would give
    # only smCoscheduledAlignment SMs per group (2 on Ampere), wasting the rest.
    result_arr = (CUdevResource * 2)()
    nb = ctypes.c_uint(2)
    min_count = total_sms // 2
    ret = libcuda.cuDevSmResourceSplitByCount(
        result_arr, ctypes.byref(nb), ctypes.byref(sm_res), None, 0, min_count
    )
    if ret != 0:
        raise RuntimeError(f"cuDevSmResourceSplitByCount failed: {ret}")

    sm0 = result_arr[0].res.sm.smCount
    sm1 = result_arr[1].res.sm.smCount

    # Generate descriptors.
    # CUdevResourceDesc is a POINTER (c_void_p), not a struct.
    # cuDevResourceGenerateDesc writes the allocated pointer into *phDesc.
    desc1 = ctypes.c_void_p()
    desc2 = ctypes.c_void_p()
    ret = libcuda.cuDevResourceGenerateDesc(
        ctypes.byref(desc1), result_arr, 1  # result_arr → pointer to elem[0]
    )
    if ret != 0:
        raise RuntimeError(f"cuDevResourceGenerateDesc[0] failed: {ret}")
    _ptr1 = ctypes.cast(
        ctypes.addressof(result_arr) + ctypes.sizeof(CUdevResource),
        ctypes.POINTER(CUdevResource),
    )
    ret = libcuda.cuDevResourceGenerateDesc(
        ctypes.byref(desc2), _ptr1, 1  # offset pointer to elem[1]
    )
    if ret != 0:
        raise RuntimeError(f"cuDevResourceGenerateDesc[1] failed: {ret}")

    # Create green contexts — pass desc by value (it's a pointer)
    gctx1 = CUgreenCtx()
    gctx2 = CUgreenCtx()
    ret = libcuda.cuGreenCtxCreate(
        ctypes.byref(gctx1), desc1, device, CU_GREEN_CTX_DEFAULT_STREAM
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxCreate[0] failed: {ret}")
    ret = libcuda.cuGreenCtxCreate(
        ctypes.byref(gctx2), desc2, device, CU_GREEN_CTX_DEFAULT_STREAM
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxCreate[1] failed: {ret}")

    # Create streams bound to green contexts.
    # CU_STREAM_NON_BLOCKING (0x1) is REQUIRED by cuGreenCtxStreamCreate;
    # passing flags=0 causes CUDA_ERROR_INVALID_VALUE.
    raw_s1 = CUstream()
    raw_s2 = CUstream()
    ret = libcuda.cuGreenCtxStreamCreate(
        ctypes.byref(raw_s1), gctx1, CU_STREAM_NON_BLOCKING, 0
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxStreamCreate[0] failed: {ret}")
    ret = libcuda.cuGreenCtxStreamCreate(
        ctypes.byref(raw_s2), gctx2, CU_STREAM_NON_BLOCKING, 0
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxStreamCreate[1] failed: {ret}")

    stream1 = torch.cuda.ExternalStream(raw_s1.value)
    stream2 = torch.cuda.ExternalStream(raw_s2.value)

    return stream1, stream2, (sm0, sm1, total_sms)


def setup_green_ctx_sm_wq(libcuda):
    """
    Create two SM+WQ-balanced green contexts (CUDA 13.1 / driver ≥ 590.x).

    Uses CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG (1000) to set
    CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED, which prevents scheduling
    interference between the two green contexts' work queues.

    Returns (stream1, stream2, sm_counts) or raises RuntimeError.
    """
    device = ctypes.c_int()
    libcuda.cuInit(0)
    libcuda.cuDeviceGet(ctypes.byref(device), 0)

    # SM resource — split into 2 equal halves
    sm_res = CUdevResource()
    ret = libcuda.cuDeviceGetDevResource(
        device, ctypes.byref(sm_res), CU_DEV_RESOURCE_TYPE_SM
    )
    if ret != 0:
        raise RuntimeError(f"cuDeviceGetDevResource(SM) failed: {ret}")

    total_sms = sm_res.res.sm.smCount
    result_arr = (CUdevResource * 2)()
    nb = ctypes.c_uint(2)
    min_count = total_sms // 2
    ret = libcuda.cuDevSmResourceSplitByCount(
        result_arr, ctypes.byref(nb), ctypes.byref(sm_res), None, 0, min_count
    )
    if ret != 0:
        raise RuntimeError(f"cuDevSmResourceSplitByCount(SM) failed: {ret}")

    # Work-queue configuration resource (CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG=1000).
    # This is a CONFIGURATION (scope hint), not a splittable resource.
    # Setting sharingScope=BALANCED requests non-overlapping WQ scheduling.
    wq_res = CUdevResource()
    ret = libcuda.cuDeviceGetDevResource(
        device, ctypes.byref(wq_res), CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
    )
    if ret != 0:
        raise RuntimeError(
            f"cuDeviceGetDevResource(WORKQUEUE_CONFIG) failed: {ret} "
            "(driver < 590.x / CUDA < 13.1)"
        )
    wq_res.res.wqConfig.sharingScope = CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED

    # Build resource pairs [SM_partition, WQ_config] for each green context.
    # Use memmove to copy CUdevResource structs (ctypes assignment is shallow).
    res_pair0 = (CUdevResource * 2)()
    res_pair1 = (CUdevResource * 2)()
    _sz = ctypes.sizeof(CUdevResource)
    ctypes.memmove(ctypes.addressof(res_pair0[0]), ctypes.addressof(result_arr[0]), _sz)
    ctypes.memmove(ctypes.addressof(res_pair0[1]), ctypes.addressof(wq_res), _sz)
    ctypes.memmove(ctypes.addressof(res_pair1[0]), ctypes.addressof(result_arr[1]), _sz)
    ctypes.memmove(ctypes.addressof(res_pair1[1]), ctypes.addressof(wq_res), _sz)

    desc1 = ctypes.c_void_p()
    desc2 = ctypes.c_void_p()
    ret = libcuda.cuDevResourceGenerateDesc(ctypes.byref(desc1), res_pair0, 2)
    if ret != 0:
        raise RuntimeError(f"cuDevResourceGenerateDesc[0] (SM+WQ) failed: {ret}")
    ret = libcuda.cuDevResourceGenerateDesc(ctypes.byref(desc2), res_pair1, 2)
    if ret != 0:
        raise RuntimeError(f"cuDevResourceGenerateDesc[1] (SM+WQ) failed: {ret}")

    gctx1 = CUgreenCtx()
    gctx2 = CUgreenCtx()
    ret = libcuda.cuGreenCtxCreate(
        ctypes.byref(gctx1), desc1, device, CU_GREEN_CTX_DEFAULT_STREAM
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxCreate[0] (SM+WQ) failed: {ret}")
    ret = libcuda.cuGreenCtxCreate(
        ctypes.byref(gctx2), desc2, device, CU_GREEN_CTX_DEFAULT_STREAM
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxCreate[1] (SM+WQ) failed: {ret}")

    raw_s1 = CUstream()
    raw_s2 = CUstream()
    ret = libcuda.cuGreenCtxStreamCreate(
        ctypes.byref(raw_s1), gctx1, CU_STREAM_NON_BLOCKING, 0
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxStreamCreate[0] (SM+WQ) failed: {ret}")
    ret = libcuda.cuGreenCtxStreamCreate(
        ctypes.byref(raw_s2), gctx2, CU_STREAM_NON_BLOCKING, 0
    )
    if ret != 0:
        raise RuntimeError(f"cuGreenCtxStreamCreate[1] (SM+WQ) failed: {ret}")

    stream1 = torch.cuda.ExternalStream(raw_s1.value)
    stream2 = torch.cuda.ExternalStream(raw_s2.value)

    sm0 = result_arr[0].res.sm.smCount
    sm1 = result_arr[1].res.sm.smCount
    return stream1, stream2, (sm0, sm1, total_sms)


def make_green_stream_fn(stream1, stream2):
    def _fn():
        with torch.cuda.stream(stream1):
            flash_attn_with_kvcache(q_dec, k_cache, v_cache, causal=True)
        with torch.cuda.stream(stream2):
            flash_attn_varlen_func(
                q_pre,
                k_pre,
                v_pre,
                cu_seqlens_q=cu_seqlens_pre,
                cu_seqlens_k=cu_seqlens_pre,
                max_seqlen_q=L_PRE,
                max_seqlen_k=L_PRE,
                causal=True,
            )
        torch.cuda.current_stream().wait_stream(stream1)
        torch.cuda.current_stream().wait_stream(stream2)

    return _fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        sys.exit("No CUDA device found.")

    gpu_name = torch.cuda.get_device_name(0)
    drv_ver = torch.cuda.get_device_properties(0).major  # SM major (not driver)
    sm_count_total = torch.cuda.get_device_properties(0).multi_processor_count
    # Driver version from nvidia-smi not easily available; use torch version info
    cuda_rt_ver = torch.version.cuda  # e.g. "12.4"

    sep = "=" * 64
    print(sep)
    print("bench_prefill_decode_collocation.py")
    print(
        f"Config: decode={N_DEC}×(q=1,kv={L_DEC})  "
        f"prefill={N_PRE}×(q={L_PRE})"
    )
    print(
        f"        heads={H}  head_dim={D}  dtype=BF16  "
        f"warmup={WARMUP}  iters={ITERS}"
    )
    print(f"GPU: {gpu_name}  CUDA runtime {cuda_rt_ver}  SM count: {sm_count_total}")
    print(sep)
    print()

    results = {}

    # ------------------------------------------------------------------
    # Throughput helpers — computed once, used throughout
    # ------------------------------------------------------------------
    # Attention FLOPs per colocation step (forward pass only):
    #   Prefill (causal): Q@K^T + A@V with triangular mask → eff. seq_k = L/2
    #     FLOPs = 2 × (2 × N_PRE × H × L_PRE × (L_PRE/2) × D)
    #           = 2 × N_PRE × H × L_PRE² × D
    #   Decode: S_q=1, full KV-cache, N_DEC requests
    #     FLOPs = 2 × (2 × N_DEC × H × 1 × L_DEC × D)
    #           = 4 × N_DEC × H × L_DEC × D
    FLOPS_PRE  = 2 * N_PRE * H * L_PRE**2 * D
    FLOPS_DEC  = 4 * N_DEC * H * L_DEC  * D
    FLOPS_STEP = FLOPS_PRE + FLOPS_DEC

    def tflops(us):
        return FLOPS_STEP / (us * 1e-6) / 1e12

    def dec_tps(us):  # new tokens generated per second (N_DEC per step)
        return N_DEC / (us * 1e-6)

    def pre_tps(us):  # prefill tokens processed per second (N_PRE × L_PRE per step)
        return N_PRE * L_PRE / (us * 1e-6)

    # ------------------------------------------------------------------
    # Baseline: decode-only and prefill-only (for overlap analysis)
    # ------------------------------------------------------------------
    print("Baseline (separate attention timings, no colocation):")
    med_dec_only, _, _ = bench(
        lambda: flash_attn_with_kvcache(q_dec, k_cache, v_cache, causal=True)
    )
    med_pre_only, _, _ = bench(
        lambda: flash_attn_varlen_func(
            q_pre,
            k_pre,
            v_pre,
            cu_seqlens_q=cu_seqlens_pre,
            cu_seqlens_k=cu_seqlens_pre,
            max_seqlen_q=L_PRE,
            max_seqlen_k=L_PRE,
            causal=True,
        )
    )
    ideal_parallel = max(med_dec_only, med_pre_only)
    serial_sum     = med_dec_only + med_pre_only

    print(
        f"  decode-only  ({N_DEC}×q=1,kv={L_DEC}):  "
        f"{med_dec_only:7.1f}µs  "
        f"{FLOPS_DEC/(med_dec_only*1e-6)/1e12:.2f} TFLOPS  "
        f"{dec_tps(med_dec_only):,.0f} new-tok/s"
    )
    print(
        f"  prefill-only ({N_PRE}×q={L_PRE}):       "
        f"{med_pre_only:7.1f}µs  "
        f"{FLOPS_PRE/(med_pre_only*1e-6)/1e12:.2f} TFLOPS  "
        f"{pre_tps(med_pre_only):,.0f} pre-tok/s"
    )
    print(
        f"  serial-sum (no overlap):      {serial_sum:7.1f}µs"
    )
    print(
        f"  ideal-overlap (perfect):      {ideal_parallel:7.1f}µs  "
        f"= max(decode, prefill)"
    )
    print()

    # ------------------------------------------------------------------
    # Mode 0: serial
    # ------------------------------------------------------------------
    print("Mode 0  serial           decode→prefill, default stream")
    med, p5, p95 = bench(run_serial)
    results["serial"] = (med, p5, p95, None)
    print(
        f"  serial                  median={med:8.1f}µs  p5={p5:.0f}  p95={p95:.0f}"
        f"  {tflops(med):.2f} TFLOPS"
    )
    print()

    # ------------------------------------------------------------------
    # Mode 1: varlen-fused
    # ------------------------------------------------------------------
    print("Mode 1  varlen-fused     single flash_attn_varlen_func (continuous batching)")
    med, p5, p95 = bench(run_varlen_fused)
    results["varlen-fused"] = (med, p5, p95, None)
    print(
        f"  varlen-fused            median={med:8.1f}µs  p5={p5:.0f}  p95={p95:.0f}"
        f"  {tflops(med):.2f} TFLOPS"
    )
    print()

    # ------------------------------------------------------------------
    # Mode 2: dual-stream
    # ------------------------------------------------------------------
    print("Mode 2  dual-stream      two CUDA streams, no SM partition")
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    fn_ds = run_dual_stream(s1, s2)
    med, p5, p95 = bench(fn_ds)
    results["dual-stream"] = (med, p5, p95, None)
    print(
        f"  dual-stream             median={med:8.1f}µs  p5={p5:.0f}  p95={p95:.0f}"
        f"  {tflops(med):.2f} TFLOPS"
    )
    print()

    # ------------------------------------------------------------------
    # Mode 3: green-ctx-sm  (CUDA ≥ 12.4)
    # ------------------------------------------------------------------
    libcuda = _load_libcuda()
    if libcuda is None:
        print("Mode 3  green-ctx-sm     [SKIPPED: libcuda.so not found]")
    else:
        print("Mode 3  green-ctx-sm     SM-partitioned green contexts, CUDA 12.4+")
        try:
            gs1, gs2, sm_info = setup_green_ctx_sm(libcuda)
            fn_gcsm = make_green_stream_fn(gs1, gs2)
            med, p5, p95 = bench(fn_gcsm)
            results["green-ctx-sm"] = (med, p5, p95, sm_info)
            print(
                f"  green-ctx-sm            median={med:8.1f}µs  "
                f"p5={p5:.0f}  p95={p95:.0f}  "
                f"{tflops(med):.2f} TFLOPS  "
                f"(SM split: {sm_info[0]} + {sm_info[1]})"
            )
        except (RuntimeError, AttributeError) as e:
            print(f"  [SKIPPED: {e}]")
    print()

    # ------------------------------------------------------------------
    # Mode 4: green-ctx-sm-wq  (CUDA ≥ 13.1 / driver ≥ 575.x)
    # ------------------------------------------------------------------
    if libcuda is None:
        print("Mode 4  green-ctx-sm-wq  [SKIPPED: libcuda.so not found]")
    else:
        print(
            "Mode 4  green-ctx-sm-wq  SM + work-queue partition, CUDA 13.1+  "
            "(driver ≥ 575.x)"
        )
        try:
            gs1_wq, gs2_wq, sm_info_wq = setup_green_ctx_sm_wq(libcuda)
            fn_gcsmwq = make_green_stream_fn(gs1_wq, gs2_wq)
            med, p5, p95 = bench(fn_gcsmwq)
            results["green-ctx-sm-wq"] = (med, p5, p95, sm_info_wq)
            print(
                f"  green-ctx-sm-wq         median={med:8.1f}µs  "
                f"p5={p5:.0f}  p95={p95:.0f}  "
                f"{tflops(med):.2f} TFLOPS  "
                f"(SM split: {sm_info_wq[0]} + {sm_info_wq[1]})"
            )
        except (RuntimeError, AttributeError) as e:
            drv_msg = "driver 550.x < 575.x required for CUDA 13.1 WQ partition"
            print(f"  [SKIPPED: {drv_msg}]")
            print(f"    (detail: {e})")
    print()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    row_order = [
        "serial",
        "varlen-fused",
        "dual-stream",
        "green-ctx-sm",
        "green-ctx-sm-wq",
    ]
    serial_med = results["serial"][0]

    print(sep)
    print("Summary  —  throughput & latency")
    print(sep)
    hdr = (
        f"  {'Mode':<20} {'Latency':>9}  {'Spdup':>5}  "
        f"{'TFLOPS':>6}  {'Dec-tok/s':>10}  {'Pre-tok/s':>10}  {'Overlap-eff':>11}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in row_order:
        if key not in results:
            continue
        med, p5, p95, sm_info = results[key]
        speedup   = serial_med / med
        tf        = tflops(med)
        dtps      = dec_tps(med)
        ptps      = pre_tps(med)
        # Overlap efficiency: how close the parallel modes come to ideal concurrency.
        # ideal = max(decode_only, prefill_only); efficiency = ideal / actual.
        # Serial and varlen-fused don't use explicit parallelism → show "—".
        if key in ("serial", "varlen-fused"):
            ov_str = "—"
        else:
            ov_eff = ideal_parallel / med * 100
            ov_str = f"{ov_eff:6.1f}%"
        sm_tag = ""
        if sm_info is not None:
            sm_tag = f"  [{sm_info[0]}+{sm_info[1]} SMs]"
        print(
            f"  {key:<20} {med:7.1f}µs  {speedup:5.2f}x  "
            f"{tf:6.2f}  {dtps:>10,.0f}  {ptps:>10,.0f}  {ov_str:>11}{sm_tag}"
        )
    print()

    # ------------------------------------------------------------------
    # Generation-throughput projection
    # ------------------------------------------------------------------
    # Model: N_DEC concurrent decode streams are always running.  Every step
    # a prefill batch (N_PRE requests, L_PRE tokens each) is also served.
    # Attention is the measured bottleneck; all other work is ignored.
    #
    # decode_overhead = extra latency imposed on decode by adding prefill:
    #   (mode_latency - decode_only) / decode_only × 100 %
    # Effective decode generation rate = N_DEC / mode_latency
    print(sep)
    print("Generation-throughput projection  (attention bottleneck, 1 prefill/step)")
    print(sep)
    # Dec overhead: extra latency added to decode by collocating prefill,
    #   relative to decode-only baseline.  Higher = worse for decode latency.
    # Δ vs serial: step latency difference relative to serial colocation.
    #   Negative = this mode is slower than serial (costs more per step).
    print(
        f"  {'Mode':<20} {'Step-lat':>9}  {'Dec-tok/s':>10}  "
        f"{'Pre-tok/s':>10}  {'Dec overhead':>13}  {'Δ vs serial':>12}"
    )
    print("  " + "-" * 80)
    for key in row_order:
        if key not in results:
            continue
        med = results[key][0]
        dtps_val = dec_tps(med)
        ptps_val = pre_tps(med)
        dec_overhead = (med - med_dec_only) / med_dec_only * 100
        delta_vs_serial = med - serial_med   # positive = slower than serial
        print(
            f"  {key:<20} {med:7.1f}µs  {dtps_val:>10,.0f}  "
            f"{ptps_val:>10,.0f}  {dec_overhead:>+12.1f}%  {delta_vs_serial:>+10.1f}µs"
        )
    print()
    print(
        f"  Decode-only baseline: {med_dec_only:.1f}µs  →  "
        f"{dec_tps(med_dec_only):,.0f} new-tok/s  (no prefill colocation)"
    )
    print(
        f"  Ideal-overlap target: {ideal_parallel:.1f}µs  →  "
        f"{dec_tps(ideal_parallel):,.0f} new-tok/s  "
        f"(perfect concurrency, {ideal_parallel/serial_sum*100:.0f}% of serial-sum)"
    )
    print()
    print(
        "  Note: Dec overhead < 0% would mean free prefill (impossible without true\n"
        "  spatial parallelism). For green-ctx to win, each partition must saturate\n"
        "  its SM allocation — try larger --n-decode / --decode-ctx / --prefill-len."
    )
    print()

    # Upgrade hint for mode 4
    if "green-ctx-sm-wq" not in results:
        print(
            "Tip: upgrade to NVIDIA driver ≥ 575.x (CUDA 13.1) to enable mode 4\n"
            "     (SM + work-queue partition eliminates scheduling contention)"
        )


if __name__ == "__main__":
    main()
