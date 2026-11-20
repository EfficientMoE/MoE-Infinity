# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Custom-kernel entry points run on the device of their tensors (#362).

With the layers of a model on several GPUs the calling thread's *current*
CUDA device is often not the device of the tensors. A Triton launch (or a
native kernel using ``getCurrentCUDAStream()`` / the NULL stream) then ran in
the other GPU's context on its default stream: peer access lets it touch the
tensors, but nothing orders it against the stream that produces its inputs.

Each case below builds small synthetic inputs on ``cuda:1``, calls the entry
point with ``cuda:0`` current and checks that

* every CUDA kernel/memcpy the call enqueued ran on device 1 (CUPTI device
  index from ``torch.profiler``),
* the result equals the call made with ``cuda:1`` current (the call is made
  with a non-default current stream on device 0, so a kernel that takes "the
  current stream" of the wrong device cannot work by accident), and
* the caller's current device is unchanged afterwards.

Needs two GPUs; cases that need the RTX PRO 6000 kernels or an extension that
is not built are skipped.
"""

import importlib

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu]

DEV = 1  # tensors live here; the thread's current device is 0


def _need_two_gpus():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 CUDA devices")


def _mod(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"{name} unavailable: {exc}")


def _ext(name):
    mod = _mod(f"moe_infinity.{name}")
    return mod


def _pro6000():
    from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports

    if not supports(RTX_PRO_6000, DEV):
        pytest.skip("RTX PRO 6000 kernels only")


def _bf16(*shape, scale=1.0):
    return (torch.randn(*shape, device=f"cuda:{DEV}") * scale).bfloat16()


# --- cases: name -> builder returning (callable, args, kwargs) -------------


def case_launch_softmax():
    m = _mod("moe_infinity.kernel.router")
    x = torch.randn(8, 64, device=f"cuda:{DEV}")
    return m.launch_softmax, (x,), {}


def case_launch_fused_softmax_topk_nobias():
    m = _mod("moe_infinity.kernel.router")
    return (
        m.launch_fused_softmax_topk_nobias,
        (_bf16(4, 256), _bf16(64, 256, scale=0.05), 8, True),
        {},
    )


def case_fused_route():
    _pro6000()
    m = _mod("moe_infinity.kernel.router")
    return (
        m.fused_route,
        (_bf16(1, 512), _bf16(64, 512, scale=0.05), 8),
        {"backend": "triton", "dense": True},
    )


def case_fused_ffn():
    m = _mod("moe_infinity.kernel.fused_ffn")
    return (
        m.fused_ffn,
        (
            _bf16(4, 1024),
            _bf16(2048, 1024, scale=0.03),
            _bf16(2048, 1024, scale=0.03),
            _bf16(1024, 2048, scale=0.03),
        ),
        {},
    )


def case_fused_qkv_proj():
    m = _mod("moe_infinity.kernel.fused_qkv")
    return (
        m.fused_qkv_proj,
        (_bf16(4, 1024), _bf16(1024, 1536, scale=0.03), 16, 4, 64),
        {},
    )


def case_gqa_decode_attention_pro6000():
    _pro6000()
    m = _mod("moe_infinity.kernel.gqa_attention")
    q = _bf16(1, 8, 1, 64)
    k = _bf16(1, 2, 96, 64)
    v = _bf16(1, 2, 96, 64)
    return m.gqa_decode_attention_pro6000, (q, k, v, 0.125), {}


def case_fused_decode_attention():
    m = _mod("moe_infinity.kernel.fused_decode_attn")
    blocks, bs, kvh, hd = 8, 16, 2, 64
    kc = _bf16(blocks, bs, kvh, hd)
    vc = _bf16(blocks, bs, kvh, hd)
    tables = torch.arange(blocks, dtype=torch.int32, device=f"cuda:{DEV}")
    tables = tables.view(2, 4)
    seq_lens = torch.tensor([50, 33], dtype=torch.int32, device=f"cuda:{DEV}")
    q = _bf16(2, 1, 8, hd)
    return m.fused_decode_attention, (q, kc, vc, tables, seq_lens, 0.125), {}


def case_paged_kv_scatter():
    m = _mod("moe_infinity.kernel.paged_kv_write")
    blocks, heads, d, bs, x = 4, 2, 64, 16, 8
    dev = f"cuda:{DEV}"
    k_cache = torch.zeros(
        blocks, heads, d // x, bs, x, dtype=torch.bfloat16, device=dev
    )
    v_cache = torch.zeros(
        blocks, heads, d, bs, dtype=torch.bfloat16, device=dev
    )
    key = _bf16(5, heads, d)
    value = _bf16(5, heads, d)
    slots = torch.tensor([3, 4, 17, 18, 40], dtype=torch.int64, device=dev)

    def call(k_cache, v_cache, key, value, slots):
        m.paged_kv_scatter_(k_cache, v_cache, key, value, slots)
        return k_cache.clone(), v_cache.clone()

    return call, (k_cache, v_cache, key, value, slots), {}


def case_native_silu_and_mul():
    store = _ext("_store")
    x = _bf16(4, 256)
    out = torch.empty(4, 128, dtype=torch.bfloat16, device=f"cuda:{DEV}")

    def call(out, x):
        store.silu_and_mul(out, x)
        return out.clone()

    return call, (out, x), {}


def case_native_fused_moe_ffn_into():
    store = _ext("_store")
    dev = f"cuda:{DEV}"
    h, inter = 256, 128
    args = (
        _bf16(2, h),
        _bf16(inter, h, scale=0.05),
        _bf16(inter, h, scale=0.05),
        _bf16(h, inter, scale=0.05),
    )

    def call(x, g, u, d):
        gate_buf = torch.empty(2, inter, dtype=torch.bfloat16, device=dev)
        fused = torch.empty(2, inter, dtype=torch.bfloat16, device=dev)
        out = torch.zeros(2, h, dtype=torch.bfloat16, device=dev)
        store.fused_moe_ffn_into(x, g, u, d, gate_buf, fused, out)
        return out

    return call, args, {}


def case_native_grouped_moe_decode_pro6000():
    _pro6000()
    store = _ext("_store")
    if not hasattr(store, "grouped_moe_decode_pro6000"):
        pytest.skip("built without grouped_moe_decode_pro6000")
    dev = f"cuda:{DEV}"
    tokens, experts, hidden, inter = 1, 8, 256, 128
    ws = [
        tuple(
            _bf16(*s, scale=0.05)
            for s in ((inter, hidden), (inter, hidden), (hidden, inter))
        )
        for _ in range(4)
    ]
    table = torch.tensor(
        [[t.data_ptr() for t in w] + [e] for e, w in enumerate(ws)],
        dtype=torch.int64,
        device=dev,
    )
    mask = torch.zeros(tokens, experts, dtype=torch.bool, device=dev)
    mask[:, :4] = True
    weights = mask.float() * 0.25
    x = _bf16(tokens, hidden)

    def call(x, mask, weights, table):
        out = torch.zeros(tokens, hidden, device=dev)
        store.grouped_moe_decode_pro6000(
            x, mask, weights, table, inter, 4, out, zero_out=True
        )
        return out

    return call, (x, mask, weights, table), {"_keep": ws}


def case_native_router_fused_route():
    _pro6000()
    router = _ext("_router")
    return (
        router.fused_route,
        (
            _bf16(1, 512),
            _bf16(64, 512, scale=0.05),
            8,
            0,  # softmax
            None,
            1,
            1,
            True,
            1.0,
            0.0,
            True,
            torch.bfloat16,
            torch.bfloat16,
            False,
            True,
            False,
            True,
            0,
        ),
        {},
    )


def case_native_engine_fused_silu_gemm():
    engine = _ext("_engine")
    return (
        engine.fused_silu_gemm,
        (_bf16(4, 128), _bf16(512, 128, scale=0.05)),
        {},
    )


def case_native_v4_fp8_dequant_blockwise():
    v4 = _ext("_v4_fp4")
    dev = f"cuda:{DEV}"
    w = torch.randint(0, 120, (128, 256), dtype=torch.uint8, device=dev)
    s = torch.rand(1, 2, device=dev) + 0.5
    return v4.fp8_dequant_blockwise, (w, s), {}


CASES = {
    name[len("case_") :]: fn
    for name, fn in sorted(globals().items())
    if name.startswith("case_")
}


def _flatten(out):
    if isinstance(out, torch.Tensor):
        return [out.detach().clone()]
    if isinstance(out, (list, tuple)):
        flat = []
        for item in out:
            flat.extend(_flatten(item))
        return flat
    return []


def _run(call, args, kwargs):
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    out = call(*args, **kwargs)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    return _flatten(out)


@pytest.mark.parametrize("name", sorted(CASES))
def test_entry_point_runs_on_tensor_device(name):
    _need_two_gpus()
    torch.manual_seed(1234)
    call, args, kwargs = CASES[name]()
    # Warm-up with the right device current (JIT compile, autotune, handles).
    with torch.cuda.device(DEV):
        expected = _run(call, args, kwargs)

    torch.cuda.set_device(0)
    from torch.profiler import ProfilerActivity, profile

    # A non-default current stream on device 0: a kernel that picks "the
    # current stream" of the wrong device now hands a device-0 stream to a
    # launch on device 1 instead of (accidentally valid) NULL.
    side0 = torch.cuda.Stream(device=0)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.cuda.stream(side0):
            actual = _run(call, args, kwargs)
    assert torch.cuda.current_device() == 0, "entry point changed the device"

    devices = {
        ev.device_index
        for ev in prof.events()
        if ev.device_type == torch.autograd.DeviceType.CUDA
    }
    assert devices, "profiler saw no CUDA activity"
    assert devices == {DEV}, f"kernels ran on devices {sorted(devices)}"
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if got.dtype.is_floating_point:
            nan_equal = torch.equal(torch.isnan(got), torch.isnan(want))
            assert nan_equal
            mask = ~torch.isnan(got)
            assert torch.equal(got[mask], want[mask])
        else:
            assert torch.equal(got, want)


def test_device_guard_helper_restores_device():
    _need_two_gpus()
    from moe_infinity.kernel._device import device_guard

    seen = []

    @device_guard
    def probe(t, extra=None):
        seen.append(torch.cuda.current_device())
        return t

    torch.cuda.set_device(0)
    probe(torch.zeros(1, device="cuda:1"))
    probe([torch.zeros(1, device="cuda:1")])  # nested in a list
    probe(None, extra=torch.zeros(1, device="cuda:1"))  # keyword
    probe(torch.zeros(1))  # CPU: untouched
    assert seen == [1, 1, 1, 0]
    assert torch.cuda.current_device() == 0
