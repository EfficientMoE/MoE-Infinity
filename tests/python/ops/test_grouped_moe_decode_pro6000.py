"""Grouped decode MoE kernels (#306): parity with the per-expert path.

RTX PRO 6000 only (gate RTX_PRO_6000); skipped on other GPUs, where the
dispatcher keeps per-expert execution (tests/python/unit/
test_grouped_moe_pro6000_gate.py covers that choice on the CPU).
"""

import gc
import time
from unittest.mock import MagicMock

import pytest
import torch

from moe_infinity.utils.hw_profile import RTX_PRO_6000, supports
from tests.python.ops.conftest import requires_cuda

requires_pro6000 = pytest.mark.skipif(
    not supports(RTX_PRO_6000), reason="grouped decode is RTX PRO 6000 only"
)

DEEPSEEK_EXPERT_TYPE = 5
DTYPE_BF16 = 0


def _store():
    try:
        import moe_infinity._store as store
    except Exception as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"moe_infinity._store unavailable: {exc}")
    if isinstance(store, MagicMock) or not hasattr(
        store, "grouped_moe_decode_pro6000"
    ):
        pytest.skip(
            "moe_infinity._store built without grouped_moe_decode_pro6000"
        )
    return store


def _per_expert_reference(x, mask, weights, experts):
    """fused_moe_ffn_into + OutputFunc math, GEMMs in fp32.

    Rounding points: bf16 gate, bf16 silu(gate) * up, bf16 down output,
    fp32 routing-weight multiply and accumulate.
    """
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e, (gate, up, down) in experts.items():
        idx = mask[:, e].nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        xe = x[idx].float()
        g = (xe @ gate.float().T).bfloat16().float()
        u = xe @ up.float().T
        a = (g * (1.0 / (1.0 + torch.exp(-g))) * u).bfloat16().float()
        o = (a @ down.float().T).bfloat16().float()
        out.index_add_(0, idx, o * weights[idx, e].float().unsqueeze(1))
    return out


def _routing(tokens, num_experts, top_k, generator, device):
    logits = torch.randn(
        tokens, num_experts, generator=generator, device=device
    )
    top = logits.topk(top_k, dim=1).indices
    mask = torch.zeros(tokens, num_experts, dtype=torch.bool, device=device)
    mask.scatter_(1, top, True)
    weights = torch.softmax(logits, dim=1) * mask
    return mask, weights


def _assert_bf16_close(actual, expected):
    # The kernels and the reference sum in different orders, so a bf16
    # rounding point can flip by one ulp; bound by a few bf16 ulps of the
    # output scale.
    scale = expected.abs().max().clamp_min(1e-6)
    err = (actual - expected).abs().max()
    assert err <= 4 * 2**-8 * scale, f"max |diff| {err} vs scale {scale}"


@requires_cuda
@requires_pro6000
@pytest.mark.parametrize(
    "tokens,num_experts,top_k,hidden,inter",
    [
        (1, 16, 4, 256, 128),
        (3, 8, 2, 512, 96),
        (16, 64, 8, 1024, 256),
        (64, 16, 4, 512, 512),
        (256, 8, 2, 256, 128),
        (200, 128, 8, 768, 160),
    ],
)
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_grouped_kernel_matches_per_expert_math(
    tokens, num_experts, top_k, hidden, inter, weight_dtype
):
    store = _store()
    gen = torch.Generator(device="cuda").manual_seed(tokens * 131 + hidden)
    mask, weights = _routing(tokens, num_experts, top_k, gen, "cuda")
    weights = weights.to(weight_dtype)
    x = torch.randn(tokens, hidden, generator=gen, device="cuda").bfloat16()

    active = mask.any(0).nonzero().squeeze(1).tolist()
    experts = {}
    for e in active:
        experts[e] = tuple(
            (torch.randn(*s, generator=gen, device="cuda") * 0.05).bfloat16()
            for s in ((inter, hidden), (inter, hidden), (hidden, inter))
        )
    table = torch.tensor(
        [[t.data_ptr() for t in experts[e]] + [e] for e in active],
        dtype=torch.int64,
        device="cuda",
    )
    expected = _per_expert_reference(x, mask, weights, experts)

    out = torch.full((tokens, hidden), 3.0, device="cuda")
    store.grouped_moe_decode_pro6000(
        x, mask, weights, table, inter, top_k, out, zero_out=True
    )
    _assert_bf16_close(out, expected)

    # Accumulates into the existing output (dispatcher contract).
    store.grouped_moe_decode_pro6000(x, mask, weights, table, inter, top_k, out)
    _assert_bf16_close(out, 2 * expected)


@requires_cuda
@requires_pro6000
def test_grouped_kernel_subset_of_experts_and_strided_inputs():
    """Experts outside the table are left to the per-expert path."""
    store = _store()
    gen = torch.Generator(device="cuda").manual_seed(7)
    tokens, num_experts, top_k, hidden, inter = 12, 32, 4, 256, 64
    mask, weights = _routing(tokens, num_experts, top_k, gen, "cuda")
    # Non-contiguous router tensors (column slices of wider buffers).
    mask_wide = torch.zeros(
        tokens, 2 * num_experts, dtype=torch.bool, device="cuda"
    )
    mask_wide[:, ::2] = mask
    weights_wide = torch.zeros(tokens, 2 * num_experts, device="cuda")
    weights_wide[:, ::2] = weights
    x_wide = torch.randn(
        tokens, 2 * hidden, generator=gen, device="cuda"
    ).bfloat16()
    x = x_wide[:, :hidden]

    active = mask.any(0).nonzero().squeeze(1).tolist()
    grouped = active[::2]
    experts = {
        e: tuple(
            (torch.randn(*s, generator=gen, device="cuda") * 0.05).bfloat16()
            for s in ((inter, hidden), (inter, hidden), (hidden, inter))
        )
        for e in grouped
    }
    table = torch.tensor(
        [[t.data_ptr() for t in experts[e]] + [e] for e in grouped],
        dtype=torch.int64,
        device="cuda",
    )
    out = torch.zeros(tokens, hidden, device="cuda")
    store.grouped_moe_decode_pro6000(
        x, mask_wide[:, ::2], weights_wide[:, ::2], table, inter, top_k, out
    )
    _assert_bf16_close(out, _per_expert_reference(x, mask, weights, experts))


def _make_dispatcher(tmp_path, num_experts, hidden, inter, generator):
    try:
        from moe_infinity.runtime.model_offload import _load_prefetch_lib

        prefetch_lib = _load_prefetch_lib()
    except Exception as exc:
        pytest.skip(f"native Archer extension unavailable: {exc}")
    if isinstance(prefetch_lib, MagicMock):
        pytest.skip("native Archer extension was stubbed during collection")
    if not hasattr(prefetch_lib.expert_dispatcher, "set_grouped_decode"):
        pytest.skip("native extension built without the grouped decode path")

    engine = prefetch_lib.prefetch_handle(f"{tmp_path}/", 0.5)
    experts = {}
    expert_tensor_ids = []
    for e in range(num_experts):
        weights = tuple(
            (torch.randn(*s, generator=generator) * 0.05).bfloat16()
            for s in ((inter, hidden), (inter, hidden), (hidden, inter))
        )
        ids = [3 * e, 3 * e + 1, 3 * e + 2]
        for tensor_id, tensor in zip(ids, weights):
            engine.offload(tensor, tensor_id)
        experts[e] = tuple(t.cuda() for t in weights)
        expert_tensor_ids.append(ids)
    dense_ids = list(
        range(3 * num_experts, 3 * num_experts + 2 * torch.cuda.device_count())
    )
    for tensor_id in dense_ids:
        engine.offload(torch.zeros(1, dtype=torch.bfloat16), tensor_id)
    split = len(dense_ids) // 2
    engine.set_topology(
        [(f"model.dense.{i}", [[t]]) for i, t in enumerate(dense_ids[:split])]
        + [("model.layers.0.mlp.experts", expert_tensor_ids)]
        + [
            (f"model.dense.{i + split}", [[t]])
            for i, t in enumerate(dense_ids[split:])
        ]
    )
    dispatcher = prefetch_lib.expert_dispatcher(
        num_experts, 1, DTYPE_BF16, DEEPSEEK_EXPERT_TYPE, 4
    )
    for e, ids in enumerate(expert_tensor_ids):
        dispatcher.register_expert(0, e, ids, "")
    return engine, dispatcher, experts


@pytest.fixture
def native_moe(tmp_path):
    """Builds engine + dispatcher; tears down the dispatcher (and its worker
    threads) before the engine whose globals they use."""
    made = []

    def make(num_experts, hidden, inter, generator):
        engine, dispatcher, experts = _make_dispatcher(
            tmp_path, num_experts, hidden, inter, generator
        )
        made.append([engine, dispatcher])
        return dispatcher, experts

    yield make
    while made:
        pair = made.pop()
        pair[1] = None
        gc.collect()
        pair[0] = None
        gc.collect()


def _dispatch(dispatcher, x, mask, weights, native):
    dispatcher.set_inputs(x, mask, weights)
    if native:
        dispatcher.dispatch_experts(0)
    else:
        active = mask.any(0).nonzero().squeeze(1).tolist()
        dispatcher.set_expected_queue(len(active))
        for e in active:
            dispatcher.enqueue_expert(0, e, 0, False)
        dispatcher.notify_fetch_start()
    try:
        return dispatcher.wait_expert().clone()
    finally:
        # Experts return to IDLE on the retirement thread after the GPU work;
        # enqueueing the same expert again before that is a fatal CAS failure
        # (real models never reuse a layer's expert within one step), so
        # back-to-back test dispatches wait for retirement.
        torch.cuda.synchronize()
        time.sleep(0.05)


@requires_cuda
@requires_pro6000
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("tokens", [1, 5])
def test_dispatcher_grouped_path_matches_per_expert_path(
    native_moe, native, tokens
):
    torch.cuda.set_device(0)
    num_experts, top_k, hidden, inter = 8, 2, 256, 128
    gen = torch.Generator().manual_seed(11)
    dispatcher, experts = native_moe(num_experts, hidden, inter, gen)
    dispatcher.set_grouped_decode(True, top_k)
    x = torch.randn(tokens, hidden, generator=gen).bfloat16().cuda()
    mask, weights = _routing(tokens, num_experts, top_k, None, "cpu")
    mask, weights = mask.cuda(), weights.to(torch.bfloat16).cuda()
    expected = _per_expert_reference(x, mask, weights, experts)

    # First call fetches the experts: per-expert path.
    cold = _dispatch(dispatcher, x, mask, weights, native)
    assert dispatcher.get_routing_stats()["grouped_experts"] == 0
    # Now resident: one grouped launch pair.
    warm = _dispatch(dispatcher, x, mask, weights, native)
    stats = dispatcher.get_routing_stats()
    active = int(mask.any(0).sum())
    assert stats["grouped_experts"] == active
    assert stats["grouped_batches"] == 1
    # Resident again with the grouped path disabled: per-expert CUTLASS.
    dispatcher.set_grouped_decode(False, top_k)
    per_expert = _dispatch(dispatcher, x, mask, weights, native)
    assert dispatcher.get_routing_stats()["grouped_experts"] == active

    _assert_bf16_close(cold, expected)
    _assert_bf16_close(per_expert, expected)
    _assert_bf16_close(warm, per_expert)


@requires_cuda
@requires_pro6000
def test_dispatcher_mixes_grouped_and_fetched_experts(native_moe):
    """Resident experts run grouped while in-flight ones use the old path."""
    torch.cuda.set_device(0)
    num_experts, hidden, inter, tokens = 8, 256, 64, 6
    gen = torch.Generator().manual_seed(5)
    dispatcher, experts = native_moe(num_experts, hidden, inter, gen)
    dispatcher.set_grouped_decode(True, 4)
    x = torch.randn(tokens, hidden, generator=gen).bfloat16().cuda()

    warm_mask = torch.zeros(tokens, num_experts, dtype=torch.bool)
    warm_mask[:, :2] = True
    warm_mask = warm_mask.cuda()
    _dispatch(dispatcher, x, warm_mask, warm_mask.float(), native=False)

    mask = torch.zeros(tokens, num_experts, dtype=torch.bool)
    mask[:, [0, 1, 4, 5]] = True  # 0, 1 resident; 4, 5 fetched
    mask = mask.cuda()
    weights = torch.rand(tokens, num_experts, generator=gen).cuda() * mask
    out = _dispatch(dispatcher, x, mask, weights, native=False)
    assert dispatcher.get_routing_stats()["grouped_experts"] == 2
    _assert_bf16_close(out, _per_expert_reference(x, mask, weights, experts))


@requires_cuda
@requires_pro6000
@pytest.mark.parametrize(
    "fault", ["exec", "output", "completion_event", "retirement_launch"]
)
def test_grouped_path_failure_rethrows_and_recovers(native_moe, fault):
    torch.cuda.set_device(0)
    num_experts, hidden, inter, tokens = 4, 256, 64, 3
    gen = torch.Generator().manual_seed(3)
    dispatcher, experts = native_moe(num_experts, hidden, inter, gen)
    dispatcher.set_grouped_decode(True, 2)
    x = torch.randn(tokens, hidden, generator=gen).bfloat16().cuda()
    mask = torch.zeros(tokens, num_experts, dtype=torch.bool)
    mask[:, :2] = True
    mask = mask.cuda()
    weights = mask.float() * 0.5
    _dispatch(dispatcher, x, mask, weights, native=True)  # make resident

    dispatcher._set_dispatch_fault_for_test(fault)
    with pytest.raises(RuntimeError, match=f"injected {fault} failure"):
        _dispatch(dispatcher, x, mask, weights, native=True)
    stats = dispatcher.get_routing_stats()
    assert stats["pending"] == 0 and stats["route_pending"] == 0

    out = _dispatch(dispatcher, x, mask, weights, native=True)
    _assert_bf16_close(out, _per_expert_reference(x, mask, weights, experts))
    assert dispatcher.get_routing_stats()["grouped_batches"] >= 1
