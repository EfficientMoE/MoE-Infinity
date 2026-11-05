"""The fused routing kernels (#315) are tuned for the RTX PRO 6000: on any
other GPU the routing must fall back to the previous torch implementation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from moe_infinity.kernel import router as R  # noqa: E402
from moe_infinity.utils import hw_profile  # noqa: E402

MIB = 1024 * 1024
KIB = 1024

DEVICES = {
    "rtx_pro_6000": SimpleNamespace(
        major=12,
        minor=0,
        multi_processor_count=188,
        L2_cache_size=128 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "rtx_5090": SimpleNamespace(
        major=12,
        minor=0,
        multi_processor_count=170,
        L2_cache_size=96 * MIB,
        shared_memory_per_block_optin=99 * KIB,
    ),
    "h100": SimpleNamespace(
        major=9,
        minor=0,
        multi_processor_count=132,
        L2_cache_size=50 * MIB,
        shared_memory_per_block_optin=227 * KIB,
    ),
}


@pytest.fixture
def fake_gpu(monkeypatch):
    def use(name):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: DEVICES[name]
        )
        hw_profile._gates.cache_clear()
        R._PRO6000_ROUTING.clear()

    monkeypatch.delenv("MOE_INFINITY_UNVALIDATED_HW", raising=False)
    yield use
    hw_profile._gates.cache_clear()
    R._PRO6000_ROUTING.clear()


@pytest.mark.parametrize(
    "name, expected",
    [("rtx_pro_6000", True), ("rtx_5090", False), ("h100", False)],
)
def test_router_gate(fake_gpu, name, expected):
    fake_gpu(name)
    assert R._pro6000_routing(torch.device("cuda", 0)) is expected
    assert R.router_supported("cuda:0") is expected
    assert R._pro6000_routing(torch.device("cpu")) is False


def test_unvalidated_hw_lifts_gate(fake_gpu, monkeypatch):
    fake_gpu("rtx_5090")
    monkeypatch.setenv("MOE_INFINITY_UNVALIDATED_HW", "1")
    assert R.router_supported(torch.device("cuda", 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("name", ["rtx_5090", "h100"])
def test_other_gpus_take_previous_routing(fake_gpu, monkeypatch, name):
    """Real CUDA tensors, device reported as another GPU: neither the
    extension nor a Triton kernel is reached; fused_route and make_router run
    the torch reference."""
    x = torch.randn(5, 64, device="cuda").to(torch.bfloat16)
    w = torch.randn(16, 64, device="cuda").to(torch.bfloat16)
    fake_gpu(name)

    class _NoKernels:
        def __getattr__(self, attr):
            raise AssertionError(f"tuned kernel used on {name}: {attr}")

    def _fail(*a, **k):
        raise AssertionError(f"tuned kernel used on {name}")

    monkeypatch.setattr(R, "_CUDA_EXT", _NoKernels())
    monkeypatch.setattr(R, "_route_prefill", _fail)
    monkeypatch.setattr(R, "_fused_route_triton", _fail)
    calls = []
    real = R._route_reference
    monkeypatch.setattr(
        R, "_route_reference", lambda *a: calls.append(1) or real(*a)
    )
    for backend in ("auto", "cuda", "triton", "prefill"):
        R.fused_route(x, w, 4, backend=backend)
    R.make_router(w, 4, active_flags=True)(x)
    assert len(calls) == 5


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("n_group", [1, 4])
def test_reference_matches_torch_routers(scoring, n_group):
    torch.manual_seed(1)
    T, H, E, k = 7, 32, 16, 4
    x = torch.randn(T, H)
    w = torch.randn(E, H)
    bias = torch.randn(E) if scoring != "softmax" else None
    out = R.fused_route(
        x,
        w,
        k,
        scoring=scoring,
        bias=bias,
        n_group=n_group,
        topk_group=2 if n_group > 1 else 1,
        eps=1e-20,
        return_logits=True,
    )
    logits = F.linear(x, w)
    if scoring == "softmax":
        scores = logits.softmax(-1)
        sel = scores.clone()
    else:
        scores = (
            logits.sigmoid()
            if scoring == "sigmoid"
            else F.softplus(logits).sqrt()
        )
        sel = scores + bias
    if n_group > 1:
        g = sel.view(T, n_group, -1).topk(2, -1)[0].sum(-1)
        keep = torch.zeros_like(g).scatter_(1, g.topk(2, -1)[1], 1.0)
        sel = sel.masked_fill(
            keep.repeat_interleave(E // n_group, 1) == 0, float("-inf")
        )
    idx = sel.topk(k, -1)[1]
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
    assert torch.equal(out.topk_ids.long(), idx)
    torch.testing.assert_close(out.topk_weights, wts)
    assert out.router_mask.sum().item() == T * k
    torch.testing.assert_close(out.routing_weights.gather(1, idx).float(), wts)
    torch.testing.assert_close(out.logits.float(), logits)
